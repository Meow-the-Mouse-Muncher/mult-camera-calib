
from tkinter import N
import numpy as np
import cv2
import os
import glob
from scipy import io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch
from torch import Tensor
import torchvision
from metavision_core.event_io import EventsIterator,RawReader
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
def undistort_image(img, K, dist_coeffs):
    """
    去除图像畸变
    """
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
    return undistorted
def create_event_frame(events, height, width, accumulation_time_us=10000):
    """
    将事件数据转换为帧图像
    参数:
    - events: 事件数据
    - height, width: 输出图像尺寸
    - accumulation_time_us: 累积时间(微秒)
    返回:
    - 彩色事件帧图像
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 正事件用红色(0,0,255)，负事件用蓝色(255,0,0)
    pos_events = events[events['p'] == 1]
    neg_events = events[events['p'] == 0]
    
    frame[pos_events['y'], pos_events['x'], 2] = 255  # 红色通道
    frame[neg_events['y'], neg_events['x'], 0] = 255  # 蓝色通道
    
    return frame
def main():
    
    # 加载相机参数
    stereo_params = io.loadmat('./stereoParams_FE.mat')['stereoParams']
    
    # 提取相机内参和畸变系数
    # 提取相机内参和畸变系数
    K_flir = stereo_params['K1'][0, 0]  
    K_event = stereo_params['K2'][0, 0]  
    dist_flir = np.hstack([
        stereo_params['RadialDistortion1'][0, 0].flatten(), 
        stereo_params['TangentialDistortion1'][0, 0].flatten()
    ])
    
    dist_event = np.hstack([
        stereo_params['RadialDistortion2'][0, 0].flatten(), 
        stereo_params['TangentialDistortion2'][0, 0].flatten()
    ])
    # 获取相机投影矩阵
    R_flir = stereo_params['R1'][0, 0]  # flir相机的旋转矩阵
    R_event = stereo_params['R2'][0, 0]  # event相机的旋转矩阵
    T_flir = stereo_params['T1'][0, 0]  # flir相机的平移向量
    T_event = stereo_params['T2'][0, 0]  # event相机的平移向量

    P_flir = K_flir @ np.hstack([R_flir, T_flir.reshape(3, 1)])  # 世界到flir相机投影矩阵
    P_event = K_event @ np.hstack([R_event, T_event.reshape(3, 1)])  # 世界到event相机投影矩阵

   
    # 获取所有子目录
    floder = None
    base_path =f'./data'
    floders = [x for x in os.listdir(base_path)]
    flir_h, flir_w = 1800,1800
    event_h, event_w = 600,600
    print("计算投影映射...")
    # 创建坐标网格 先横后竖 xy
    x_coords,y_coords = np.meshgrid(np.arange(event_w), np.arange(event_h))
    # 默认z平面
    Z=0
    # 创建均匀坐标 [3, h*w]
    p11, p12, p14 = P_event[0, 0], P_event[0, 1], P_event[0, 3]
    p21, p22, p24 = P_event[1, 0], P_event[1, 1], P_event[1, 3]
    p31, p32, p34 = P_event[2, 0], P_event[2, 1], P_event[2, 3]
    # 获取所有像素坐标
    u = x_coords.flatten()
    v = y_coords.flatten()

    print("求解线性方程组...")
    # 分批处理以减少内存使用
    batch_size = 100000  # 每批处理的像素点数量
    num_batches = (len(u) + batch_size - 1) // batch_size
    
    X_result = np.zeros_like(u)
    Y_result = np.zeros_like(u)
    lambda_result = np.zeros_like(u)
    
    for i in tqdm(range(num_batches), desc="处理像素批次"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(u))
        
        # 当前批次的像素坐标
        u_batch = u[start_idx:end_idx]
        v_batch = v[start_idx:end_idx]
        
        # 构建当前批次的系数矩阵 [batch_size, 3, 3]
        A_batch = np.zeros((end_idx - start_idx, 3, 3))
        A_batch[:, 0, 0] = p11
        A_batch[:, 0, 1] = p12
        A_batch[:, 0, 2] = -u_batch
        A_batch[:, 1, 0] = p21
        A_batch[:, 1, 1] = p22
        A_batch[:, 1, 2] = -v_batch
        A_batch[:, 2, 0] = p31
        A_batch[:, 2, 1] = p32
        A_batch[:, 2, 2] = -1
        
        # 构建常数向量 [batch_size, 3]
        b_batch = np.tile([-p14, -p24, -p34], (end_idx - start_idx, 1))
        
        # 求解当前批次的线性方程组
        X_batch = np.linalg.solve(A_batch, b_batch)
        
        # 保存结果
        X_result[start_idx:end_idx] = X_batch[:, 0]
        Y_result[start_idx:end_idx] = X_batch[:, 1]
        lambda_result[start_idx:end_idx] = X_batch[:, 2]

    
    # 构建世界坐标 [X,Y,Z,1]
    world_coords = np.zeros((4, len(u)))
    world_coords[0, :] = X_result
    world_coords[1, :] = Y_result
    world_coords[2, :] = Z  # Z平面坐标
    world_coords[3, :] = 1  # Z平面坐标
    
    # 投影flir相机坐标系
    print("投影到flir相机坐标系...")
    flir_proj = P_flir @ world_coords
    flir_proj = flir_proj / flir_proj[2, :]  # 齐次坐标归一化
    
    # 获取映射坐标
    map_x = flir_proj[0, :].reshape(event_h, event_w)
    map_y = flir_proj[1, :].reshape(event_h, event_w)
    # 这里得到的 坐标问题不大
    
    '''
    开始解释这段代码
    λ * [x; y; 1] = P_flir * [X; Y; Z; 1]   （原始投影方程）
    当 z=0
    λ * [x; y; 1] = P_flir_part * [X; Y] + P_flir(:,4)
    p11*X + p12*Y + p14 = λx
    p21*X + p22*Y + p24 = λy
    p31*X + p32*Y + p34 = λ
    开始重组
    p11*X + p12*Y - λx = -p14
    p21*X + p22*Y - λy = -p24
    p31*X + p32*Y - λ  = -p34
    矩阵形式
    [p11 p12  -u]   [X]   [ -p14 ]
    [p21 p22  -v] * [Y] = [ -p24 ]
    [p31 p32  -1]   [λ]   [ -p34 ]
    '''
    
    # 处理所有flir图像
    for floder in tqdm(floders,desc="处理文件夹"):
        # subdirs = [x for x in os.listdir(os.path.join(base_path, floder))]
        # subdirs.sort(key=lambda x: tuple(map(int, x.split())))
        # for flir_subdir in subdirs:
        flir_dir = os.path.join(base_path,floder, 'flir')
        flir_files = glob.glob(os.path.join(flir_dir, '*.png'))
        flir_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        all_images = []
        for flir_file in flir_files:
            #灰度图
            I_flir = cv2.imread(flir_file, cv2.IMREAD_GRAYSCALE)  # 灰度图
            # 如果图像不是,则调整大小
            if I_flir.shape[0] != flir_h or I_flir.shape[1] != flir_w:
                I_flir = cv2.resize(I_flir, (flir_h, flir_w))
            # 去畸变
            I_flir = undistort_image(I_flir, K_flir, dist_event)
            all_images.append(I_flir)
        for i, img in enumerate(all_images):
            os.makedirs(f'projection_outputs/flir/{floder}', exist_ok=True)  # 为flir图像创建输出文件夹
            # 创建映射矩阵 (float32类型)
            map_x_float = map_x.astype(np.float32)
            map_y_float = map_y.astype(np.float32)
            # 使用OpenCV的remap函数进行图像变形
            warped = cv2.remap(img, map_x_float, map_y_float, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # 创建有效区域掩码
            mask = ((map_x >= 0) & (map_x < img.shape[1]) & 
                    (map_y >= 0) & (map_y < img.shape[0]))
            mask = mask.astype(np.uint8) * 255
            # 应用掩码
            result = cv2.bitwise_and(warped, warped, mask=mask)
            # 获取文件名（不包含路径和扩展名）
            # 保存结果
            cv2.imwrite(f'projection_outputs/flir/{floder}/{i:04d}.png', result)


    print("\n处理完成！结果已保存到 projection_outputs 文件夹")
if __name__ == "__main__":
    main()