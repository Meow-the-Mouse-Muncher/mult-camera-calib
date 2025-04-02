
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
def dcn_warp(voxelgrid: Tensor, flow_x: Tensor, flow_y: Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs, ts, H, W = voxelgrid.shape
    flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,ts,2,H,W]
    flow = flow.reshape(bs, ts * 2, H, W)  # [bs,ts*2,H,W]
    #  单位矩阵 保证了 只对一张图处理
    weight = torch.eye(ts, device=flow.device).float().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
    # 单位卷积核
    return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,ts,H,W]

def undistort_image(img, K, dist_coeffs):
    """
    去除图像畸变
    """
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
    return undistorted

def main():
    # 创建输出文件夹
    os.makedirs('Outputs/flir', exist_ok=True)
    os.makedirs('Outputs/overlay', exist_ok=True)
    
    # 加载相机参数
    stereo_params = io.loadmat('matlab/stereo_camera_parameters.mat')['stereo_params']
    
    # 提取相机内参和畸变系数
    # 提取相机内参和畸变系数
    K_flir = stereo_params['K2'][0, 0]  
    K_event = stereo_params['K1'][0, 0]  
    dist_flir = np.hstack([
        stereo_params['RadialDistortion2'][0, 0].flatten(), 
        stereo_params['TangentialDistortion2'][0, 0].flatten()
    ])
    
    dist_event = np.hstack([
        stereo_params['RadialDistortion1'][0, 0].flatten(), 
        stereo_params['TangentialDistortion1'][0, 0].flatten()
    ])
    # 获取相机投影矩阵
    P_flir = stereo_params['P_flir'][0, 0]  # 世界到flir相机投影矩阵
    P_event = stereo_params['P_event'][0, 0]  # 世界到event相机投影矩阵

    # 获取所有图像文件
    event_files = sorted(glob.glob('evk_result/*.png'))  # 读取event图像文件
    flir_files = sorted(glob.glob('flir_result/*.png'))  # 读取flir图像文件
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
    print("读取所有flir图像...")
    all_images = []
    for flir_file in tqdm(flir_files):
        # 读取event图像为灰度图
        I_flir = cv2.imread(flir_file, cv2.IMREAD_GRAYSCALE)  # 灰度图
        # 如果图像不是,则调整大小
        if I_flir.shape[0] != flir_h or I_flir.shape[1] != flir_w:
            I_flir = cv2.resize(I_flir, (flir_h, flir_w))
        # 去畸变
        I_flir = undistort_image(I_flir, K_flir, dist_event)
        all_images.append(I_flir)

    # 创建输出文件夹
    os.makedirs('Outputs/flir', exist_ok=True)  # 为flir图像创建输出文件夹
    os.makedirs('Outputs/overlay_f2e', exist_ok=True)  # 为重叠图像创建输出文件夹
    
    warped_images = []
    for i, img in enumerate(tqdm(all_images, desc="变形处理")):
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
        warped_images.append(result)
        # 获取文件名（不包含路径和扩展名）
        base_name = os.path.splitext(os.path.basename(flir_files[i]))[0]
        # 保存结果
        cv2.imwrite(f'Outputs/flir/flir_in_event_view_{base_name}.png', result)
        # 读取对应的event图像
        if i < len(event_files):
            event_img = cv2.imread(event_files[i], cv2.IMREAD_GRAYSCALE)
            # 创建彩色重叠图像
            overlay_img = np.zeros((event_h, event_w, 3), dtype=np.uint8)
            # 将变形后的FLIR图像放在绿色通道
            overlay_img[:, :, 1] = result
            # 将event图像放在蓝色通道
            overlay_img[:, :, 0] = event_img
            # 保存重叠图像
            cv2.imwrite(f'Outputs/overlay_f2e/overlay_{base_name}.png', overlay_img)
            # 可选：创建带有标签的可视化图像
            vis_img = np.zeros((event_h + 30, event_w, 3), dtype=np.uint8)
            vis_img[:event_h, :, :] = overlay_img
    print("\n处理完成！结果已保存到 Outputs 文件夹")
if __name__ == "__main__":
    main()