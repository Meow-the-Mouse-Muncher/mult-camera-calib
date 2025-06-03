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

def main():
    
    # 加载相机参数
    stereo_params = io.loadmat('./stereoParams_FI.mat')['stereoParams']
    
    # 提取相机内参和畸变系数
    # FLIR相机参数
    K_flir = stereo_params['K1'][0, 0]  
    dist_flir = np.hstack([
        stereo_params['RadialDistortion1'][0, 0].flatten(), 
        stereo_params['TangentialDistortion1'][0, 0].flatten()
    ])
    
    # Thermal红外相机参数
    K_thermal = stereo_params['K2'][0, 0]  
    dist_thermal = np.hstack([
        stereo_params['RadialDistortion2'][0, 0].flatten(), 
        stereo_params['TangentialDistortion2'][0, 0].flatten()
    ])
    
    # 获取相机投影矩阵
    R_flir = stereo_params['R1'][0, 0]  # FLIR相机的旋转矩阵
    R_thermal = stereo_params['R2'][0, 0]  # Thermal相机的旋转矩阵
    T_flir = stereo_params['T1'][0, 0]  # FLIR相机的平移向量
    T_thermal = stereo_params['T2'][0, 0]  # Thermal相机的平移向量

    P_flir = K_flir @ np.hstack([R_flir, T_flir.reshape(3, 1)])  # 世界到FLIR相机投影矩阵
    P_thermal = K_thermal @ np.hstack([R_thermal, T_thermal.reshape(3, 1)])  # 世界到Thermal相机投影矩阵

    # 获取所有子目录
    base_path = './data'
    subdirs = [x for x in os.listdir(base_path)]
    subdirs.sort(key=lambda x: tuple(map(int, x.split())))
    
    # 图像尺寸设置
    flir_h, flir_w = 1800, 1800
    thermal_h, thermal_w = 640, 512  # thermal图像尺寸
    
    print("计算投影映射...")
    # 创建thermal图像的坐标网格 (目标视角)
    x_coords, y_coords = np.meshgrid(np.arange(thermal_w), np.arange(thermal_h))
    
    # 默认z平面
    Z = 0
    
    # 使用thermal相机的投影矩阵进行反投影
    p11, p12, p14 = P_thermal[0, 0], P_thermal[0, 1], P_thermal[0, 3]
    p21, p22, p24 = P_thermal[1, 0], P_thermal[1, 1], P_thermal[1, 3]
    p31, p32, p34 = P_thermal[2, 0], P_thermal[2, 1], P_thermal[2, 3]
    
    # 获取thermal图像的所有像素坐标
    u = x_coords.flatten()
    v = y_coords.flatten()

    print("求解线性方程组...")
    # 分批处理以减少内存使用
    batch_size = 100000  # 每批处理的像素点数量
    num_batches = (len(u) + batch_size - 1) // batch_size
    
    X_result = np.zeros_like(u, dtype=np.float64)
    Y_result = np.zeros_like(u, dtype=np.float64)
    lambda_result = np.zeros_like(u, dtype=np.float64)
    
    for i in tqdm(range(num_batches), desc="处理像素批次"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(u))
        
        # 当前批次的像素坐标
        u_batch = u[start_idx:end_idx]
        v_batch = v[start_idx:end_idx]
        
        # 构建当前批次的系数矩阵 [batch_size, 3, 3]
        A_batch = np.zeros((end_idx - start_idx, 3, 3), dtype=np.float64)
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
        
        try:
            # 求解当前批次的线性方程组
            X_batch = np.linalg.solve(A_batch, b_batch)
            
            # 保存结果
            X_result[start_idx:end_idx] = X_batch[:, 0]
            Y_result[start_idx:end_idx] = X_batch[:, 1]
            lambda_result[start_idx:end_idx] = X_batch[:, 2]
        except np.linalg.LinAlgError:
            print(f"警告: 批次 {i} 中存在奇异矩阵，跳过该批次")
            continue

    # 构建世界坐标 [X,Y,Z,1]
    world_coords = np.zeros((4, len(u)), dtype=np.float64)
    world_coords[0, :] = X_result
    world_coords[1, :] = Y_result
    world_coords[2, :] = Z  # Z平面坐标
    world_coords[3, :] = 1  # 齐次坐标
    
    # 投影到FLIR相机坐标系
    print("投影到FLIR相机坐标系...")
    flir_proj = P_flir @ world_coords
    
    # 检查分母是否为零
    denominator = flir_proj[2, :]
    valid_mask = np.abs(denominator) > 1e-10
    
    # 齐次坐标归一化（只对有效点）
    flir_proj_normalized = np.zeros_like(flir_proj)
    flir_proj_normalized[:, valid_mask] = flir_proj[:, valid_mask] / denominator[valid_mask]
    
    # 获取映射坐标（从thermal视角到FLIR图像的映射）
    map_x = flir_proj_normalized[0, :].reshape(thermal_h, thermal_w)
    map_y = flir_proj_normalized[1, :].reshape(thermal_h, thermal_w)
    
    # 创建有效性掩码
    valid_mapping_mask = valid_mask.reshape(thermal_h, thermal_w)
    
    '''
    解释投影过程：
    1. 对于thermal图像的每个像素点(u,v)
    2. 通过thermal相机的投影矩阵反投影到世界坐标系的Z=0平面
    3. 再通过FLIR相机的投影矩阵投影到FLIR图像坐标
    4. 得到从thermal视角采样FLIR图像的映射关系
    '''
    
    # 处理所有子目录
    for subdir in tqdm(subdirs, desc="处理子文件夹"):
        subdir_path = os.path.join(base_path, subdir)
        
        # FLIR图像路径
        flir_dir = os.path.join(subdir_path, 'flir')
        flir_files = glob.glob(os.path.join(flir_dir, '*.png'))
        flir_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        # Thermal图像路径
        thermal_dir = os.path.join(subdir_path, 'thermal')
        thermal_files = glob.glob(os.path.join(thermal_dir, '*.png'))
        thermal_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        if not flir_files:
            print(f"警告：在 {flir_dir} 中未找到FLIR图像")
            continue
            
        if not thermal_files:
            print(f"警告：在 {thermal_dir} 中未找到thermal图像")
            continue
        
        num_images = min(len(flir_files), len(thermal_files))
        print(f"处理子目录 {subdir}：{num_images} 张图像")
        
        # 处理图像对
        for i in range(num_images):
            # 读取FLIR图像
            I_flir = cv2.imread(flir_files[i], cv2.IMREAD_GRAYSCALE)
            if I_flir.shape[0] != flir_h or I_flir.shape[1] != flir_w:
                I_flir = cv2.resize(I_flir, (flir_w, flir_h))
            
            # 去畸变FLIR图像
            I_flir_undistorted = undistort_image(I_flir, K_flir, dist_flir)
            
            # 读取thermal图像（作为参考）
            I_thermal = cv2.imread(thermal_files[i], cv2.IMREAD_GRAYSCALE)
            if I_thermal.shape[0] != thermal_h or I_thermal.shape[1] != thermal_w:
                I_thermal = cv2.resize(I_thermal, (thermal_w, thermal_h))
            
            # 去畸变thermal图像
            # I_thermal_undistorted = undistort_image(I_thermal, K_thermal, dist_thermal)
            I_thermal_undistorted = I_thermal  # 假设thermal图像没有畸变·
            # 创建输出目录
            os.makedirs(f'projection_outputs/flir_to_thermal/{subdir}', exist_ok=True)
            os.makedirs(f'projection_outputs/thermal_undistorted/{subdir}', exist_ok=True)  # 新增thermal目录
            os.makedirs(f'projection_outputs/overlay_f2t/{subdir}', exist_ok=True)
            
            # 将FLIR图像映射到thermal视角
            map_x_float = map_x.astype(np.float32)
            map_y_float = map_y.astype(np.float32)
            
            # 使用remap将FLIR图像变换到thermal视角
            flir_in_thermal_view = cv2.remap(
                I_flir_undistorted, 
                map_x_float, 
                map_y_float, 
                cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            # 创建有效区域掩码
            mask = ((map_x >= 0) & (map_x < flir_w) & 
                    (map_y >= 0) & (map_y < flir_h) &
                    valid_mapping_mask)
            mask = mask.astype(np.uint8) * 255
            
            # 应用掩码
            flir_in_thermal_view = cv2.bitwise_and(flir_in_thermal_view, flir_in_thermal_view, mask=mask)
            
            # 保存映射后的FLIR图像
            cv2.imwrite(f'projection_outputs/flir_to_thermal/{subdir}/{i:04d}.png', flir_in_thermal_view)
            
            # 保存去畸变后的thermal图像
            cv2.imwrite(f'projection_outputs/thermal_undistorted/{subdir}/{i:04d}.png', I_thermal_undistorted)
            
            # 创建叠加图像（thermal为绿色，FLIR为红色）
            overlay = np.zeros((thermal_h, thermal_w, 3), dtype=np.uint8)
            overlay[:, :, 1] = I_thermal_undistorted  # 绿色通道：thermal图像
            overlay[:, :, 2] = flir_in_thermal_view   # 红色通道：映射后的FLIR图像
            
            # 保存叠加结果
            cv2.imwrite(f"projection_outputs/overlay_f2t/{subdir}/{i:04d}.png", overlay)

    print("\n处理完成！结果已保存到 projection_outputs 文件夹")
    print("- flir_to_thermal/: FLIR图像映射到thermal视角")
    print("- thermal_undistorted/: 去畸变后的thermal原图像")  # 新增说明
    print("- overlay_f2t/: thermal(绿)和FLIR(红)的叠加图像")

if __name__ == "__main__":
    main()