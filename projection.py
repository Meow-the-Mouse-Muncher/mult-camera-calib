import numpy as np
import cv2
import os
import glob
from scipy import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

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
    os.makedirs('Outputs/event', exist_ok=True)
    os.makedirs('Outputs/overlay', exist_ok=True)
    
    # 加载相机参数
    stereo_params = io.loadmat('matlab/stereo_camera_parameters.mat')['stereo_params']
    
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

    # 获取投影矩阵
    P_flir = stereo_params['P_flir'][0, 0]
    P_event = stereo_params['P_event'][0, 0]
    
    # 获取所有图像文件
    flir_files = sorted(glob.glob('flir_result/*.png'))
    event_files = sorted(glob.glob('evk_result/*.png'))
    
    # 设置投影平面Z值
    Z = 0
    
    # 预先计算常量矩阵
    P_flir_part = P_flir[:, :2]
    P_flir_const = -Z * P_flir[:, 2:3] - P_flir[:, 3:4]
    
<<<<<<< HEAD
    print("计算投影映射...")
    # 创建坐标网格 先横后竖 xy
    x_coords,y_coords = np.meshgrid(np.arange(flir_w), np.arange(flir_h))
    
    # 创建均匀坐标 [3, h*w]
    flir_pixels = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())], axis=0)
    
    # 使用内参矩阵的逆矩阵将像素坐标转换为归一化相机坐标
    flir_rays = np.matmul(K_flir_inv, flir_pixels)  # [3, h*w]
    
    # 将flir相机坐标系中的射线转换到event相机坐标系
    flir_rays = np.matmul(R_f2e, flir_rays) + T_f2e # [3, h*w]
    # 使用内参矩阵将3d坐标转换为像素坐标
    flir_rays = np.matmul(K_event, flir_rays)  # [3, h*w]
    # 进行归一化 f2e_pixels->u v 1
    f2e_pixels = np.zeros_like(flir_pixels)
    # 0-x 1-y 2-z
    f2e_pixels[0, :] = flir_rays[0, :] / flir_rays[2, :]
    f2e_pixels[1, :] = flir_rays[1, :] / flir_rays[2, :]
    # 重塑回原始形状
    map_x = f2e_pixels[0, :].reshape(flir_h, flir_w)
    map_y = f2e_pixels[1, :].reshape(flir_h, flir_w)
    
    # 计算位移场 (对于DCN需要的是位移而不是绝对坐标)
    flow_x =map_x-x_coords
    flow_y =map_y-y_coords
    
    # 处理所有FLIR图像
=======
>>>>>>> matlab版本
    print("读取所有图像...")
    # 读取所有图像并预处理
    all_flir_images = []
    all_event_images = []
    for flir_file, event_file in tqdm(zip(flir_files, event_files)):
        I_flir = cv2.imread(flir_file)
        I_event = cv2.imread(event_file)
        I_flir_undist = undistort_image(I_flir, K_flir, dist_flir)
        I_event_undist = undistort_image(I_event, K_event, dist_event)
        all_flir_images.append(I_flir_undist)
        all_event_images.append(I_event_undist)

    # 获取图像尺寸
    h, w = all_flir_images[0].shape[:2]
    batch_size = len(all_flir_images)

    # 创建坐标网格
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())], axis=0)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为tensor进行批处理
    pixels_tensor = torch.from_numpy(pixels).float().to(device)
    P_flir_tensor = torch.from_numpy(P_flir).float().to(device)
    P_event_tensor = torch.from_numpy(P_event).float().to(device)
    
    # 构建方程组
    P_flir_part_tensor = P_flir_tensor[:, :2]
    P_flir_const_tensor = -Z * P_flir_tensor[:, 2:3] - P_flir_tensor[:, 3:4]
    
    # 批量求解3D点
    print("计算投影映射...")
    A = torch.zeros(pixels.shape[1], 3, 3, device=device)
    for idx in range(pixels.shape[1]):
        A[idx, :, :2] = P_flir_part_tensor
        A[idx, :, 2] = -pixels_tensor[:, idx]
    
    b = P_flir_const_tensor.repeat(pixels.shape[1], 1, 1)
    X = torch.zeros((pixels.shape[1], 3), device=device)
    for idx in range(pixels.shape[1]):
        X[idx] = torch.linalg.solve(A[idx], b[idx].squeeze(-1))
    
    # 构建齐次坐标
    X_homogeneous = torch.ones(X.shape[0], 4, device=device)
    X_homogeneous[:, :3] = X
    X_homogeneous[:, 2] = Z
    
    # 投影到event相机
    P_event_points = torch.matmul(P_event_tensor, X_homogeneous.t())  # [3, N]
    P_event_points = P_event_points / P_event_points[2:3, :]
    
    # 计算映射场
    map_x = P_event_points[0].reshape(h, w).cpu().numpy().astype(np.float32)
    map_y = P_event_points[1].reshape(h, w).cpu().numpy().astype(np.float32)
    
    # 创建有效区域掩码 - 修正掩码计算
    event_h, event_w = all_event_images[0].shape[:2]
    valid_mask = ((map_x >= 0) & (map_x < event_w) & 
                  (map_y >= 0) & (map_y < event_h))
    
    print("批量处理图像变形...")
    for idx, (I_flir_undist, I_event_undist, flir_file) in enumerate(zip(all_flir_images, all_event_images, flir_files)):
        print(f"处理图像 {idx+1}/{len(flir_files)}")
        
        # 确保warped图像与event图像尺寸匹配
        event_h, event_w = I_event_undist.shape[:2]
        
        # 将FLIR图像投影到EVENT视角
        warped = cv2.remap(I_flir_undist, map_x, map_y, 
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 应用掩码 - 确保掩码尺寸匹配
        mask_resized = cv2.resize(valid_mask.astype(np.uint8), (warped.shape[1], warped.shape[0]))
        warped[mask_resized == 0] = 0
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(flir_file))[0]
        cv2.imwrite(f'Outputs/flir/flir_{base_name}.png', I_flir_undist)
        cv2.imwrite(f'Outputs/event/flir_in_event_view_{base_name}.png', warped)
        
        # 创建重叠图像：EVENT图像和投影后的FLIR图像叠加
        # 确保两个图像尺寸一致
        if I_event_undist.shape != warped.shape:
            I_event_undist = cv2.resize(I_event_undist, (warped.shape[1], warped.shape[0]))
        
        overlay = cv2.addWeighted(I_event_undist, 0.5, warped, 0.5, 0)
        cv2.imwrite(f'Outputs/overlay/overlay_{base_name}.png', overlay)

    print("\n处理完成！结果已保存到 Outputs 文件夹:")
    print("- flir: 原始FLIR图像")
    print("- event: 投影到EVENT视角的FLIR图像")
    print("- overlay: EVENT图像与投影后FLIR图像的叠加")

if __name__ == "__main__":
    main()