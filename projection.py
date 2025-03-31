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
    """
    对灰度图进行变形
    voxelgrid: [bs,1,H,W] - 单通道灰度图
    flow_x/flow_y: [bs,1,H,W] - 位移场
    """
    bs, c, H, W = voxelgrid.shape
    flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,1,2,H,W]
    flow = flow.reshape(bs, c * 2, H, W)  # [bs,2,H,W]
    # 单位矩阵权重
    weight = torch.eye(c, device=flow.device).to(torch.float32).reshape(c, c, 1, 1)
    return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,1,H,W]
# def dcn_warp(voxelgrid: Tensor, flow_x: Tensor, flow_y: Tensor):
#     # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
#     bs, ts, H, W = voxelgrid.shape
#     flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,ts,2,H,W]
#     flow = flow.reshape(bs, ts * 2, H, W)  # [bs,ts*2,H,W]
#     #  单位矩阵 保证了 只对一张图处理
#     weight = torch.eye(ts, device=flow.device).double().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
#     # 单位卷积核
#     return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,ts,H,W]

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
    K_flir = stereo_params['K2'][0, 0]  # flir相机内参
    K_flir_inv = np.linalg.inv(K_flir)  # flir相机逆内参
    K_event = stereo_params['K1'][0, 0]  # event相机内参
    K_event_inv = np.linalg.inv(K_event)  # event相机逆内参
    dist_flir = np.hstack([
        stereo_params['RadialDistortion2'][0, 0].flatten(), 
        stereo_params['TangentialDistortion2'][0, 0].flatten()
    ])
    
    dist_event = np.hstack([
        stereo_params['RadialDistortion1'][0, 0].flatten(), 
        stereo_params['TangentialDistortion1'][0, 0].flatten()
    ])
    # 获取旋转矩阵和平移矩阵
    R_flir = stereo_params['R2'][0, 0]  # flir相机旋转矩阵
    T_flir = stereo_params['T2'][0, 0]  # flir相机平移向量
    R_event = stereo_params['R1'][0, 0]  # event相机旋转矩阵
    T_event = stereo_params['T1'][0, 0]  # event相机平移向量
    # 获取所有图像文件
    flir_files = sorted(glob.glob('flir_result/*.png'))
    R_f2e = np.matmul(R_event, R_flir.T)
    # 单位是mm 
    T_f2e = T_event - np.matmul(np.matmul(R_event, R_flir.T), T_flir)
    
    flir_h, flir_w = 1800,1800
    # 设置投影平面Z值 默认1
    # Z = 1
    
    print("计算投影映射...")
    # 创建坐标网格 先横后竖 xy
    x_coords,y_coords = np.meshgrid(np.arange(flir_w), np.arange(flir_h))
    
    # 创建均匀坐标 [3, h*w]
    flir_pixels = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())], axis=0)
    
    # 使用内参矩阵的逆矩阵将像素坐标转换为归一化相机坐标
    flir_rays = np.matmul(K_event_inv, flir_pixels)  # [3, h*w]
    
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
    flow_x = map_x - x_coords
    flow_y = map_y - y_coords
    
    # 处理每张FLIR图像
    for i, flir_file in enumerate(flir_files):
        print(f"处理图像 {i+1}/{len(flir_files)}: {os.path.basename(flir_file)}")
        
        # 读取FLIR图像为灰度图
        I_flir = cv2.imread(flir_file, cv2.IMREAD_GRAYSCALE)  # 灰度图
        
        # # 去畸变
        # I_flir_undist = undistort_image(I_flir, K_flir, dist_flir)
        
        # # 获取FLIR图像的实际尺寸
        # flir_h, flir_w = I_flir_undist.shape
        
        # # 调整位移场大小以匹配FLIR图像
        # flow_x_resized = cv2.resize(flow_x, (flir_w, flir_h), interpolation=cv2.INTER_LINEAR)
        # flow_y_resized = cv2.resize(flow_y, (flir_w, flir_h), interpolation=cv2.INTER_LINEAR)
        
        # # 根据尺寸比例缩放位移值
        # flow_x_resized = flow_x_resized * (flir_w / flir_w)
        # flow_y_resized = flow_y_resized * (flir_h / flir_h)
        
        # 准备输入tensor (灰度图为单通道，不需要permute通道维度)
        I_flir_tensor = torch.from_numpy(I_flir).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        flow_x_tensor = torch.from_numpy(flow_x).float().unsqueeze(0).unsqueeze(0)
        flow_y_tensor = torch.from_numpy(flow_y).float().unsqueeze(0).unsqueeze(0)
        # 使用DCN进行变形
        warped_tensor = dcn_warp(I_flir_tensor, flow_x_tensor, flow_y_tensor)
        
        # 转回numpy格式 (单通道图像不需要permute)
        warped_img = warped_tensor.squeeze().numpy().astype(np.uint8)
        
        # 创建有效区域掩码并调整尺寸以匹配warped_img
        projection_mask = ((map_x >= 0) & (map_x < I_flir.shape[1]) & 
                          (map_y >= 0) & (map_y < I_flir.shape[0]))
        projection_mask = projection_mask.astype(np.uint8) * 255

        # 调整掩码大小以匹配warped_img
        projection_mask_resized = cv2.resize(projection_mask, 
                                           (warped_img.shape[1], warped_img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

        # 应用调整后的掩码
        result_img = cv2.bitwise_and(warped_img, warped_img, 
                                    mask=projection_mask_resized)
        
        # 获取文件名（不包含路径和扩展名）
        base_name = os.path.splitext(os.path.basename(flir_file))[0]
        
        # 保存结果
        cv2.imwrite(f'Outputs/flir/flir_in_event_view_{base_name}.png', result_img)
    

    print("\n处理完成！结果已保存到 Outputs 文件夹:")
    
if __name__ == "__main__":
    main()