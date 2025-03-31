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
    weight = torch.eye(ts, device=flow.device).double().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
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
    K_flir = stereo_params['K2'][0, 0]  # flir相机内参
    K_event = stereo_params['K1'][0, 0]  # event相机内参
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
    T_f2e = T_event - np.matmul(np.matmul(R_event, R_flir.T), T_flir)
    event_h, event_w = 600,600
    # 设置投影平面Z值 (可调整)
    Z = 1
    
    # 为每个EVENT像素位置计算投影映射
    print("计算投影映射...")
    y_coords, x_coords = np.meshgrid(np.arange(event_h), np.arange(event_w), indexing='ij')
    
    # 创建映射数组
    map_x = np.zeros((event_h, event_w), dtype=np.float32)
    map_y = np.zeros((event_h, event_w), dtype=np.float32)
    
    # 计算EVENT到FLIR的映射
    for y in tqdm(range(event_h)):
        for x in range(event_w):
            # 将EVENT像素反投影到3D (归一化坐标)
            x_norm = (x - K_event[0, 2]) / K_event[0, 0]
            y_norm = (y - K_event[1, 2]) / K_event[1, 1]
            
            # 3D点在EVENT坐标系
            X_event = np.array([x_norm * Z, y_norm * Z, Z]).reshape(3, 1)
            
            # 从EVENT坐标系转换到FLIR坐标系
            X_flir = np.matmul(R_f2e.T, X_event - T_f2e)
            
            # 投影到FLIR图像平面
            if X_flir[2] > 0:  # 确保点在相机前方
                u_flir = K_flir[0, 0] * (X_flir[0] / X_flir[2]) + K_flir[0, 2]
                v_flir = K_flir[1, 1] * (X_flir[1] / X_flir[2]) + K_flir[1, 2]
                
                map_x[y, x] = u_flir
                map_y[y, x] = v_flir
    
    # 计算位移场 (对于DCN需要的是位移而不是绝对坐标)
    base_x, base_y = np.meshgrid(np.arange(event_w), np.arange(event_h))
    flow_x = map_x - base_x.T  # 注意转置以匹配y_coords, x_coords的indexing='ij'
    flow_y = map_y - base_y.T
    
    # 处理每张FLIR图像
    for i, flir_file in enumerate(flir_files):
        print(f"处理图像 {i+1}/{len(flir_files)}: {os.path.basename(flir_file)}")
        
        # 读取FLIR图像
        I_flir = cv2.imread(flir_file)
        
        # 去畸变
        I_flir_undist = undistort_image(I_flir, K_flir, dist_flir)
        
        # 准备输入tensor
        I_flir_tensor = torch.from_numpy(I_flir_undist).permute(2, 0, 1).float().unsqueeze(0)  # [1, 3, H, W]
        flow_x_tensor = torch.from_numpy(flow_x).float().unsqueeze(0).unsqueeze(0)
        flow_y_tensor = torch.from_numpy(flow_y).float().unsqueeze(0).unsqueeze(0)
        
        # 使用DCN进行变形
        warped_tensor = dcn_warp(I_flir_tensor, flow_x_tensor, flow_y_tensor)
        
        # 转回numpy格式
        warped_img = warped_tensor.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
        
        # 创建有效区域掩码 (去除超出范围的像素)
        valid_mask = ((map_x >= 0) & (map_x < I_flir.shape[1]) & 
                      (map_y >= 0) & (map_y < I_flir.shape[0]))
        valid_mask = valid_mask.astype(np.uint8) * 255
        
        # 应用掩码
        result_img = cv2.bitwise_and(warped_img, warped_img, mask=valid_mask)
        
        # 获取文件名（不包含路径和扩展名）
        base_name = os.path.splitext(os.path.basename(flir_file))[0]
        
        # 保存结果
        cv2.imwrite(f'Outputs/flir/flir_in_event_view_{base_name}.png', result_img)
        
        # 创建并保存重叠图像 (使用matplotlib)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(I_flir, cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), alpha=0.5)
        plt.axis('off')
        plt.title(f'FLIR与投影结果叠加 - {base_name}')
        plt.savefig(f'Outputs/overlay/overlay_{base_name}.png', bbox_inches='tight')
        plt.close()
    
    print("\n处理完成！结果已保存到 Outputs 文件夹:")
    
    


if __name__ == "__main__":
    main()