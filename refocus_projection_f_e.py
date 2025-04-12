# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from operator import index
import re
from scipy import io
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
import os
import torch
from torch import Tensor
import torchvision
import torch.nn.functional as F
from tqdm import tqdm 
# 自适应直方图均衡化  AHE
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def undistort_image(img, K, dist_coeffs):
    """
    去除图像畸变
    """
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
    return undistorted

def nonzero_mean(tensor: Tensor, dim: int, keepdim: bool = False, eps=1e-3):
    numel = torch.sum((tensor > eps).float(), dim=dim, keepdim=keepdim)
    value = torch.sum(tensor, dim=dim, keepdim=keepdim)
    return value / (numel + eps)
def read_images(directory_path,read_image_files, image_size):
    global N
    N = len(read_image_files)
    # 初始化矩阵
    images_matrix = np.zeros((len(read_image_files), *image_size), dtype=np.uint8)
    # 读取每张图像并存储到矩阵中
    for i, image_file in enumerate(read_image_files):
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images_matrix[i] = np.array(image)

    return images_matrix

def refocus(image_matrix, R_matrix, T_matrix,K,depth, center_idx=None):
    """
    使用相机参数和深度值对图像进行重聚焦
    参数:
    - image_matrix: 输入图像矩阵 [N, H, W]，取值范围 0-255
    - R_matrix: 旋转矩阵 [N, 3, 3]
    - T_matrix: 平移向量 [N, 3, 1]
    - K: 相机内参矩阵 [3, 3]
    - depth: 重聚焦深度
    - center_idx: 中心图像索引，如果为None则使用中间图像
    返回:
    - 重聚焦后的图像
    """
    with torch.no_grad():
        # 如果未指定中心索引，使用中间图像
        if center_idx is None:
            center_idx = R_matrix.shape[0] // 2
        N,H,W = image_matrix.shape
        K_inv = torch.inverse(K)
        R_C_inv = R_matrix[center_idx,...].transpose(1,0)
        R_C2x = R_matrix@R_C_inv
        T_c2x = T_matrix - R_C2x@T_matrix[center_idx,...]
        c_pix_y,c_pix_x = torch.meshgrid(torch.arange(H), torch.arange(W),indexing='ij')
        c_pix_x = c_pix_x.flatten().to(device, dtype=torch.float32)
        c_pix_y = c_pix_y.flatten().to(device, dtype=torch.float32)
        ones_pix = torch.ones_like(c_pix_x).to(device, dtype=torch.float32)
        c_pix = torch.stack([c_pix_x,c_pix_y,ones_pix],dim=0)  # 3,H*W
        c_ray = K_inv@c_pix   # 3x3 X 3XH*W -> 3XH*W
        c_ray = c_ray.unsqueeze(0).repeat(N,1,1) # N,3,H*W
        x_ray = R_C2x@c_ray + T_c2x/depth
        x_pix = K@x_ray
        # 这里不用归一化的
        # x_pix = x_pix/x_pix[2,:]
        x_pix_x = x_pix[:,0,:]
        x_pix_y = x_pix[:,1,:]
        x_pix_x = x_pix_x.reshape(N,H,W)
        x_pix_y = x_pix_y.reshape(N,H,W)
        # 归一化坐标到 [-1, 1] 范围，这是 grid_sample 要求的
        norm_x = 2.0 * x_pix_x / (W - 1) - 1.0  # 使用 W 
        norm_y = 2.0 * x_pix_y / (H - 1) - 1.0  # 使用 H 
        # 构建采样网格 [N, H, W, 2]
        grid = torch.stack([norm_x,norm_y], dim=-1)
        # 准备输入图像 [N, 1, H, W]
        input_images = image_matrix.unsqueeze(1)
        # 使用 grid_sample 进行重投影
        output = F.grid_sample(
            input_images, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
    return output.squeeze(1) # N,H,W

def projection(flir_image):
    flir_image = flir_image.cpu().numpy()
    # 加载相机参数
    stereo_params = io.loadmat('./data_4.2/stereo_camera_parameters.mat')['stereo_params']
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

    N,flir_h, flir_w = flir_image.shape
    event_h, event_w = 600,600
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

    batch_size = 100000  # 每批处理的像素点数量
    num_batches = (len(u) + batch_size - 1) // batch_size
    
    X_result = np.zeros_like(u)
    Y_result = np.zeros_like(u)
    lambda_result = np.zeros_like(u)
    
    for i in range(num_batches):
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
    flir_proj = P_flir @ world_coords
    flir_proj = flir_proj / flir_proj[2, :]  # 齐次坐标归一化
    
    # 获取映射坐标
    map_x = flir_proj[0, :].reshape(event_h, event_w)
    map_y = flir_proj[1, :].reshape(event_h, event_w)
    map_x_float = map_x.astype(np.float32)
    map_y_float = map_y.astype(np.float32)
    # 使用OpenCV的remap函数进行图像变形
    warped_img = np.zeros((N,event_h, event_w), dtype=np.uint8)
    for i in range(N):
        img = flir_image[i,...]
        img = undistort_image(img, K_flir, dist_event)
        # 创建映射矩阵 (float32类型)
        map_x_float = map_x.astype(np.float32)
        warped = cv2.remap(img, map_x_float, map_y_float, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # 创建有效区域掩码
        mask = ((map_x >= 0) & (map_x < img.shape[1]) & 
                (map_y >= 0) & (map_y < img.shape[0]))
        mask = mask.astype(np.uint8) * 255
        # 应用掩码
        result = cv2.bitwise_and(warped, warped, mask=mask)
        warped_img[i,...] = result
    return warped_img

if __name__ == '__main__':
# 4703.863770,4750.683105,750.655371,491.003268
# 2000 1000 6472.86 6419.42 1000 500
# 7191.73748782809,7190.76441594966,897.056114098032,862.573591961378
# 3463.13471858099	0	0
# 0	3463.11625407825	0
# 347.866576798087	325.714566240859	1
    fx,fy,cx,cy= 3463.13471858099,3463.11625407825,347.866576798087, 325.714566240859
    K=np.array([[3463.13471858099,  0,   0],
                [0   ,3463.11625407825,  0],
                [347.866576798087,325.714566240859,1]]).T
    # K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    h, w = 1800, 1800
    
    event_raw =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'data','0000','events.raw'))
    save_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'results','0000','000'))
    # image_path =  os.path.join(os.path.dirname(__file__),'projection_outputs', 'flir','0000','000')
    image_path =  os.path.abspath(os.path.join(os.path.dirname(__file__),'colmap','0000','000'))
    os.makedirs(save_path, exist_ok=True)
    files_num = len(os.listdir(save_path))
    extrinsic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'colmap','0000','000'))
    #读取文件夹中的图像序号
    read_image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
    read_image_files = sorted(read_image_files,key=lambda x :  int(x.split('.')[0]))
    # 读取图像并存储到矩阵中
    image_matrix = read_images(image_path, read_image_files,image_size=(h, w))
    # 读取外参
    # 从images.txt文件中提取相机参数
    file_path = os.path.join(extrinsic_path, 'images.txt') 
    with open(file_path, 'r') as file:
        lines = file.readlines()
    Qw_list = []
    Qx_list = []
    Qy_list = []
    Qz_list = []
    Tx_list = []
    Ty_list = []
    Tz_list = []
    name_list = []
    for line in lines:
        if '.jpg' in line or '.png' in line:
            data = line.strip().split()
            # 提取四元数和平移向量
            q_values = np.float64(data[-9:-5])  # QW, QX, QY, QZ
            t_values = np.float64(data[-5:-2])  # TX, TY, TZ
            
            # 添加到对应列表
            Qw_list.append(q_values[0])
            Qx_list.append(q_values[1])
            Qy_list.append(q_values[2])
            Qz_list.append(q_values[3])
            Tx_list.append(t_values[0])
            Ty_list.append(t_values[1])
            Tz_list.append(t_values[2])
            # 提取图像名称并转为整数
            name_list.append(int(data[-1].split('.')[0]))
    
    # 将名称列表转换为numpy数组
    name_list = np.array(name_list)
    
    # 根据名称排序所有参数
    indices = np.argsort(name_list)
    name_list = name_list[indices]
    
    # 使用numpy的索引功能一次性对所有列表进行排序
    param_lists = [Tx_list, Ty_list, Tz_list, Qx_list, Qy_list, Qz_list, Qw_list]
    param_lists = [np.array(param_list)[indices] for param_list in param_lists]
    Tx_list, Ty_list, Tz_list, Qx_list, Qy_list, Qz_list, Qw_list = param_lists
    
    # 直接转换为torch张量并设置数据类型
    Tx_list = torch.tensor(Tx_list, dtype=torch.float32).to(device)
    Ty_list = torch.tensor(Ty_list, dtype=torch.float32).to(device)
    Tz_list = torch.tensor(Tz_list, dtype=torch.float32).to(device)
    
    # 创建四元数张量
    quat = torch.stack([
        torch.tensor(Qx_list), 
        torch.tensor(Qy_list), 
        torch.tensor(Qz_list), 
        torch.tensor(Qw_list)
    ], dim=1)
    
    R_matrix = R.from_quat(quat).as_matrix() # N,3,3
    R_matrix = torch.tensor(R_matrix).to(device, dtype=torch.float32)
    T_matrix = torch.stack([Tx_list, Ty_list, Tz_list], dim=1).unsqueeze(2)  # N,3,1
    T_matrix = T_matrix.detach().to(device, dtype=torch.float32)
    
    # 将图像矩阵转换为torch张量
    image_matrix = torch.from_numpy(image_matrix).to(device, dtype=torch.float32)
    # 找到中间图像的索引
    center_idx = 1
    center_idx = torch.tensor(center_idx, device=device, dtype=torch.float32)
    print(f"使用索引 {center_idx} 的图像作为参考图像")
    depth =15
    K = torch.tensor(K).to(device, dtype=torch.float32)

    with torch.no_grad():
        for best_focus_depth in tqdm(np.arange(2000, 9000, 1000), desc="focus depth"):
            warped_save_path = os.path.join(save_path, f"warped_{best_focus_depth:.3f}")
            depth = torch.tensor(best_focus_depth).to(device, dtype=torch.float32)
            refocus_images= refocus(image_matrix, R_matrix, T_matrix,K,depth, center_idx=None)
            refocus_images= projection(refocus_images)
            os.mkdir(warped_save_path) if not os.path.exists(warped_save_path) else None
            for i in range(refocus_images.shape[0]):
                warped_img = refocus_images[i]
                # 归一化并转换为uint8
                warped_img = warped_img / warped_img.max() if warped_img.max() > 0 else warped_img
                warped_img = np.uint8(np.clip(warped_img * 255.0, 0, 255))
                # 保存图像
                cv2.imwrite(f"{warped_save_path}/{i:03d}.png", warped_img)
            # ref_image = nonzero_mean(refocus_images, dim=0)
            # ref_image = ref_image
            # image = ref_image / ref_image.max()
            # new_image = np.uint8(np.clip(image*255., 0, 255))
            # cv2.imwrite(f"{save_path}/{best_focus_depth:.3f}.png", new_image)
    print(f'OK, is done!')

    