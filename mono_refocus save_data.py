# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from operator import index
import re
import h5py
from cv2.gapi.streaming import timestamp
from numpy.core.defchararray import center
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
from metavision_core.event_io import EventsIterator,RawReader
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
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
            print(center_idx)
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
        output = output.squeeze(1) # N,H,W 
        output = output / (output.max() + 1e-8) if output.max() > 0 else output
    return output

def event_refocus(event_path,exposure_times_path,name_list,R_matrix,T_matrix,K,depth,center_idx=None):
    """
    """
    if center_idx is None:
        center_idx = R_matrix.shape[0] // 2
    device=R_matrix.device
    print(device)
    N = R_matrix.shape[0]
    triggers = None
    # 读取触发时间 txt文件
    exposure_times =  np.loadtxt(exposure_times_path).reshape(-1)  ## 图像问题 以后改回去
    with RawReader(event_path, do_time_shifting=False) as ev_data:
        while not ev_data.is_done():
            ev_data.load_n_events(1000000)
        triggers = ev_data.get_ext_trigger_events()
    triggers = triggers[triggers['p'] == 0].copy()
    triggers = triggers['t'] + exposure_times 
    # 只保留 namelist的事件
    # 从1 开始命名的图像所以减一
    triggers = triggers[name_list-1]
    mean_time_us = (triggers[-1]-triggers[0])/(triggers.shape[0]-1) #得到平均时间间隔
    star_t = triggers[0]
    end_t = triggers[-1]
    mv_iterator = EventsIterator(input_path=event_path,mode="delta_t",delta_t=100000)
    # 收集所有事件
    all_events = []
    for evs in mv_iterator:
        x_min = evs['x'].min()
        y_min = evs['y'].min()
        evs['x'] = evs['x'] - 340
        evs['y'] = evs['y'] - 60 
        all_events.append(evs)
    # 合并所有事件批次
    all_x = np.concatenate([ev['x'] for ev in all_events])
    all_y = np.concatenate([ev['y'] for ev in all_events])
    all_p = np.concatenate([ev['p'] for ev in all_events])
    all_t = np.concatenate([ev['t'] for ev in all_events])
    # 筛选只保留star_t到end_t之间的事件
    time_mask = (all_t >= star_t) & (all_t <= end_t)
    all_x = all_x[time_mask]
    all_y = all_y[time_mask]
    all_p = all_p[time_mask]
    all_t = all_t[time_mask]
    # 转换为PyTorch张量
    x = torch.tensor(all_x, dtype=torch.float32, device=device)
    y = torch.tensor(all_y, dtype=torch.float32, device=device)
    p = torch.tensor(all_p, dtype=torch.float32, device=device)
    t = torch.tensor(all_t, dtype=torch.float32, device=device)
    H,W = 600,600
    # 从R_matrix转换回四元数用于插值
    quats = []
    for i in range(R_matrix.shape[0]):
        r = R_matrix[i].cpu().numpy()
        quat = R.from_matrix(r).as_quat()  # 返回 [x, y, z, w]
        quats.append(quat)
    quats = np.array(quats)
    # 创建四元数插值器
    # 注意：scipy的Slerp需要numpy数组，所以我们在CPU上进行插值
    center_time = triggers[center_idx]
    slerp = Slerp(triggers, R.from_quat(quats))
    # 对每个事件的时间戳进行姿态插值
    event_times_np = t.cpu().numpy()
    # 计算每个事件与中心时间的时间差，找到最接近的事件索引
    C_idx = np.abs(event_times_np - center_time).argmin()
    # 计算每个事件的旋转矩阵
    event_rotations = slerp(event_times_np)
    event_R_matrices = event_rotations.as_matrix()
    event_R_matrices = torch.tensor(event_R_matrices, dtype=torch.float32, device=device) # Nx3x3
    # 对平移向量进行线性插值
    T_np = T_matrix.cpu().numpy()
    # 为每个事件创建插值的平移向量
    event_T_matrices = np.zeros((len(event_times_np), 3, 1))
    for i in range(3):  # x, y, z三个维度
        # 使用numpy的interp函数进行线性插值
        event_T_matrices[:, i, 0] = np.interp(event_times_np, triggers, T_np[:, i, 0])
    event_T_matrices = torch.tensor(event_T_matrices, dtype=torch.float32, device=device) #eventsX3X1
    # event_T_matrices = event_T_matrices.permute(2,1,0).squeeze(0)  # 3Xevents
    # refocus event 
    with torch.no_grad():
         # 如果未指定中心索引，使用中间图像
        if center_idx is None:
            center_idx = R_matrix.shape[0] // 2
        K_inv = torch.inverse(K)
        pix_x,pix_y = x,y
        ones_pix = torch.ones_like(pix_x).to(device, dtype=torch.float32)
        pix = torch.stack([pix_x,pix_y,ones_pix],dim=1)  # evensx3
        pix = pix.unsqueeze(2)  # evensx3x1
        R_x2c = event_R_matrices[C_idx,...]@event_R_matrices.permute(0,2,1) # 求逆
        T_x2c = event_T_matrices[C_idx] - R_x2c@event_T_matrices  # Nx3x3 X Nx3x1 -> Nx3x1
        x_ray = K_inv.unsqueeze(0)@pix   # 1x3x3 X eventsx3x1 -> eventsx3x1
        x_ray = R_x2c@x_ray + T_x2c/depth # eventsX3x3 X 3Xevents -> 3Xevents
        x_pix = K.unsqueeze(0)@x_ray
        x =  x_pix[:,0,0]
        y =  x_pix[:,1,0]
        # 创建用于存储压缩帧的张量，按极性分开
        pos_frames = torch.zeros((len(triggers), H, W), dtype=torch.float32, device=device).flatten()
        neg_frames = torch.zeros((len(triggers), H, W), dtype=torch.float32, device=device).flatten()
        # 计算时间窗口边界（基于触发时间中点）
        triggers_tensor = torch.tensor(triggers, device=device, dtype=torch.float32)
        mid_points = (triggers_tensor[1:] + triggers_tensor[:-1]) / 2.0
        time_bins = torch.cat([
            triggers_tensor[0].unsqueeze(0) - 1,  # 第一个窗口左边界（稍微扩展）
            mid_points,
            triggers_tensor[-1].unsqueeze(0) + 1  # 最后一个窗口右边界（稍微扩展）
        ])
        
        # 将事件坐标四舍五入为整数
        x_int = torch.round(x).long()
        y_int = torch.round(y).long()
        
        # 筛选出有效范围内的事件（在图像边界内）
        valid_events = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H)
        
        # 向量化分配事件到时间窗口，并确保索引在有效范围内
        frame_indices = torch.bucketize(t, time_bins) - 1
        
        # 确保所有索引都在有效范围内
        valid_indices = (frame_indices >= 0) & (frame_indices < len(triggers))
        
        # 并行化极性处理（同时考虑坐标有效性和时间窗口有效性）
        pos_mask = valid_events & valid_indices & (p == 1)
        neg_mask = valid_events & valid_indices & (p == 0)
        # 计算一维索引: frame_idx * H * W + y * W + x
        pos_linear_indices = frame_indices[pos_mask] * H * W + y_int[pos_mask] * W + x_int[pos_mask]
        neg_linear_indices = frame_indices[neg_mask] * H * W + y_int[neg_mask] * W + x_int[neg_mask]
        
        # 使用一维索引直接累加事件
        pos_frames.index_add_(0, pos_linear_indices, torch.ones_like(pos_linear_indices, dtype=torch.float32))
        neg_frames.index_add_(0, neg_linear_indices, -torch.ones_like(neg_linear_indices, dtype=torch.float32))
        
        # 将一维张量重塑为所需的输出格式 [N, H, W]
        pos_frames = pos_frames.reshape(N, H, W)
        neg_frames = neg_frames.reshape(N, H, W)
        
        # 合并正负极性事件帧  这里的归一化考虑了整个时间范围内的
        pos_frames = pos_frames / (pos_frames.max() + 1e-8) if pos_frames.max() > 0 else pos_frames
        neg_frames = neg_frames / (abs(neg_frames.min() - 1e-8)) if neg_frames.min() < 0 else neg_frames
        event_frame = torch.stack([pos_frames, neg_frames], dim=1)  # [C, 2, H, W]
        return event_frame

        
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
    H, W = 600, 600
    folder_path = '0000'
    sub_floder = '000'
    # 找到中间图像的索引
    center_idx = 11
    # 保存数据
    num = 2
    best_focus_depth= 404
    frame_npy_path = os.path.join("./dataset/Frame", f'{num:04d}.h5')
    event_npy_path = os.path.join("./dataset/Event", f'{num:04d}.h5')
    

    save_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'results',folder_path,sub_floder))
    os.makedirs(save_path, exist_ok=True)
    event_path =  os.path.abspath(os.path.join(os.path.dirname(__file__), 'data',folder_path,'events.raw'))
    exposure_times_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data',folder_path,'exposure_times.txt'))
    image_path =  os.path.join(os.path.dirname(__file__),'projection_outputs', 'flir',folder_path,sub_floder)
    extrinsic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'colmap',folder_path,sub_floder))
    #读取文件夹中的图像序号
    read_image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
    read_image_files = sorted(read_image_files,key=lambda x :  int(x.split('.')[0]))
    # 读取图像并存储到矩阵中
    image_matrix = read_images(image_path, read_image_files,image_size=(H, W))
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

    K = torch.tensor(K).to(device, dtype=torch.float32)

    with torch.no_grad():
        warped_save_path = os.path.join(save_path, f"warped_{best_focus_depth:.3f}")
        depth = torch.tensor(best_focus_depth).to(device, dtype=torch.float32)
        refocus_images= refocus(image_matrix, R_matrix, T_matrix,K,depth, center_idx=center_idx)
        os.mkdir(warped_save_path) if not os.path.exists(warped_save_path) else None
        event_frame=event_refocus(event_path,exposure_times_path,name_list,R_matrix,T_matrix,K,depth,center_idx=center_idx)
        for i in range(refocus_images.shape[0]):
            warped_img = refocus_images[i].cpu().numpy()
            # 归一化并转换为uint8
            warped_img = warped_img / warped_img.max() if warped_img.max() > 0 else warped_img
            warped_img = np.uint8(np.clip(warped_img * 255.0, 0, 255))
            # 处理event_frame
            event_curr = event_frame[i].cpu().numpy()  # 获取当前帧 [2,H,W]
            event_curr = abs(event_curr)/event_curr.max()
            event_curr = np.uint8(np.clip(event_curr * 255.0, 0, 255)).transpose(1,2,0)  # [H,W,2]
            # 堆叠通道
            event_display = np.stack([ event_curr[...,1],np.zeros_like(warped_img),event_curr[...,0]], axis=-1)  # [H,W,3]
            overlay = cv2.addWeighted(
                cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR), 0.3,
                event_display, 0.7, 0)
            cv2.imwrite(f"{warped_save_path}/{i:03d}_overlay.png", overlay)
        ref_image = nonzero_mean(refocus_images, dim=0).cpu().numpy()
        ref_image = ref_image
        image = ref_image / ref_image.max()
        new_image = np.uint8(np.clip(image*255., 0, 255))
        cv2.imwrite(f"{save_path}/{best_focus_depth:.3f}.png", new_image)
        refocus_images = refocus_images.cpu().numpy()
        event_frame = event_frame.cpu().numpy()
        with h5py.File(frame_npy_path, 'w') as f:
            f.create_dataset('frame', data=refocus_images)
            f.create_dataset('depth', data=best_focus_depth)
            f.create_dataset('center_idx', data=center_idx)
        
        with h5py.File(event_npy_path, 'w') as f:
            f.create_dataset('pos', data=event_frame[:,0,...])
            f.create_dataset('neg', data=event_frame[:,1,...])
            f.create_dataset('depth', data=best_focus_depth)
            f.create_dataset('center_idx', data=center_idx)

    print(f'OK, is done!')

    