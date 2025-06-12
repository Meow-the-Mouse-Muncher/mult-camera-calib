import sys
import re
from numpy.core.defchararray import center
from scipy import io
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
import os
import torch
from torch import Tensor
import torchvision
import torch.nn.functional as F
sys.path.append("/usr/lib/python3/dist-packages")
from metavision_core.event_io import EventsIterator,RawReader
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from tqdm import tqdm 

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nonzero_mean(tensor: Tensor, dim: int, keepdim: bool = False, eps=1e-3):
    numel = torch.sum((tensor > eps).float(), dim=dim, keepdim=keepdim)
    value = torch.sum(tensor, dim=dim, keepdim=keepdim)
    return value / (numel + eps)

def read_images_by_ids(directory_path, image_ids, image_size):
    """
    根据图像ID列表读取对应的图像
    """
    N = len(image_ids)
    images_matrix = np.zeros((N, *image_size), dtype=np.uint8)
    
    for i, img_id in enumerate(image_ids):
        image_path = os.path.join(directory_path, f"{img_id:04d}.png")
        if not os.path.exists(image_path):
            print(f"警告: 图像 {image_path} 不存在，使用零填充")
            continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images_matrix[i] = np.array(image)
        else:
            print(f"警告: 无法读取图像 {image_path}")
    return images_matrix

def refocus(image_matrix, R_matrix, T_matrix,K,depth, center_idx=None):
    """
    使用相机参数和深度值对图像进行重聚焦
    """
    with torch.no_grad():
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
        c_pix = torch.stack([c_pix_x,c_pix_y,ones_pix],dim=0)
        c_ray = K_inv@c_pix
        c_ray = c_ray.unsqueeze(0).repeat(N,1,1)
        x_ray = R_C2x@c_ray + T_c2x/depth
        x_pix = K@x_ray
        x_pix_x = x_pix[:,0,:]
        x_pix_y = x_pix[:,1,:]
        x_pix_x = x_pix_x.reshape(N,H,W)
        x_pix_y = x_pix_y.reshape(N,H,W)
        norm_x = 2.0 * x_pix_x / (W - 1) - 1.0
        norm_y = 2.0 * x_pix_y / (H - 1) - 1.0
        grid = torch.stack([norm_x,norm_y], dim=-1)
        input_images = image_matrix.unsqueeze(1)
        output = F.grid_sample(
            input_images, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        output = output.squeeze(1)
        output = output / (output.max() + 1e-8) if output.max() > 0 else output
    return output

def event_refocus(event_path,exposure_times_path,name_list,R_matrix,T_matrix,K,depth,center_idx=None):
    """
    Event refocus处理
    """
    if center_idx is None:
        center_idx = R_matrix.shape[0] // 2
    device=R_matrix.device
    N = R_matrix.shape[0]
    
    # 读取触发时间
    exposure_times = np.loadtxt(exposure_times_path).reshape(-1)  
    with RawReader(event_path, do_time_shifting=False) as ev_data:
        while not ev_data.is_done():
            ev_data.load_n_events(1000000)
        triggers = ev_data.get_ext_trigger_events()
    triggers = triggers[triggers['p'] == 0].copy()
    triggers = triggers['t']
    triggers = triggers[name_list-1]
    
    star_t = triggers[0]
    end_t = triggers[-1]
    mv_iterator = EventsIterator(input_path=event_path,mode="delta_t",delta_t=100000)
    
    # 收集所有事件
    all_events = []
    for evs in mv_iterator:
        evs['x'] = evs['x'] - 340
        evs['y'] = evs['y'] - 60 
        all_events.append(evs)
    
    # 合并所有事件批次
    all_x = np.concatenate([ev['x'] for ev in all_events])
    all_y = np.concatenate([ev['y'] for ev in all_events])
    all_p = np.concatenate([ev['p'] for ev in all_events])
    all_t = np.concatenate([ev['t'] for ev in all_events])
    
    # 筛选时间范围内的事件
    time_mask = (all_t >= star_t) & (all_t <= end_t)
    all_x = all_x[time_mask]
    all_y = all_y[time_mask]
    all_p = all_p[time_mask]
    all_t = all_t[time_mask]
    
    # 转换为PyTorch张量
    x = torch.tensor(all_x.astype(np.float32), dtype=torch.float32, device=device)
    y = torch.tensor(all_y.astype(np.float32), dtype=torch.float32, device=device)
    p = torch.tensor(all_p.astype(np.float32), dtype=torch.float32, device=device)
    t = torch.tensor(all_t.astype(np.int64), dtype=torch.float32, device=device)
    
    H,W = 600,600
    
    # 从R_matrix转换回四元数用于插值
    quats = []
    for i in range(R_matrix.shape[0]):
        r = R_matrix[i].cpu().numpy()
        quat = R.from_matrix(r).as_quat()
        quats.append(quat)
    quats = np.array(quats)
    
    center_time = triggers[center_idx]
    slerp = Slerp(triggers, R.from_quat(quats))
    
    event_times_np = t.cpu().numpy()
    C_idx = np.abs(event_times_np - center_time).argmin()
    
    event_rotations = slerp(event_times_np)
    event_R_matrices = event_rotations.as_matrix()
    event_R_matrices = torch.tensor(event_R_matrices, dtype=torch.float32, device=device)
    
    # 对平移向量进行线性插值
    T_np = T_matrix.cpu().numpy()
    event_T_matrices = np.zeros((len(event_times_np), 3, 1))
    for i in range(3):
        event_T_matrices[:, i, 0] = np.interp(event_times_np, triggers, T_np[:, i, 0])
    event_T_matrices = torch.tensor(event_T_matrices, dtype=torch.float32, device=device)
    
    # refocus event 
    with torch.no_grad():
        K_inv = torch.inverse(K)
        pix_x,pix_y = x,y
        ones_pix = torch.ones_like(pix_x).to(device, dtype=torch.float32)
        pix = torch.stack([pix_x,pix_y,ones_pix],dim=1)
        pix = pix.unsqueeze(2)
        R_x2c = event_R_matrices[C_idx,...]@event_R_matrices.permute(0,2,1)
        T_x2c = event_T_matrices[C_idx] - R_x2c@event_T_matrices
        x_ray = K_inv.unsqueeze(0)@pix
        x_ray = R_x2c@x_ray + T_x2c/depth
        x_pix = K.unsqueeze(0)@x_ray
        x =  x_pix[:,0,0]
        y =  x_pix[:,1,0]
        
        # 创建用于存储压缩帧的张量
        pos_frames = torch.zeros((len(triggers), H, W), dtype=torch.float32, device=device).flatten()
        neg_frames = torch.zeros((len(triggers), H, W), dtype=torch.float32, device=device).flatten()
        
        triggers_tensor = torch.tensor(triggers, device=device, dtype=torch.float32)
        mid_points = (triggers_tensor[1:] + triggers_tensor[:-1]) / 2.0
        time_bins = torch.cat([
            triggers_tensor[0].unsqueeze(0) - 1,
            mid_points,
            triggers_tensor[-1].unsqueeze(0) + 1
        ])
        
        x_int = torch.round(x).long()
        y_int = torch.round(y).long()
        
        valid_events = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H)
        frame_indices = torch.bucketize(t, time_bins) - 1
        valid_indices = (frame_indices >= 0) & (frame_indices < len(triggers))
        
        pos_mask = valid_events & valid_indices & (p == 1)
        neg_mask = valid_events & valid_indices & (p == 0)
        
        pos_linear_indices = frame_indices[pos_mask] * H * W + y_int[pos_mask] * W + x_int[pos_mask]
        neg_linear_indices = frame_indices[neg_mask] * H * W + y_int[neg_mask] * W + x_int[neg_mask]
        
        pos_frames.index_add_(0, pos_linear_indices, torch.ones_like(pos_linear_indices, dtype=torch.float32))
        neg_frames.index_add_(0, neg_linear_indices, -torch.ones_like(neg_linear_indices, dtype=torch.float32))
        
        pos_frames = pos_frames.reshape(N, H, W)
        neg_frames = neg_frames.reshape(N, H, W)
        
        pos_frames = pos_frames / (pos_frames.max() + 1e-8) if pos_frames.max() > 0 else pos_frames
        neg_frames = neg_frames / (abs(neg_frames.min() - 1e-8)) if neg_frames.min() < 0 else neg_frames
        event_frame = torch.stack([pos_frames, neg_frames], dim=1)
        
        return event_frame

def visualize_and_save(refocus_images, save_dir, prefix, depth, modality='image'):
    """保存重聚焦后的图像 - 只保存累积结果，深度信息在文件名中"""
    os.makedirs(save_dir, exist_ok=True)
    
    if modality == 'event':
        # Event数据 [N, 2, H, W] - 只保存平均事件帧
        pos_mean = refocus_images[:, 0, :, :].mean(dim=0).cpu().numpy()
        neg_mean = refocus_images[:, 1, :, :].mean(dim=0).cpu().numpy()
        pos_mean = pos_mean / (pos_mean.max() + 1e-8)
        neg_mean = abs(neg_mean) / (abs(neg_mean).max() + 1e-8)
        event_mean_display = np.stack([neg_mean, np.zeros_like(pos_mean), pos_mean], axis=-1)
        event_mean_display = np.uint8(np.clip(event_mean_display * 255.0, 0, 255))
        cv2.imwrite(f"{save_dir}/refocus_depth_{depth:.1f}.png", event_mean_display)
        
    else:
        # 普通图像数据 [N, H, W] - 只保存平均图像
        ref_image = nonzero_mean(refocus_images, dim=0).cpu().numpy()
        ref_image = ref_image / (ref_image.max() + 1e-8)
        new_image = np.uint8(np.clip(ref_image * 255., 0, 255))
        cv2.imwrite(f"{save_dir}/refocus_depth_{depth:.1f}.png", new_image)

if __name__ == '__main__':
    # === 直接在代码中设置参数 ===
    modalities = ['flir', 'thermal', 'event']  # 可修改：选择要处理的模态
    sequence = '0003'                          # 可修改：序列名称
    pose_subdir = '001'                        # 可修改：位姿子目录
    
    # === 深度设置 - 直接在代码中修改 ===
    depth_min = 10     # 最小深度
    depth_max = 800     # 最大深度
    depth_step = 20    # 深度步长
    depth_range = np.arange(depth_min, depth_max + depth_step, depth_step)
    print(f"深度范围: {depth_range}")
    print(f"处理序列: {sequence}/{pose_subdir}")
    print(f"处理模态: {modalities}")

    # 加载相机参数
    stereo_params = io.loadmat('./stereoParams_FE.mat')['stereoParams']
    K_flir = stereo_params['K2'][0, 0]
    K_event = stereo_params['K1'][0, 0]
    
    stereo_params_thermal = io.loadmat('./stereoParams_FI.mat')['stereoParams']
    K_thermal = stereo_params_thermal['K2'][0, 0]
    
    H, W = 600, 600
    center_idx = 15
    center_idx = torch.tensor(center_idx, device=device, dtype=torch.int8)

    # 路径设置 - 修改为新的保存结构
    base_dir = os.path.abspath(os.path.dirname(__file__))
    proj_dir = os.path.join(base_dir, 'projection_outputs')
    save_root = os.path.join(base_dir, 'results_refocus', sequence, pose_subdir)
    os.makedirs(save_root, exist_ok=True)

    # 读取位姿文件
    extrinsic_path = os.path.join("./colmap", sequence, pose_subdir, 'images.txt')
    
    if not os.path.exists(extrinsic_path):
        print(f"错误: 位姿文件不存在: {extrinsic_path}")
        exit(1)
        
    with open(extrinsic_path, 'r') as file:
        lines = file.readlines()
    
    Qw_list, Qx_list, Qy_list, Qz_list = [], [], [], []
    Tx_list, Ty_list, Tz_list = [], [], []
    name_list = []
    
    for line in lines:
        if '.jpg' in line or '.png' in line:
            data = line.strip().split()
            # COLMAP images.txt格式: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            q_values = np.float64(data[1:5])  # QW, QX, QY, QZ
            t_values = np.float64(data[5:8])  # TX, TY, TZ
            
            Qw_list.append(q_values[0])
            Qx_list.append(q_values[1])
            Qy_list.append(q_values[2])
            Qz_list.append(q_values[3])
            Tx_list.append(t_values[0])
            Ty_list.append(t_values[1])
            Tz_list.append(t_values[2])
            name_list.append(int(data[-1].split('.')[0]))  # 提取图像ID

    # 排序
    name_list = np.array(name_list)
    indices = np.argsort(name_list)
    name_list = name_list[indices]
    
    param_lists = [Tx_list, Ty_list, Tz_list, Qx_list, Qy_list, Qz_list, Qw_list]
    param_lists = [np.array(param_list)[indices] for param_list in param_lists]
    Tx_list, Ty_list, Tz_list, Qx_list, Qy_list, Qz_list, Qw_list = param_lists
    
    # 转换为torch张量
    Tx_list = torch.tensor(Tx_list, dtype=torch.float32).to(device)
    Ty_list = torch.tensor(Ty_list, dtype=torch.float32).to(device)
    Tz_list = torch.tensor(Tz_list, dtype=torch.float32).to(device)
    
    quat = torch.stack([
        torch.tensor(Qx_list), 
        torch.tensor(Qy_list), 
        torch.tensor(Qz_list), 
        torch.tensor(Qw_list)
    ], dim=1)
    
    R_matrix = R.from_quat(quat).as_matrix()
    R_matrix = torch.tensor(R_matrix).to(device, dtype=torch.float32)
    T_matrix = torch.stack([Tx_list, Ty_list, Tz_list], dim=1).unsqueeze(2)
    T_matrix = T_matrix.detach().to(device, dtype=torch.float32)

    print(f"加载了 {len(name_list)} 个位姿，图像ID范围: {name_list.min()}-{name_list.max()}")

    # 处理每个模态
    for modality in modalities:
        print(f"处理模态: {modality}")
        
        # 创建模态文件夹
        modality_dir = os.path.join(save_root, modality)
        os.makedirs(modality_dir, exist_ok=True)
        
        if modality == 'flir':
            image_dir = os.path.join(proj_dir, 'flir', sequence)
            image_matrix = read_images_by_ids(image_dir, name_list, (H, W))
            image_matrix = torch.from_numpy(image_matrix).to(device, dtype=torch.float32)
            K = torch.tensor(K_flir).to(device, dtype=torch.float32)
            
            for depth in tqdm(depth_range, desc=f"FLIR refocus"):
                with torch.no_grad():
                    refocus_images = refocus(image_matrix, R_matrix, T_matrix, K, torch.tensor(depth).to(device), center_idx=center_idx)
                    visualize_and_save(refocus_images, modality_dir, 'flir', depth)

        elif modality == 'thermal':
            image_dir = os.path.join(proj_dir, 'thermal', sequence)
            image_matrix = read_images_by_ids(image_dir, name_list, (H, W))
            image_matrix = torch.from_numpy(image_matrix).to(device, dtype=torch.float32)
            K = torch.tensor(K_thermal).to(device, dtype=torch.float32)
            
            for depth in tqdm(depth_range, desc=f"Thermal refocus"):
                with torch.no_grad():
                    refocus_images = refocus(image_matrix, R_matrix, T_matrix, K, torch.tensor(depth).to(device), center_idx=center_idx)
                    visualize_and_save(refocus_images, modality_dir, 'thermal', depth)

        elif modality == 'event':
            # Event数据路径
            event_path = os.path.join(proj_dir, 'event', sequence, 'events.raw')
            exposure_times_path = os.path.join(proj_dir, 'event', sequence, 'exposure_times.txt')
            
            if not os.path.exists(event_path):
                print(f"警告: Event数据文件不存在: {event_path}")
                continue
                
            K = torch.tensor(K_event).to(device, dtype=torch.float32)
            
            for depth in tqdm(depth_range, desc=f"Event refocus"):
                with torch.no_grad():
                    event_frame = event_refocus(event_path, exposure_times_path, name_list, R_matrix, T_matrix, K, torch.tensor(depth).to(device), center_idx=center_idx)
                    visualize_and_save(event_frame, modality_dir, 'event', depth, modality='event')

    print(f'所有模态refocus完成！结果保存在: {save_root}')
    print("\n保存结构:")
    print("results_refocus/")
    print("└── 0000/")
    print("    └── 000/")
    print("        ├── flir/")
    print("        │   ├── refocus_depth_15.0.png")
    print("        │   ├── refocus_depth_17.5.png")
    print("        │   ├── refocus_depth_20.0.png")
    print("        │   └── ...")
    print("        ├── thermal/")
    print("        │   ├── refocus_depth_15.0.png")
    print("        │   └── ...")
    print("        └── event/")
    print("            ├── refocus_depth_15.0.png")
    print("            └── ...")