import numpy as np
import cv2
import os
import glob
from scipy import io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import time
import torch
from torch import Tensor
import sys
import torchvision
# sys.path.append("/home/nvidia/openeb/sdk/modules/core/python/pypkg")
# sys.path.append("/home/nvidia/openeb/build/py3")
sys.path.append("/usr/lib/python3/dist-packages")
from metavision_core.event_io import EventsIterator, RawReader
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm

def undistort_image(img, K, dist_coeffs):
    """
    去除图像畸变
    """
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)
    return undistorted

def compute_projection_mapping(P_source, P_target, target_h, target_w, Z=0):
    """
    计算从源相机到目标相机的投影映射
    
    参数:
    - P_source: 源相机投影矩阵
    - P_target: 目标相机投影矩阵
    - target_h, target_w: 目标图像尺寸
    - Z: 世界坐标系中的Z平面
    
    返回:
    - map_x, map_y: 映射坐标
    """
    # 创建目标相机的坐标网格
    x_coords, y_coords = np.meshgrid(np.arange(target_w), np.arange(target_h))
    
    # 使用目标相机的投影矩阵进行反投影
    p11, p12, p14 = P_target[0, 0], P_target[0, 1], P_target[0, 3]
    p21, p22, p24 = P_target[1, 0], P_target[1, 1], P_target[1, 3]
    p31, p32, p34 = P_target[2, 0], P_target[2, 1], P_target[2, 3]
    
    # 获取目标图像的所有像素坐标
    u = x_coords.flatten()
    v = y_coords.flatten()

    print("求解线性方程组...")
    # 分批处理以减少内存使用
    batch_size = 100000
    num_batches = (len(u) + batch_size - 1) // batch_size
    
    X_result = np.zeros_like(u, dtype=np.float64)
    Y_result = np.zeros_like(u, dtype=np.float64)
    lambda_result = np.zeros_like(u, dtype=np.float64)
    
    for i in tqdm(range(num_batches), desc="处理像素批次"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(u))
        
        u_batch = u[start_idx:end_idx]
        v_batch = v[start_idx:end_idx]
        
        # 构建系数矩阵
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
        
        b_batch = np.tile([-p14, -p24, -p34], (end_idx - start_idx, 1))
        
        try:
            X_batch = np.linalg.solve(A_batch, b_batch)
            X_result[start_idx:end_idx] = X_batch[:, 0]
            Y_result[start_idx:end_idx] = X_batch[:, 1]
            lambda_result[start_idx:end_idx] = X_batch[:, 2]
        except np.linalg.LinAlgError:
            print(f"警告: 批次 {i} 中存在奇异矩阵，跳过该批次")
            continue

    # 构建世界坐标
    world_coords = np.zeros((4, len(u)), dtype=np.float64)
    world_coords[0, :] = X_result
    world_coords[1, :] = Y_result
    world_coords[2, :] = Z
    world_coords[3, :] = 1
    
    # 投影到源相机坐标系
    source_proj = P_source @ world_coords
    source_proj = source_proj / source_proj[2, :]
    
    # 获取映射坐标
    map_x = source_proj[0, :].reshape(target_h, target_w)
    map_y = source_proj[1, :].reshape(target_h, target_w)
    
    return map_x, map_y

def compute_thermal_to_flir_mapping(P_thermal, P_flir, flir_h, flir_w, Z=0):
    """
    计算从Thermal相机到FLIR相机的投影映射
    参考projection_f2ir.py的正确实现
    """
    # 创建FLIR图像的坐标网格 (目标视角)
    x_coords, y_coords = np.meshgrid(np.arange(flir_w), np.arange(flir_h))
    
    # 使用FLIR相机的投影矩阵进行反投影
    p11, p12, p14 = P_flir[0, 0], P_flir[0, 1], P_flir[0, 3]
    p21, p22, p24 = P_flir[1, 0], P_flir[1, 1], P_flir[1, 3]
    p31, p32, p34 = P_flir[2, 0], P_flir[2, 1], P_flir[2, 3]
    
    # 获取FLIR图像的所有像素坐标
    u = x_coords.flatten()
    v = y_coords.flatten()

    print("计算Thermal到FLIR映射...")
    # 分批处理
    batch_size = 100000
    num_batches = (len(u) + batch_size - 1) // batch_size
    
    X_result = np.zeros_like(u, dtype=np.float64)
    Y_result = np.zeros_like(u, dtype=np.float64)
    lambda_result = np.zeros_like(u, dtype=np.float64)
    
    for i in tqdm(range(num_batches), desc="处理Thermal->FLIR映射"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(u))
        
        u_batch = u[start_idx:end_idx]
        v_batch = v[start_idx:end_idx]
        
        # 构建系数矩阵
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
        
        b_batch = np.tile([-p14, -p24, -p34], (end_idx - start_idx, 1))
        
        try:
            X_batch = np.linalg.solve(A_batch, b_batch)
            X_result[start_idx:end_idx] = X_batch[:, 0]
            Y_result[start_idx:end_idx] = X_batch[:, 1]
            lambda_result[start_idx:end_idx] = X_batch[:, 2]
        except np.linalg.LinAlgError:
            print(f"警告: 批次 {i} 中存在奇异矩阵，跳过该批次")
            continue

    # 构建世界坐标
    world_coords = np.zeros((4, len(u)), dtype=np.float64)
    world_coords[0, :] = X_result
    world_coords[1, :] = Y_result
    world_coords[2, :] = Z
    world_coords[3, :] = 1
    
    # 投影到Thermal相机坐标系
    thermal_proj = P_thermal @ world_coords
    thermal_proj = thermal_proj / thermal_proj[2, :]
    
    # 获取映射坐标
    map_x = thermal_proj[0, :].reshape(flir_h, flir_w)
    map_y = thermal_proj[1, :].reshape(flir_h, flir_w)
    
    return map_x, map_y

def generate_event_frames(event_file, triggers, height=600, width=600):
    """
    生成事件帧
    """
    event_frames = []
    
    # Instantiate the frame generator
    mv_iterator = EventsIterator(input_path=event_file, delta_t=1e6)
    on_demand_gen = OnDemandFrameGenerationAlgorithm(height, width, accumulation_time_us=10000)
    
    # 预处理事件数据
    for evs in mv_iterator:
        # 根据参考代码调整事件坐标
        evs['x'] = evs['x'] - 340
        evs['y'] = evs['y'] - 60 
        on_demand_gen.process_events(evs)
    
    # 为每个触发时间生成帧
    for i, trigger_time in enumerate(tqdm(triggers, desc="生成事件帧")):
        frame = np.zeros((height, width, 3), np.uint8)
        timestamp = int(trigger_time['t'])
        on_demand_gen.generate(timestamp, frame)
        event_frames.append(frame)
    
    return event_frames

def main():
    # 加载 FLIR-Event 相机参数
    stereo_params_fe = io.loadmat('./stereoParams_FE.mat')['stereoParams']
    
    # FLIR 相机参数
    K_flir = stereo_params_fe['K1'][0, 0]
    dist_flir = np.hstack([
        stereo_params_fe['RadialDistortion1'][0, 0].flatten(),
        stereo_params_fe['TangentialDistortion1'][0, 0].flatten()
    ])
    
    # Event 相机参数
    K_event = stereo_params_fe['K2'][0, 0]
    dist_event = np.hstack([
        stereo_params_fe['RadialDistortion2'][0, 0].flatten(),
        stereo_params_fe['TangentialDistortion2'][0, 0].flatten()
    ])
    
    # 获取相机投影矩阵
    R_flir_fe = stereo_params_fe['R1'][0, 0]
    R_event = stereo_params_fe['R2'][0, 0]
    T_flir_fe = stereo_params_fe['T1'][0, 0]
    T_event = stereo_params_fe['T2'][0, 0]

    P_flir_fe = K_flir @ np.hstack([R_flir_fe, T_flir_fe.reshape(3, 1)])
    P_event = K_event @ np.hstack([R_event, T_event.reshape(3, 1)])

    # 加载 FLIR-Thermal 相机参数
    stereo_params_fi = io.loadmat('./stereoParams_FI.mat')['stereoParams']
    
    # Thermal 相机参数
    K_thermal = stereo_params_fi['K2'][0, 0]
    dist_thermal = np.hstack([
        stereo_params_fi['RadialDistortion2'][0, 0].flatten(),
        stereo_params_fi['TangentialDistortion2'][0, 0].flatten()
    ])
    
    # FLIR-Thermal 投影矩阵
    R_flir_fi = stereo_params_fi['R1'][0, 0]
    R_thermal = stereo_params_fi['R2'][0, 0]
    T_flir_fi = stereo_params_fi['T1'][0, 0]
    T_thermal = stereo_params_fi['T2'][0, 0]

    P_flir_fi = K_flir @ np.hstack([R_flir_fi, T_flir_fi.reshape(3, 1)])
    P_thermal = K_thermal @ np.hstack([R_thermal, T_thermal.reshape(3, 1)])

    # 图像尺寸设置
    flir_h, flir_w = 1800, 1800
    thermal_h, thermal_w = 640, 512
    event_h, event_w = 600, 600

    # 修正投影映射计算
    print("计算 FLIR -> Event 投影映射...")
    map_x_flir2event, map_y_flir2event = compute_projection_mapping(
        P_flir_fe, P_event, event_h, event_w, Z=0
    )

    print("计算 Thermal -> FLIR 投影映射...")
    # 使用修正后的函数
    map_x_thermal2flir, map_y_thermal2flir = compute_thermal_to_flir_mapping(
        P_thermal, P_flir_fi, flir_h, flir_w, Z=0
    )

    print("计算 Thermal -> Event 投影映射...")
    map_x_thermal2event, map_y_thermal2event = compute_projection_mapping(
        P_thermal, P_event, event_h, event_w, Z=0
    )

    # 获取所有子目录
    base_path = './data'
    subdirs = [x for x in os.listdir(base_path)]
    subdirs.sort(key=lambda x: tuple(map(int, x.split())))

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
        
        # Event数据文件
        event_file = os.path.join(subdir_path, 'event', 'events.raw')
        event_dir = os.path.join(subdir_path, 'event')
        
        if not flir_files:
            print(f"警告：在 {flir_dir} 中未找到FLIR图像")
            continue
            
        if not thermal_files:
            print(f"警告：在 {thermal_dir} 中未找到thermal图像")
            continue
        
        if not os.path.exists(event_file):
            print(f"警告：在 {subdir_path} 中未找到event数据文件")
            continue
        
        num_images = min(len(flir_files), len(thermal_files))
        print(f"处理子目录 {subdir}：{num_images} 张图像")
        
        # 创建输出目录
        os.makedirs(f'projection_outputs/flir/{subdir}', exist_ok=True)
        os.makedirs(f'projection_outputs/thermal_to_flir/{subdir}', exist_ok=True)
        os.makedirs(f'projection_outputs/thermal/{subdir}', exist_ok=True)
        os.makedirs(f'projection_outputs/event/{subdir}', exist_ok=True)
        os.makedirs(f'projection_outputs/overlay_all/{subdir}', exist_ok=True)
        os.makedirs(f'projection_outputs/overlay_flir_event/{subdir}', exist_ok=True)
        
        # === 复制Event文件夹内容 ===
        output_event_dir = f'projection_outputs/event/{subdir}'
        print(f"复制Event数据从 {event_dir} 到 {output_event_dir}")
        
        # 复制events.raw文件
        if os.path.exists(event_file):
            shutil.copy2(event_file, os.path.join(output_event_dir, 'events.raw'))
            print(f"  - 复制 events.raw")
        
        # 复制exposure_times.txt文件
        exposure_times_file = os.path.join(event_dir, 'exposure_times.txt')
        if os.path.exists(exposure_times_file):
            shutil.copy2(exposure_times_file, os.path.join(output_event_dir, 'exposure_times.txt'))
            print(f"  - 复制 exposure_times.txt")
        
        # 复制其他Event相关文件（如果存在）
        for item in os.listdir(event_dir):
            item_path = os.path.join(event_dir, item)
            if os.path.isfile(item_path) and item not in ['events.raw', 'exposure_times.txt']:
                shutil.copy2(item_path, os.path.join(output_event_dir, item))
                print(f"  - 复制 {item}")
        
        # === 处理Event数据 ===
        try:
            # 读取曝光时间
            exposure_times = np.loadtxt(exposure_times_file).reshape(-1)
            print(f"找到 {len(exposure_times)} 个曝光时间")
            
            # 读取触发事件
            with RawReader(event_file, do_time_shifting=False) as ev_data:
                while not ev_data.is_done():
                    ev_data.load_n_events(1000000)
                triggers = ev_data.get_ext_trigger_events()
                triggers = triggers[triggers['p'] == 0].copy()
                print(f"读取到 {len(triggers)} 个触发事件")
                if len(triggers) ==151:
                    triggers = triggers[1:]  # 之后
                triggers['t'] = triggers['t'] + exposure_times
            
            # 生成事件帧
            event_frames = generate_event_frames(event_file, triggers, event_h, event_w)
            
        except Exception as e:
            print(f"处理Event数据时出错: {e}")
            print(f"文件夹 {subdir} 的Event处理失败，创建空白帧")
            # 如果Event处理失败，创建空白帧
            event_frames = [np.zeros((event_h, event_w, 3), np.uint8) for _ in range(num_images)]
        
        # 处理图像对
        for i in range(num_images):
            # === 处理 FLIR -> Event ===
            I_flir = cv2.imread(flir_files[i], cv2.IMREAD_GRAYSCALE)
            if I_flir.shape[0] != flir_h or I_flir.shape[1] != flir_w:
                I_flir = cv2.resize(I_flir, (flir_w, flir_h))
            
            # 去畸变FLIR图像
            # I_flir_undistorted = undistort_image(I_flir, K_flir, dist_flir)
            
            # FLIR -> Event 投影
            map_x_float = map_x_flir2event.astype(np.float32)
            map_y_float = map_y_flir2event.astype(np.float32)

            shift_x = -150  # 负数向右平移
            shift_y = -35  # 向上平移10像素（正数表示向上，负数表示向下）
            
            flir_in_event_view = cv2.remap(
                I_flir,
                map_x_float,
                map_y_float,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # 创建有效区域掩码
            mask_flir = ((map_x_flir2event >= 0) & (map_x_flir2event < flir_w) & 
                        (map_y_flir2event >= 0) & (map_y_flir2event < flir_h))
            mask_flir = mask_flir.astype(np.uint8) * 255
            
            flir_in_event_view = cv2.bitwise_and(flir_in_event_view, flir_in_event_view, mask=mask_flir)
            cv2.imwrite(f'projection_outputs/flir/{subdir}/{i+1:04d}.png', flir_in_event_view)

            # === 处理 Thermal -> FLIR -> Event ===
            I_thermal = cv2.imread(thermal_files[i], cv2.IMREAD_GRAYSCALE)
            if I_thermal.shape[0] != thermal_h or I_thermal.shape[1] != thermal_w:
                I_thermal = cv2.resize(I_thermal, (thermal_w, thermal_h))
            
            # 去畸变thermal图像
            # I_thermal_undistorted = undistort_image(I_thermal, K_thermal, dist_thermal)
            
            # Thermal -> FLIR 映射
            map_x_t2f_float = map_x_thermal2flir.astype(np.float32)
            map_y_t2f_float = map_y_thermal2flir.astype(np.float32)
            
            thermal_in_flir_view = cv2.remap(
                I_thermal,
                map_x_t2f_float,
                map_y_t2f_float,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # 修正掩码计算
            mask_t2f = ((map_x_thermal2flir >= 0) & (map_x_thermal2flir < thermal_w) & 
                       (map_y_thermal2flir >= 0) & (map_y_thermal2flir < thermal_h))
            mask_t2f = mask_t2f.astype(np.uint8) * 255
            
            thermal_in_flir_view = cv2.bitwise_and(thermal_in_flir_view, thermal_in_flir_view, mask=mask_t2f)
            cv2.imwrite(f'projection_outputs/thermal_to_flir/{subdir}/{i+1:04d}.png', thermal_in_flir_view)
            
            # 步骤2: (Thermal->FLIR) -> Event with 人为平移
            shift_x = -150  # 可调整的平移参数
            shift_y = -35
            map_x_shifted = map_x_float + shift_x
            map_y_shifted = map_y_float + shift_y
            
            thermal_flir_in_event_view = cv2.remap(
                thermal_in_flir_view,
                map_x_shifted,
                map_y_shifted,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            thermal_in_event_view = cv2.bitwise_and(thermal_flir_in_event_view, thermal_flir_in_event_view, mask=mask_flir)
            
            cv2.imwrite(f'projection_outputs/thermal/{subdir}/{i+1:04d}.png', thermal_in_event_view)

            # === 保存Event帧 ===
            if i < len(event_frames):
                event_frame = event_frames[i]
                cv2.imwrite(f'projection_outputs/event/{subdir}/{i:04d}.png', event_frame)
            else:
                event_frame = np.zeros((event_h, event_w, 3), np.uint8)

            # === 创建多种叠加图像 ===
            
            # 1. FLIR(红色) + Thermal(绿色) + Event(蓝色)
            overlay_all = np.zeros((event_h, event_w, 3), dtype=np.uint8)
            overlay_all[:, :, 2] = flir_in_event_view      # 红色通道：FLIR -> Event
            overlay_all[:, :, 1] = thermal_in_event_view   # 绿色通道：Thermal -> Event
            overlay_all[:, :, 0] = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)  # 蓝色通道：Event
            cv2.imwrite(f'projection_outputs/overlay_all/{subdir}/{i+1:04d}.png', overlay_all)
            
            # 2. FLIR + Event 叠加 (参考projection_f2e_overlay.py)
            flir_bgr = cv2.cvtColor(flir_in_event_view, cv2.COLOR_GRAY2BGR)
            overlay_flir_event = cv2.addWeighted(flir_bgr, 0.3, event_frame, 0.7, 0)
            cv2.imwrite(f'projection_outputs/overlay_flir_event/{subdir}/{i+1:04d}.png', overlay_flir_event)

    print("\n处理完成！结果已保存到 projection_outputs 文件夹")
    print("- flir_to_event/: FLIR图像投影到Event视角")
    print("- thermal_to_flir/: Thermal图像投影到FLIR视角")  
    print("- thermal_to_event/: Thermal图像投影到Event视角")
    print("- event/: Event帧 + 原始Event数据文件")
    print("- overlay_all/: FLIR(红)+Thermal(绿)+Event(蓝)叠加")
    print("- overlay_flir_event/: FLIR+Event加权叠加")

if __name__ == "__main__":
    main()