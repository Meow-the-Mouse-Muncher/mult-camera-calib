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
from metavision_core.event_io import EventsIterator, RawReader
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm

def process_event_file(event_file, timestamps_file, output_dir):
    """
    处理单个事件文件，生成事件帧
    
    参数:
    event_file: event.raw文件路径
    timestamps_file: TimeStamps.txt文件路径
    output_dir: 输出目录
    """
    print(f"处理文件: {event_file}")
    
    # 检查文件是否存在
    if not os.path.exists(event_file):
        print(f"警告: 事件文件不存在 {event_file}")
        return
    
    if not os.path.exists(timestamps_file):
        print(f"警告: 时间戳文件不存在 {timestamps_file}")
        return
    
    try:
        # 读取时间戳
        exposure_times = np.loadtxt(timestamps_file).reshape(-1)
        print(f"找到 {len(exposure_times)} 个曝光时间")
        with RawReader(event_file, do_time_shifting=False) as ev_data:
            while not ev_data.is_done():
                ev_data.load_n_events(1000000)
            triggers = ev_data.get_ext_trigger_events()
            triggers = triggers[triggers['p'] == 0].copy()
            triggers['t'] = triggers['t']
        print(f"找到 {len(triggers)} 个触发器事件")
        event_frames = []
        # 实例化帧生成器
        height, width = 600, 600
        mv_iterator = EventsIterator(input_path=event_file, delta_t=1e6)
        on_demand_gen = OnDemandFrameGenerationAlgorithm(height, width, accumulation_time_us=10000)
        frame = np.zeros((height, width, 3), np.uint8)
        
        # 处理事件数据
        for evs in mv_iterator:
            if evs.size > 0:  # 确保事件数组不为空
                evs['x'] = evs['x'] - 340
                evs['y'] = evs['y'] - 60
                # PROPHESEE_ROI_X0 = 340
                # PROPHESEE_ROI_Y0 = 60
                # PROPHESEE_ROI_X1 = 939
                # PROPHESEE_ROI_Y1 = 659
                on_demand_gen.process_events(evs)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成事件帧
        for i, trigger_time in enumerate(tqdm(triggers, desc=f"生成事件帧 - {os.path.basename(output_dir)}")):
            # 创建帧缓冲区
            frame = np.zeros((height, width, 3), np.uint8)
            # 在指定时间戳生成事件帧
            timestamp = int(trigger_time['t'])
            on_demand_gen.generate(timestamp, frame)
            # 保存事件帧
            event_frames.append(frame)
            output_path = os.path.join(output_dir, f"{i:04d}.png")
            cv2.imwrite(output_path, frame)
        
        print(f"完成处理: {len(triggers)} 帧已保存到 {output_dir}")
        
    except Exception as e:
        print(f"处理文件时出错 {event_file}: {str(e)}")

def batch_process_event_files(base_path):
    """
    批量处理指定路径下所有子目录中的事件文件
    
    参数:
    base_path: 基础路径，包含多个子目录
    """
    print(f"开始批量处理目录: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"错误: 目录不存在 {base_path}")
        return
    
    # 获取所有子目录
    subdirs = [x for x in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, x))]
    subdirs.sort(key=lambda x: tuple(map(int, x.split('_'))))
    subdirs.sort(key=lambda x: tuple(map(int, x.split('_'))))
    
    
    print(f"找到 {len(subdirs)} 个子目录: {subdirs}")
    
    processed_count = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
                # 获取所有子目录

        
        # 构建文件路径
        event_file = os.path.join(subdir_path, "event", "event.raw")
        timestamps_file = os.path.join(subdir_path, "flir", "exposure_times.txt")
        
        # 检查必要文件是否存在
        if os.path.exists(event_file):
            # 创建输出目录
            output_dir = os.path.join("EFrame", subdir)
            
            # 处理事件文件
            process_event_file(event_file, timestamps_file, output_dir)
            processed_count += 1
        else:
            print(f"跳过目录 {subdir}: 未找到 event.raw 文件")
    
    print(f"批量处理完成! 共处理了 {processed_count} 个目录")

if __name__ == "__main__":
    # 设置基础路径
    base_path = r"D:\sai\calib_sai\biaoding\data5.27"
    
    # 执行批量处理
    batch_process_event_files(base_path)