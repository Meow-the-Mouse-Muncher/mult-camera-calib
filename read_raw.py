# import sys
# sys.path.append('/usr/lib/python3/dist-packages')

# # 如果需要使用 ROS 包
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

from metavision_core.event_io import EventsIterator
import h5py
import torch
import tqdm
import os 
import numpy as np


if __name__ == '__main__':
    #path = "/mnt/sda/XDP/dataout/in_ex01_cab02/event/event.raw"
    path="/mnt/sda/sjy/camera_2_calibration/data3.22/3/event/event.raw"
    path = os.path.abspath(path)
    print(path)
    print(os.path.exists(path))

    # 预先计算大致的事件数量
    # 假设每秒约100万个事件（根据实际情况调整）
    estimated_events = int(60 * 1e6)  # 60秒 * 每秒100万个事件
    
    # 预分配numpy数组
    x_array = np.zeros(estimated_events, dtype=np.int32)
    y_array = np.zeros(estimated_events, dtype=np.int32)
    p_array = np.zeros(estimated_events, dtype=np.int32)
    t_array = np.zeros(estimated_events, dtype=np.int64)
    
    current_idx = 0
    chunk_size = 1000000  # 每次处理的事件数量

    mv_iterator = EventsIterator(input_path=path, delta_t=1000000, start_ts=0,
                                max_duration=1e6 * 60)
    total_steps = int((1e6 * 60) // 1000000)

    print("开始读取事件数据...")
    for evs in tqdm.tqdm(mv_iterator, total=total_steps, desc="读取事件"):
        # 获取当前批次的事件数量
        batch_events = len(evs)
        
        # 如果预分配的空间不够，扩展数组
        if current_idx + batch_events > len(x_array):
            new_size = len(x_array) + estimated_events
            x_array.resize(new_size, refcheck=False)
            y_array.resize(new_size, refcheck=False)
            p_array.resize(new_size, refcheck=False)
            t_array.resize(new_size, refcheck=False)
        
        # 批量赋值
        x_array[current_idx:current_idx+batch_events] = evs['x']
        y_array[current_idx:current_idx+batch_events] = evs['y']
        p_array[current_idx:current_idx+batch_events] = evs['p']
        t_array[current_idx:current_idx+batch_events] = evs['t']
        
        current_idx += batch_events

    # 裁剪到实际大小
    x_array = x_array[:current_idx]
    y_array = y_array[:current_idx]
    p_array = p_array[:current_idx]
    t_array = t_array[:current_idx]

    print("保存到HDF5文件...")
    #save_path = "./data/events1.h5"
    save_path = "/mnt/sda/sjy/camera_2_calibration/data3.22/3/event/event/event.h5"
    with h5py.File(save_path, 'w') as f:
        # 使用压缩来节省存储空间
        f.create_dataset('x', data=x_array, compression='gzip', compression_opts=1)
        f.create_dataset('y', data=y_array, compression='gzip', compression_opts=1)
        f.create_dataset('p', data=p_array, compression='gzip', compression_opts=1)
        f.create_dataset('t', data=t_array, compression='gzip', compression_opts=1)
    
    print(f"数据已保存到 {save_path}")
    print(f"事件总数: {current_idx}")