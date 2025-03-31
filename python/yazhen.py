#import dv_processing as dv
import cv2
import numpy as np
import torch
import os
import glob
import argparse
from tqdm import tqdm
from enum import Enum
from metavision_core.event_io import EventsIterator
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class Filter(Enum):
    NONE = 0
    ACTIVITY = 1
    STC = 2
    TRAIL = 3

# 全局参数设置
FILTER_TYPE = 'STC'  # 可选: 'NONE', 'ACTIVITY', 'STC', 'TRAIL'
'''
          "  - A: Filter events using the activity noise filter algorithm\n"
          "  - T: Filter events using the trail filter algorithm\n"
          "  - S: Filter events using the spatio temporal contrast algorithm\n"
          "  - N: Show all events\n"'
'''
ACTIVITY_TIME_THS = 20000  # 活动过滤时间窗口(微秒)
ACTIVITY_THS = 2  # 邻域最小事件数
STC_FILTER_THS = 25000  # STC过滤时间窗口(微秒) 1s=1e3ms=1e6us
TRAIL_FILTER_THS = 1000000  # 轨迹过滤时间窗口(微秒)

def parse_argument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## dir params
    parser.add_argument('--raw_file_dir', type=str,
                        default='../data3.22/',
                        help='The path of a raw dir')
    parser.add_argument('--save_dir', type=str,
                        default='../image',
                        help='output dir')
    parser.add_argument('--time_interval_us', type=int, default=30000,
                        help='reconstruction frequency 20Hz')
    parser.add_argument('--show', default=True, help='if show detection and tracking process')
    parser.add_argument('--output_pattern', type=str, default='Light',
                        help='output pattern .choose your event is dark or light')
    return parser

def get_reader(file_path, time_interval_us):
    assert os.path.exists(file_path), 'The file \'{}\' is not exist'.format(file_path)
    mv_iterator = EventsIterator(input_path=file_path, delta_t=time_interval_us)

    return mv_iterator

def initialize_filters(width, height):
    """Initialize noise filters with hard-coded parameters"""
    filters = {
        Filter.ACTIVITY: ActivityNoiseFilterAlgorithm(width, height, ACTIVITY_TIME_THS),
        Filter.TRAIL: TrailFilterAlgorithm(width, height, TRAIL_FILTER_THS),
        Filter.STC: SpatioTemporalContrastAlgorithm(width, height, STC_FILTER_THS, True)
    }
    return filters

def main():
    ## Get params
    args, _ = parse_argument().parse_known_args(None)
    print(args)

    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(args.raw_file_dir) 
              if os.path.isdir(os.path.join(args.raw_file_dir, d))]
    print(f"找到子文件夹: {subdirs}")

    for subdir in tqdm(subdirs, desc="处理子文件夹"):
        raw_dir_path = os.path.join(args.raw_file_dir, subdir)
        save_dir_path = os.path.join(args.save_dir, subdir)
        
        # 获取当前子文件夹中的所有.raw文件
        raw_path_list = glob.glob(os.path.join(raw_dir_path,'event', '*.raw'))
        print(f"\n处理文件夹 {subdir} 中的 {len(raw_path_list)} 个RAW文件")

        for raw_path in tqdm(raw_path_list, desc=f"处理{subdir}中的文件"):
            assert os.path.exists(raw_path)
            base_name = os.path.basename(raw_path).split('.')[0]

            # 在对应子文件夹下创建保存路径
            ev_save_path = save_dir_path
            if not os.path.exists(ev_save_path):
                os.makedirs(ev_save_path)

            reader = get_reader(raw_path, args.time_interval_us)
            height, width = 600,600

            # 初始化过滤器
            filters = initialize_filters(width, height)
            filter_type = Filter[FILTER_TYPE]
            events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

            ev_cnt = 0
            for evs in reader:
                # 首先对原始事件数据进行坐标平移
                x_min = evs['x'].min()
                y_min = evs['y'].min()
                evs['x'] = evs['x'] - x_min
                evs['y'] = evs['y'] - y_min
                # 应用过滤器
                if filter_type != Filter.NONE:
                    filters[filter_type].process_events(evs, events_buf)
                    filtered_evs = events_buf.numpy() 
                else:
                    filtered_evs = evs

                mask = filtered_evs['p'] > 0
                if args.output_pattern == 'Dark':
                    if len(filtered_evs['t']) > 0:
                        min_t, max_t = np.min(filtered_evs['t']), np.max(filtered_evs['t'])
                        event_image = np.zeros((height, width))
                        np.add.at(event_image, (filtered_evs['y'][mask], filtered_evs['x'][mask]), 1)
                        cv2.imwrite('{}/{:06d}_{}_{}.png'.format(ev_save_path, ev_cnt, min_t, max_t), 
                                  (event_image * 60).astype(np.uint8))
                        ev_cnt += 1
                    else:
                        ev_cnt += 1
                        print(f"Warning: Empty events at index {ev_cnt} in {raw_path}")

                elif args.output_pattern == 'Light':
                    if len(filtered_evs['t']) > 0:
                        min_t, max_t = np.min(filtered_evs['t']), np.max(filtered_evs['t'])
                        event_image = np.ones((height, width)) * 255
                        np.add.at(event_image, (filtered_evs['y'][mask], filtered_evs['x'][mask]), 16)
                        cv2.imwrite('{}/{:06d}_{}_{}.png'.format(ev_save_path, ev_cnt, min_t, max_t), 
                                  event_image.astype(np.uint8))
                        ev_cnt += 1
                    else:
                        ev_cnt += 1
                        print(f"Warning: Empty events at index {ev_cnt} in {raw_path}")
            
            print(f"完成文件 {raw_path} 的处理，共生成 {ev_cnt} 帧")

if __name__ == '__main__':
    main()

