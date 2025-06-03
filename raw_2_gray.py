import cv2 as cv
from h5py._hl import base
import numpy as np
import os
import glob
import argparse
import shutil

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_rg8_to_gray(input_path, output_folder=None):
    """
    将RG8格式的图像转换为gray格式
    :param input_path: 输入文件夹路径，包含images.raw文件
    :param output_folder: 输出文件夹路径，用于保存转换后的图像。如果为None，则在输入目录下创建文件夹
    """
    # 如果output_folder为None，则在输入目录下创建rgb文件夹
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(input_path), "gray")
    
    # 确保输出目录存在
    ensure_dir(output_folder)
    
    try:
        # 读取raw文件
        with open(input_path, 'rb') as f:
            # 读取文件头信息
            header = np.fromfile(f, dtype=np.int32, count=5)
            num_images = header[0]
            offset_x = header[1]
            offset_y = header[2]
            width = header[3]
            height = header[4]
                 
            # 计算Bayer模式的偏移
            x_odd = (offset_x % 2) == 1
            y_odd = (offset_y % 2) == 1
            
            # 根据偏移选择正确的Bayer模式
            bayer_patterns = {
                (False, False): cv.COLOR_BAYER_RG2RGB,  # 偶数行偶数列 -> RGGB
                (True, False): cv.COLOR_BAYER_GR2RGB,   # 奇数行偶数列 -> GRBG
                (False, True): cv.COLOR_BAYER_GB2RGB,   # 偶数行奇数列 -> GBRG
                (True, True): cv.COLOR_BAYER_BG2RGB     # 奇数行奇数列 -> BGGR
            }
            bayer_pattern = bayer_patterns[(x_odd, y_odd)]
            
            # 读取并处理每张图像
            for i in range(num_images):
                # 读取单张图像数据
                bayer_data = np.fromfile(f, dtype=np.uint8, count=width*height)
                if len(bayer_data) != width*height:
                    print(f"警告：图像 {i} 数据不完整，跳过")
                    continue
                    
                bayer_data = bayer_data.reshape((height, width))
                
                # 转换为RGB
                rgb_image = cv.cvtColor(bayer_data, bayer_pattern)
                gray_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_image = clahe.apply(gray_image)
                
                # 保存转换后的图像
                output_path = os.path.join(output_folder, f"{i+1:04d}.png")
                cv.imwrite(output_path, gray_image)
                
            print(f"处理完成！共转换 {num_images} 张图像")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

def copy_thermal_images(thermal_dir, output_dir):
    """
    复制红外图像到输出目录并逆时针旋转90度
    :param thermal_dir: 红外图像源目录
    :param output_dir: 输出目录
    """
    if not os.path.exists(thermal_dir):
        print(f"警告：红外图像目录不存在 {thermal_dir}")
        return
    
    ensure_dir(output_dir)
    
    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    thermal_files = []
    
    for ext in image_extensions:
        thermal_files.extend(glob.glob(os.path.join(thermal_dir, f"*{ext}")))
        thermal_files.extend(glob.glob(os.path.join(thermal_dir, f"*{ext.upper()}")))
    
    thermal_files.sort()  # 按文件名排序
    
    if not thermal_files:
        print(f"警告：在 {thermal_dir} 中未找到图像文件")
        return
    
    # 复制文件
    copied_count = 0
    for i, thermal_file in enumerate(thermal_files):
        try:
            # 使用统一的命名格式
            output_filename = f"{i+1:04d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 读取图像
            img = cv.imread(thermal_file, cv.IMREAD_UNCHANGED)
            if img is not None:
                # 逆时针旋转90度
                rotated_img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                
                # 保存旋转后的图像为PNG格式
                cv.imwrite(output_path, rotated_img)
                copied_count += 1
            else:
                print(f"警告：无法读取图像 {thermal_file}")
                continue
            
        except Exception as e:
            print(f"复制红外图像时出错 {thermal_file}: {str(e)}")
    
    print(f"红外图像复制完成！共复制并旋转 {copied_count} 张图像")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将RG8格式的图像转换为灰度格式并整理文件结构')
    parser.add_argument('--input_path', default='./data5.25', help='输入文件路径，包含子目录结构')
    parser.add_argument('--output', '-o', default='data', help='输出文件夹路径（可选）')
    parser.add_argument('--copy_thermal', action='store_true', help='是否复制红外图像')
    
    # 解析命令行参数
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input_path):
        # 如果输入是目录，则查找子目录
        base_path = args.input_path
        
        # 获取所有子目录
        subdirs = [x for x in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, x))]
        subdirs.sort(key=lambda x: tuple(map(int, x.split('_'))))
        print(f"找到子目录: {subdirs}")
        
        # 收集所有文件路径
        raw_files = []
        event_raw_files = []
        exposure_times_files = []
        thermal_dirs = []
        
        # 在每个子目录中查找相关文件
        for subdir in subdirs:
            # FLIR raw文件
            raw_file = os.path.join(base_path, subdir, 'flir', "images.raw")
            if os.path.isfile(raw_file):
                raw_files.append(raw_file)
            else:
                print(f"警告：未找到 {raw_file}")
                raw_files.append(None)
            
            # Event raw文件
            event_raw = os.path.join(base_path, subdir, "event", "event.raw")
            event_raw_files.append(event_raw)
            
            # 曝光时间文件
            exposure_times_file = os.path.join(base_path, subdir, 'flir', "exposure_times.txt")
            exposure_times_files.append(exposure_times_file)
            
            # 红外图像目录
            thermal_dir = os.path.join(base_path, subdir, 'thermal')
            thermal_dirs.append(thermal_dir)
        
        # 检查是否找到了有效的文件
        valid_files = [f for f in raw_files if f is not None]
        if not valid_files:
            print(f"错误：在目录 {args.input_path} 中找不到任何有效的 images.raw 文件")
            return
        
        print(f"开始处理 {len(subdirs)} 个子目录...")
        
        # 处理每个子目录
        for i, (raw_file, event_raw, exposure_times_file, thermal_dir) in enumerate(
            zip(raw_files, event_raw_files, exposure_times_files, thermal_dirs)
        ):
            print(f"\n处理第 {i+1}/{len(subdirs)} 个目录: {subdirs[i]}")
            
            # 创建输出目录结构
            base_output_dir = os.path.join(args.output, f'{i:04d}')
            
            # 处理FLIR图像
            if raw_file and os.path.exists(raw_file):
                flir_output_dir = os.path.join(base_output_dir, 'flir')
                ensure_dir(flir_output_dir)
                print(f"  转换FLIR图像: {raw_file}")
                convert_rg8_to_gray(raw_file, flir_output_dir)
            else:
                print(f"  跳过FLIR图像转换（文件不存在）")
            
            # 处理Event数据
            event_output_dir = os.path.join(base_output_dir, "event")
            ensure_dir(event_output_dir)
            
            # 复制event.raw文件
            if os.path.exists(event_raw):
                try:
                    shutil.copyfile(event_raw, os.path.join(event_output_dir, "events.raw"))
                    print(f"  复制Event数据: {event_raw}")
                except Exception as e:
                    print(f"  复制Event数据失败: {str(e)}")
            else:
                print(f"  警告：Event文件不存在 {event_raw}")
            
            # 复制exposure_times.txt文件
            if os.path.exists(exposure_times_file):
                try:
                    shutil.copyfile(exposure_times_file, os.path.join(event_output_dir, "exposure_times.txt"))
                    print(f"  复制曝光时间文件: {exposure_times_file}")
                except Exception as e:
                    print(f"  复制曝光时间文件失败: {str(e)}")
            else:
                print(f"  警告：曝光时间文件不存在 {exposure_times_file}")
            
            # 处理红外图像
            if args.copy_thermal or True:  # 默认复制红外图像
                thermal_output_dir = os.path.join(base_output_dir, 'thermal')
                print(f"  复制红外图像: {thermal_dir}")
                copy_thermal_images(thermal_dir, thermal_output_dir)
        
        print(f"\n全部处理完成！输出目录: {args.output}")
        print(f"处理了 {len(subdirs)} 个子目录")
            
    else:
        print(f"错误：输入路径 {args.input_path} 不存在")
        return

if __name__ == "__main__":
    main()