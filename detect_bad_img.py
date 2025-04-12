import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def is_black_image(image, threshold=5):
    """
    判断图像是否为全黑图像
    
    参数:
    - image: 输入图像
    - threshold: 平均像素阈值，低于此值视为全黑
    
    返回:
    - True 如果图像是全黑的，否则 False
    """
    if image is None:
        return True
    
    # 计算平均像素值
    mean_value = np.mean(image)
    
    # 判断是否为全黑图像
    return mean_value < threshold

def main():
    # 数据根目录
    data_root = "data"
    base_path = os.path.join("d:\\sai\\calib_sai\\Preprocess", data_root)
    
    # 确保数据目录存在
    if not os.path.exists(base_path):
        print(f"错误: 找不到目录 {base_path}")
        return
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subdirs.sort()
    
    print(f"找到 {len(subdirs)} 个子目录")
    
    # 处理每个子目录
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        
        # 查找图像 - 支持多种格式和位置
        image_patterns = [
            os.path.join(subdir_path, "*.png"),
            os.path.join(subdir_path, "*.jpg"),
            os.path.join(subdir_path, "*.bmp"),
            os.path.join(subdir_path, "gray", "*.png"),
            os.path.join(subdir_path, "gray", "*.jpg")
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(pattern))
        
        # 如果没有找到图像，尝试寻找images.raw文件
        if not image_files and os.path.exists(os.path.join(subdir_path, "images.raw")):
            print(f"目录 {subdir} 只有raw文件，需要先转换为图像格式")
            continue
        
        if not image_files:
            print(f"在目录 {subdir} 中未找到图像文件")
            continue
        
        # 排序图像文件
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        print(f"处理目录 {subdir}，包含 {len(image_files)} 张图像")
        
        # 检测全黑图像
        black_image_indices = []
        for i, img_path in enumerate(tqdm(image_files, desc=f"检测 {subdir} 中的全黑图像")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if is_black_image(img):
                # 获取图像序号
                img_index = int(os.path.splitext(os.path.basename(img_path))[0])
                black_image_indices.append(img_index)
        
        # 保存结果
        result_file = os.path.join(subdir_path, "black_images.txt")
        with open(result_file, 'w') as f:
            f.write("# black image indices\n")
            for idx in black_image_indices:
                f.write(f"{idx}\n")
        
        print(f"目录 {subdir} 中检测到 {len(black_image_indices)} 张全黑图像")
        print(f"结果已保存到 {result_file}")
        
        # 对比exposure_times.txt中的零值
        exp_time_file = os.path.join(subdir_path, "exposure_times.txt")
        if os.path.exists(exp_time_file):
            try:
                exp_times = np.loadtxt(exp_time_file)
                zero_exp_indices = np.where(exp_times == 0)[0] + 1  # 假设索引从1开始
                
                print(f"exposure_times.txt中有 {len(zero_exp_indices)} 个曝光时间为0的项")
                
                # 检查是否有不一致情况
                set_black = set(black_image_indices)
                set_zero_exp = set(zero_exp_indices)
                
                if set_black != set_zero_exp:
                    print("注意: 检测到的全黑图像与曝光时间为0的项不完全一致")
                    print(f"只在检测中发现的全黑图像: {set_black - set_zero_exp}")
                    print(f"只在曝光时间中发现的零值项: {set_zero_exp - set_black}")
            except:
                print(f"无法解析 {exp_time_file}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()