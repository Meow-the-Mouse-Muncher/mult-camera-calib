import os
import glob
import shutil
from tqdm import tqdm

def sort_and_copy_images(src_root, dst_root):
    # 确保目标文件夹存在
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    
    # 获取所有数字命名的子文件夹
    subdirs = [d for d in os.listdir(src_root) 
              if os.path.isdir(os.path.join(src_root, d)) and d.isdigit()]
    
    # 按数字顺序排序文件夹
    subdirs.sort(key=lambda x: int(x))
    
    # 用于追踪目标文件的计数
    dst_counter = 1
    
    # 遍历每个子文件夹
    for subdir in tqdm(subdirs, desc="处理文件夹"):
        src_dir = os.path.join(src_root, subdir)
        
        # 查找PNG文件
        png_files = glob.glob(os.path.join(src_dir, "*.png"))
        if not png_files:
            print(f"Warning: No PNG files found in {src_dir}.")
            continue
            
        # 对当前文件夹中的PNG文件按名称排序
        png_files.sort()
        
        # 复制文件到目标文件夹，并按序号重命名
        for png_file in png_files:
            dst_name = f"{dst_counter}.png"
            dst_path = os.path.join(dst_root, dst_name)
            shutil.copy2(png_file, dst_path)
            dst_counter += 1

if __name__ == "__main__":
    # 源文件夹和目标文件夹路径
    src_root = r"d:\sai\配准标定\biaoding\data3.22"
    dst_root = r"d:\sai\配准标定\biaoding\flir_result"
    
    # 执行排序和复制
    sort_and_copy_images(src_root, dst_root)
    # print(f"完成！共处理了 {dst_counter-1} 个文件")