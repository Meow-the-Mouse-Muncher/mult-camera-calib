import os
from PIL import Image # ImageOps 可能不再直接需要，除非有其他用途
import tqdm
import re
import cv2 # 导入 OpenCV
import numpy as np # 导入 NumPy

def natural_sort_key(s):
    """
    为自然排序生成键。
    例如：'image1.jpg' -> ['image', 1, '.jpg']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def crop_image(input_image_path, output_image_path, target_width, target_height, 
               clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    """
    对单张图片进行处理：
    1. 转换为灰度图。
    2. 应用 CLAHE 以增强局部对比度。
    3. 如果图片尺寸大于目标尺寸，则进行中心截取。
    4. 保存处理后的图片到输出路径。
    
    参数:
    input_image_path (str): 输入图片路径
    output_image_path (str): 输出图片路径
    target_width (int): 目标宽度 (用于裁剪)
    target_height (int): 目标高度 (用于裁剪)
    clahe_clip_limit (float): CLAHE 的对比度限制阈值
    clahe_tile_grid_size (tuple): CLAHE 的网格大小 (行, 列)
    """
    try:
        # 打开图片
        pil_image = Image.open(input_image_path)
        
        # 1. 转换为灰度图 (PIL)
        grayscale_pil_image = pil_image.convert('L')
        
        # 将 PIL 灰度图转换为 OpenCV 格式 (NumPy 数组)
        grayscale_cv_image = np.array(grayscale_pil_image)
        
        # 2. 应用 CLAHE
        # 创建 CLAHE 对象
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        # 应用 CLAHE
        enhanced_cv_image = clahe.apply(grayscale_cv_image)
        
        # 将增强后的 OpenCV 图像转换回 PIL 图像
        enhanced_pil_image = Image.fromarray(enhanced_cv_image)
        
        original_width, original_height = enhanced_pil_image.size # 获取增强后图像的尺寸
        
        image_to_save = enhanced_pil_image # 默认为增强后的完整图像

        # 3. 判断是否需要裁剪
        if not (original_width == target_width and original_height == target_height) and \
           not (original_width < target_width or original_height < target_height):
            left = (original_width - target_width) // 2
            top = (original_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            image_to_save = enhanced_pil_image.crop((left, top, right, bottom))
            
        # 4. 保存处理后的图片
        image_to_save.save(output_image_path)
    except FileNotFoundError:
        print(f"错误：输入图片 {input_image_path} 未找到。")
    except Exception as e:
        print(f"处理图片 {input_image_path} 时发生错误: {e}")

def batch_crop_images(input_folder, output_folder, target_width, target_height,
                      clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    """
    对输入文件夹中的所有图片进行批量处理（灰度化、CLAHE、裁剪），并保存到输出文件夹。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files = sorted(all_files, key=natural_sort_key)
    print(f"找到 {len(image_files)} 张图片，准备进行处理。排序后列表预览: {image_files[:20]}")
    
    for idx, image_file in tqdm.tqdm(enumerate(image_files), desc="处理图片", ncols=100, unit="张"):
        input_image_path = os.path.join(input_folder, image_file)
        output_image_filename = f"{idx+1}.png" 
        output_image_path = os.path.join(output_folder, output_image_filename)

        try:
            crop_image(input_image_path, output_image_path, target_width, target_height,
                       clahe_clip_limit, clahe_tile_grid_size)
        except Exception as e:
            print(f"错误处理图片 {image_file}: {e}")
    
    print(f"所有图片已处理并保存到 {output_folder}")

if __name__ == "__main__":
    input_folder = "./flir_result"
    output_folder = "./flir_result_clahe_cropped"  # 建议修改输出文件夹名
    
    target_crop_width = 1800
    target_crop_height = 1800

    # CLAHE 参数 (可根据实际情况调整)
    clahe_clip = 2.0  # 对比度限制，值越大对比度越高，但可能引入噪声。常用 1.0-4.0
    clahe_grid = (8, 8) # 网格大小，影响局部处理的范围。常用 (8,8)

    batch_crop_images(input_folder, output_folder, 
                      target_crop_width, target_crop_height,
                      clahe_clip_limit=clahe_clip, 
                      clahe_tile_grid_size=clahe_grid)
