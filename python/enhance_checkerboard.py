import os
import tqdm
import cv2
import numpy as np

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def enhance_checkerboard_simple(image):
    """
    只做局部对比度增强（CLAHE），不二值化，保留灰度细节
    """
    # 1. CLAHE自适应直方图均衡化，增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    eq = clahe.apply(image)
    # 2. 轻微去噪
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    # 3. 不做二值化，直接返回增强后的灰度图
    return blur

def batch_enhance_checkerboard(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files = sorted(all_files, key=natural_sort_key)
    print(f"找到 {len(image_files)} 张图片，开始增强...")

    for idx, image_file in tqdm.tqdm(enumerate(image_files), desc="增强棋盘格", ncols=100, unit="张"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"{idx+1:04d}.png")
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            print(f"跳过无效图片: {image_file}")
            continue
        enhanced = enhance_checkerboard_simple(img)
        enhanced = cv2.rotate(enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(output_path, enhanced)
    print(f"全部处理完成，结果保存在: {output_folder}")

if __name__ == "__main__":
    input_folder = "./thermal"
    output_folder = "./thermal_enhanced"
    batch_enhance_checkerboard(input_folder, output_folder)