import cv2
import numpy as np
import os

def enhance_checkerboard(image_path, output_path):
    # 读取灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取图片")
        return

    # 1. CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    # 2. 轻微锐化（可选）
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharp = cv2.filter2D(img_clahe, -1, kernel)

    # 保存结果
    cv2.imwrite(output_path, img_sharp)
    print(f"增强后的图片已保存至: {output_path}")
def crop_image(image_path, output_path):
    # 读取图像 
    img = cv2.imread(image_path)
    if img.height != 1800 or img.width != 1800:
        star_width = (img.width - 1800) // 2
        star_height = (img.height - 1800) // 2
        img = img[star_height:star_height + 1800, star_width:star_width + 1800]
        cv2.imwrite(output_path, img)
        print(f"裁剪后的图片已保存至: {output_path}")

# 用法示例
input_image = '001.jpg'  # 替换为你的图片名
output_image = 'enhanced_' + input_image
image_flies = os.listdir('./')
for image_file in image_flies:
    if image_file.endswith('.jpg'):
        input_image = image_file
        output_image = 'enhanced_' + input_image
        enhance_checkerboard(input_image, output_image)


rgb_images = os.listdir('./rgb/')
for rgb_image in rgb_images:
    input_image = rgb_image
    output_image = 'enhanced_' + input_image
    crop_image(input_image, output_image)

