import os
from PIL import Image
import tqdm

def crop_image(input_image_path, output_image_path, x_min, x_max, y_min, y_max):
    """
    对单张图片进行截取，并保存到输出路径。
    
    参数:
    input_image_path (str): 输入图片路径
    output_image_path (str): 输出图片路径
    x_min, x_max (int): 截取区域的 x 轴范围
    y_min, y_max (int): 截取区域的 y 轴范围
    """
    try:
        # 打开图片
        image = Image.open(input_image_path)
        
        # 截取图片
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # 保存截取后的图片
        cropped_image.save(output_image_path)
    except Exception as e:
        print(f"处理图片 {input_image_path} 时发生错误: {e}")

def batch_crop_images(input_folder, output_folder, x_min, x_max, y_min, y_max):
    """
    对输入文件夹中的所有图片进行批量截取，并保存到输出文件夹。
    
    参数:
    input_folder (str): 包含需要截取的图片的文件夹路径
    output_folder (str): 保存截取后图片的文件夹路径
    x_min, x_max (int): 截取区域的 x 轴范围
    y_min, y_max (int): 截取区域的 y 轴范围
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件夹中的所有图片文件，并按文件名排序
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # 遍历图片文件并进行截取
    for idx, image_file in tqdm.tqdm(enumerate(image_files), desc="处理图片", ncols=100):
        input_image_path = os.path.join(input_folder, image_file)
        
        # 生成新的输出文件名，按顺序编号
        output_image_path = os.path.join(output_folder, f"{idx}.png")  # 可以根据需要修改格式（如 .jpg）

        try:
            # 对每张图片进行截取
            crop_image(input_image_path, output_image_path, x_min, x_max, y_min, y_max)
        except Exception as e:
            print(f"错误处理图片 {image_file}: {e}")
    
    print(f"所有图片已处理并保存到 {output_folder}")

if __name__ == "__main__":
    # 设置输入文件夹和输出文件夹路径
    input_folder = "./flir_result"  # 替换为你的输入文件夹路径
    output_folder = "./flir_result_1"  # 替换为你的输出文件夹路径
    
    # 设置截取区间
    x_min, x_max = 324, 2124  # 设置你想截取的 x 轴范围
    y_min, y_max = 124, 1924  # 设置你想截取的 y 轴范围

    # 执行批量截取操作
    batch_crop_images(input_folder, output_folder, x_min, x_max, y_min, y_max)
