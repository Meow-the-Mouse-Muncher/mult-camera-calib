import os
from PIL import Image
import tqdm
import shutil # 导入 shutil 用于复制文件
import re # 导入正则表达式模块

def natural_sort_key(s):
    """
    为自然排序生成键。
    例如：'image1.jpg' -> ['image', 1, '.jpg']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def crop_image(input_image_path, output_image_path, target_width, target_height):
    """
    对单张图片进行中心截取到目标尺寸，并保存到输出路径。
    如果图片已经是目标尺寸，或者图片小于目标尺寸，则直接复制原图。
    
    参数:
    input_image_path (str): 输入图片路径
    output_image_path (str): 输出图片路径
    target_width (int): 目标宽度
    target_height (int): 目标高度
    """
    try:
        # 打开图片
        image = Image.open(input_image_path)
        original_width, original_height = image.size
        
        # 检查图片是否已经是目标尺寸
        if original_width == target_width and original_height == target_height:
            print(f"图片 {input_image_path} 已经是目标尺寸 {target_width}x{target_height}，直接复制。")
            # 如果输出路径和输入路径可能相同，或者为了保留所有元数据，使用shutil.copy2
            # 为简单起见，如果只是格式转换或简单复制，image.save()也可以
            shutil.copy2(input_image_path, output_image_path)
            return

        # 检查图片是否小于目标尺寸（任一维度）
        if original_width < target_width or original_height < target_height:
            print(f"图片 {input_image_path} ({original_width}x{original_height}) 小于目标尺寸 {target_width}x{target_height}，直接复制。")
            shutil.copy2(input_image_path, output_image_path)
            return
            
        # 计算中心截取坐标
        # PIL's crop box is (left, upper, right, lower)
        left = (original_width - target_width) // 2
        top = (original_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        # 截取图片
        cropped_image = image.crop((left, top, right, bottom))
        
        # 保存截取后的图片
        cropped_image.save(output_image_path)
        # print(f"图片 {input_image_path} 已中心裁剪并保存到 {output_image_path}") # tqdm 会显示进度
    except Exception as e:
        print(f"处理图片 {input_image_path} 时发生错误: {e}")

def batch_crop_images(input_folder, output_folder, target_width, target_height):
    """
    对输入文件夹中的所有图片进行批量中心截取到目标尺寸，并保存到输出文件夹。
    
    参数:
    input_folder (str): 包含需要截取的图片的文件夹路径
    output_folder (str): 保存截取后图片的文件夹路径
    target_width (int): 目标宽度
    target_height (int): 目标高度
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件夹中的所有图片文件
    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    # 按自然顺序排序
    image_files = sorted(all_files, key=natural_sort_key)
    print(f"找到 {len(image_files)} 张图片，准备进行截取。排序后列表预览: {image_files[:20]}") # 打印图片数量和排序后的部分列表
    # 遍历图片文件并进行截取
    for idx, image_file in tqdm.tqdm(enumerate(image_files), desc="处理图片", ncols=100, unit="张"):
        input_image_path = os.path.join(input_folder, image_file)
        
        # 生成新的输出文件名，按顺序编号，并保留原扩展名或统一为png
        # 这里保持原逻辑，统一输出为 .png
        output_image_filename = f"{idx+1}.png"
        output_image_path = os.path.join(output_folder, output_image_filename)

        try:
            # 对每张图片进行截取或复制
            crop_image(input_image_path, output_image_path, target_width, target_height)
        except Exception as e:
            print(f"错误处理图片 {image_file}: {e}")
    
    print(f"所有图片已处理并保存到 {output_folder}")

if __name__ == "__main__":
    # 设置输入文件夹和输出文件夹路径
    input_folder = "./flir_result"  # 替换为你的输入文件夹路径
    output_folder = "./flir_result_1"  # 替换为你的输出文件夹路径
    
    # 设置目标截取尺寸
    target_crop_width = 1800
    target_crop_height = 1800

    # 执行批量截取操作
    batch_crop_images(input_folder, output_folder, target_crop_width, target_crop_height)
