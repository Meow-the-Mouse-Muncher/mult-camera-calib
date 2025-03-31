import subprocess
# def run_offline_reconstruction(h5file, upsample_rate, timestamps_file, height, width):
def run_offline_reconstruction(h5file, freq_hz, upsample_rate,output_folder,height, width):
    # 构造命令
    command = [
        "python", "offline_reconstruction.py", 
        "--h5file", h5file,
        "--freq_hz", str(freq_hz),
        "--upsample_rate", str(upsample_rate),
        # "--timestamps_file", timestamps_file,
        # "--use_gpu",use_gpu,
        # "--gpu_id",gpu_id,
        "--output_folder",output_folder,
        "--height", str(height),
        "--width", str(width)
    ]
    
    # 执行命令
    try:
        subprocess.run(command, check=True)
        print("Offline reconstruction executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # 参数设置
    h5file = "/mnt/sda/sjy/camera_2_calibration/data3.22/3/event/event/event.h5"
    freq_hz = 15
    upsample_rate = 1
    # timestamps_file = "/mnt/sda/sjy/data0217/2025021704/event/T.txt"
    output_folder = "/mnt/sda/sjy/camera_2_calibration/data3.22/3/event"
    height = 720
    width = 1280

    # 调用函数执行命令
    # run_offline_reconstruction(h5file, upsample_rate, timestamps_file, height, width)
    run_offline_reconstruction(h5file, freq_hz, upsample_rate, output_folder,height, width)
