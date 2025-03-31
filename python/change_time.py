# 打开文件，读取内容
with open('/mnt/sda/sjy/calibration/双目/1.11/8/event/TimeStamps.txt', 'r') as infile:
    lines = infile.readlines()

# 处理每一行，提取时间戳
timestamps = []
for line in lines:
    # 分割每行，提取时间戳部分
    parts = line.split(' ')
    timestamp = float(parts[0].split(':')[1])  # 获取时间戳（冒号后面的部分）并转换为浮动数值
    timestamps.append(int(timestamp * 1e6))  # 乘以 10 的 9 次方并转换为整数
    # timestamps.append(timestamp)

# 将提取的时间戳保存到新文件中
with open('/mnt/sda/XDP/SAI_code/data/two_calib1-11/8.txt', 'w') as outfile:
    for timestamp in timestamps:
        outfile.write(f"{timestamp}\n")  # 使用 f-string 格式化输出

print("处理完成！")
