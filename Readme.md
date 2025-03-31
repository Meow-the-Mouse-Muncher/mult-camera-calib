# 标定流程

## 使用e2calib压帧

1. 通过read_raw.py将.raw文件转为.h5文件

2. 使用./python/run_reconstruction.py对上步处理好的.h5文件进行压帧重建，分为按频率重建和按触发时间戳重建

   - 按频率重建

     修改`freq_hz`为你实际的频率，不需要格外设置

   - 按触发时间戳重建

     需要`timestamps_file`，格式为txt，每一行为具体触发时间，在run_reconstruction.py中把相应代码注释解除，并将`freq_hz`注释掉

## 直接对.raw压帧

使用./python/yazhen.py进行压帧，代码比较简单，直接使用即可

## 使用matlab进行标定（固定内参版）

1.使用matlab自带工具箱进行内参标定，使用savemat.m存为.mat文件，maltab文件夹中的camera_intrinsics0.mat和camera_intrinsics1.mat分别是event，flir两个相机的内参

2.使用calib.m进行标定

3.使用projection.m进行红蓝映射观察标定结果

ps：也有python的投影变换版本

