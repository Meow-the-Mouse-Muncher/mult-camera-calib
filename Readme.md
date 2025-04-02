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
ps： matlab代码为屎山代码，请务必按照格式使用0为event，1为flir
ps：也有python的投影变换版本

flir相机坐标系到event相机坐标系的变换公式：
$X_e=R_{f2e}X_f+T_{f2e}$
又有内参的相关公式
$x_e=\frac{1}{z_e}K_eX_e$
$x_f=\frac{1}{z_f}K_fX_f$
则从flir像素坐标到event的像素坐标转换为:
$x_e=\frac{z_f}{z_e}K_eR_{f2e}K^{-1}_{f}X_f+\frac{1}{z_e}K_eT_{f2e}$
