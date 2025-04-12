## 数据预处理
1. raw_2_gray.py  原始图像转灰度图  存储到data目录
2. projetion_f2e.py  flir投影到event视角 存储到projection_outputs目录
3. colmap  1800x1800原图重建但是有些图有问题，需要切分事件和图像段落 colmap/0000/001 这种目录
4. mono_refocus.py  event视角下的图refoucs 存储到result目录
4. 数据集设置 假如说30张图为一组，为了所有的图都用上，需要大概60张图重建。然后中间30张的事件流和时间戳，refocus之后保存事件帧和图像
*ps：为了重建，colmap使用的图像从1开始命名，其他从0开始命名*
## colmap 设置参数
flir内参 fx fy cx cy
data4.2 
7191.73748782809,7190.76441594966,897.056114098032,862.573591961378
event 内参
3463.13471858099,3463.11625407825,347.866576798087, 325.714566240859
重建设置
init_max_forward_motion  1.0
init_min_tri_angle  0.5
输出格式
图像的重建姿态定义为从世界坐标系到相机坐标系的投影，利用四元数（QW,QX,QY,QZ）以及旋转向量（TX,TY,TZ）表示。	
图像的重建姿态是使用四元数（QW、QX、QY、QZ）和平移矢量（TX、TY、TZ）从世界坐标系到图像的摄像机坐标系的投影。	
