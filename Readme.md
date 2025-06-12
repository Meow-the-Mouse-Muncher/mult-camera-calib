## 标定处理
1. 压帧-使用./python/yazhen.py得到事件帧
2. 对flir图像进行裁剪-crop_images.py
3. thermal 进行旋转-rotate_images.py
4. matlab进行单目标定 然后在calib_F_I.m  calib_F_E.m填上内参
5. matlab进行标定-calib_F_I.m  calib_F_E.m
6. 保存文件 stereoParams_RGB 和stereoParams_thermal
## 数据预处理
ps 所有图像从001开始命名
转成data目录了之后一定要备份，因为需要对data进行更改
1. raw_2_gray.py  原始图像转灰度图  存储到data目录，都是原分辨率
    ```
        data
        └── 0000
            ├── event
            │   ├── events.raw
            │   └── exposure_times.txt
            ├── flir
            │   ├── 0001.png
            │   ├── 0002.png
            │   ├── 0003.png
            │   └── ...
            └── theraml
                ├── 0001.png
                ├── 0002.png
                ├── 0003.png
                └── ...
    ```
2. projetion_f2e.py projection_f2ir.py flir/thermal投影到event视角 存储到projection_outputs目录 event自己复制过去，然后data就可以删掉了
    ```
        projection_outputs
        └── 0000
            ├── event
            │   ├── events.raw
            │   └── exposure_times.txt
            ├── flir
            │   ├── 0001.png
            │   ├── 0002.png
            │   ├── 0003.png
            │   └── ...
            └── theraml
                ├── 0001.png
                ├── 0002.png
                ├── 0003.png
                └── ...
    ```
3. colmap  1800x1800原图重建但是有些图有问题，需要切分事件和图像段落 colmap/0000/001 这种目录
4. mono_refocus.py  event视角下的图refoucs 存储到result目录、缺少一个批量的三相机的refocus的代码   为什么不先重聚焦后投影呢？refocus_projection_f_e.py
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

# data 4.24
thermal
fx fy cx cy 
1722.95047233710,1724.29721341806,342.114302615478,285.372759710394
-0.104446080733295,-16.3708035986313,0
RGB
7027.68435336543,7025.83124373155,905.259765268915,920.719437822226
0.224547921379496,7.05294755153471,0
EVENT
3463.34473431084,3463.31965116414,347.854207157689,325.521471998322
-0.223341934313151,7.48438933606789,0


# data 5.27
flir
3653.25566560442,3651.46376148352,915.409511717337,877.707297841537
