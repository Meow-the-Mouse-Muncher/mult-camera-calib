% 假设 cameraParams 是已有的 cameraParameters 对象
% 0-event 1-flir 2-thermal
% 提取内参矩阵和径向畸变系数


intrinsicMatrix1 = cameraParams.Intrinsics.IntrinsicMatrix;
radialDistortion1 = cameraParams.RadialDistortion;

% 可选：提取更多参数（如切向畸变、图像尺寸等）
tangentialDistortion1 = cameraParams.TangentialDistortion;
imageSize1 = cameraParams.ImageSize;

% 将提取的数据保存到 .mat 文件中
save('camera_intrinsics1.mat', 'intrinsicMatrix1', 'radialDistortion1', ...
    'tangentialDistortion1', 'imageSize1');

% 输出确认信息
disp('内参数据已成功保存到 camera_intrinsics1.mat 文件中。');
%{ 
% 提取内参矩阵和径向畸变系数
intrinsicMatrix0 = cameraParams.Intrinsics.IntrinsicMatrix;
radialDistortion0 = cameraParams.RadialDistortion;

% 可选：提取更多参数（如切向畸变、图像尺寸等）
tangentialDistortion0 = cameraParams.TangentialDistortion;
imageSize0 = cameraParams.ImageSize;

% 将提取的数据保存到 .mat 文件中
save('camera_intrinsics0.mat', 'intrinsicMatrix0', 'radialDistortion0', ...
    'tangentialDistortion0', 'imageSize0');

% 输出确认信息
disp('内参数据已成功保存到 camera_intrinsics0.mat 文件中。');
%}