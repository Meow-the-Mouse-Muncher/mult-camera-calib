% 创建参数结构体
stereo_params = struct();

% 相机内参
stereo_params.K1 = K1;  % 相机1内参矩阵
stereo_params.K2 = K2;  % 相机2内参矩阵

% 相机外参
stereo_params.R1 = R1;  % 相机1旋转矩阵
stereo_params.T1 = T1;  % 相机1平移向量
stereo_params.R2 = R2;  % 相机2旋转矩阵
stereo_params.T2 = T2;  % 相机2平移向量

% 相机投影矩阵
stereo_params.Lcam = Lcam;  % 相机1投影矩阵
stereo_params.Rcam = Rcam;  % 相机2投影矩阵

% 相机位置
stereo_params.CL = CL;  % 相机1在世界坐标系中的位置
stereo_params.CR = CR;  % 相机2在世界坐标系中的位置

% 畸变参数
stereo_params.RadialDistortion1 = param.CameraParameters1.RadialDistortion;
stereo_params.TangentialDistortion1 = param.CameraParameters1.TangentialDistortion;
stereo_params.RadialDistortion2 = param.CameraParameters2.RadialDistortion;
stereo_params.TangentialDistortion2 = param.CameraParameters2.TangentialDistortion;

% 相机间的相对位姿
stereo_params.TranslationOfCamera2 = param.TranslationOfCamera2;
stereo_params.RotationOfCamera2 = param.RotationOfCamera2;

% 图像尺寸信息
stereo_params.ImageSize1 = size(I1);
stereo_params.ImageSize2 = size(I2);

% 保存参数
save('stereo_camera_parameters.mat', 'stereo_params');

% 可选：同时保存为JSON格式（便于其他语言读取）
params_json = jsonencode(stereo_params);
fid = fopen('stereo_camera_parameters.json', 'w');
fprintf(fid, '%s', params_json);
fclose(fid);

% 打印确认信息
fprintf('相机参数已保存到 stereo_camera_parameters.mat 和 stereo_camera_parameters.json\n');