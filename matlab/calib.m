%% Written by Muhammet Balcilar, France, muhammetbalcilar@gmail.com
% All rights reserved
%%%%%%%%%%%%

clear all
close all
% 0-event 1-flir 2-thermal
% 动态获取event和flir文件夹下的所有png文件
eventFiles = dir('evk_result/*.png');
flirFiles = dir('flir_result/*.png');

% 提取文件名中的数字并排序
[~, eventOrder] = sort(str2double(regexp({eventFiles.name}, '\d+', 'match', 'once')));
[~, flirOrder] = sort(str2double(regexp({flirFiles.name}, '\d+', 'match', 'once')));

% 重新排序文件列表
eventFiles = eventFiles(eventOrder);
flirFiles = flirFiles(flirOrder);

% 构建完整的文件路径
file2 = fullfile({flirFiles.folder}, {flirFiles.name});    % flir (2)
file1 = fullfile({eventFiles.folder}, {eventFiles.name});  % event (1)

% 明确指定棋盘格大小
boardSize = [7, 10]; % 行数和列数

% Detect checkerboards in images
[imagePoints{2}, boardSize2, imagesUsed2] = detectCheckerboardPoints(file2);   % flir
[imagePoints{1}, boardSize1, imagesUsed1] = detectCheckerboardPoints(file1);   % event

% 显示检测结果统计
disp(['FLIR相机成功检测数量: ' num2str(sum(imagesUsed2))]);
disp(['Event相机成功检测数量: ' num2str(sum(imagesUsed1))]);
disp(['两相机都成功检测的数量: ' num2str(sum(imagesUsed1 & imagesUsed2))]);

% 显示检测失败的图像编号
disp('FLIR相机检测失败的图像:');
disp(find(~imagesUsed2));
disp('Event相机检测失败的图像:');
disp(find(~imagesUsed1));

% 验证检测到的棋盘格大小是否符合预期
if ~isequal(boardSize2, boardSize) || ~isequal(boardSize1, boardSize)
    warning('检测到的棋盘格大小与预期不符');
    disp(['检测到的大小: ' num2str(boardSize2) ' 和 ' num2str(boardSize1)]);
    disp(['预期大小: ' num2str(boardSize)]);
end

% 确保commonValidIdx的长度与检测结果匹配
commonValidIdx = imagesUsed1 & imagesUsed2;
if length(commonValidIdx) > size(imagePoints{2}, 3)
    commonValidIdx = commonValidIdx(1:size(imagePoints{2}, 3));
end

% 只保留两个相机都成功检测到棋盘格的图像
imagePoints{2} = imagePoints{2}(:,:,commonValidIdx);  % flir
imagePoints{1} = imagePoints{1}(:,:,commonValidIdx);  % event
file2 = file2(commonValidIdx);  % flir
file1 = file1(commonValidIdx);  % event

% 检查imagePoints中是否存在无效值
disp('检查imagePoints中的无效值：');
disp(['Camera 2 (FLIR) NaN数量: ' num2str(sum(isnan(imagePoints{2}(:))))]);
disp(['Camera 1 (Event) NaN数量: ' num2str(sum(isnan(imagePoints{1}(:))))]);
disp(['Camera 2 (FLIR) Inf数量: ' num2str(sum(isinf(imagePoints{2}(:))))]);
disp(['Camera 1 (Event) Inf数量: ' num2str(sum(isinf(imagePoints{1}(:))))]);

% 如果存在无效值，移除包含无效值的图像对
validPoints = all(isfinite(imagePoints{2}), [1,2]) & all(isfinite(imagePoints{1}), [1,2]);
if ~all(validPoints)
    disp(['移除 ' num2str(sum(~validPoints)) ' 对包含无效值的图像']);
    imagePoints{2} = imagePoints{2}(:,:,validPoints);
    imagePoints{1} = imagePoints{1}(:,:,validPoints);
    file2 = file2(validPoints);
    file1 = file1(validPoints);
end

% 检查是否有足够的有效图像对
if size(imagePoints{2}, 3) < 3
    error('需要至少3对有效的棋盘格图像进行标定');
end

disp(['最终使用的有效图像对数量: ' num2str(size(imagePoints{2}, 3))]);

% Generate world coordinates of the checkerboards keypoints
squareSize = 125.5;  % in units of 'mm'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% 加载之前保存的相机内参
load('camera_intrinsics0.mat');   % event相机
load('camera_intrinsics1.mat');   % flir相机

% 重新排列imagePoints数组以适应标准函数输入格式
imagePointsForCalib{1} = imagePoints{2};  % flir放在第一位
imagePointsForCalib{2} = imagePoints{1};  % event放在第二位

[param, pairsUsed, estimationErrors] = my_estimateCameraParameters(imagePointsForCalib, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'mm', ...
    'InitialIntrinsicMatrix', intrinsicMatrix1, 'InitialIntrinsicMatrix2', intrinsicMatrix0, ...
    'InitialRadialDistortion', radialDistortion1, 'InitialRadialDistortion2', radialDistortion0);

% 提取并保存相机参数到工作区

% 读取第一对图像以获取尺寸信息
I2 = imread(file2{1});  % flir
I1 = imread(file1{1});  % event

% 相机内参
K2 = param.CameraParameters1.IntrinsicMatrix';  % flir
K1 = param.CameraParameters2.IntrinsicMatrix';  % event

% 相机外参
R2 = param.CameraParameters1.RotationMatrices(:,:,1)';  % flir
T2 = param.CameraParameters1.TranslationVectors(1,:)';  % flir
R1 = param.CameraParameters2.RotationMatrices(:,:,1)';  % event
T1 = param.CameraParameters2.TranslationVectors(1,:)';  % event

% 计算投影矩阵
P_flir = K2 * [R2 T2];  % flir投影矩阵
P_event = K1 * [R1 T1];  % event投影矩阵

% 计算相机在世界坐标系中的位置
C_flir = -R2' * T2;  % flir相机中心
C_event = -R1' * T1;  % event相机中心

% View reprojection errors
h1=figure; showReprojectionErrors(param);

% Visualize pattern locations
h2=figure; showExtrinsics(param, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, param);

% 显示关键参数
disp('Camera 2 (FLIR) 内参矩阵 (K2):');
disp(K2);
disp('Camera 1 (Event) 内参矩阵 (K1):');
disp(K1);
disp('相机间距离 (mm):');
disp(norm(C_event-C_flir));

% 保存立体相机参数到文件
stereo_params = struct();
stereo_params.K2 = K2;  % flir
stereo_params.K1 = K1;  % event
%外参，从世界坐标系到相机坐标系的变换。
stereo_params.R2 = R2;  % flir
stereo_params.T2 = T2;  % flir
stereo_params.R1 = R1;  % event
stereo_params.T1 = T1;  % event
stereo_params.P_flir = P_flir;  
stereo_params.P_event = P_event;
stereo_params.RadialDistortion2 = param.CameraParameters1.RadialDistortion;
stereo_params.RadialDistortion1 = param.CameraParameters2.RadialDistortion;
stereo_params.TangentialDistortion2 = param.CameraParameters1.TangentialDistortion;
stereo_params.TangentialDistortion1 = param.CameraParameters2.TangentialDistortion;

save('stereo_camera_parameters.mat', 'stereo_params');
disp('立体相机参数已保存到 stereo_camera_parameters.mat');
