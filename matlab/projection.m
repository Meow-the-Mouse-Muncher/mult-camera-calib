% event 投影到flir视角
% 创建输出文件夹
if ~exist('Outputs/flir', 'dir')
    mkdir('Outputs/flir');
end
if ~exist('Outputs/event', 'dir')
    mkdir('Outputs/event');
end
if ~exist('Outputs/overlay', 'dir')
    mkdir('Outputs/overlay');
end

% 加载相机参数
load('stereo_camera_parameters.mat');

% 创建相机参数对象
cam1 = cameraParameters('IntrinsicMatrix', stereo_params.K2', ...  % flir (2)
                       'RadialDistortion', stereo_params.RadialDistortion2, ...
                       'TangentialDistortion', stereo_params.TangentialDistortion2);
                   
cam0 = cameraParameters('IntrinsicMatrix', stereo_params.K1', ...  % event (1)
                       'RadialDistortion', stereo_params.RadialDistortion1, ...
                       'TangentialDistortion', stereo_params.TangentialDistortion1);

% 获取所有图像文件
flirFiles = dir('flir_result/*.png');
eventFiles = dir('evk_result/*.png');

% 确保图像对数量匹配
numImages = min(length(flirFiles), length(eventFiles));

% 设置投影平面
Z = 0;

% 处理每对图像
for i = 1:numImages
    tic;  % 开始计时
    fprintf('处理图像对 %d/%d... ', i, numImages);
    
    % 读取图像对
    I1 = imread(fullfile(flirFiles(i).folder, flirFiles(i).name));  % flir
    I0 = imread(fullfile(eventFiles(i).folder, eventFiles(i).name));  % event
    
    % 去畸变
    I1_undist = undistortImage(I1, cam1);
    I0_undist = undistortImage(I0, cam0);
    
    
    % 生成FLIR图像中所有像素的网格坐标
    [X_grid, Y_grid] = meshgrid(1:size(I1,2), 1:size(I1,1));
    
    % 预先计算常量矩阵
    P_flir_part = stereo_params.P_flir(:,1:2); %P_flir = K2 * [R2 T2]; 从世界到flir的投影矩阵
    P_flir_const = -Z*stereo_params.P_flir(:,3)-stereo_params.P_flir(:,4);
    
    % 映射数组 表示FLIR图像中每个像素对应到EVENT图像中的位置
    map_x = zeros(size(X_grid));
    map_y = zeros(size(Y_grid));
    
    % 按行处理
    for y = 1:size(I1,1)
        x_coords = X_grid(y,:);
        
        % 构建正确的最后一列
        last_col = [-x_coords; -y*ones(1,length(x_coords)); -ones(1,length(x_coords))];
        
        % 一次处理整行
        X_batch = zeros(3, length(x_coords));
        for j = 1:length(x_coords)
            X_batch(:,j) = inv([P_flir_part, last_col(:,j)]) * P_flir_const;
        end
        
        % 执行投影
        for j = 1:length(x_coords)
            P = stereo_params.P_event * [X_batch(1,j); X_batch(2,j); Z; 1];
            P = P / P(3);
            
            map_x(y,j) = P(1);
            map_y(y,j) = P(2);
        end
    end
    
    % 使用映射进行图像重采样
    % 判断EVENT图像是否在有效范围内
    valid_indices = map_x >= 1 & map_x <= size(I0,2) & map_y >= 1 & map_y <= size(I0,1);
    
    % 对每个颜色通道进行采样
    if size(I0, 3) == 3
        for c = 1:3
            % 使用线性插值进行像素采样
            channel = double(I0(:,:,c));
            O_channel = interp2(1:size(I0,2), 1:size(I0,1), channel, map_x, map_y, 'linear', 0);
            O(:,:,c) = uint8(O_channel .* valid_indices);
        end
    else
        % 灰度图像
        channel = double(I0);
        O_channel = interp2(1:size(I0,2), 1:size(I0,1), channel, map_x, map_y, 'linear', 0);
        O = uint8(O_channel .* valid_indices);
    end
    
    % 获取文件名（不包含扩展名）
    [~, name] = fileparts(flirFiles(i).name);
    
    % 保存结果
    imwrite(I1, fullfile('Outputs/flir', sprintf('flir_%s.png', name)));
    imwrite(O, fullfile('Outputs/event', sprintf('event_with_flir_content_%s.png', name)));
    
    % 创建并保存重叠图像
    figure('Visible', 'off');
    overlay = imshowpair(I1, O, 'falsecolor', 'ColorChannels', 'red-cyan');
    frame = getframe(gca);
    overlay_img = frame.cdata;
    imwrite(overlay_img, fullfile('Outputs/overlay', sprintf('overlay_%s.png', name)));
    close;
    
    % 计算并显示处理时间
    elapsed = toc;
    fprintf('完成! (%.2f 秒)\n', elapsed);
end

fprintf('\n处理完成！结果已保存到 Outputs 文件夹下的子文件夹中：\n');
fprintf('- flir: 原始FLIR图像\n');
fprintf('- event: 投影后的EVENT图像（含FLIR内容）\n');
fprintf('- overlay: 重叠效果图像\n');