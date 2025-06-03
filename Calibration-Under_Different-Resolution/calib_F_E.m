%% Written by Muhammet Balcilar, France, muhammetbalcilar@gmail.com
% All rights reserved
%%%%%%%%%%%%

clear all
close all

% 1-flir 2-evk
% 读取evk_result文件夹中的所有图片
evkDir = 'event';
evkFiles = dir(fullfile(evkDir, '*.bmp'));
if isempty(evkFiles)
    evkFiles = dir(fullfile(evkDir, '*.jpg'));
end
if isempty(evkFiles)
    evkFiles = dir(fullfile(evkDir, '*.png'));
end

% 读取flir_result文件夹中的所有图片
flirDir = 'flir-event';
flirFiles = dir(fullfile(flirDir, '*.bmp'));
if isempty(flirFiles)
    flirFiles = dir(fullfile(flirDir, '*.jpg'));
end
if isempty(flirFiles)
    flirFiles = dir(fullfile(flirDir, '*.png'));
end
% 对文件名进行排序
evkNames = {evkFiles.name};
[~, sortIdx] = sort(evkNames);
evkFiles = evkFiles(sortIdx);

flirNames = {flirFiles.name};
[~, sortIdx] = sort(flirNames);
flirFiles = flirFiles(sortIdx);
% 创建文件路径列表
file1 = cell(length(flirFiles), 1);
for i = 1:length(flirFiles)
    file1{i} = fullfile(flirDir, flirFiles(i).name);
end

file2 = cell(length(evkFiles), 1);
for i = 1:length(evkFiles)
    file2{i} = fullfile(evkDir, evkFiles(i).name);
end


% 显示找到的图片数量
fprintf('从evk_result文件夹中找到%d张图片\n', length(file2));
fprintf('从flir_result文件夹中找到%d张图片\n', length(file1));


% Detect checkerboards in images
[imagePoints{1}, boardSize, imagesUsed1] = detectCheckerboardPoints(file1);
[imagePoints{2}, boardSize, imagesUsed2] = detectCheckerboardPoints(file2);

% Generate world coordinates of the checkerboards keypoints
squareSize = 126.0;  % in units of 'mm'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);
% FLIR相机的内参矩阵
K1 = [
   3653.25566560442,    0,                  915.409511717337;
    0,                3651.46376148352,      877.707297841537;
    0,                  0,                       1
];

% EVK相机的内参矩阵
K2 = [
    1755.44073514260,    0,                  318.374004280142;
    0,               1754.58574195715,       293.312723627602;
    0,                   0,                          1
];


% 畸变参数 [k1, k2] - 径向畸变系数
% FLIR相机的畸变参数
radialDistortion1 = [-0.157414231108809,0.328213958313925];
% EVK相机的畸变参数
radialDistortion2 = [-0.110800662548499,0.576532212622538];

[param, pairsUsed, estimationErrors] = my_estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'mm', ...
    'InitialIntrinsicMatrix', {K1, K2}, 'InitialRadialDistortion', {radialDistortion1, radialDistortion2});


% 保存相机标定参数
% 创建包含所有参数的结构体
stereoParams = struct();

% 保存内参
stereoParams.K1 = param.CameraParameters1.IntrinsicMatrix';  % FLIR相机内参
stereoParams.K2 = param.CameraParameters2.IntrinsicMatrix';  % EVK相机内参

% 保存畸变参数
stereoParams.RadialDistortion1 = param.CameraParameters1.RadialDistortion;  % FLIR相机径向畸变
stereoParams.RadialDistortion2 = param.CameraParameters2.RadialDistortion;  % EVK相机径向畸变
stereoParams.TangentialDistortion1 = param.CameraParameters1.TangentialDistortion;  % FLIR相机切向畸变
stereoParams.TangentialDistortion2 = param.CameraParameters2.TangentialDistortion;  % EVK相机切向畸变

% 保存外参
stereoParams.Rof2 = param.RotationOfCamera2;  % 相机2相对于相机1的旋转矩阵
stereoParams.Tof2 = param.TranslationOfCamera2;  % 相机2相对于相机1的平移向量
%stereoParams.R1=param.CameraParameters1.RotationMatrices;  %相机到世界坐标系 3x3
stereoParams.R1 = param.CameraParameters1.RotationMatrices;


stereoParams.T1 = param.CameraParameters1.TranslationVectors';
stereoParams.R2 = param.CameraParameters2.RotationMatrices;
stereoParams.T2 = param.CameraParameters2.TranslationVectors';

% 保存其他有用信息
stereoParams.WorldPoints = param.WorldPoints;  % 世界坐标点
stereoParams.WorldUnits = param.WorldUnits;  % 世界坐标单位
stereoParams.ReprojectionErrors = estimationErrors;  % 重投影误差

% 保存到MAT文件
save('./stereoParams.mat', 'stereoParams');
fprintf('相机标定参数已保存到 stereoParams.mat\n');
%-----------------------------------------------------------------------------------------------%



% You can use the calibration data to undistort images
I1 = imread(file1{14});
I2 = imread(file2{14});
% select displayed checkeroard detection point grount truth 
% estimated point positions and camera positions.
cno=1;

Wpoints=[worldPoints zeros(size(worldPoints,1),1)];
figure;hold on;
axis vis3d; axis image;
grid on;
plot3(Wpoints(:,1),Wpoints(:,2),Wpoints(:,3),'b.','MarkerSize',20)

K1=param.CameraParameters1.IntrinsicMatrix';
R1=param.CameraParameters1.RotationMatrices';
T1=param.CameraParameters1.TranslationVectors';

Lcam=K1*[R1 T1];

K2=param.CameraParameters2.IntrinsicMatrix';
R2=param.CameraParameters2.RotationMatrices';
T2=param.CameraParameters2.TranslationVectors';


Rcam=K2*[R2 T2];

[points3d] = mytriangulate(imagePoints{1}(:,:,cno), imagePoints{2}(:,:,cno), Lcam,Rcam );
plot3(points3d(:,1),points3d(:,2),points3d(:,3),'r.')


% referencePoint(0,0,0)= R*Camera+T, So Camera=-inv(R)*T;
CL=-R1'*T1;
CR=-R2'*T2;

plot3(CR(1),CR(2),CR(3),'gs','MarkerFaceColor','g');
plot3(CL(1),CL(2),CL(3),'cs','MarkerFaceColor','c');
legend({'ground truth point locations','Calculated point locations','Camera2 position','Camera1 Position'});


% calculate relative distance from camera1 to camera2 in two different way
dist_1=norm(param.TranslationOfCamera2)
dist_2=norm(CR-CL)




% set the projection plane. I just project all pixel on to Z=0 plane
Z=0;

O=zeros(size(I1));
% remapping
for i=1:size(I1,1)
    i
    for j=1:size(I1,2)
        X=inv([Lcam(:,1:2) [-1*j;-1*i;-1]])*(-Z*Lcam(:,3)-Lcam(:,4));
        P=Rcam*[X(1);X(2);Z;1];
        P=fix(P/P(end));
        if P(1)>0 & P(2)<size(I2,1) & P(2)>0 & P(1)<size(I2,2)
            O(i,j,:)=I2(P(2),P(1),:);
        end
    end
end
figure;imshow(uint8(O))
figure;imshowpair(I1,uint8(O),'falsecolor','ColorChannels','red-cyan');










