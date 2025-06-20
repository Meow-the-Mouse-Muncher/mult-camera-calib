function [cameraParams, imagesUsed, estimationErrors] = my_estimateCameraParameters(varargin)
%% Modified by Muhammet Balcilar, France, muhammetbalcilar@gmail.com

%estimateCameraParameters Calibrate a single camera or a stereo camera
%
%   [cameraParams, imagesUsed, estimationErrors] =
%   estimateCameraParameters(imagePoints, worldPoints) estimates intrinsic,
%   extrinsic, and distortion parameters of a single camera.
% 
%   Inputs:
%   -------
%   imagePoints - an M-by-2-by-P array of [x,y] intrinsic image coordinates 
%                 of keypoints on the calibration pattern. M > 3 
%                 is the number of keypoints in the pattern. P > 2 is the
%                 number of images containing the calibration pattern.
%
%   worldPoints - an M-by-2 array of [x,y] world coordinates of 
%                 keypoints on the calibration pattern. The pattern 
%                 must be planar, so all z-coordinates are assumed to be 0.
%   
%   Outputs:
%   --------
%   cameraParams     - a cameraParameters object containing the camera parameters. 
%
%   imagesUsed       - a P-by-1 logical array indicating which images were 
%                      used to estimate the camera parameters. P > 2 is the
%                      number of images containing the calibration pattern. 
%
%   estimationErrors - a cameraCalibrationErrors object containing the 
%                      standard errors of the estimated camera parameters.
% 
%   [stereoParams, pairsUsed, estimationErrors] =
%   estimateCameraParameters(imagePoints, worldPoints) estimates parameters
%   of a stereo camera.
% 
%   Inputs:
%   -------
%   imagePoints - An M-by-2-by-numPairs-by-2 array of [x,y] intrinsic image 
%                 coordinates of keypoints of the calibration pattern. M > 3 
%                 is the number of keypoints in the pattern. numPairs is the
%                 number of stereo pairs of images containing the
%                 calibration pattern. imagePoints(:,:,:,1) are the points 
%                 from camera 1, and imagePoints(:,:,:,2) are the points 
%                 from camera 2.
%
%   worldPoints - An M-by-2 array of [x,y] world coordinates of 
%                 keypoints on the calibration pattern. The pattern 
%                 must be planar, so all z-coordinates are assumed to be 0.
% 
%   Outputs:
%   --------
%   stereoParams     - A stereoParameters object containing the parameters
%                      of the stereo camera system. 
%
%   pairsUsed        - A numPairs-by-1 logical array indicating which image
%                      pairs were used to estimate the camera parameters. 
%                      numPairs > 2 is the number of image pairs containing
%                      the calibration pattern. 
%
%   estimationErrors - A stereoCalibrationErrors object containing the 
%                      standard errors of the estimated stereo parameters. 
%
%   cameraParams = estimateCameraParameters(..., Name, Value)  
%   specifies additional name-value pair arguments described below.
% 
%   Parameters include:
%   -------------------
%   'WorldUnits'                      A string that describes the units in 
%                                     which worldPoints are specified.
%
%                                     Default: 'mm'
%
%   'EstimateSkew'                    A logical scalar that specifies whether 
%                                     image axes skew should be estimated.
%                                     When set to false, the image axes are
%                                     assumed to be exactly perpendicular.
%
%                                     Default: false
%
%   'NumRadialDistortionCoefficients' 2 or 3. Specifies the number of radial 
%                                     distortion coefficients to be estimated. 
%
%                                     Default: 2
%
%   'EstimateTangentialDistortion'    A logical scalar that specifies whether 
%                                     tangential distortion should be estimated.
%                                     When set to false, tangential distortion 
%                                     is assumed to be negligible.
%
%                                     Default: false
%
%   'InitialIntrinsicMatrix'          A 3-by-3 matrix containing the initial
%                                     guess for the camera intrinsics. If the 
%                                     value is empty, the initial intrinsic 
%                                     matrix is computed using linear least 
%                                     squares.
%
%                                     Default: []
%
%   'InitialRadialDistortion'         A vector of 2 or 3 elements containing
%                                     the initial guess for radial distortion 
%                                     coefficients. If the value is empty, 
%                                     0 is used as the initial value for all
%                                     the coefficients.
%                                     
%                                     Default: []
%
%   Class Support
%   -------------
%   worldPoints and imagePoints must be double or single.
%
%   Notes
%   -----
%   estimateCameraParameters computes a homography between the world points
%   and the points detected in each image. If the homography computation
%   fails for any image, the function will issue a warning and it will not
%   use the points from that image for estimating the camera parameters. To
%   determine which images were actually used for estimating the parameters,
%   use imagesUsed output.
%
%   Example 1: Single Camera Calibration
%   ------------------------------------
%   % Create a set of calibration images.
%   images = imageDatastore(fullfile(toolboxdir('vision'), 'visiondata', ...
%     'calibration', 'fishEye'));
%   imageFileNames = images.Files;
%
%   % Detect calibration pattern.
%   [imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames);
%
%   % Generate world coordinates of the corners of the squares.
%   squareSize = 29; % millimeters
%   worldPoints = generateCheckerboardPoints(boardSize, squareSize);
%
%   % Calibrate the camera.
%   params = estimateCameraParameters(imagePoints, worldPoints);
%
%   % Visualize calibration accuracy.
%   figure;
%   showReprojectionErrors(params);
%
%   % Visualize camera extrinsics.
%   figure;
%   showExtrinsics(params);
%   drawnow;
%
%   % Plot detected and reprojected points.
%   figure; 
%   imshow(imageFileNames{1}); 
%   hold on
%   plot(imagePoints(:, 1, 1), imagePoints(:, 2, 1), 'go');
%   plot(params.ReprojectedPoints(:, 1, 1), params.ReprojectedPoints(:, 2, 1), 'r+');
%   legend('Detected Points', 'ReprojectedPoints');
%   hold off
%
%   Example 2: Stereo Calibration
%   -----------------------------
%   % Specify calibration images.
%   leftImages = imageDatastore(fullfile(toolboxdir('vision'), 'visiondata', ...
%       'calibration', 'stereo', 'left'));
%   rightImages = imageDatastore(fullfile(toolboxdir('vision'), 'visiondata', ...
%       'calibration', 'stereo', 'right'));
%
%   % Detect the checkerboards.
%   [imagePoints, boardSize] = ...
%     detectCheckerboardPoints(leftImages.Files, rightImages.Files);
%
%   % Specify world coordinates of checkerboard keypoints.
%   squareSize = 108; % in millimeters
%   worldPoints = generateCheckerboardPoints(boardSize, squareSize);
%
%   % Calibrate the stereo camera system.
%   params = estimateCameraParameters(imagePoints, worldPoints);
%
%   % Visualize calibration accuracy.
%   figure;
%   showReprojectionErrors(params);
%
%   % Visualize camera extrinsics.
%   figure;
%   showExtrinsics(params);
%
%   See also cameraCalibrator, stereoCameraCalibrator, detectCheckerboardPoints, 
%     generateCheckerboardPoints, showExtrinsics, showReprojectionErrors,
%     undistortImage, cameraParameters, stereoParameters,
%     cameraCalibrationErrors, stereoCalibrationErrors

%   Copyright 2013-2014 MathWorks, Inc.

% References:
%
% [1] Z. Zhang. A flexible new technique for camera calibration. 
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 
% 22(11):1330-1334, 2000.
%
% [2] Janne Heikkila and Olli Silven. A Four-step Camera Calibration Procedure 
% with Implicit Image Correction, IEEE International Conference on Computer
% Vision and Pattern Recognition, 1997.
%
% [3] Bouguet, JY. "Camera Calibration Toolbox for Matlab." 
% Computational Vision at the California Institute of Technology. 
% http://www.vision.caltech.edu/bouguetj/calib_doc/
%
% [4] G. Bradski and A. Kaehler, "Learning OpenCV : Computer Vision with
% the OpenCV Library," O'Reilly, Sebastopol, CA, 2008.


% Calibrate the camera
if iscell(varargin{1}) 
    itmp=varargin{1};    
    varargin{1}=itmp{2};    
    [imagePoints, worldPoints, worldUnits, cameraModel, calibrationParams] = ...
    parseInputs(varargin{:});
    imagePoints={itmp{1},imagePoints};    
else
    [imagePoints, worldPoints, worldUnits, cameraModel, calibrationParams] = ...
    parseInputs(varargin{:});
end

    
    


calibrationParams.shouldComputeErrors = (nargout >= 3);

if iscell(imagePoints) == 0 % single camera
    [cameraParams, imagesUsed, estimationErrors] = calibrateOneCamera(imagePoints, ...
        worldPoints, cameraModel, worldUnits, calibrationParams);
else % 2-camera stereo
    shouldComputeErrors = calibrationParams.shouldComputeErrors;
    calibrationParams.shouldComputeErrors = false;

    % 为第一个相机设置正确的初始参数
    calibParams1.initIntrinsics = calibrationParams.initIntrinsics1;
    calibParams1.initRadial = calibrationParams.initRadial1;
    calibParams1.showProgressBar = calibrationParams.showProgressBar;
    calibParams1.shouldComputeErrors=calibrationParams.shouldComputeErrors;

    
    % 为第二个相机设置正确的初始参数
    calibParams2.initIntrinsics = calibrationParams.initIntrinsics2;
    calibParams2.initRadial = calibrationParams.initRadial2;
    calibParams2.showProgressBar = calibrationParams.showProgressBar;
    calibParams2.shouldComputeErrors=calibrationParams.shouldComputeErrors;

    % 分别标定两个相机
    [cameraParams1, imagesUsed1, estimationErrors1] = calibrateOneCamera(imagePoints{1}, ...
        worldPoints, cameraModel, worldUnits, calibParams1);
    [cameraParams2, imagesUsed2, estimationErrors2] = calibrateOneCamera(imagePoints{2}, ...
        worldPoints, cameraModel, worldUnits, calibParams2);
    
    imagesUsed = imagesUsed1 & imagesUsed2;
    cameraParams1 = removeUnusedExtrinsics(cameraParams1, imagesUsed, imagesUsed1);
    cameraParams2 = removeUnusedExtrinsics(cameraParams2, imagesUsed, imagesUsed2);

    % Compute the initial estimate of translation and rotation of camera 2
    [R, t] = estimateInitialTranslationAndRotation(cameraParams1, cameraParams2);

    cameraParams = stereoParameters(cameraParams1, cameraParams2, R, t);
    ip1=imagePoints{1};
    ip2=imagePoints{2};
    
    % 新代码：使用固定内参的优化方法
    [stereoParams,estimationErrors] = optimizeExtrinsicsOnly(cameraParams, ip1(:, :, imagesUsed), ip2(:, :, imagesUsed), shouldComputeErrors);
    clear cameraParams ;
    cameraParams = stereoParams;
    % estimationErrors = refine(cameraParams, ip1(:, :, imagesUsed), ip2(:, :, imagesUsed), shouldComputeErrors);
    
end



%--------------------------------------------------------------------------
function [imagePoints, worldPoints, worldUnits, cameraModel, calibrationParams] = ...
    parseInputs(varargin)
parser = inputParser;
parser.addRequired('imagePoints', @checkImagePoints);
parser.addRequired('worldPoints', @checkWorldPoints);
parser.addParameter('WorldUnits', 'mm', @checkWorldUnits);
parser.addParameter('EstimateSkew', false, @checkEstimateSkew);
parser.addParameter('EstimateTangentialDistortion', false, ...
    @checkEstimateTangentialDistortion);
parser.addParameter('NumRadialDistortionCoefficients', 2, ...
    @checkNumRadialDistortionCoefficients);
parser.addParameter('InitialIntrinsicMatrix', {}, @checkInitialIntrinsicMatrix);
parser.addParameter('InitialRadialDistortion', {}, @checkInitialRadialDistortion);
parser.addParameter('ShowProgressBar', false, @checkShowProgressBar);

parser.parse(varargin{:});

imagePoints = parser.Results.imagePoints;
worldPoints = parser.Results.worldPoints;
if size(imagePoints, 1) ~= size(worldPoints, 1)
    error(message('vision:calibrate:numberOfPointsMustMatch'));
end

worldUnits  = parser.Results.WorldUnits;
cameraModel.EstimateSkew = parser.Results.EstimateSkew;
cameraModel.EstimateTangentialDistortion = ...
    parser.Results.EstimateTangentialDistortion;
cameraModel.NumRadialDistortionCoefficients = ...
    parser.Results.NumRadialDistortionCoefficients;

% 获取初始内参和畸变参数
initIntrinsics = parser.Results.InitialIntrinsicMatrix;
initRadial = parser.Results.InitialRadialDistortion;

% 添加对cell数组形式参数的处理
if iscell(initIntrinsics) && length(initIntrinsics) == 2
    % 保存两个相机的内参
    calibrationParams.initIntrinsics1 = double(initIntrinsics{1});
    calibrationParams.initIntrinsics2 = double(initIntrinsics{2});
else
    calibrationParams.initIntrinsics1 = double(initIntrinsics);
    calibrationParams.initIntrinsics2 = double(initIntrinsics);
end

if iscell(initRadial) && length(initRadial) == 2
    % 保存两个相机的畸变参数
    calibrationParams.initRadial1 = double(initRadial{1});
    calibrationParams.initRadial2 = double(initRadial{2});
else
    calibrationParams.initRadial1 = double(initRadial);
    calibrationParams.initRadial2 = double(initRadial);
end

calibrationParams.showProgressBar = parser.Results.ShowProgressBar;

%--------------------------------------------------------------------------
function tf = checkImagePoints(imagePoints)
validateattributes(imagePoints, {'double'}, ...
    {'finite', 'nonsparse', 'ncols', 2}, ...
    mfilename, 'imagePoints');

if ndims(imagePoints) == 4
    validateattributes(imagePoints, {'double'}, {'size', [nan, nan, nan, 2]}, ...
        mfilename, 'imagePoints');
end

if ndims(imagePoints) > 4
    error(message('vision:calibrate:imagePoints3Dor4D'));
end

if size(imagePoints, 3) < 2
    error(message('vision:calibrate:minNumPatterns'));
end

minNumPoints = 4;
if size(imagePoints, 1) < minNumPoints
    error(message('vision:calibrate:minNumImagePoints', minNumPoints-1));
end

tf = true;

%--------------------------------------------------------------------------
function tf = checkWorldPoints(worldPoints)
validateattributes(worldPoints, {'double'}, ...
    {'finite', 'nonsparse', '2d', 'ncols', 2}, ...
    mfilename, 'worldPoints');

minNumPoints = 4;
if size(worldPoints, 1) < minNumPoints
    error(message('vision:calibrate:minNumWorldPoints', minNumPoints-1));
end

tf = true;

%--------------------------------------------------------------------------
function tf = checkWorldUnits(worldUnits)
validateattributes(worldUnits, {'char'}, {'vector'}, mfilename, 'worldUnits');
tf = true;

%--------------------------------------------------------------------------
function tf = checkEstimateSkew(esitmateSkew)
validateattributes(esitmateSkew, {'logical'}, {'scalar'}, ...
    mfilename, 'EstimateSkew');
tf = true;

%--------------------------------------------------------------------------
function tf = checkEstimateTangentialDistortion(estimateTangential)
validateattributes(estimateTangential, {'logical'}, {'scalar'}, mfilename, ...
    'EstimateTangentialDistortion');
tf = true;

%--------------------------------------------------------------------------
function tf = checkNumRadialDistortionCoefficients(numRadialCoeffs)
validateattributes(numRadialCoeffs, {'numeric'}, ...
   {'scalar', 'integer', '>=', 2, '<=', 3}, ...
   mfilename, 'NumRadialDistortionCoefficients');
tf = true;

%--------------------------------------------------------------------------
function tf = checkInitialIntrinsicMatrix(K)
if ~isempty(K)
    % if iscell(K)
    %     % 处理cell数组形式的内参矩阵
    %     for i = 1:length(K)
    %         validateattributes(K{i}, {'single', 'double'}, ...
    %             {'real', 'nonsparse', 'finite', 'size', [3 3]}, ...
    %             mfilename, 'InitialIntrinsicMatrix');
    %     end
    % else
    %     validateattributes(K, {'single', 'double'}, ...
    %         {'real', 'nonsparse', 'finite', 'size', [3 3]}, ...
    %         mfilename, 'InitialIntrinsicMatrix');
    % end
end
tf = true;

%--------------------------------------------------------------------------
function tf = checkInitialRadialDistortion(P)
if ~isempty(P)
    % if iscell(P)
    %     % 处理cell数组形式的畸变参数
    %     for i = 1:length(P)
    %         validateattributes(P{i}, {'single', 'double'},...
    %             {'real', 'nonsparse', 'finite', 'vector'},...
    %             mfilename, 'InitialRadialDistortion');
            
    %         if numel(P{i}) ~= 2 && numel(P{i}) ~= 3
    %             error(message('vision:calibrate:invalidRadialDistortion'));
    %         end
    %     end
    % else
    %     validateattributes(P, {'single', 'double'},...
    %         {'real', 'nonsparse', 'finite', 'vector'},...
    %         mfilename, 'InitialRadialDistortion');
        
    %     if numel(P) ~= 2 && numel(P) ~= 3
    %         error(message('vision:calibrate:invalidRadialDistortion'));
    %     end
    % end
end
tf = true;

%--------------------------------------------------------------------------
function tf = checkShowProgressBar(showProgressBar)
vision.internal.inputValidation.validateLogical(showProgressBar, 'ShowProgressBar');
tf = true;

%--------------------------------------------------------------------------
function [cameraParams, imagesUsed, errors] = calibrateOneCamera(imagePoints, ...
    worldPoints, cameraModel, worldUnits, calibrationParams)

progressBar = createSingleCameraProgressBar(calibrationParams.showProgressBar);

% compute the initial "guess" of intrinisc and extrinsic camera parameters
% in closed form ignoring distortion
[cameraParams, imagesUsed] = computeInitialParameterEstimate(...
    worldPoints, imagePoints, cameraModel, worldUnits, ...
    calibrationParams.initIntrinsics, calibrationParams.initRadial);
imagePoints = imagePoints(:, :, imagesUsed);

progressBar.update();

% refine the initial estimate and compute distortion coefficients using
% non-linear least squares minimization
errors = refine(cameraParams, imagePoints, calibrationParams.shouldComputeErrors);
progressBar.update();
progressBar.delete();
%--------------------------------------------------------------------------
function [iniltialParams, validIdx] = computeInitialParameterEstimate(...
    worldPoints, imagePoints, cameraModel, worldUnits, initIntrinsics, initRadial)
% Solve for the camera intriniscs and extrinsics in closed form ignoring
% distortion.

[H, validIdx] = computeHomographies(imagePoints, worldPoints);

if isempty(initIntrinsics)
    V = computeV(H);
    B = computeB(V);
    A = computeIntrinsics(B);
else
    % initial guess for the intrinsics has been provided. No need to solve.
    A = initIntrinsics;
end

[rvecs, tvecs] = computeExtrinsics(A, H);

if isempty(initRadial)
    radialCoeffs = zeros(1, cameraModel.NumRadialDistortionCoefficients);
else
    radialCoeffs = initRadial;
end

iniltialParams = cameraParameters('IntrinsicMatrix', A', ...
    'RotationVectors', rvecs, ...
    'TranslationVectors', tvecs, 'WorldPoints', worldPoints, ...
    'WorldUnits', worldUnits, 'EstimateSkew', cameraModel.EstimateSkew,...
    'NumRadialDistortionCoefficients', cameraModel.NumRadialDistortionCoefficients,...
    'EstimateTangentialDistortion', cameraModel.EstimateTangentialDistortion,...
    'RadialDistortion', radialCoeffs);

%--------------------------------------------------------------------------
function H = computeHomography(imagePoints, worldPoints)
% Compute projective transformation from worldPoints to imagePoints

H = fitgeotrans(worldPoints, imagePoints, 'projective');
H = (H.T)';
H = H / H(3,3);


%--------------------------------------------------------------------------
function [homographies, validIdx] = computeHomographies(points, worldPoints)
% Compute homographies for all images

w1 = warning('Error', 'MATLAB:nearlySingularMatrix'); %#ok
w2 = warning('Error', 'images:maketform:conditionNumberofAIsHigh'); %#ok

numImages = size(points, 3);
validIdx = true(numImages, 1);
homographies = zeros(3, 3, numImages);
for i = 1:numImages
    try    
        homographies(:, :, i) = ...
            computeHomography(double(points(:, :, i)), worldPoints);
    catch 
        validIdx(i) = false;
    end
end
warning(w1);
warning(w2);
homographies = homographies(:, :, validIdx);
if ~all(validIdx)
    warning(message('vision:calibrate:invalidHomographies', ...
        numImages - size(homographies, 3), numImages));
end

if size(homographies, 3) < 2
    error(message('vision:calibrate:notEnoughValidHomographies'));
end

%--------------------------------------------------------------------------
function V = computeV(homographies)
% Vb = 0

numImages = size(homographies, 3);
V = zeros(2 * numImages, 6);
for i = 1:numImages
    H = homographies(:, :, i)';
    V(i*2-1,:) = computeLittleV(H, 1, 2);
    V(i*2, :) = computeLittleV(H, 1, 1) - computeLittleV(H, 2, 2);
end

%--------------------------------------------------------------------------
function v = computeLittleV(H, i, j)
    v = [H(i,1)*H(j,1), H(i,1)*H(j,2)+H(i,2)*H(j,1), H(i,2)*H(j,2),...
         H(i,3)*H(j,1)+H(i,1)*H(j,3), H(i,3)*H(j,2)+H(i,2)*H(j,3), H(i,3)*H(j,3)];

%--------------------------------------------------------------------------     
function B = computeB(V)
% lambda * B = inv(A)' * inv(A), where A is the intrinsic matrix

[~, ~, U] = svd(V);
b = U(:, end);

% b = [B11, B12, B22, B13, B23, B33]
B = [b(1), b(2), b(4); b(2), b(3), b(5); b(4), b(5), b(6)];

%--------------------------------------------------------------------------
function A = computeIntrinsics(B)
% Compute the intrinsic matrix
cy = (B(1,2)*B(1,3) - B(1,1)*B(2,3)) / (B(1,1)*B(2,2)-B(1,2)^2);
lambda = B(3,3) - (B(1,3)^2 + cy * (B(1,2)*B(1,3) - B(1,1)*B(2,3))) / B(1,1);
fx = sqrt(lambda / B(1,1));
fy = sqrt(lambda * B(1,1) / (B(1,1) * B(2,2) - B(1,2)^2));
skew = -B(1,2) * fx^2 * fy / lambda;
cx = skew * cy / fx - B(1,3) * fx^2 / lambda;
A = vision.internal.calibration.constructIntrinsicMatrix(fx, fy, cx, cy, skew);
if ~isreal(A)
    error(message('vision:calibrate:complexCameraMatrix'));
end

%--------------------------------------------------------------------------
function [rotationVectors, translationVectors] = ...
    computeExtrinsics(A, homographies)
% Compute translation and rotation vectors for all images

numImages = size(homographies, 3);
rotationVectors = zeros(3, numImages);
translationVectors = zeros(3, numImages); 
Ainv = inv(A);
for i = 1:numImages;
    H = homographies(:, :, i);
    h1 = H(:, 1);
    h2 = H(:, 2);
    h3 = H(:, 3);
    lambda = 1 / norm(Ainv * h1); %#ok
    
    % 3D rotation matrix
    r1 = lambda * Ainv * h1; %#ok
    r2 = lambda * Ainv * h2; %#ok
    r3 = cross(r1, r2);
    R = [r1,r2,r3];
    
    rotationVectors(:, i) = vision.internal.calibration.rodriguesMatrixToVector(R);
    
    % translation vector
    t = lambda * Ainv * h3;  %#ok
    translationVectors(:, i) = t;
end

rotationVectors = rotationVectors';
translationVectors = translationVectors';

%--------------------------------------------------------------------------
function [stereoParams, pairsUsed, errors] = calibrateTwoCameras(imagePoints,...
    worldPoints, cameraModel, worldUnits, calibrationParams)

imagePoints1 = imagePoints(:, :, :, 1);
imagePoints2 = imagePoints(:, :, :, 2);

showProgressBar = calibrationParams.showProgressBar;
progressBar = createStereoCameraProgressBar(showProgressBar);
calibrationParams.showProgressBar = false;

% Calibrate each camera separately
shouldComputeErrors = calibrationParams.shouldComputeErrors;
calibrationParams.shouldComputeErrors = false;
[cameraParameters1, imagesUsed1] = calibrateOneCamera(imagePoints1, ...
    worldPoints, cameraModel, worldUnits, calibrationParams);

progressBar.update();

[cameraParameters2, imagesUsed2] = calibrateOneCamera(imagePoints2, ...
    worldPoints, cameraModel, worldUnits, calibrationParams);

progressBar.update();

% Account for possible mismatched pairs
pairsUsed = imagesUsed1 & imagesUsed2;
cameraParameters1 = removeUnusedExtrinsics(cameraParameters1, pairsUsed, ...
    imagesUsed1);
cameraParameters2 = removeUnusedExtrinsics(cameraParameters2, pairsUsed, ...
    imagesUsed2);

% Compute the initial estimate of translation and rotation of camera 2
[R, t] = estimateInitialTranslationAndRotation(cameraParameters1, ...
    cameraParameters2);

stereoParams = stereoParameters(cameraParameters1, ...
    cameraParameters2, R, t);

errors = refine(stereoParams, imagePoints1(:, :, pairsUsed), ...
    imagePoints2(:, :, pairsUsed), shouldComputeErrors);

progressBar.update();
delete(progressBar);
%--------------------------------------------------------------------------
function cameraParams = removeUnusedExtrinsics(cameraParams, pairsUsed, ...
    imagesUsed)
% Remove the extrinsics corresponding to the images that were not used by
% the other camera
rotationVectors = zeros(numel(pairsUsed), 3);
rotationVectors(imagesUsed, :) = cameraParams.RotationVectors;

translationVectors = zeros(numel(pairsUsed), 3);
translationVectors(imagesUsed, :) = cameraParams.TranslationVectors;

cameraParams.setExtrinsics(...
    rotationVectors(pairsUsed, :), translationVectors(pairsUsed, :));

%--------------------------------------------------------------------------
% Compute the initial estimate of translation and rotation of camera 2
% relative to camera 1.
%--------------------------------------------------------------------------
function [R, t] = estimateInitialTranslationAndRotation(cameraParameters1, ...
    cameraParameters2)
% Now the number of images in both cameras should be the same.
numImages = cameraParameters1.NumPatterns;

rotationVectors = zeros(numImages, 3);
translationVectors = zeros(numImages, 3);

% For each set of extrinsics, compute R (rotation) and T (translation)
% between the two cameras.
for i = 1:numImages
    R = cameraParameters2.RotationMatrices(:, :, i)* ...
        cameraParameters1.RotationMatrices(:, :, i)';
    rotationVectors(i, :) = vision.internal.calibration.rodriguesMatrixToVector(R);
    translationVectors(i, :) = (cameraParameters2.TranslationVectors(i, :)' - ...
        R * cameraParameters1.TranslationVectors(i, :)')';
end

% Take the median rotation and translation as the initial guess.
r = median(rotationVectors, 1);
R = vision.internal.calibration.rodriguesVectorToMatrix(r)';
t = median(translationVectors, 1);

%--------------------------------------------------------------------------
function progressBar = createSingleCameraProgressBar(isEnabled)
messages = {'vision:calibrate:initialGuess', 'vision:calibrate:jointOptimization', ...
            'vision:calibrate:calibrationComplete'};
percentages = [0, 0.25, 1];        
progressBar = vision.internal.calibration.CalibrationProgressBar(isEnabled,...
    messages, percentages);

%--------------------------------------------------------------------------
function progressBar = createStereoCameraProgressBar(isEnabled)
messages = {'vision:calibrate:calibratingCamera1', ...
    'vision:calibrate:calibratingCamera2', ...
    'vision:calibrate:jointOptimization', 'vision:calibrate:calibrationComplete'};
percentages = [0, 0.25, .5, 1];        
progressBar = vision.internal.calibration.CalibrationProgressBar(isEnabled,...
    messages, percentages);
%--------------------------------------------------------------------------
function [stereoParams,errors] = optimizeExtrinsicsOnly(stereoParams, imagePoints1, imagePoints2, shouldComputeErrors)
% 固定内参，只优化外参的函数

% 保存原始内参
intrinsics1 = stereoParams.CameraParameters1.K;
intrinsics2 = stereoParams.CameraParameters2.K;
radialDist1 = stereoParams.CameraParameters1.RadialDistortion;
radialDist2 = stereoParams.CameraParameters2.RadialDistortion;
tangentialDist1 = stereoParams.CameraParameters1.TangentialDistortion;
tangentialDist2 = stereoParams.CameraParameters2.TangentialDistortion;

% 提取当前外参作为初始值
numImages = size(imagePoints1, 3);
rotVecs1 = stereoParams.CameraParameters1.RotationVectors;
transVecs1 = stereoParams.CameraParameters1.TranslationVectors;
rotVecs2 = stereoParams.CameraParameters2.RotationVectors;
transVecs2 = stereoParams.CameraParameters2.TranslationVectors;
R = stereoParams.RotationOfCamera2;
t = stereoParams.TranslationOfCamera2;
worldPoints = stereoParams.WorldPoints;
WorldPoints = [worldPoints, zeros(size(worldPoints,1), 1)];


% 将所有外参合并为一个向量用于优化
initialParams = [rotVecs1(:); transVecs1(:); rotVecs2(:); transVecs2(:); vision.internal.calibration.rodriguesMatrixToVector(R); t(:)];

% 设置优化选项
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', ...
                      'Display', 'iter', 'MaxIterations', 160,'StepTolerance', 1e-10);

% 定义目标函数：计算重投影误差
objectiveFunction = @(params) computeReprojectionErrorsFixedIntrinsics(params, ...
                                intrinsics1, intrinsics2, ...
                                radialDist1, radialDist2, ...
                                tangentialDist1, tangentialDist2, ...
                                imagePoints1, imagePoints2, numImages,WorldPoints);

% 执行非线性最小二乘优化
[optimizedParams, ~, residuals] = lsqnonlin(objectiveFunction, initialParams, [], [], options);
rotVecs1_opt = reshape(optimizedParams(1:3*numImages), numImages, 3);
transVecs1_opt = reshape(optimizedParams(3*numImages+1:6*numImages), numImages, 3);
rotVecs2_opt = reshape(optimizedParams(6*numImages+1:9*numImages), numImages, 3);
transVecs2_opt = reshape(optimizedParams(9*numImages+1:12*numImages), numImages, 3);
R_opt = vision.internal.calibration.rodriguesVectorToMatrix(optimizedParams(12*numImages+1:12*numImages+3));
t_opt = optimizedParams(12*numImages+4:12*numImages+6)';
% 计算每一对图像的重投影误差，找出误差最小的那一对
minError = inf;
bestIdx = 1;
errorPerPair = zeros(numImages, 1);

for i = 1:numImages
    % 创建临时相机参数对象用于计算误差
    tempCamParams1 = cameraParameters('IntrinsicMatrix', intrinsics1', ...
                                 'RotationVectors', rotVecs1_opt(i,:), ...
                                 'TranslationVectors', transVecs1_opt(i,:), ...
                                 'RadialDistortion', radialDist1, ...
                                 'TangentialDistortion', tangentialDist1);
                             
    tempCamParams2 = cameraParameters('IntrinsicMatrix', intrinsics2', ...
                                 'RotationVectors', rotVecs2_opt(i,:), ...
                                 'TranslationVectors', transVecs2_opt(i,:), ...
                                 'RadialDistortion', radialDist2, ...
                                 'TangentialDistortion', tangentialDist2);
    
    % 计算当前图像对的重投影误差
    rotMatrix1 = vision.internal.calibration.rodriguesVectorToMatrix(rotVecs1_opt(i,:));
    rotMatrix2 = vision.internal.calibration.rodriguesVectorToMatrix(rotVecs2_opt(i,:));
    
    projPoints1 = worldToImage(tempCamParams1, rotMatrix1, transVecs1_opt(i,:)', WorldPoints);
    projPoints2 = worldToImage(tempCamParams2, rotMatrix2, transVecs2_opt(i,:)', WorldPoints);
    
    errors1 = imagePoints1(:,:,i) - projPoints1;
    errors2 = imagePoints2(:,:,i) - projPoints2;
    
    totalError = sum(errors1(:).^2) + sum(errors2(:).^2);
    errorPerPair(i) = totalError;
    
    if totalError < minError
        minError = totalError;
        bestIdx = i;
    end
end

% 输出最佳外参对的索引和误差
fprintf('最小误差的外参对索引: %d, 误差值: %f\n', bestIdx, minError);

% 使用误差最小的那一对外参计算外参
best_rotVec1 = rotVecs1_opt(bestIdx,:);
best_transVec1 = transVecs1_opt(bestIdx,:);
best_rotVec2 = rotVecs2_opt(bestIdx,:);
best_transVec2 = transVecs2_opt(bestIdx,:);

% 创建新的相机参数对象
camParams1 = cameraParameters('IntrinsicMatrix', stereoParams.CameraParameters1.K', ...
    'RotationVectors', best_rotVec1, ...
    'TranslationVectors', best_transVec1, ...
    'RadialDistortion', stereoParams.CameraParameters1.RadialDistortion, ...
    'TangentialDistortion', stereoParams.CameraParameters1.TangentialDistortion);

camParams2 = cameraParameters('IntrinsicMatrix', stereoParams.CameraParameters2.K', ...
    'RotationVectors', best_rotVec2, ...
    'TranslationVectors', best_transVec2, ...
    'RadialDistortion', stereoParams.CameraParameters2.RadialDistortion, ...
    'TangentialDistortion', stereoParams.CameraParameters2.TangentialDistortion);

% 创建新的stereoParameters对象
stereoParams = stereoParameters(camParams1, camParams2, R_opt, t_opt);

% 计算最终误差
if shouldComputeErrors
    %errors = computeCalibrationErrors(stereoParams, imagePoints1, imagePoints2,worldPoints);
    errors = minError;
else
    errors = [];
end

%--------------------------------------------------------------------------
function residuals = computeReprojectionErrorsFixedIntrinsics(params, ...
                     intrinsics1, intrinsics2, ...
                     radialDist1, radialDist2, ...
                     tangentialDist1, tangentialDist2, ...
                     imagePoints1, imagePoints2, numImages,WorldPoints)
% 计算固定内参情况下的重投影误差

% 从参数向量中提取外参
rotVecs1 = reshape(params(1:3*numImages), numImages, 3);
transVecs1 = reshape(params(3*numImages+1:6*numImages), numImages, 3);
rotVecs2 = reshape(params(6*numImages+1:9*numImages), numImages, 3);
transVecs2 = reshape(params(9*numImages+1:12*numImages), numImages, 3);
R = vision.internal.calibration.rodriguesVectorToMatrix(params(12*numImages+1:12*numImages+3));
t = params(12*numImages+4:12*numImages+6)';

% 创建临时相机参数对象
camParams1 = cameraParameters('IntrinsicMatrix', intrinsics1', ...
                             'RotationVectors', rotVecs1, ...
                             'TranslationVectors', transVecs1, ...
                             'RadialDistortion', radialDist1, ...
                             'TangentialDistortion', tangentialDist1);
                         
camParams2 = cameraParameters('IntrinsicMatrix', intrinsics2', ...
                             'RotationVectors', rotVecs2, ...
                             'TranslationVectors', transVecs2, ...
                             'RadialDistortion', radialDist2, ...
                             'TangentialDistortion', tangentialDist2);

% 获取世界坐标点
worldPoints = WorldPoints;
% 初始化投影点数组
projectedPoints1 = zeros(size(imagePoints1));
projectedPoints2 = zeros(size(imagePoints2));

% 逐帧计算投影点
for i = 1:numImages
    % 将旋转向量转换为旋转矩阵
    rotMatrix1 = vision.internal.calibration.rodriguesVectorToMatrix(rotVecs1(i,:));
    rotMatrix2 = vision.internal.calibration.rodriguesVectorToMatrix(rotVecs2(i,:));
    
    % 使用旋转矩阵和平移向量计算投影点
    projectedPoints1(:,:,i) = worldToImage(camParams1, rotMatrix1, transVecs1(i,:)', worldPoints);
    projectedPoints2(:,:,i) = worldToImage(camParams2, rotMatrix2, transVecs2(i,:)', worldPoints);
end

% 计算重投影误差
residuals1 = imagePoints1 - projectedPoints1;
residuals2 = imagePoints2 - projectedPoints2;

% 将误差整合为一个向量
residuals = [residuals1(:); residuals2(:)];
%--------------------------------------------------------------------------
function errors = computeCalibrationErrors(stereoParams, imagePoints1, imagePoints2,worldPoints)
% 计算标定误差

% 获取世界坐标点
worldPoints = [worldPoints, zeros(size(worldPoints,1), 1)];
numImages = size(imagePoints1, 3);

% 初始化投影点数组
projectedPoints1 = zeros(size(imagePoints1));
projectedPoints2 = zeros(size(imagePoints2));

% 逐帧计算投影点
for i = 1:numImages
    % 将旋转向量转换为旋转矩阵
    rotMatrix1 = vision.internal.calibration.rodriguesVectorToMatrix(stereoParams.CameraParameters1.RotationVectors(i,:));
    rotMatrix2 = vision.internal.calibration.rodriguesVectorToMatrix(stereoParams.CameraParameters2.RotationVectors(i,:));
    
    % 使用旋转矩阵和平移向量计算投影点
    projectedPoints1(:,:,i) = worldToImage(stereoParams.CameraParameters1, ...
                             rotMatrix1, ...
                             stereoParams.CameraParameters1.TranslationVectors(i,:)', ...
                             worldPoints);
    projectedPoints2(:,:,i) = worldToImage(stereoParams.CameraParameters2, ...
                             rotMatrix2, ...
                             stereoParams.CameraParameters2.TranslationVectors(i,:)', ...
                             worldPoints);
end

% 计算误差
errors1 = imagePoints1 - projectedPoints1;
errors2 = imagePoints2 - projectedPoints2;

% 计算RMS误差
rmsError1 = sqrt(mean(sum(errors1.^2, 2), 'all'));
rmsError2 = sqrt(mean(sum(errors2.^2, 2), 'all'));

% 创建误差结构体
errors.CameraParameters1 = rmsError1;
errors.CameraParameters2 = rmsError2;
errors.MeanReprojectionError = (rmsError1 + rmsError2) / 2;




