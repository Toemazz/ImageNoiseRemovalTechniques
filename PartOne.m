%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT: PartOne Program
clear; clc; close all;

% ----------------------------------------------------------------
% BOTTLE CAP MISSING
% Get 'bottle cap missing' images
imagesDirBCM = 'images/BottleCapMissing/';
fileDataBCM = GetFileDataFromDirectory(imagesDirBCM);
% ----------------------------------------------------------------

% ----------------------------------------------------------------
% BOTTLE UNDERFILLED
% Get 'bottle underfilled' images
imagesDirBU = 'images/BottleUnderfilled/';
fileDataBU = GetFileDataFromDirectory(imagesDirBU);
% ----------------------------------------------------------------

% ----------------------------------------------------------------
% LABEL MISSING
% Get 'label missing' images
imagesDirLM = 'images/LabelMissing/';
fileDataLM = GetFileDataFromDirectory(imagesDirLM);
% ----------------------------------------------------------------

% Assume each folder of images has the same number of images
numFiles = length(fileDataBCM);

% Define noise levels and number of times to executed the tests
noiseLevels = 0.0:0.0035:0.35;
numTests = 1;
results = zeros(3, length(noiseLevels));

% Loop over number of tests
for k = 1:numTests
    % Loop over noise levels
    for j = 1:length(noiseLevels)
        % Initialize fault counts
        faultCountBCM = 0;
        faultCountBU = 0;
        faultCountLM = 0;
        
        % Loop over number of files
        for i = 1:numFiles
            % ------------------------------------------------------------
            % BOTTLE CAP MISSING
            % Load image and add noise
            filePath = fullfile(imagesDirBCM, fileDataBCM(i).name);
            image = imread(filePath);
            imageWithNoise = imnoise(image, 'gaussian', 0, noiseLevels(j));
            
            % Check if fault detected
            bottleCapMissing = CheckIfBottleCapMissing(imageWithNoise);
            faultCountBCM = faultCountBCM + bottleCapMissing;
            % ------------------------------------------------------------

            % ------------------------------------------------------------
            % BOTTLE UNDERFILLED
            % Load image and add noise
            filePath = fullfile(imagesDirBU, fileDataBU(i).name);
            image = imread(filePath);
            imageWithNoise = imnoise(image, 'gaussian', 0, noiseLevels(j));
            
            % Check if fault detected
            bottleUnderfilled = CheckIfBottleUnderfilled(imageWithNoise);
            faultCountBU = faultCountBU  + bottleUnderfilled;
            % ------------------------------------------------------------

            % ------------------------------------------------------------
            % LABEL MISSING
            % Load image and add noise
            filePath = fullfile(imagesDirLM, fileDataLM(i).name);
            image = imread(filePath);
            imageWithNoise = imnoise(image, 'gaussian', 0, noiseLevels(j));
            
            % Check if fault detected
            labelMissing = CheckIfLabelMissing(imageWithNoise);
            faultCountLM = faultCountLM  + labelMissing;
            % ------------------------------------------------------------
        end
        
        % Add fault classifcation % to the 'results' array
        results(1, j) = results(1, j) + (100*(faultCountBCM/numFiles));
        results(2, j) = results(2, j) + (100*(faultCountBU/numFiles));
        results(3, j) = results(3, j) + (100*(faultCountLM/numFiles));
    end
end

% Divide each element of the 'results' array by the number of tests
results = results ./ numTests;

% ----------------------------------------------------------------
% BOTTLE CAP MISSING
% Plot bar chart
figure;
bar(noiseLevels, results(1, :), 1, 'r');
title('Fault Detection Performance');
xlabel('Noise Level')
ylabel('Fault Detection %');
ylim([0,  105])
grid on;
legend('Bottle Cap Missing');
% ----------------------------------------------------------------

% ----------------------------------------------------------------
% BOTTLE UNDERFILLED
% Plot bar chart
figure;
bar(noiseLevels, results(2, :), 1, 'b');
title('Fault Detection Performance');
xlabel('Noise Level')
ylabel('Fault Detection %');
ylim([0,  105])
grid on;
legend('Bottle Underfilled');
% ----------------------------------------------------------------

% ----------------------------------------------------------------
% LABEL MISSING
% Plot bar chart
figure;
bar(noiseLevels, results(3, :), 1, 'g');
title('Fault Detection Performance');
xlabel('Noise Level')
ylabel('Fault Detection %');
ylim([0,  105])
grid on;
legend('Label Missing');
% ----------------------------------------------------------------

% ----------------------------------------------------------------
% OVERALL PERFORMANCE
% Plot graph
figure;
plot(noiseLevels, results(1, :), 'r', 'LineWidth', 2); hold on;
plot(noiseLevels, results(2, :), 'b', 'LineWidth', 2); hold on;
plot(noiseLevels, results(3, :), 'g', 'LineWidth', 2); hold on;
title('Fault Detection Performance');
xlabel('Noise Level')
ylabel('Fault Detection %');
ylim([0,  105])
grid on;
legend('Bottle Cap Missing', 'Bottle Underfilled', 'Label Missing');
% ----------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: Used to load images from a specified directory
function fileData = GetFileDataFromDirectory(dirPath)
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(dirPath)
    errorMessage = sprintf('[ERROR]: The following folder does not exist:\n%s', dirPath);
    uiwait(warndlg(errorMessage));
    return;
end

% Get a list of all '.jpg' files in the directory
filePattern = fullfile(dirPath, '*.jpg');
fileData = dir(filePattern);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: Used to extract a ROI (Region of Interest) from an image
function imageOut = ExtractROI(imageIn, y1, x1, y2, x2)
% Check if any of the points are '0'
if x1 == 0 || x2 == 0 || y1 == 0 || y2 == 0
    errorMessage = sprintf('[ERROR]: Ooops you forgot MATLAB indices start at 1!\n');
    uiwait(warndlg(errorMessage));
    return;
end

% Get image dimensions
[h, w, ~] = size(imageIn);

if x1 > w || x2 > w || y1 > h || y2 > h
    errorMessage = sprintf('[ERROR]: Images dimensions (%d, %d) exceeded!\n', h, w);
    uiwait(warndlg(errorMessage));
    return;
end

imageOut = imageIn(y1:y2, x1:x2, :);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: Used to detect cases where the bottle cap is missing
function result = CheckIfBottleCapMissing(image)
    % Convert to greyscale
    image = rgb2gray(image);
    
    % Extract the ROI for the bottle cap 
    roi = ExtractROI(image, 5, 150, 45, 200);
    
    % Convert to a binary image using a greyscale threshold of '150' 
    roiBinary = imbinarize(roi, double(150/256));

    % Calculate the percentage of black pixels in the binary image
    blackPercentage = 100 * (sum(roiBinary(:) == 0) / numel(roiBinary(:)));
    
    % Fault detected if % black pixels is less than 25%
    result = blackPercentage < 25;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: Used to detect cases where the bottle is underfilled
function result = CheckIfBottleUnderfilled(image)
    % Convert to greyscale
    image = rgb2gray(image);
    
    % Extract the ROI just under the ideal liquid level in the bottle
    roi = ExtractROI(image, 130, 140, 170, 220);
    
    % Convert to a binary image using a greyscale threshold of '150' 
    roiBinary = imbinarize(roi, double(150/256));
    
    % Calculate the percentage of black pixels in the binary image
    blackPercentage = 100 * (sum(roiBinary(:) == 0) / numel(roiBinary(:)));

    % Fault detected if % black pixels is less than 25%
    result = blackPercentage < 25;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: Used to detect cases where the label is missing
function result = CheckIfLabelMissing(image)
    % Convert to greyscale
    image = rgb2gray(image);
    
    % Extract the ROI for the label
    roi = ExtractROI(image, 180, 110, 280, 240);
    
    % Convert to a binary image using a greyscale threshold of '50' 
    roiBinary = imbinarize(roi, double(50/256));
    
    % Calculate the percentage of black pixels in the binary image
    blackPercentage = 100 * (sum(roiBinary(:) == 0) / numel(roiBinary(:)));
    
    % Fault detected if % black pixels is greater than 50%
    result = blackPercentage > 50;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
