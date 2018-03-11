%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT: PartTwo_B Program
clear; clc; close all;

% Get 'bottle underfilled' images
imagesDir = 'images/BottleUnderfilled/';
fileData = GetFileDataFromDirectory(imagesDir);
numFiles = length(fileData);

% Define number of tests, and filter parameters
numTests = 10;
N = 5;
standardDev = 2.5;
normalCutOffFreq = 0.1;

% Define noise levels
noiseLevels = 0.0:0.01:1.0;
results = zeros(5, length(noiseLevels));

% Loop over number of tests
for k = 1:numTests
    % Loop over the noise levels
    for j = 1:length(noiseLevels)
        faultCount = 0;
        faultCountMean = 0;
        faultCountGauss = 0;
        faultCountMed = 0;
        faultCountFreq = 0;
        
        % Loop over number of files
        for i = 1:numFiles
            % Load image
            filePath = fullfile(imagesDir, fileData(i).name);
            image = imread(filePath);

            % Add noise
            imageWithNoise = imnoise(image, 'gaussian', 0, noiseLevels(j));

            % --------------------------------------------------------
            % NOISE REMOVAL TECHNIQUES
            % Apply a mean filter
            imageWithMeanFilt = imfilter(imageWithNoise, ones(N, N)/N^2);

            % Apply a Gaussian filter
            imageWithGaussFilt = imgaussfilt(imageWithNoise, standardDev);

            % Apply a median filter
            imageWithMedFilt = medfilt2(rgb2gray(imageWithNoise), [N, N]);

            % Apply a simple 'low-pass' filter (Frequency Domain)
            imageWithFreqFilt = IdealLowPassFilt(imageWithNoise, normalCutOffFreq);
            % --------------------------------------------------------

            % Check if fault detected
            bu      = CheckIfBottleUnderfilled(imageWithNoise);
            buMean  = CheckIfBottleUnderfilled(imageWithMeanFilt);
            buMed   = CheckIfBottleUnderfilled(imageWithMedFilt);
            buGauss = CheckIfBottleUnderfilled(imageWithGaussFilt);
            buFreq  = CheckIfBottleUnderfilled(imageWithFreqFilt);

            faultCount      = faultCount      + bu;
            faultCountMean  = faultCountMean  + buMean;
            faultCountMed   = faultCountMed   + buMed;
            faultCountGauss = faultCountGauss + buGauss;
            faultCountFreq  = faultCountFreq  + buFreq;
        end
        
        % Add fault classifcation % to the 'results' array
        results(1, j) = results(1, j) + (100*(faultCount      / numFiles));
        results(2, j) = results(2, j) + (100*(faultCountMean  / numFiles));
        results(3, j) = results(3, j) + (100*(faultCountMed   / numFiles));
        results(4, j) = results(4, j) + (100*(faultCountGauss / numFiles));
        results(5, j) = results(5, j) + (100*(faultCountFreq  / numFiles));
    end
end

% Divide each element of the 'results' array by the number of tests
results = results ./ numTests;

% ----------------------------------------------------------------
% OVERALL PERFORMANCE
% Plot graph
figure;
plot(noiseLevels, results(1, :), 'b', 'LineWidth', 2); hold on;
plot(noiseLevels, results(2, :), 'k', 'LineWidth', 2); hold on;
plot(noiseLevels, results(3, :), 'y', 'LineWidth', 2); hold on;
plot(noiseLevels, results(4, :), 'm', 'LineWidth', 2); hold on;
plot(noiseLevels, results(5, :), 'c', 'LineWidth', 2); hold on;
title('Fault Detection Performance');
xlabel('Noise Level')
ylabel('Fault Detection %');
ylim([0,  105])
grid on;
legend({'No Filter', 'Mean', 'Median', 'Gaussian', 'Low-Pass'}, 'Location', 'south');
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
% FUNCTION: Used to detect cases where the bottle is underfilled
function result = CheckIfBottleUnderfilled(image)
    % Convert to greyscale
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
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
% FUNCTION: Used to apply a low-pass (frequency domain) filter to an image
function filteredImage = IdealLowPassFilt(image, cutoffFreq)
    % Get image dimensions
    [h, w, c] = size(image);

    % Get centered version of discrete Fourier transform
    DFT = fftshift(fft2(image));
    
    % Calculate image centerpoint
    hr = (h-1)/2; 
    hc = (w-1)/2; 
    [X, Y] = meshgrid(-hc:hc, -hr:hr);
    
    % Construct ideal low-pass filter
    freqFilt = sqrt((X/hc).^2 + (Y/hr).^2); 
    freqFilt = double(freqFilt <= cutoffFreq);
    
    % Construct the RGB output of the centered filter
    imageOut = zeros(size(DFT)); 
    for channel = 1:c 
        imageOut(:, :, channel) = DFT(:, :, channel) .* freqFilt; 
    end 
    
    % Centred filter on the spectrum
    filteredImage = abs(ifft2(ifftshift(imageOut)));

    % Normalize to the range [1, 256]
    filteredImage = uint8(256 * mat2gray(filteredImage));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
