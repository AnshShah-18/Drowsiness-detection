function downloadMRLEyeDataset(maxImagesPerClass)
%DOWNLOADMRLEYEDATASET Download and prepare the public MRL Eye Dataset.
%
% Usage:
%   downloadMRLEyeDataset
%   downloadMRLEyeDataset(5000)
%
% The MRL Eye Dataset encodes eye state in each filename. The fifth
% underscore-separated token is the eye state: 0 = closed, 1 = open.

arguments
    maxImagesPerClass (1,1) double {mustBePositive, mustBeInteger} = 6000
end

projectRoot = fullfile(fileparts(mfilename("fullpath")), "..");
rawDir = fullfile(projectRoot, "raw");
zipPath = fullfile(rawDir, "mrlEyes_2018_01.zip");
extractDir = fullfile(rawDir, "mrlEyes_2018_01");
dataDir = fullfile(projectRoot, "data");
openDir = fullfile(dataDir, "open");
closedDir = fullfile(dataDir, "closed");

datasetUrl = "https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip";

ensureDir(rawDir);
ensureDir(extractDir);
ensureDir(openDir);
ensureDir(closedDir);

if ~isfile(zipPath)
    fprintf("Downloading MRL Eye Dataset from:\n%s\n", datasetUrl);
    fprintf("This archive is about 326 MB and may take a few minutes.\n");
    websave(zipPath, datasetUrl);
else
    fprintf("Using existing archive: %s\n", zipPath);
end

if isempty(dir(fullfile(extractDir, "**", "*.png")))
    fprintf("Extracting dataset to: %s\n", extractDir);
    unzip(zipPath, extractDir);
else
    fprintf("Using existing extracted dataset: %s\n", extractDir);
end

imageFiles = dir(fullfile(extractDir, "**", "*.png"));
if isempty(imageFiles)
    error("No PNG images found after extraction. Check the downloaded archive.");
end

openCount = 0;
closedCount = 0;

fprintf("Preparing up to %d images per class...\n", maxImagesPerClass);

for index = 1:numel(imageFiles)
    sourcePath = fullfile(imageFiles(index).folder, imageFiles(index).name);
    eyeState = eyeStateFromFilename(imageFiles(index).name);

    if eyeState == "open"
        if openCount >= maxImagesPerClass
            continue;
        end
        openCount = openCount + 1;
        destinationPath = fullfile(openDir, sprintf("mrl_open_%05d.png", openCount));
    elseif eyeState == "closed"
        if closedCount >= maxImagesPerClass
            continue;
        end
        closedCount = closedCount + 1;
        destinationPath = fullfile(closedDir, sprintf("mrl_closed_%05d.png", closedCount));
    else
        continue;
    end

    if ~isfile(destinationPath)
        image = imread(sourcePath);
        image = im2gray(image);
        image = imresize(image, [64 128]);
        imwrite(image, destinationPath);
    end

    if openCount >= maxImagesPerClass && closedCount >= maxImagesPerClass
        break;
    end
end

fprintf("Prepared %d open-eye images and %d closed-eye images.\n", openCount, closedCount);
fprintf("Dataset is ready for trainEyeStateCNN.\n");
end

function ensureDir(path)
if ~exist(path, "dir")
    mkdir(path);
end
end

function eyeState = eyeStateFromFilename(filename)
[~, name] = fileparts(filename);
tokens = split(string(name), "_");

if numel(tokens) < 5
    eyeState = "unknown";
    return;
end

stateToken = tokens(5);
if stateToken == "1"
    eyeState = "open";
elseif stateToken == "0"
    eyeState = "closed";
else
    eyeState = "unknown";
end
end
