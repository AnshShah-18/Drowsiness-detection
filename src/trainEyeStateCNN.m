function trainEyeStateCNN()
%TRAINEYESTATECNN Train a CNN to classify open and closed eye images.

projectRoot = fullfile(fileparts(mfilename("fullpath")), "..");
dataDir = fullfile(projectRoot, "data");
modelDir = fullfile(projectRoot, "models");
modelPath = fullfile(modelDir, "eyeStateCNN.mat");

if ~exist(modelDir, "dir")
    mkdir(modelDir);
end

imageSize = [64 128 1];
dataset = imageDatastore(dataDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames", ...
    "ReadFcn", @(filename) preprocessEyeImage(filename, imageSize));

if numel(dataset.Files) < 20
    error("Not enough training images. Collect images with collectEyeDataset first.");
end

labelCounts = countEachLabel(dataset);
disp(labelCounts);

[trainData, validationData] = splitEachLabel(dataset, 0.8, "randomized");

layers = [
    imageInputLayer(imageSize, "Name", "input", "Normalization", "none")

    convolution2dLayer(3, 16, "Padding", "same", "Name", "conv_1")
    batchNormalizationLayer("Name", "bn_1")
    reluLayer("Name", "relu_1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_1")

    convolution2dLayer(3, 32, "Padding", "same", "Name", "conv_2")
    batchNormalizationLayer("Name", "bn_2")
    reluLayer("Name", "relu_2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_2")

    convolution2dLayer(3, 64, "Padding", "same", "Name", "conv_3")
    batchNormalizationLayer("Name", "bn_3")
    reluLayer("Name", "relu_3")
    dropoutLayer(0.25, "Name", "dropout")

    fullyConnectedLayer(2, "Name", "fc")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classoutput")
];

options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-4, ...
    "MaxEpochs", 12, ...
    "MiniBatchSize", 32, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", validationData, ...
    "ValidationFrequency", 10, ...
    "Verbose", true, ...
    "Plots", "training-progress");

net = trainNetwork(trainData, layers, options);
classNames = string(net.Layers(end).Classes);

predictedLabels = classify(net, validationData);
actualLabels = validationData.Labels;
validationAccuracy = mean(predictedLabels == actualLabels);

save(modelPath, "net", "imageSize", "classNames", "validationAccuracy");
fprintf("Saved trained model to %s\n", modelPath);
fprintf("Validation accuracy: %.2f%%\n", validationAccuracy * 100);
end

function image = preprocessEyeImage(filename, imageSize)
image = imread(filename);
image = im2gray(image);
image = imresize(image, imageSize(1:2));
image = im2single(image);
end
