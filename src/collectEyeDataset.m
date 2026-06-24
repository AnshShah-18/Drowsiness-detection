function collectEyeDataset(labelName, sampleCount)
%COLLECTEYEDATASET Capture eye crops from a webcam for CNN training.
%
% Usage:
%   collectEyeDataset("open", 200)
%   collectEyeDataset("closed", 200)

arguments
    labelName (1,1) string {mustBeMember(labelName, ["open", "closed"])}
    sampleCount (1,1) double {mustBePositive, mustBeInteger} = 200
end

outputDir = fullfile(fileparts(mfilename("fullpath")), "..", "data", labelName);
if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

cam = webcam;
faceDetector = vision.CascadeObjectDetector("FrontalFaceCART");
eyeDetector = vision.CascadeObjectDetector("EyePairBig");

previewFigure = figure( ...
    "Name", "Eye Dataset Collection", ...
    "NumberTitle", "off", ...
    "Color", "w");

fprintf("Collecting %d '%s' eye samples. Press Ctrl+C to stop early.\n", sampleCount, labelName);
pause(2);

captured = 0;
while captured < sampleCount && ishandle(previewFigure)
    frame = snapshot(cam);
    grayFrame = im2gray(frame);

    faceBox = largestBox(faceDetector(grayFrame));
    displayFrame = frame;

    if ~isempty(faceBox)
        faceImage = imcrop(grayFrame, faceBox);
        eyeBox = largestBox(eyeDetector(faceImage));

        if ~isempty(eyeBox)
            eyeImage = imcrop(faceImage, eyeBox);
            eyeImage = imresize(eyeImage, [64 128]);

            captured = captured + 1;
            filename = sprintf("%s_%04d.png", labelName, captured);
            imwrite(eyeImage, fullfile(outputDir, filename));

            fullEyeBox = eyeBox;
            fullEyeBox(1:2) = fullEyeBox(1:2) + faceBox(1:2);
            displayFrame = insertShape(displayFrame, "Rectangle", faceBox, "Color", "green", "LineWidth", 3);
            displayFrame = insertShape(displayFrame, "Rectangle", fullEyeBox, "Color", "yellow", "LineWidth", 3);
        end
    end

    displayFrame = insertText(displayFrame, [12 12], ...
        sprintf("%s samples: %d / %d", labelName, captured, sampleCount), ...
        "FontSize", 18, "BoxOpacity", 0.7);
    imshow(displayFrame);
    drawnow;
end

clear cam;
fprintf("Saved %d samples to %s\n", captured, outputDir);
end

function box = largestBox(boxes)
if isempty(boxes)
    box = [];
    return;
end

areas = boxes(:, 3) .* boxes(:, 4);
[~, index] = max(areas);
box = boxes(index, :);
end
