function runDrowsinessDetector()
%RUNDROWSINESSDETECTOR Run live webcam drowsiness detection.

projectRoot = fullfile(fileparts(mfilename("fullpath")), "..");
modelPath = fullfile(projectRoot, "models", "eyeStateCNN.mat");

if ~isfile(modelPath)
    error("Model not found. Train it first with trainEyeStateCNN.");
end

loadedModel = load(modelPath, "net", "imageSize", "classNames");
net = loadedModel.net;
imageSize = loadedModel.imageSize;
classNames = loadedModel.classNames;

closedFrameLimit = 18;
predictionThreshold = 0.70;
closedFrameCount = 0;
lastAlertTime = datetime("now") - seconds(5);

cam = webcam;
faceDetector = vision.CascadeObjectDetector("FrontalFaceCART");
eyeDetector = vision.CascadeObjectDetector("EyePairBig");

appFigure = figure( ...
    "Name", "Real-Time Driver Drowsiness Detection", ...
    "NumberTitle", "off", ...
    "Color", "w", ...
    "KeyPressFcn", @(~, event) setappdata(gcf, "stopRequested", strcmp(event.Key, "escape")));
setappdata(appFigure, "stopRequested", false);

fprintf("Drowsiness detector running. Press Escape or close the window to stop.\n");

while ishandle(appFigure) && ~getappdata(appFigure, "stopRequested")
    frame = snapshot(cam);
    grayFrame = im2gray(frame);
    faceBox = largestBox(faceDetector(grayFrame));

    labelText = "Face not found";
    labelColor = "red";
    displayFrame = frame;

    if ~isempty(faceBox)
        faceImage = imcrop(grayFrame, faceBox);
        eyeBox = largestBox(eyeDetector(faceImage));
        displayFrame = insertShape(displayFrame, "Rectangle", faceBox, "Color", "green", "LineWidth", 3);

        if ~isempty(eyeBox)
            eyeImage = imcrop(faceImage, eyeBox);
            eyeImage = imresize(eyeImage, imageSize(1:2));
            eyeImage = im2single(eyeImage);

            [predictedLabel, scores] = classify(net, eyeImage);
            scoreTable = table(classNames(:), scores(:), ...
                "VariableNames", ["Class", "Score"]);
            closedScore = scoreForLabel(scoreTable, "closed");

            if predictedLabel == "closed" && closedScore >= predictionThreshold
                closedFrameCount = closedFrameCount + 1;
            else
                closedFrameCount = max(0, closedFrameCount - 1);
            end

            fullEyeBox = eyeBox;
            fullEyeBox(1:2) = fullEyeBox(1:2) + faceBox(1:2);
            displayFrame = insertShape(displayFrame, "Rectangle", fullEyeBox, "Color", "yellow", "LineWidth", 3);

            if closedFrameCount >= closedFrameLimit
                labelText = sprintf("DROWSINESS ALERT  closed frames: %d", closedFrameCount);
                labelColor = "red";
                if seconds(datetime("now") - lastAlertTime) > 1.5
                    beep;
                    lastAlertTime = datetime("now");
                end
            else
                labelText = sprintf("Eyes: %s  closed score: %.2f  count: %d", ...
                    string(predictedLabel), closedScore, closedFrameCount);
                labelColor = "green";
            end
        else
            labelText = "Eyes not found";
            labelColor = "yellow";
            closedFrameCount = max(0, closedFrameCount - 1);
        end
    else
        closedFrameCount = max(0, closedFrameCount - 1);
    end

    displayFrame = insertText(displayFrame, [12 12], labelText, ...
        "FontSize", 18, ...
        "TextColor", "white", ...
        "BoxColor", labelColor, ...
        "BoxOpacity", 0.75);
    displayFrame = insertText(displayFrame, [12 52], "Press Esc to stop", ...
        "FontSize", 14, ...
        "TextColor", "white", ...
        "BoxColor", "black", ...
        "BoxOpacity", 0.55);

    imshow(displayFrame);
    drawnow;
end

clear cam;
if ishandle(appFigure)
    close(appFigure);
end
fprintf("Drowsiness detector stopped.\n");
end

function score = scoreForLabel(scoreTable, labelName)
row = string(scoreTable.Class) == labelName;
if any(row)
    score = scoreTable.Score(row);
else
    score = 0;
end
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
