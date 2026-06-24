# Real-Time Driver Drowsiness Detection using MATLAB

This project detects driver drowsiness from a live webcam feed by locating the face and eyes, classifying eye state with a CNN, and triggering an alert when the eyes remain closed for too many consecutive frames.

## Requirements

- MATLAB R2021a or newer
- Computer Vision Toolbox
- Deep Learning Toolbox
- Image Acquisition Toolbox Support Package for OS Generic Video Interface

Optional but recommended:

- A webcam with decent lighting
- A few hundred eye crops per class for better CNN accuracy

## Project Structure

```text
matlab-drowsiness-detection/
  README.md
  src/
    collectEyeDataset.m
    downloadMRLEyeDataset.m
    trainEyeStateCNN.m
    runDrowsinessDetector.m
  data/
    open/
    closed/
  models/
```

## Quick Start with Public Dataset

The fastest public-dataset workflow uses the MRL Eye Dataset. The project includes a script that downloads the public ZIP archive, extracts it, and converts the images into the expected `data/open` and `data/closed` folders.

1. Open MATLAB in this folder.
2. Add the source folder to the MATLAB path:

```matlab
addpath("src")
```

3. Download and prepare a balanced MRL subset:

```matlab
downloadMRLEyeDataset(6000)
```

4. Train the CNN:

```matlab
trainEyeStateCNN
```

5. Start real-time detection:

```matlab
runDrowsinessDetector
```

The MRL archive is about 326 MB, so the first download can take a few minutes.

## Quick Start with Your Webcam Dataset

1. Open MATLAB in this folder.
2. Add the source folder to the MATLAB path:

```matlab
addpath("src")
```

3. Collect training images:

```matlab
collectEyeDataset("open", 200)
collectEyeDataset("closed", 200)
```

Keep your eyes open for the first command and closed for the second command.

4. Train the CNN:

```matlab
trainEyeStateCNN
```

5. Start real-time detection:

```matlab
runDrowsinessDetector
```

## How It Works

- `collectEyeDataset.m` captures eye crops from the webcam and saves them to `data/open` or `data/closed`.
- `downloadMRLEyeDataset.m` downloads and prepares the public MRL Eye Dataset.
- `trainEyeStateCNN.m` trains a small CNN to classify eye images as open or closed.
- `runDrowsinessDetector.m` detects the face and eye region in real time, classifies eye state, and plays an alert when the closed-eye count crosses a threshold.

## Public Dataset

This project supports the MRL Eye Dataset:

- URL: `https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip`
- Size: about 326 MB
- Filename label convention: the fifth underscore-separated filename token is eye state, where `0` means closed and `1` means open.

## Tuning

In `runDrowsinessDetector.m`, adjust these values if needed:

- `closedFrameLimit`: number of consecutive closed-eye frames before alerting.
- `predictionThreshold`: confidence required before a frame is counted as closed.
- `inputSize`: must match the trained CNN input size.

Better lighting and a camera facing the driver directly will improve detection quality.
