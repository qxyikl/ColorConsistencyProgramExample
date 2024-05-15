# ColorConsistencyProgramExample

## Introduction

This project is developed for optimizing the color consistency between images. We establish a color mapping model between overlapping areas of images based on convolutional neural networks, and apply this color mapping model to the entire image of the target image to achieve color consistency.

Here is an example for demonstration below:

<img src="example.png" width="900px"/>

## Usage

The procedure and sample datasets can be downloaded from https://pan.baidu.com/s/11tplAx3WaHbkBvmreP85IA?pwd=20zj (Access code: 20zj) 

### 1. Sample datasets:
The training data is the patches of the overlapping area of the two images, which is stored in the folder 'datasets/trainingdata'. The input folder contains the target image data, and the label folder contains the reference image data.
The prediction data is the patches of the whole target image, which is stored in the folder 'datasets/predictingdata/predict'.
The folder 'datasets/predictingdata/result' is used to store the results after color consistency processing.

### 2. Project Configure:
This procedure is developed on python 3.6 (conda 4.10) under Window 10 system.
The program interface is as follows：

<img src="Program interface diagram.png" width="900px"/>

After entering the training data, epochs, and model save path, you can start training the model.

After inputting the prediction data and color mapping model, you can start predicting the data mapped by the model.
