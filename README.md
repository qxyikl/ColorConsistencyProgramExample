# ColorConsistencyProgramExample

## Introduction

This project is developed for optimizing the color consistency between images. We establish a color mapping model between overlapping areas of images based on convolutional neural networks, and apply this color mapping model to the entire image of the target image to achieve color consistency.

Here is an example for demonstration below:

<img src="example.png" width="900px"/>

## Usage

The sample datasets can be downloaded from https://pan.baidu.com/s/1TiUMhyIEcrypE7y7gQnFWQ?pwd=rBiy (Access code: rBiy) 

### 1. Sample datasets:
The training data is the patches of the overlapping area of the two images, which is stored in the folder 'datasets/trainingdata'. The input folder contains the target image data, and the label folder contains the reference image data.
The prediction data is the patches of the whole target image, which is stored in the folder 'datasets/predictingdata/predict'.
The folder 'datasets/predictingdata/result' is used to store the results after color consistency processing.

### 2. Project Configure:
This procedure is developed on python 3.6 (conda 4.10) under Window 10 system.

config.json: enter the training data (xxx/dataset/trainingdata), epochs, model save path (xxx/xxx.pkl), the prediction data (xxx/dataset/predictingdata) and color mapping model (xxx/xxx.pkl).

After entering the corresponding attributes in the config.json, run the train&eval.py. You can see the color consistency results in xxx/dataset/predictingdata/result.
