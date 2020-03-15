# Car Make and Model classification example with YOLOv3 object detector

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A C++ example for using [Spectrico's car make and model classifier](http://spectrico.com/car-make-model-recognition.html). It consists of object detector for finding the cars, and a classifier to recognize the makes and models of the detected cars. The object detector is an implementation of YOLOv3 (OpenCV DNN backend). It doesn't use GPU and one frame takes 1s to process on Intel Core i5-7600 CPU. YOLOv3 weights were downloaded from [YOLO website](https://pjreddie.com/darknet/yolo/). The classifier is based on Mobilenet v2 (OpenCV DNN backend). It takes 35 milliseconds on Intel Core i5-7600 CPU for single classification. The light version of the classifier is slightly less accuracy but is 4 times faster. It is optimized for speed and is recommended for edge devices. The demo doesn't include the classifier for car make and model recognition. It is a commercial product and is available for purchase at [http://spectrico.com/car-make-model-recognition.html](http://spectrico.com/car-make-model-recognition.html). A free version of the classifier with lower accuracy is available for download at [http://spectrico.com/spectrico-mmr-mobilenet-64x64-531A7126.zip](http://spectrico.com/spectrico-mmr-mobilenet-64x64-531A7126.zip).

![image](https://github.com/spectrico/car-make-model-classifier-yolo3-cpp/blob/master/car-make-model.png?raw=true)

---
## Object Detection and Classification in images
This example takes an image as input, detects the cars using YOLOv3 object detector, crops the car images, makes them square while keeping the aspect ratio, resizes them to the input size of the classifier, and recognizes the make and model of each car. The result is printed on the display.


#### Usage
```
$ car-make-model-classifier-yolo3-cpp cars.jpg
```
The output is printed to the console:
```
------------------------------------------------
Object box: [440 x 193 from (606, 144)]
Inference time, ms: 52.5417
Top 3 probabilities:
make: Volkswagen        model: Arteon   confidence: 94.5597 %
make: Volkswagen        model: Passat   confidence: 0.495846 %
make: Audi      model: A7       confidence: 0.426329 %
------------------------------------------------

------------------------------------------------
Object box: [318 x 158 from (958, 157)]
Inference time, ms: 35.1407
Top 3 probabilities:
make: Volkswagen        model: Polo     confidence: 89.0573 %
make: Volkswagen        model: T-ROC    confidence: 5.95808 %
make: Volkswagen        model: Arteon   confidence: 1.97434 %
------------------------------------------------

------------------------------------------------
Object box: [277 x 176 from (362, 137)]
Inference time, ms: 34.6848
Top 3 probabilities:
make: Volkswagen        model: Tiguan   confidence: 95.8008 %
make: Volkswagen        model: Passat   confidence: 0.732364 %
make: Volkswagen        model: Polo     confidence: 0.701915 %
------------------------------------------------
```

---
## Requirements
  - C++ compiler
  - OpenCV
  - yolov3.weights must be downloaded from [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and saved into the project folder

---
## Credits
The YOLOv3 object detector is from: [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
The make and model classifier is based on MobileNetV2 mobile architecture: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


