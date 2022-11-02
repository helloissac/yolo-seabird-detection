# YOLOv3 Model for Seabird Detection and Counting

## Installation
- Install requirements and download pretrained weights in Terminal
```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```
## Required libraries
- numpy>=1.16.0
- pillow==6.2.0
- scipy==1.1.0
- wget==3.2
- seaborn==0.9.0
- easydict==1.9
- grpcio>=1.24.3
- tensorflow==2.0.0.

## Image Dataset
- The images should be stored in the `/docs` directory.
- Change the image path before running imagedetection
- Open `image_demo.py` and change the image path using `image_path   = "./docs/DSC03040.jpg"`
- Due to copyright issues, only sample images were included.

## YOLO Object Detection
In this part, we will use pretrained weights to make predictions on seabird images.
- The neural network for the  YOLOv3 model is implemented in `yolov3.py`
```
$ python image_demo.py
```
## Sample Output
- The sample output shows the predicted bounding boxes. 
- Further statistical analysis would provide deeper insights into the classification accuracy of object detection.
<img src="https://github.com/issacjohannli/yolo-seabird-detection/blob/main/docs/sample_output.png">
Figure 1: Sample images with predicted bounding boxes.
