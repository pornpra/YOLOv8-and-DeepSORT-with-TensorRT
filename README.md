# YOLOv8-and-DeepSORT-with-TensorRT


### Project Overview :rocket: ###
The objective of this project is to convert object detection model (YOLOv8) and object tracking model (DeepSORT) into the TensorRT format and inference converted models by using the NVIDIA Jetson Orin Nano. The results from object detection (bounding box) and object tracking (object id) are saved into mp4 file for visualization. Good luck :blush:


### Prerequisites :star: ###

#### Platform ####
1. Device: NVIDIA Jetson Orin Nano Development Kit (8GB) <br />
[Jetson Orin Nano Developer Kit Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
2. Jetpack: 5.1.1  <br />
3. Python: 3.8.10 (pre-installed by Jetpack) <br />

#### Libraries ####
1. CUDA: 11.4.315 (pre-installed by Jetpack) <br />
2. cuDNN: 8.6.0.166 (pre-installed by Jetpack)  <br />
3. TensorRT: 8.5.2.2 (pre-installed by Jetpack) <br />
4. Torch: torch-2.0.0+nv23.05 <br />
5. Torchvision: 0.15.2a0+fa99a53 <br />
6. Ultralytics: 8.0.190 <br />


### Steps to Run Code :computer: ###

* Please make sure your platform and libraries follow prerequisites :white_check_mark:
  
* Clone the repository
```
git clone https://github.com/pornpra/YOLOv8-and-DeepSORT-with-TensorRT.git
```

* Goto the cloned folder

```
cd YOLOv8-and-DeepSORT-with-TensorRT
```

* Convert YOLOv8 from Pytorch model to TensorRT model (fp16 precision)

```
python3 yolov8s_torch_to_engine.py
```

Don't forget to check and rename converted model to yolov8s_fp16.engine :exclamation:

* Convert DeepSORT's ReID from Pytorch model to TensorRT model
1. Download DeepSORT files (including reid.pt, reid.onnx and reid_fp16.trt) from Google Drive. After downloading, unzip it and move the deep_sort_tensorrt folder under YOLOv8-and-DeepSORT-with-TensorRT folder <br />

```
https://drive.google.com/drive/folders/10hXfbdwDXn7AF4NG-gHWDoYpTNrfZ2XO?usp=sharing
```

2. Convert DeepSORT's ReID from Pytorch model to ONNX model (dynamic batch but static width and height) <br />

```
python3 reid_torch_to_onnx.py
```

3. Convert DeepSORT's ReID from ONNX model to TensorRT model (fp16 precision and dynamic batch) using trtexec command <br />
3.1 cd to your ONNX model directory <br />
3.2 Run this command: <br />

```
/usr/src/tensorrt/bin/trtexec --onnx=reid.onnx --saveEngine=reid_fp16.trt --minShapes=input:1x3x128x64 --optShapes=input:5x3x128x64 --maxShapes=input:30x3x128x64 --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
```

* Inference 

```
python3 main.py
```
![me](https://github.com/pornpra/YOLOv8-and-DeepSORT-with-TensorRT/blob/main/output.gif)

References
1. [Ultralytics](https://docs.ultralytics.com/) 
2. [YOLOv8-DeepSORT-Object-Tracking](https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking)
3. [Deploy YOLOv8 on NVIDIA Jetson using TensorRT](https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/)
