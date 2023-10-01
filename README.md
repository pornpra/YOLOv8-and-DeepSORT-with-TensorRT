# YOLOv8-and-DeepSORT-with-TensorRT
## This project uses YOLOv8 under AGPL 3.0 License (for open source and academic projects) ##

### Project Overview ###
This project converts object detection (YOLOv8) and object tracking (DeepSORT) into TensorRT format and inferences converted models by NVIDIA Jetson Orin Nano. The objective of this project are to convert YOLOv8 and DeepSORT with TensorRT and test with edge device only (no object counting and other applications in main.py). The exact outputs from main.py are bouding box from object detection and object id from object tracking. After that it's depend on your applications based on these extracted infomration. Good luck :blush:


### :star: Prerequisites :star: ###

#### Platform ####
1. Device: NVIDIA Jetson Orin Nano Development Kit (8GB) <br />
(https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
2. Machine: aarch64  <br />
3. System: Linux  <br />
4. Distribution: Ubuntu 20.04 focal  <br />
5. Release: 5.10.104-tegra  <br />
6. Jetpack: 5.1.1  <br />
7. Python: 3.8.10 (pre-installed by Jetpack) <br />

#### Libraries ####
1. CUDA: 11.4.315 (pre-installed by Jetpack) <br />
2. cuDNN: 8.6.0.166 (pre-installed by Jetpack)  <br />
3. TensorRT: 8.5.2.2 (pre-installed by Jetpack) <br />
4. Torch: torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64  <br />
5. Torchvision: 0.15.2 <br />
6. Ultralytics: 8.0.184 <br />
7. ONNX: 1.14.1 <br />
8. ONNXruntime: 1.16.0 <br />
9. PyCUDA: 2022.2.2  <br />
10. Numpy: 1.22.2  <br />
11. Plotly: xxxx  <br />
12. Easydict: xxxx  <br />

### Steps to Run Code ###

* Please make sure your platform and libraries follow prerequisites !!! :white_check_mark:
  
* Clone the repository
```
git clone https://github.com/pornpra/YOLOv8-and-DeepSORT-with-TensorRT.git
```

* Goto the cloned folder

```
cd YOLOv8-and-DeepSORT-with-TensorRT
```
    
* Convert YOLOv8 from Pytorch model to TensorRT model 

    * Export pre-trained YOLOv8 from Pytorch model to TensorRT model (fp16 precision) using 3-4 lines of code <br />
    * See more details in yolov8s_torch_to_engine.py

* Convert DeepSORT's ReID from Pytorch model to TensorRT model
1. Download DeepSORT files (including reid.pt, reid.onnx and reid_fp16.trt) from Google Drive. After downloading, unzip it and move the deep_sort_tensorrt folder under YOLOv8-and-DeepSORT-with-TensorRT folder. <br />

```
https://drive.google.com/drive/folders/10hXfbdwDXn7AF4NG-gHWDoYpTNrfZ2XO?usp=sharing
```

2. Convert DeepSORT's ReID from Pytorch model to ONNX model (dynamic batch but static width and height) using reid_torch_to_onnx.py <br />
3. Convert DeepSORT's ReID from ONNX model to TensorRT model (fp16 precision and dynamic batch) using trtexec command <br />
3.1 cd to your ONNX model directory <br />
3.2 Run this command: <br />

```
/usr/src/tensorrt/bin/trtexec --onnx=reid.onnx --saveEngine=reid_fp16.trt --minShapes=input:1x3x128x64 --optShapes=input:5x3x128x64 --maxShapes=input:30x3x128x64 --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
```

* Time to Inference !!! 
  * Run main.py to inference (test with test_video.mp4)
