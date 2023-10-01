# YOLOv8-and-DeepSORT-with-TensorRT-
## This project uses YOLOv8 under AGPL 3.0 License (for open source and academic projects) ##

### Project Overview ###
This project converts object detection (YOLOv8) with TensorRT and deploys converted models into NVIDIA Jetson Orin Nano. 

### Prerequisites ###

#### Platform ####
1. Device: NVIDIA Jetson Orin Nano Development Kit (8GB) <br />
(https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
2. Machine: aarch64  <br />
3. System: Linux  <br />
4. Distribution: Ubuntu 20.04 focal  <br />
5. Release: 5.10.104-tegra  <br />
6. Jetpack: 5.1.1  <br />
7. Python: 3.8.10 (pre-installed by Jetpack)  <br />

#### Libraries ####
1. CUDA: 11.4.315 (pre-installed by Jetpack) <br />
2. cuDNN: 8.6.0.166 (pre-installed by Jetpack)  <br />
3. TensorRT: 8.5.2.2 (pre-installed by Jetpack) <br />
4. Torch: torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64  <br />
5. Torchvision: 0.15.2 <br />
6. Ultralytics: 8.0.184 <br />
7. Onnxruntime: 1.16.0 <br />
8. PyCUDA: xxx  <br />


### Convert YOLOv8 from Pytorch model to TensorRT model ###

Export pre-trained YOLOv8 from Pytorch model to TensorRT model (fp16 precision) using 3-4 lines of code <br />
See more details in yolov8_export.py 

### Convert DeepSORT's ReID from Pytorch model to TensorRT model ###

### Time to Inference !!! ###
Run main.py to inference
