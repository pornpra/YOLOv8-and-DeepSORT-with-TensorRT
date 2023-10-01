import torch
import cv2
import onnxruntime as onnxrt
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load a custom trained
model_weight = './yolov8s.pt'
model = YOLO(model_weight)

# Export the model
model.export(format='engine', device=device, half=True)

## Don't forget to check and rename converted model to yolov8s_fp16.engine ##

# Inference with tensorrt

#input_img = './test_image.jpg'
#img = cv2.imread(input_img)
#pred = YOLO('./yolov8s_fp16.engine', task='detect').predict(img, device=device)
#res_plotted = pred[0].plot()
#cv2.imshow('res_plotted', res_plotted)
#pred[0].boxes.data


