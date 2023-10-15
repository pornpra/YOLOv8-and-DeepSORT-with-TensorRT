
import os
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime as onnxrt

from track import *

source = './input.mp4'
video_out_path = './output.mp4'
cap = cv2.VideoCapture(source)
ret, frame = cap.read()
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('F','M','P','4'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

class_mapping = {}
class_mapping[0] = 'Person'

deepsort = init_tracker()
        
results = object_detection.predict(source, stream=True, device=device, classes=classes, verbose=False, stream_buffer=True)

for pred in results:

  frame = pred.orig_img
  det = pred.boxes.data

  if det is not None and len(det):
    
    xywh_bboxs, confs, oids, track_outputs = [], [], [], []

    xywh_bboxs = [list(xyxy_to_xywh(*xyxy)) for xyxy in det[:, :4]]
    confs = [[conf.item()] for conf in det[:, -2]]
    oids = [int(cls) for cls in det[:, -1]]

    xywhs = torch.Tensor(xywh_bboxs)
    confss = torch.Tensor(confs)

    # update the tracker with the new detections
    track_outputs = deepsort.update(xywhs, confss, oids, frame)

    # loop over the tracks
    if len(track_outputs) > 0:

      for track in track_outputs:

        bbox_xyxy, object_id, object_class = track[0:4], track[-2], track[-1]
        xmin, ymin, xmax, ymax = int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3])
        w, h = xmax - xmin, ymax - ymin

        colors = sns.color_palette('colorblind')
        color = colors[object_id%10]
        color = tuple([x * 255 for x in color])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(frame, (xmin-10, ymin-25), (xmax+10, ymin), color, -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'ID: ' + str(object_id) + '   {}'.format(class_mapping[int(object_class)])
        textsize = cv2.getTextSize(text, font, 0.5, 2)[0]
        textX = ((xmax - xmin + 20) - textsize[0]) // 2
        textY = (25 + textsize[1]) // 2
                        
        cv2.putText(frame, text, (xmin - 10 + textX, ymin - 25 + textY), font, 0.5, (255,255,255), 2)
  
  else:
    deepsort.increment_ages()
  
  cap_out.write(frame)

cap_out.release()
    



 