
import os
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime as onnxrt

from track import *

source = './test_video.mp4'

deepsort = init_tracker()
        
results = vehicle_detection.predict(source, stream=True, device=device, verbose=False) 

for pred in results:

    frame = pred.orig_img
    offset = int(frame.shape[0]*0.25)
    det = pred.boxes.data

    if det is not None and len(det):
        xywh_bboxs = []
        confs = []
        oids = []
        track_outputs = []
        
        for *xyxy, conf, cls in det:
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # update the tracker with the new detections
        track_outputs = deepsort.update(xywhs, confss, oids, frame)

        # loop over the tracks
        if len(track_outputs) > 0:
            for track in track_outputs:
                bbox_xyxy = track[0:4]
                object_id = track[-2]
                object_class = track[-1]

                xmin, ymin, xmax, ymax = int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3])
                w = xmax - xmin
                h = ymax - ymin

                ### this end of object tracking with yolov8 and deepsort part ###
                ### we can extract bounding box and object id ###
                ### next step is depend on your application !! ###

    else:
        deepsort.increment_ages()



 