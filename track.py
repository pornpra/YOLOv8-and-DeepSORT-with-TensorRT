import os
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import torch
import cv2
from PIL import Image
import onnxruntime as onnxrt

from deep_sort_tensorrt.utils.parser import get_config
from deep_sort_tensorrt.deep_sort import DeepSort
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


object_detection = YOLO('./yolov8s_fp16.engine', task='detect')


def init_tracker():

  deepsort = None

  cfg_deep = get_config()
  cfg_deep_path = './deep_sort_tensorrt/configs/deep_sort.yaml'
  cfg_deep.merge_from_file(cfg_deep_path)

  deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                      max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                      nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                      use_cuda=True)

  return deepsort

def xyxy_to_xywh(*xyxy):
  """" Calculates the relative bounding box from absolute pixel values. """
  bbox_left = min([xyxy[0].item(), xyxy[2].item()])
  bbox_top = min([xyxy[1].item(), xyxy[3].item()])
  bbox_w = abs(xyxy[0].item() - xyxy[2].item())
  bbox_h = abs(xyxy[1].item() - xyxy[3].item())
  x_c = (bbox_left + bbox_w / 2)
  y_c = (bbox_top + bbox_h / 2)
  w = bbox_w
  h = bbox_h
  return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
  tlwh_bboxs = []
  for i, box in enumerate(bbox_xyxy):
    x1, y1, x2, y2 = [int(i) for i in box]
    top = x1
    left = y1
    w = int(x2 - x1)
    h = int(y2 - y1)
    tlwh_obj = [top, left, w, h]
    tlwh_bboxs.append(tlwh_obj)
  return tlwh_bboxs