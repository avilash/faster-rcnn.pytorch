# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

from flask import Flask
from flask import request
import json

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def runDetection(class_name, im, vis=False):
  max_per_image = 100
  thresh = 0.05
  detect_thresh = 0.5
  cls_dets = None

  im_in = im
  im = im_in[:,:,::-1]
  blobs, im_scales = _get_image_blob(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)

  im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
  im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
  gt_boxes.data.resize_(1, 1, 5).zero_()
  num_boxes.data.resize_(1).zero_()

  rois, cls_prob, bbox_pred, \
  rpn_loss_cls, rpn_loss_box, \
  RCNN_loss_cls, RCNN_loss_bbox, \
  rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

  scores = cls_prob.data
  boxes = rois.data[:, :, 1:5]

  if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred.data
      if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # Optionally normalize targets by a precomputed mean and stdev
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

      pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
      pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
  else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  pred_boxes /= im_scales[0]
  scores = scores.squeeze()
  pred_boxes = pred_boxes.squeeze()

  if vis:
    im2show = np.copy(im)
  for j in xrange(1, len(pascal_classes)):
    if class_name not in pascal_classes:
        continue
    else:
      cls_ind = np.where(pascal_classes==class_name)
      if cls_ind != j:
        pass
    inds = torch.nonzero(scores[:,j]>thresh).view(-1)
    # if there is det
    if inds.numel() > 0:
      cls_scores = scores[:,j][inds]
      _, order = torch.sort(cls_scores, 0, True)
      cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
      cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
      # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
      cls_dets = cls_dets[order]
      # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
      keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
      cls_dets = cls_dets[keep.view(-1).long()]
      if vis:
        im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), detect_thresh)

  if vis:
    cv2.imwrite("result.jpg", im2show)

  final_class_dets = []
  if cls_dets is not None:
    cls_dets = cls_dets.cpu().numpy()
    for det in cls_dets:
      if det[4] > detect_thresh:
        bbox = det[:4]
        score = det[4]
        bbox_obj = {'xmin' : int(bbox[0]), 'ymin' : int(bbox[1]), 'xmax' : int(bbox[2]), 'ymax' : int(bbox[3]), 'cls': class_name}
        final_class_dets.append(bbox_obj)

  return final_class_dets

cfg_from_file("prod.yml")

cfg.USE_GPU_NMS = True
cfg.CUDA = True

print('Using config:')
pprint.pprint(cfg)
np.random.seed(cfg.RNG_SEED)

# Classes
pascal_classes = np.asarray(['__background__',
                     # 'aeroplane', 'bicycle', 'bird', 'boat',
                     # 'bottle', 'bus', 'car', 'cat', 'chair',
                     # 'cow', 'diningtable', 'dog', 'horse',
                     # 'motorbike', 'person', 'pottedplant',
                     # 'sheep', 'sofa', 'train', 'tvmonitor'])
                     '1'])

# Network Init
fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
fasterRCNN.create_architecture()
fasterRCNN.cuda()
fasterRCNN.eval()

# Load weights
load_name = 'models/res101/pascal_voc/faster_rcnn_1_8_215.pth'
print("load checkpoint %s" % (load_name))
checkpoint = torch.load(load_name)
fasterRCNN.load_state_dict(checkpoint['model'])
if 'pooling_mode' in checkpoint.keys():
  cfg.POOLING_MODE = checkpoint['pooling_mode']
print('load model successfully!')
print("load checkpoint %s" % (load_name))

# Initilize the tensor holder here.
im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

im_data = im_data.cuda()
im_info = im_info.cuda()
num_boxes = num_boxes.cuda()
gt_boxes = gt_boxes.cuda()
  
im_data = Variable(im_data, volatile=True)
im_info = Variable(im_info, volatile=True)
num_boxes = Variable(num_boxes, volatile=True)
gt_boxes = Variable(gt_boxes, volatile=True)

print ("Warming up")
im = cv2.imread("test.jpg")
result = runDetection("1", im, vis=True)
print (result)

app = Flask(__name__)

@app.route("/object_detect_torch", methods=['POST'])
def detect():
  class_name = request.args.get("class_name")
  im_file = request.files['file']
  print("Detecting object - ", class_name, ", in image ", im_file.filename)
  im = cv2.imdecode(np.fromstring(im_file.read(), np.uint8), cv2.IMREAD_COLOR)
  result = runDetection(str(class_name), im, vis=True)
  return json.dumps(result), 200


if __name__ == "__main__":
  print("Running http server ... ")
  app.run(host='0.0.0.0', port=9191)
