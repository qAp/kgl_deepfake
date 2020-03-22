import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Pytorch_Retinaface.models.retinaface import RetinaFace
from Pytorch_Retinaface.data import cfg_mnet, cfg_re50
from Pytorch_Retinaface.test_widerface import load_model
from Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms


class EasyRetinaFace:

    def __init__(self, path='Pytorch_Retinaface/weights/Resnet50_Final.pth'):
        self.cfg = cfg_re50
        self.cfg['pretrain'] = False  # Don't download pretrained ImageNet weights
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        # Download weights from Google Drive. See repo for details: https://github.com/biubug6/Pytorch_Retinaface
        self.net = load_model(self.net, path, load_to_cpu=False)
        self.net = self.net.eval()
        self.net = self.net.cuda()

    def detect_on_multiple_frames(self, frames):
        """
        Get detections from multiple frames.

        NOTE: This does not run in parallel, it simply calls self.detect() sequentially
        This is because with the default image size, we can't predict more than 2 images at a time. :(
        """

        detections = []
        for frame in frames:
            current_detections = self.detect(frame)
            detections.append(current_detections)

        return detections

    def detect(self, frame):
        """
        Get the detections for a single frame
        """
        img = np.float32(frame)
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)

        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        # Only resize if the image is bigger than we can handle
        resize = min(1.0, resize)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        scale = scale.cuda()

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.cuda()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.cuda()
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        DEFAULT_CONFIDENCE_THRESH = 0.9
        inds = np.where(scores > DEFAULT_CONFIDENCE_THRESH)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        DEFAULT_NMS_THRESH = 0.4
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, DEFAULT_NMS_THRESH)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        # Rescale dets by 120%
        x_min = dets[:, 0]
        y_min = dets[:, 1]
        x_max = dets[:, 2]
        y_max = dets[:, 3]

        center_y = (y_max + y_min) / 2
        center_x = (x_min + x_max) / 2

        width = x_max - x_min
        height = y_max - y_min

        height = height * 1.2   # increase height by 20%
        width = width * 1.2     # increase width by 20%

        y_min = center_y - (height / 2)
        y_max = center_y + (height / 2)

        x_min = center_x - (width / 2)
        x_max = center_x + (width / 2)

        dets[:, 0] = x_min
        dets[:, 1] = y_min
        dets[:, 2] = x_max
        dets[:, 3] = y_max

        return dets
