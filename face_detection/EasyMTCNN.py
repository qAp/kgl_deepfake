import cv2
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from facenet_pytorch import MTCNN
from utils import read_frame


class EasyMTCNN:

    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(device=device, post_process=False)

    def detect(self, frame):
        img = PIL.Image.fromarray(frame)
        detections, probabilities = self.detector.detect(img)
        dets_with_probs = np.append(detections, np.expand_dims(probabilities, axis=1), axis=1)
        return dets_with_probs


if __name__ == '__main__':
    # Test code
    video_path = '../data/dfdc_train_part_2/abmjszfycr_REAL.mp4'
    frame = read_frame(video_path)

    easyMTCNN = EasyMTCNN()
    face = easyMTCNN.detect(frame)
    plt.imshow(face)
    plt.show()
