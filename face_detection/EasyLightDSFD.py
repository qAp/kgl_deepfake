import torch

from lightDSFD.data import config, TestBaseTransform
from lightDSFD.light_face_ssd import build_ssd
from lightDSFD.test import infer


class EasyLightDSFD:

    def __init__(self, path="lightDSFD/weights/light_DSFD.pth"):
        cfg = config.widerface_640
        WIDERFace_CLASSES = ['face']  # We're only finding faces
        num_classes = len(WIDERFace_CLASSES) + 1  # +1 background

        self.net = build_ssd('test', cfg['min_dim'], num_classes)  # initialize SSD
        self.net.load_state_dict(torch.load(path))
        self.net = self.net.cuda()
        self.net = self.net.eval()

        # evaluation
        self.transform = TestBaseTransform((104, 117, 123))
        self.thresh = cfg['conf_thresh']


    def detect(self, frame):
        detections = infer(self.net, frame, self.transform, self.thresh, cuda=True, shrink=1)
        return detections

