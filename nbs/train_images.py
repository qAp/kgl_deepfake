from fastai.script import *
from fastai.vision import *
from fastai.core import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.callbacks.oversampling import OverSamplingCallback
from kgl_deepfake.nb_01b import * 
from kgl_deepfake.nb_07 import *
from kgl_deepfake.nb_06 import *
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

import albumentations as A

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def tensor2np(x):
    np_image = x.cpu().permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)
    
    return np_image

def alb_tfm2fastai(alb_tfm, order):
    def _alb_transformer(x):
        # tensor to numpy
        np_image = tensor2np(x)

        # apply albumentations
        transformed = alb_tfm(image=np_image)['image']

        # back to tensor
        tensor_image = pil2tensor(transformed, np.float32)
        tensor_image.div_(255)

        return tensor_image

    transformer = TfmPixel(_alb_transformer, order=order)
    
    return transformer()

tfms = []

tfms += [alb_tfm2fastai(A.MotionBlur(p=.3), order=11)]

tfms += [alb_tfm2fastai(A.GaussNoise(p=.3), order=12)]

tfms += [alb_tfm2fastai(A.JpegCompression(p=.3, quality_lower=50), order=13)]

tfms += [alb_tfm2fastai(A.Downscale(scale_min=0.25, scale_max=0.9, p=.3), order=14)]



bs, sz = 32, 256
path = Path('../data/cropped_faces')

src = ImageList.from_folder(path).split_by_folder(train='train', valid='valid')

def get_data(bs,size):
    data = (src.label_from_re('([A-Z]+).jpg$') 
        .transform(get_transforms(xtra_tfms=tfms),size=size, padding_mode="border")
        .databunch(bs=bs, device=device).normalize(imagenet_stats))
    return data


@call_parse
def main(gpu:Param("GPU to run on", str)=None,
        save_file: Param("Name of the model to save", str)=None
        ):

    data = get_data(bs, sz)

    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=data.c)
    learn = Learner(data, model, metrics=[accuracy],loss_func = LabelSmoothingCrossEntropy()).to_fp16()


    cb = OverSamplingCallback(learn)

    # Train only the head of the network
    learn.fit_one_cycle(5, 1e-3, callbacks=[cb, SaveModelCallback(learn, monitor='valid_loss', name=save_file+'_1')])

    
    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-4, callbacks=[cb, SaveModelCallback(learn, monitor='valid_loss', name=save_file+'_2')])