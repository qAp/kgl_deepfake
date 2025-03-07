# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04a_classification_videolist.ipynb (unless otherwise specified).

__all__ = ['infer_on_videolist']

# Cell
import cv2
from fastai.core import *
from fastai.vision import *
from facenet_pytorch import MTCNN
from .nb_00 import *
from .nb_01b import *
from .nb_03 import *

# Cell
def infer_on_videolist(learn:Learner, vlist:VideoFaceList):
    filenames, labels = [], []
    for i in progress_bar(range(len(vlist))):
        fn, img = vlist.items[i], vlist[i]
        y, _, _ = learn.predict(img)
        filenames.append(fn.name)
        labels.append(int(y))
    return pd.DataFrame({'filename':filenames, 'label':labels})