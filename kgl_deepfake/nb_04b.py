# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04b_inference.ipynb (unless otherwise specified).

__all__ = ['inference']

# Cell
import cv2
from fastai.core import *
from fastai.vision import *
from facenet_pytorch import MTCNN
from .nb_00 import *
from .nb_01b import *
from .nb_03 import *

# Cell
def inference(learn:Learner, path:Path, nf=15, equalize=False):

    filenames, labels = [], []

    facepipe = DetectionPipeline(nf=nf)

    for fn in progress_bar(path.ls()):
        if fn.suffix == '.mp4':
            try:
                preds = []
                iframes, faces, probs = facepipe(fn, equalize=equalize)

                for face in faces:
                    if face is not None and face.shape[0]==1: # Only do the inference if there is a face detected
                        _,_,pred = learn.predict(Image((face/255.).squeeze()))
                        preds.append(pred[0])
                    else: preds.append(0.5) # if #face <1 or >1 in a frame, predict 0.5

                mean_pred = np.mean(preds)

            except Exception:
                print(f'except: {fn}')
                mean_pred = 0.5

            filenames.append(fn.name)
            labels.append(mean_pred)
    return pd.DataFrame({'filename':filenames, 'label':labels})