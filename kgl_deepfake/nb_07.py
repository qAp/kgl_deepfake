# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_full_classification.ipynb (unless otherwise specified).

__all__ = ['get_annots', 'infer_on_videolist']

# Cell
import cv2
from fastai.core import *
from fastai.vision import *
from facenet_pytorch import MTCNN
from .nb_00 import *
from .nb_01b import *
from .nb_03 import *

# Cell
def get_annots(SOURCE):
    """
    extract the metadata from all the folders contained in SOURCE.
    """

    files = []
    annots = []

    for i in SOURCE.iterdir(): # iterate over the files in SOURCE
        if i.is_dir() and (i/'metadata.json').is_file(): # Get only the directories
            print(f'Extracting data from the {i.name} folder')
            f = get_files(i, extensions=['.json']) # Extract the metadata
            files.append(f)

            a = pd.read_json(f[0]).T
            a.reset_index(inplace=True)
            a.rename({'index':'fname'}, axis=1, inplace=True)
            a.fname = i.name + '/' + a.fname.astype(str)
            annots.append(a)


    return pd.concat(annots)

# Cell
def infer_on_videolist(learn:Learner, vlist:VideoFaceList):
    filenames, labels = [], []
    for i in progress_bar(range(len(vlist))):
        fn, img = vlist.items[i], vlist[i]
        _, _, y = learn.predict(img)
        filenames.append(fn.name)
        labels.append(float(y[0]))
    return pd.DataFrame({'filename':filenames, 'label':labels})