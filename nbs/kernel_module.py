import cv2
import pandas as pd
import torch.nn as nn
from fastai.core import *
import torch.nn.functional as F
from IPython.display import display, Video, HTML
from fastai.vision import *
from tqdm import tqdm
import time
from IPython.display import HTML
from facenet_pytorch import MTCNN


def html_vid(fname, **kwargs): return display(Video(fname, **kwargs))

def html_vid(fname):
    "Return HTML for video."
    return f'''
    <video width="300" height="250" controls>
    <source src="{fname}" type="video/mp4">
    </video>
    '''

def html_titled_vid(fname, title):
    "Return HTML for titled video."
    return f'<div><p>{title}</p><br>{html_vid(fname)}</div>'

def html_vids(fnames, titles=None, ncols=3):
    "Return HTML for table of (titled) videos."
    n = len(fnames)
    if titles is None: titles = n * ['']
    assert len(titles) == n
    rs = []
    for i in range(0, n, ncols):
        fs, ts = fnames[i:i+ncols], titles[i:i+ncols]
        xs = (html_titled_vid(f, t) for f,t in zip(fs, ts))
        xs = (f'<td>{x}</td>' for x in xs)
        r = f"<tr>{''.join(xs)}</tr>"
        rs.append(r)
    return f"<table>{''.join(rs)}</table>"

def video2frames(fname, *fs):
    '''
    fname - path to mp4 file
    fs - fractional lengths to resize original image to. e.g. (.5, .25)
    '''
    capture = cv2.VideoCapture(fname)
    imgs = []
    for i in tqdm(range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, img0 = capture.read()
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        imgs.append([img0] + [cv2.resize(img0, (0, 0), fx=f, fy=f) for f in fs])
    capture.release()
    return [np.stack(imgsz) for imgsz in zip(*imgs)]

def detect_facenet_pytorch(detector, images, batch_size):
    '''
    detector - facenet_pytorch.MTCNN
    images:  numpy.array
      array of images
    batch_size: int
      Number of images to be processed by `detector` in one go.
    '''
    faces = []
    for lb in np.arange(0, len(images), batch_size):
        imgs_pil = [PIL.Image.fromarray(image) for image in images[lb:lb+batch_size]]
        faces.extend(detector(imgs_pil))
    return torch.stack(faces)


#comes from https://www.kaggle.com/unkownhihi/starter-kernel-with-cnn-model-ll-lb-0-69235
class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class.

        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames, self.batch_size, self.resize = n_frames, batch_size, resize

    def __call__(self, filename, label=None, save_dir=None):
        """Load frames from an MP4 video and detect faces.
        Arguments:
            filename {str} -- Path to video.
        """
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None: sample = np.arange(0, v_len)
        else: sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        faces = []
        idxs, frames = [], []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success: continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = PIL.Image.fromarray(frame)

                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                idxs.append(j); frames.append(frame)

                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    if save_dir is not None:
                        save_paths = self.get_savepaths(filename, idxs, label, save_dir)
                        faces.extend(self.detector(frames, save_path=save_paths))
                    else: faces.extend(self.detector(frames))
                    idxs, frames = [], []

        v_cap.release()
        return faces

    def get_savepaths(self, filename, idxs, label=None, save_dir=None):
        if isinstance(filename, str): filename = Path(filename)
        if save_dir is None: save_dir = Path('./')
        if label is None: save_paths = [save_dir/f'{filename.stem}_{i:03d}.png' for i in idxs]
        else: save_paths = [save_dir/f'{filename.stem}_{i:03d}_{label}.png' for i in idxs]
        return [str(o) for o in save_paths]

def get_first_face(detector, fn, resize=.5):
    '''
    Returns the first detected face from a video
    '''
    v_cap = cv2.VideoCapture(str(fn))
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iframe, face = None, None
    for i in range(v_len):
        _ = v_cap.grab()
        success, frame = v_cap.retrieve()
        if not success: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        if resize is not None: frame = frame.resize([int(d * resize) for d in frame.size])
        face = detector(frame)
        if face is not None:
            iframe = i
            break
        v_cap.release()
    return iframe, face

def get_has_face(fnames, detector):
    if isinstance(fnames, (str, Path)): fnames = [fnames]
    res = []
    for i in progress_bar(range(len(fnames))):
        iframe, face = get_first_face(detector, fnames[i])
        res.append(True if iframe is not None else False)
    return res

class VideoFaceList(ImageList):
    def __init__(self, *args, detector=None, resize=.5, device=None, **kwargs):
        if device is None: device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if detector is None: detector = MTCNN(device=device, post_process=False)
        self.detector, self.resize = detector, resize
        super().__init__(*args, **kwargs)

    def get_face(self, fn:Path):
        iframe, face = get_first_face(self.detector, fn, self.resize)
        if iframe is None or face is None: raise Exception(f'No faces detected in {fn}')
        return iframe, face

    def open(self, fn:Path):
        iframe, face = self.get_face(fn)
        return Image(face / 255)

def show_faces(faces, idxs):
    ncols = 4
    nrows = int((len(idxs) - 1) / ncols) + 1
    _, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 4*nrows))
    for idx, ax in itertools.zip_longest(idxs, axs.flatten()):
        if idx:
            #img = np.array(faces[idx])
            ax.imshow(diff[idx].permute(1, 2, 0)/255.); ax.set_title(idx)
        ax.axis('off')

def plot_faces(faces, idxs):
    ncols = 4
    nrows = int((len(idxs) - 1) / ncols) + 1
    _, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(18, 4*nrows))
    for idx, ax in itertools.zip_longest(idxs, axs.flatten()):
        if idx:
            #img = np.array(faces[idx])
            ax.imshow(diff[idx].permute(1, 2, 0)/255.); ax.set_title(idx)
        ax.axis('off')


#comes from https://www.kaggle.com/unkownhihi/starter-kernel-with-cnn-model-ll-lb-0-69235
class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class.

        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                frames.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(frames))
                    frames = []

        v_cap.release()

        return faces

# By Nathan Hubens.
# Paper implementation does not use Adaptive Average Pooling. To get the exact same implementation,
# comment the avg_pool and uncomment the final max_pool layer.
class MesoNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, 1,1) # 8 x 256 x 256
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 5, 1,2) # 8 x 128 x 128
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 5, 1,2) # 8 x 64 x 64
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,16,5,1,2) # 8 x 32 x 32
        self.bn4 = nn.BatchNorm2d(16)
        self.avg_pool = nn.AdaptiveAvgPool2d((8))
        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        #x = F.max_pool2d(x, 4, 4)

        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)

        x = F.dropout(x, 0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.5)
        x = self.fc2(x)
        return x

def infer_on_videolist(learn:Learner, vlist:VideoFaceList):
    filenames, labels = [], []
    for i in progress_bar(range(len(vlist))):
        fn, img = vlist.items[i], vlist[i]
        y, _, _ = learn.predict(img)
        filenames.append(fn.name)
        labels.append(int(y))
    return pd.DataFrame({'filename':filenames, 'label':labels})

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)
