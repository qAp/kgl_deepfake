import cv2
import torch
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_frame_as_size(video_path, size=(128, 128)):
    capture = cv2.VideoCapture(str(video_path))
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    capture.release()
    return frame


def read_frame(video_path):
    capture = cv2.VideoCapture(str(video_path))
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return frame


def read_all_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    all_frames = []
    ret = True
    while True:
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        else:
            break

    capture.release()
    return all_frames


def read_random_frame(video_path):
    """
    Read a random frame from any point in the video.
    """
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # HACK: Some videos are missing the last 10 frames. No idea why.
    random_frame = int(random.random() * frame_count) - 10
    print(random_frame)
    # Set to read specific frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return frame


def read_frame_at_frame_number(video_path, frame_number):
    capture = cv2.VideoCapture(str(video_path))
    # Set to read specific frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capture.release()
    return frame


def read_random_sequential_frames(video_path, num_frames=4):
    """
    Starting at a random point in the video, read {num_frames} frames and return
    as a single numpy array
    """

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - num_frames
    random_frame = int(random.random() * frame_count)
    capture.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    frames = []
    for i in range(num_frames):
        # Set to read specific frame
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    capture.release()
    return np.array(frames)


def plot_detections(img, detections, with_keypoints=True, figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.grid(False)
    ax.imshow(img)

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    print("Found %d faces" % len(detections))

    for i in range(len(detections)):
        xmin = detections[i, 0]
        ymin = detections[i, 1]
        xmax = detections[i, 2]
        ymax = detections[i, 3]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    plt.show()
