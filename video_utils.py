import cv2
import sys
import os
import numpy as np
from tqdm import tqdm


def mp4_to_jpg(video_path, dst, skipframes=4, n_frames=99999999):
    if not os.path.isdir(dst):
        os.mkdir(dst)
    cap = cv2.VideoCapture(video_path)
    success, count = True, 0

    while success and count < n_frames:
        success, frame = cap.read()
        count += 1
        if count % skipframes == 0:
            dst_path = os.path.join(dst, f'frame{count}.jpg')
            cv2.imwrite(dst_path, frame)

    print(f'Saved {count // skipframes} frames to {dst}.')


def cut_frames(frames_dir, dst, corners):
    if not os.path.isdir(dst):
        os.mkdir(dst)
    frame_files = os.listdir(frames_dir)
    frame_paths = [os.path.join(frames_dir, frame_file)
                   for frame_file in frame_files]

    for frame_path, frame_file in tqdm(zip(frame_paths, frame_files)):
        img = cv2.imread(frame_path)
        if type(img) == np.ndarray:
            resized = resize_img(img, corners)
            dst_path = os.path.join(dst, frame_file)
            cv2.imwrite(dst_path, resized)
        else:
            print('No file read.')


def write_video(dst, imgs, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dst, fourcc, fps, (1080, 660))
    for img in imgs:
        out.write(img)
    out.release()


def resize_img(img, corners):
    (x1, y1), (x2, y2) = corners
    resized = img[y1:y2, x1:x2]
    return resized
