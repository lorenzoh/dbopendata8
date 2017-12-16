import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image


label_map = {1: "person",
             2: "bicycle"
             # 15: "bench",
             # 63: "chair"
             # 27: "backpack",
             # 28: "umbrella",
             # 31: "handbag",
             # 32: "tie",
             # 33: "suitcase"
             }

label_colors = {1: [0, 255, 0]}


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj


def visualize_preds(img_dir, pred_path, show=False, count=True):
    imgs = []
    predictions = load_pkl(pred_path)

    img_names = [f'frame{x}.jpg' for x in sorted(
        [name[5:-4] for name in predictions.keys()], key=int)]

    for img_name in tqdm(img_names):
        prediction = predictions[img_name]
        img = cv2.imread(os.path.join(img_dir, img_name))
        assert type(img) == np.ndarray
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        img = visualize_pred(img, prediction, count=count)
        if show:
            plt.imshow(img[:, :, ::-1])
            plt.title(img_name)
            plt.show()
        imgs.append(img)

    return imgs


def visualize_pred(img, pred, labels=label_map, cutoff=0.5, count=True):
    boxes = pred['boxes'][0]
    classes = pred['classes'][0]
    scores = pred['scores'][0]

    n_persons = 0

    for box, clas, score in zip(boxes, classes, scores):
        if score > cutoff:
            if clas in labels:
                img = draw_bb(img, box, clas, score)
                if clas == 1:

                    n_persons += 1

    if count:
        img = draw_n_persons(img, n_persons)

    return img


def draw_bb(img, corners, clas, score):
    height, width, _ = img.shape
    copy = img.copy()
    corner_ul = (int(corners[1] * width), int(corners[0] * height))
    corner_br = (int(corners[3] * width), int(corners[2] * height))

    color = label_colors.get(clas, [255, 0, 0])

    copy = cv2.rectangle(copy, corner_ul, corner_br, color, 2)
    if clas != 1:
        copy = draw_label(copy, corner_ul, color, clas, score)
    return copy


def draw_label(img, corner_ul, color, clas, score):
    text_loc = corner_ul[0], corner_ul[1] + 20

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    percentage = round(score * 100, 2)
    text = f'{label_map[int(clas)]}'  # ' {percentage}%'

    # draw background for text
    img = cv2.rectangle(
        img, corner_ul, (corner_ul[0] + 120, corner_ul[1] + 30), color, -1)
    # draw text

    img = cv2.putText(img, text, text_loc, font,
                      1, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def draw_n_persons(img, n_persons):
    height, width, _ = img.shape
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text = f'People in view: {n_persons}'
    img = cv2.putText(img, text, (0, height - 50), font,
                      1, (255, 255, 255), 2, cv2.LINE_AA)
    return img
