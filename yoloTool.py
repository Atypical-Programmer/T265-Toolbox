import argparse
import os
import cv2
import time
import numpy as np
import torch
from models.yolo.yolo_nano import YOLONano
from data.transforms import ValTransforms
from data.coco import coco_class_labels, coco_class_index


def plot_bbox_labels(img, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(
        img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0,
                test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, class_colors, vis_thresh=0.3):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_color = class_colors[int(cls_inds[i])]
            cls_id = coco_class_index[int(cls_inds[i])]
            mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

    return img

def detect(img):
    #img = cv2.imread("./1.png", cv2.IMREAD_COLOR)

    device = torch.device("cuda")
    cfg = {
        # backbone
        'backbone': 'sfnet_v2',
        # neck
        'neck': 'spp-dw',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    }


    print('Build YOLO-Nano ...')
    model = YOLONano(cfg=cfg,
                    device=device,
                    img_size=640,
                    num_classes=80,
                    trainable=False,
                    conf_thresh=0.1,
                    nms_thresh=0.45,
                    center_sample=False)
    model.load_state_dict(torch.load(
        'weights/yolo_nano_22.4_40.7.pth', map_location='cpu'), strict=False)
    model = model.to(device).eval()

    transform = ValTransforms(640)
    img_h, img_w = img.shape[:2]
    x, _, _, scale, offset = transform(img)
    x = x.unsqueeze(0).to(device)
    bboxes, scores, cls_inds = model(x)
    bboxes -= offset
    bboxes /= scale
    bboxes *= np.array([[img_w, img_h, img_w, img_h]])
    class_colors = [(np.random.randint(255),
                    np.random.randint(255),
                    np.random.randint(255)) for _ in range(80)]
    img_processed = visualize(img=img,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            class_colors=class_colors,
                            vis_thresh=0.3)
    return img_processed
