"""
data/voc_dataset.py  —  Pascal VOC 2007/2012 dataset loader

Augmentation (paper Section 4.1):
  "randomly choose a simplified SSD data augmentation and
   a modified YOLO data augmentation"
  Simplified SSD = random crop patches
  Modified YOLO  = jitter [-0.3, 0.1] (crop outside image boundary)
  + horizontal flip (standard)

For the faithful baseline we implement:
  - letterbox resize (maintain aspect ratio, pad to square)
  - random horizontal flip
  - basic colour jitter
  (paper's exact augmentation can be added in the high-mAP variant)
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
)
CLASS2IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_voc_xml(ann_path):
    tree = ET.parse(ann_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in CLASS2IDX:
            continue
        if int(obj.find('difficult').text):
            continue
        bb  = obj.find('bndbox')
        x1  = float(bb.find('xmin').text)
        y1  = float(bb.find('ymin').text)
        x2  = float(bb.find('xmax').text)
        y2  = float(bb.find('ymax').text)
        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS2IDX[name])
    boxes  = np.array(boxes,  dtype=np.float32) if boxes  else np.zeros((0,4), np.float32)
    labels = np.array(labels, dtype=np.int64)   if labels else np.zeros(0,     np.int64)
    return boxes, labels


def letterbox(img, boxes, target_size):
    """Resize to target_size×target_size preserving aspect ratio, pad with grey."""
    h, w = img.shape[:2]
    S = target_size
    r = min(S / h, S / w)
    nh, nw = int(h * r), int(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pt = (S - nh) // 2
    pl = (S - nw) // 2
    img = cv2.copyMakeBorder(img, pt, S-nh-pt, pl, S-nw-pl,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + pl
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + pt
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, S)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, S)
    return img, boxes, r, pl, pt


def resize_to_square(img, boxes, target_size):
    """Resize image to target square and scale boxes accordingly."""
    h, w = img.shape[:2]
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= target_size / max(w, 1)
        boxes[:, [1, 3]] *= target_size / max(h, 1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, target_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, target_size)
    return img, boxes


def _filter_and_clip_boxes(boxes, labels, crop_x1, crop_y1, crop_x2, crop_y2,
                           min_size=2.0):
    if len(boxes) == 0:
        return boxes, labels

    centers = np.stack(((boxes[:, 0] + boxes[:, 2]) * 0.5,
                        (boxes[:, 1] + boxes[:, 3]) * 0.5), axis=1)
    keep = (
        (centers[:, 0] >= crop_x1) & (centers[:, 0] <= crop_x2) &
        (centers[:, 1] >= crop_y1) & (centers[:, 1] <= crop_y2)
    )
    boxes = boxes[keep].copy()
    labels = labels[keep].copy()
    if len(boxes) == 0:
        return boxes, labels

    boxes[:, [0, 2]] -= crop_x1
    boxes[:, [1, 3]] -= crop_y1
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, crop_x2 - crop_x1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, crop_y2 - crop_y1)

    wh = boxes[:, 2:4] - boxes[:, 0:2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size)
    return boxes[keep], labels[keep]


def _box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    area1 = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    area2 = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return inter / (area1 + area2 - inter + 1e-6)


def random_ssd_crop(img, boxes, labels):
    """Simplified SSD-style crop without photometric distortion."""
    if len(boxes) == 0:
        return img, boxes, labels

    h, w = img.shape[:2]
    min_iou_choices = [None, 0.1, 0.3, 0.5, 0.7]
    for _ in range(50):
        min_iou = random.choice(min_iou_choices)
        if min_iou is None:
            return img, boxes, labels

        scale = random.uniform(0.3, 1.0)
        aspect = random.uniform(0.5, 2.0)
        crop_w = int(w * scale * np.sqrt(aspect))
        crop_h = int(h * scale / np.sqrt(aspect))
        if crop_w <= 1 or crop_h <= 1 or crop_w > w or crop_h > h:
            continue

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        crop = np.array([x1, y1, x1 + crop_w, y1 + crop_h], dtype=np.float32)

        ious = _box_iou_one_to_many(crop, boxes)
        if len(ious) == 0 or ious.max() < min_iou:
            continue

        new_boxes, new_labels = _filter_and_clip_boxes(
            boxes, labels, crop[0], crop[1], crop[2], crop[3]
        )
        if len(new_boxes) == 0:
            continue
        return img[y1:y1 + crop_h, x1:x1 + crop_w], new_boxes, new_labels

    return img, boxes, labels


def random_hsv(img, hue=0.1, sat=0.7, val=0.4):
    """HSV colour jitter, operates on uint8 BGR image."""
    img = img.astype(np.float32)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue * 180, hue * 180)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1 - sat, 1 + sat), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1 - val, 1 + val), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_yolo_jitter(img, boxes, labels, jitter_min=-0.3, jitter_max=0.1):
    """Modified YOLO-style jitter crop that may extend beyond image borders."""
    h, w = img.shape[:2]

    left = int(random.uniform(jitter_min, jitter_max) * w)
    right = int(random.uniform(jitter_min, jitter_max) * w)
    top = int(random.uniform(jitter_min, jitter_max) * h)
    bottom = int(random.uniform(jitter_min, jitter_max) * h)

    new_x1 = left
    new_y1 = top
    new_x2 = w - right
    new_y2 = h - bottom
    new_w = max(int(new_x2 - new_x1), 1)
    new_h = max(int(new_y2 - new_y1), 1)

    canvas = np.full((new_h, new_w, 3), 114, dtype=img.dtype)

    src_x1 = max(0, new_x1)
    src_y1 = max(0, new_y1)
    src_x2 = min(w, new_x2)
    src_y2 = min(h, new_y2)

    dst_x1 = max(0, -new_x1)
    dst_y1 = max(0, -new_y1)
    dst_x2 = dst_x1 + max(src_x2 - src_x1, 0)
    dst_y2 = dst_y1 + max(src_y2 - src_y1, 0)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= new_x1
        boxes[:, [1, 3]] -= new_y1
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, new_h)
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        keep = (wh[:, 0] >= 2.0) & (wh[:, 1] >= 2.0)
        boxes = boxes[keep]
        labels = labels[keep]

    return canvas, boxes, labels


class VOCDataset(Dataset):
    def __init__(self, roots, img_size=448, augment=True):
        """
        roots   : list of (voc_root, split) e.g.
                  [('/data/VOC2007','trainval'), ('/data/VOC2012','trainval')]
        img_size: 448 / 512 / 640
        """
        self.img_size = img_size
        self.augment  = augment
        self.samples  = []

        for voc_root, split in roots:
            ids_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
            with open(ids_file) as f:
                ids = [l.strip() for l in f if l.strip()]
            for img_id in ids:
                img_path = os.path.join(voc_root, 'JPEGImages',  f'{img_id}.jpg')
                ann_path = os.path.join(voc_root, 'Annotations', f'{img_id}.xml')
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    self.samples.append((img_path, ann_path))

        print(f'VOCDataset: {len(self.samples)} samples  '
              f'img_size={img_size}  augment={augment}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.full((self.img_size, self.img_size, 3), 114, np.uint8)
        boxes, labels = parse_voc_xml(ann_path)

        if self.augment:
            if random.random() < 0.5:
                img, boxes, labels = random_ssd_crop(img, boxes, labels)
            else:
                img, boxes, labels = random_yolo_jitter(img, boxes, labels)

            # HSV colour jitter
            img = random_hsv(img)

            # random horizontal flip
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                if len(boxes):
                    w = img.shape[1]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            img, boxes = resize_to_square(img, boxes, self.img_size)
        else:
            img, boxes = resize_to_square(img, boxes, self.img_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img_t    = torch.from_numpy(img.transpose(2, 0, 1))
        boxes_t  = torch.from_numpy(boxes).float()
        labels_t = torch.from_numpy(labels).long()
        return img_t, boxes_t, labels_t


def collate_fn(batch):
    imgs, boxes, labels = zip(*batch)
    return torch.stack(imgs), list(boxes), list(labels)


class VOCTestDataset(Dataset):
    def __init__(self, voc_root, split='test', img_size=448):
        self.img_size = img_size
        self.voc_root = voc_root
        ids_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
        with open(ids_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_path = os.path.join(self.voc_root, 'JPEGImages', f'{img_id}.jpg')
        img = cv2.imread(img_path)
        oh, ow = img.shape[:2]
        img, _ = resize_to_square(img, np.zeros((0,4),np.float32), self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        return img_t, img_id, ow, oh

    def get_gt(self, img_id):
        ann_path = os.path.join(self.voc_root, 'Annotations', f'{img_id}.xml')
        return parse_voc_xml(ann_path)


def test_collate(batch):
    imgs, ids, ows, ohs = zip(*batch)
    return torch.stack(imgs), ids, ows, ohs
