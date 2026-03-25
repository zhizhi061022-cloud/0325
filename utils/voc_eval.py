import numpy as np
from collections import defaultdict


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall.

    If use_07_metric is true, uses the VOC 07 11-point method.
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
        return ap

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def _iou_voc(det_box, gt_boxes):
    if len(gt_boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    ixmin = np.maximum(gt_boxes[:, 0], det_box[0])
    iymin = np.maximum(gt_boxes[:, 1], det_box[1])
    ixmax = np.minimum(gt_boxes[:, 2], det_box[2])
    iymax = np.minimum(gt_boxes[:, 3], det_box[3])

    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    uni = (
        (det_box[2] - det_box[0] + 1.0) * (det_box[3] - det_box[1] + 1.0)
        + (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0)
        * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
        - inters
    )
    return inters / np.maximum(uni, 1e-12)


def compute_map(detections, groundtruths, num_classes, iou_thresh=0.5, use_07_metric=True):
    gt_by_cls = defaultdict(lambda: defaultdict(list))
    for g in groundtruths:
        gt_by_cls[g['label']][g['img_id']].append(g['box'])

    det_by_cls = defaultdict(list)
    for d in detections:
        det_by_cls[d['label']].append(d)

    aps = []
    ap_per_class = []

    for c in range(num_classes):
        class_recs = {}
        npos = 0
        for img_id, boxes in gt_by_cls[c].items():
            bbox = np.asarray(boxes, dtype=np.float32)
            class_recs[img_id] = {
                'bbox': bbox,
                'det': [False] * len(bbox),
            }
            npos += len(bbox)

        dets = det_by_cls[c]
        if len(dets) == 0:
            ap_per_class.append((c, 0.0))
            aps.append(0.0)
            continue

        image_ids = [d['img_id'] for d in dets]
        confidence = np.asarray([d['score'] for d in dets], dtype=np.float32)
        BB = np.asarray([d['box'] for d in dets], dtype=np.float32)

        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[i] for i in sorted_ind]

        nd = len(image_ids)
        tp = np.zeros(nd, dtype=np.float32)
        fp = np.zeros(nd, dtype=np.float32)

        for d in range(nd):
            img_id = image_ids[d]
            bb = BB[d, :].astype(np.float32)
            rec = class_recs.get(img_id)
            if rec is None:
                fp[d] = 1.0
                continue

            BBGT = rec['bbox'].astype(np.float32)
            if BBGT.size > 0:
                overlaps = _iou_voc(bb, BBGT)
                ovmax = np.max(overlaps)
                jmax = int(np.argmax(overlaps))
            else:
                ovmax = -np.inf
                jmax = -1

            if ovmax > iou_thresh:
                if not rec['det'][jmax]:
                    tp[d] = 1.0
                    rec['det'][jmax] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(max(npos, 1))
        prec = tp / np.maximum(tp + fp, 1e-12)
        ap = voc_ap(rec, prec, use_07_metric=use_07_metric)
        aps.append(ap)
        ap_per_class.append((c, ap))

    return float(np.mean(aps)) if aps else 0.0, ap_per_class
