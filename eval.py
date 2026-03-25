"""
Standalone VOC evaluation for PLN checkpoints.

Usage:
    python eval.py --voc07 /data/VOC2007 --ckpt runs/exp/best.pth
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.voc_dataset import VOCTestDataset, VOC_CLASSES, test_collate
from model.decoder import PLNDecoder
from model.pln import PLN
from utils.voc_eval import compute_map


@torch.no_grad()
def evaluate_dataset(model, test_ds, decoder, device, batch=16, workers=4, progress_every=5):
    model.eval()
    loader = DataLoader(
        test_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=test_collate,
    )
    dets, gts = [], []
    n_total = len(test_ds)
    t0 = time.time()
    print(f'Eval: {n_total} images, {len(loader)} batches (batch={batch})')

    for bi, (imgs, ids, ows, ohs) in enumerate(loader):
        results = decoder(model(imgs.to(device, non_blocking=True)))
        for i, (boxes, scores, labels) in enumerate(results):
            img_id = ids[i]
            ow, oh = int(ows[i]), int(ohs[i])
            if len(boxes):
                b = boxes.cpu().numpy().copy()
                b[:, [0, 2]] = (b[:, [0, 2]] * ow / test_ds.img_size).clip(0, ow)
                b[:, [1, 3]] = (b[:, [1, 3]] * oh / test_ds.img_size).clip(0, oh)
                for box, sc, lb in zip(b, scores.cpu().numpy(), labels.cpu().numpy()):
                    dets.append(
                        dict(
                            img_id=img_id,
                            label=int(lb),
                            score=float(sc),
                            box=box.tolist(),
                        )
                    )
            for gb, gl in zip(*test_ds.get_gt(img_id)):
                gts.append(dict(img_id=img_id, label=int(gl), box=gb.tolist()))

        if (bi + 1) % progress_every == 0 or (bi + 1) == len(loader):
            done = min((bi + 1) * batch, n_total)
            elapsed = time.time() - t0
            eta = elapsed / max(bi + 1, 1) * (len(loader) - bi - 1)
            print(
                f'Eval [{bi + 1}/{len(loader)}] '
                f'imgs={done}/{n_total} '
                f'dets={len(dets)} gts={len(gts)} '
                f'elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min'
            )

    return compute_map(dets, gts, num_classes=20)


def evaluate(model, test_ds, decoder, device, batch=16, workers=4):
    return evaluate_dataset(
        model,
        test_ds,
        decoder,
        device,
        batch=batch,
        workers=workers,
        progress_every=5,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--voc07', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--img_size', type=int, default=448, choices=[448, 512, 640])
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--conf_thresh', type=float, default=0.05)
    p.add_argument('--nms_thresh', type=float, default=0.45)
    p.add_argument('--B', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f'  GPU  : {prop.name}  {prop.total_memory/1e9:.1f} GB')

    ck = torch.load(args.ckpt, map_location=device)
    ck_args = ck.get('args', {})
    img_size = ck_args.get('img_size', args.img_size)
    B = ck_args.get('B', args.B)

    print(f'Checkpoint: {args.ckpt}')
    print(f'Using img_size={img_size} B={B}')

    test_ds = VOCTestDataset(args.voc07, split='test', img_size=img_size)
    model = PLN(num_classes=20, img_size=img_size, B=B, pretrained=False).to(device)
    model.load_state_dict(ck['model'])

    dec = PLNDecoder(
        img_size=img_size,
        C=20,
        B=B,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
    )

    mAP, ap_cls = evaluate(
        model,
        test_ds,
        dec,
        device,
        batch=args.batch,
        workers=args.workers,
    )
    print(f'\nFinal mAP@0.5 = {mAP:.4f}')
    for ci, ap in ap_cls:
        print(f'  {VOC_CLASSES[ci]:<16s} {ap:.4f}')


if __name__ == '__main__':
    main()
