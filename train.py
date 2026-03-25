"""
train.py  —  PLN-VOC Faithful Baseline Training

Usage:
    python train.py --voc07 /data/VOC2007 --voc12 /data/VOC2012
    python train.py --voc07 /data/VOC2007   # VOC07 only
"""

import os, sys, argparse, time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model.pln        import PLN, _IMG_TO_S
from model.loss       import PLNLoss
from model.decoder    import PLNDecoder
from data.voc_dataset import VOCDataset, VOCTestDataset, collate_fn, VOC_CLASSES
from eval import evaluate_dataset


# ── LR schedule: paper-style warmup to target LR ──────────────────────────────

def get_lr(step, warmup_steps, warmup_lr, lr0, epoch=0, milestones=(90, 120), gamma=0.1):
    if warmup_steps > 0 and step < warmup_steps:
        alpha = (step + 1) / warmup_steps
        return warmup_lr + alpha * (lr0 - warmup_lr)
    lr = lr0
    for m in milestones:
        if epoch >= m:
            lr *= gamma
    return lr


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, test_ds, decoder, device, batch=8):
    return evaluate_dataset(
        model,
        test_ds,
        decoder,
        device,
        batch=batch,
        workers=4,
        progress_every=20,
    )



# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--voc07',    required=True)
    p.add_argument('--voc12',    default='')
    p.add_argument('--img_size', type=int,   default=448,
                   choices=[448, 512, 640])
    p.add_argument('--epochs',   type=int,   default=140)
    p.add_argument('--batch',    type=int,   default=16)
    p.add_argument('--workers',  type=int,   default=8)
    p.add_argument('--lr',       type=float, default=5e-3)
    p.add_argument('--warmup_lr', type=float, default=1e-3,
                   help='paper-style warmup start LR')
    p.add_argument('--warmup',   type=int,   default=3)
    p.add_argument('--wd',       type=float, default=4e-5,
                   help='weight decay (paper uses 0.00004)')
    p.add_argument('--val_freq', type=int,   default=10)
    p.add_argument('--ckpt_freq', type=int,  default=10,
                   help='save epoch_xxx.pth every N epochs; 0 disables it')
    p.add_argument('--save_dir', default='runs/faithful')
    p.add_argument('--resume',   default='')
    p.add_argument('--backbone', default='resnet18',
                   choices=['resnet18', 'inceptionv2', 'inceptionv4'],
                   help='feature extractor backbone (default: resnet18)')
    p.add_argument('--B',        type=int,   default=2,
                   help='point slots per type (paper B=2)')
    p.add_argument('--lr_steps', default='90,120',
                   help='epoch milestones for lr decay, e.g. "90,120"')
    p.add_argument('--lr_gamma', type=float, default=0.1,
                   help='lr decay factor at each milestone')
    p.add_argument('--cosine', action='store_true',
                   help='use cosine annealing LR instead of step decay')
    p.add_argument('--reset_scheduler', action='store_true',
                   help='ignore saved scheduler state on resume (use when extending epochs)')
    return p.parse_args()


def main():
    args     = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f'  GPU  : {prop.name}  {prop.total_memory/1e9:.1f} GB')

    S      = _IMG_TO_S[args.img_size]
    stride = args.img_size // S
    print(f'img_size={args.img_size}  S={S}  stride={stride}  B={args.B}')

    # ── data ─────────────────────────────────────────────────────────────
    roots = [(args.voc07, 'trainval')]
    if args.voc12:
        roots.append((args.voc12, 'trainval'))

    train_ds = VOCDataset(roots, img_size=args.img_size, augment=True)
    test_ds  = VOCTestDataset(args.voc07, split='test', img_size=args.img_size)

    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )

    # ── model ────────────────────────────────────────────────────────────
    model = PLN(num_classes=20, img_size=args.img_size,
                B=args.B, pretrained=True,
                backbone=args.backbone).to(device)
    print(f'Backbone : {args.backbone}')

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f'  DataParallel × {torch.cuda.device_count()}')

    crit = PLNLoss(S=S, C=20, B=args.B, stride=stride)
    dec  = PLNDecoder(img_size=args.img_size, C=20, B=args.B,
                      conf_thresh=0.05, nms_thresh=0.45)

    # ── optimiser: paper uses RMSProp ─────────────────────────────────────
    # Separate BN/bias (no weight decay) from weights (weight decay).
    # Use id() dedup to avoid the "duplicate parameters" warning that
    # occurred with the old modules() traversal.
    wd_params, no_wd_params = [], []
    seen = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if id(param) in seen or not param.requires_grad:
                continue
            seen.add(id(param))
            if isinstance(module, torch.nn.BatchNorm2d) or name == 'bias':
                no_wd_params.append(param)
            else:
                wd_params.append(param)

    optimizer = optim.RMSprop(
        [{'params': wd_params,    'weight_decay': args.wd},
         {'params': no_wd_params, 'weight_decay': 0.}],
        lr=args.lr, momentum=0.9, alpha=0.9,
    )

    # ── resume ───────────────────────────────────────────────────────────
    start, best_map = 0, 0.
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start    = ck['epoch'] + 1
        best_map = ck.get('best_map', 0.)
        print(f'Resumed epoch {ck["epoch"]}, best mAP={best_map:.4f}')

    milestones = [int(x) for x in args.lr_steps.split(',') if x.strip()]
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup, 1),
            eta_min=1e-6,
        )
        print(f'LR schedule: warmup {args.warmup} epochs ({args.warmup_lr}→{args.lr}), '
              f'then cosine annealing to 1e-6 over {args.epochs - args.warmup} epochs')
    else:
        scheduler = None
        print(f'LR schedule: warmup {args.warmup} epochs ({args.warmup_lr}→{args.lr}), '
              f'decay×{args.lr_gamma} at epochs {milestones}')

    if args.resume and scheduler is not None and not args.reset_scheduler:
        ck_sched = torch.load(args.resume, map_location=device).get('scheduler')
        if ck_sched is not None:
            scheduler.load_state_dict(ck_sched)
            print(f'Restored scheduler state from checkpoint')
    elif args.reset_scheduler:
        # Reset optimizer LR to args.lr so CosineAnnealingLR base_lr is correct
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr
        # Re-create scheduler so base_lrs reflects the reset LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup, 1),
            eta_min=1e-6,
        )
        print(f'Scheduler reset: fresh cosine over {args.epochs - args.warmup} epochs, base_lr={args.lr}')

    # ── CSV 列定义 ───────────────────────────────────────────────────────────
    _BRANCHES   = ('lt', 'rt', 'lb', 'rb')
    _SUB_KEYS   = ('exist', 'cls', 'coord', 'link', 'neg')
    _SUB_COLS   = [f'{br}_{k}' for br in _BRANCHES for k in _SUB_KEYS]
    _CLASS_COLS = list(VOC_CLASSES)
    _HEADER     = ','.join(['epoch', 'loss'] + list(_BRANCHES) + ['mAP'] + _CLASS_COLS)

    log = open(save_dir / 'log.csv', 'a')
    if start == 0:
        log.write(_HEADER + '\n')

    # ── training loop ────────────────────────────────────────────────────
    warmup_steps = args.warmup * len(loader)
    global_step = start * len(loader)

    for epoch in range(start, args.epochs):
        model.train()
        loss_sum, n_steps = 0., 0
        sub_sum = {col: 0. for col in _SUB_COLS}
        br_sum = {br: 0. for br in _BRANCHES}
        t0 = time.time()

        for step, (imgs, boxes, labels) in enumerate(loader):
            if scheduler is not None and global_step >= warmup_steps:
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = get_lr(global_step, warmup_steps, args.warmup_lr, args.lr,
                            epoch=epoch, milestones=milestones, gamma=args.lr_gamma)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            imgs = imgs.to(device, non_blocking=True)

            preds = model(imgs)
            loss, sub = crit(preds, boxes, labels)

            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at epoch={epoch} step={step}; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            loss_sum += loss.item()
            for br in _BRANCHES:
                for k in _SUB_KEYS:
                    v = sub.get(f'{br}/{k}', 0.)
                    sub_sum[f'{br}_{k}'] += v
                    br_sum[br] += v
            n_steps  += 1

            if step % 100 == 0:
                eta = (time.time()-t0)/(step+1)*(len(loader)-step-1)
                br_str = '  '.join(f'{br}={br_sum[br]/n_steps:.3f}' for br in _BRANCHES)
                print(f'[{epoch:03d}/{args.epochs}] {step}/{len(loader)} '
                      f'loss={loss_sum/n_steps:.4f}  {br_str}  lr={lr:.6f}  eta={eta/60:.1f}min')

        avg_loss = loss_sum / max(n_steps, 1)

        if scheduler is not None and epoch >= args.warmup:
            scheduler.step()

        mAP = 0.
        ap_per_class = {c: 0. for c in VOC_CLASSES}
        if (epoch+1) % args.val_freq == 0 or epoch == args.epochs-1:
            mAP, ap_cls = evaluate(model, test_ds, dec, device, batch=16)
            print(f'\n=== Epoch {epoch}  mAP@0.5 = {mAP:.4f} ===')
            for ci, ap in ap_cls:
                print(f'  {VOC_CLASSES[ci]:<16s} {ap:.4f}')
                ap_per_class[VOC_CLASSES[ci]] = ap
            print()

        n = max(n_steps, 1)
        br_avg_vals  = [f'{br_sum[br]/n:.6f}' for br in _BRANCHES]
        cls_ap_vals  = [f'{ap_per_class[c]:.6f}' for c in _CLASS_COLS]
        log.write(','.join([str(epoch), f'{avg_loss:.6f}'] + br_avg_vals +
                           [f'{mAP:.6f}'] + cls_ap_vals) + '\n')
        log.flush()

        ck = dict(epoch=epoch, model=model.state_dict(),
                  optimizer=optimizer.state_dict(),
                  scheduler=scheduler.state_dict() if scheduler is not None else None,
                  best_map=max(best_map, mAP), args=vars(args))
        if args.ckpt_freq > 0 and (epoch + 1) % args.ckpt_freq == 0:
            snap_path = save_dir / f'epoch_{epoch + 1:03d}.pth'
            torch.save(ck, snap_path)
            print(f'  Saved snapshot: {snap_path.name}')
            for old in save_dir.glob('epoch_*.pth'):
                if old != snap_path:
                    old.unlink()
        if mAP > best_map:
            best_map = mAP
            torch.save(ck, save_dir / 'best.pth')
            print(f'  ★ New best mAP = {best_map:.4f}')

    log.close()
    print(f'\nDone.  Best mAP = {best_map:.4f}')
    print(f'Checkpoint: {save_dir}/best.pth')


if __name__ == '__main__':
    main()
