"""
model/loss.py  —  PLN loss, faithful to paper Eq.(2) and Eq.(3)

Paper uses Euclidean (MSE) loss on activated outputs (sigmoid, softmax).
However, MSE on softmax has vanishing gradient issues in practice.

This version supports two modes:
  - mode='mse'    : original paper formulation (MSE on activated outputs)
  - mode='hybrid' : BCE for P, CE for Q/Lx/Ly, MSE for xy (recommended)

The 'hybrid' mode is numerically equivalent in intent but converges much
better because BCE/CE operate on logits with stronger gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.target import build_targets, _BRANCH_NAMES


class PLNLoss(nn.Module):
    def __init__(self, S, C, B=2, stride=32,
                 w_class=1.0, w_coord=5.0, w_link=1.0,
                 mode='hybrid'):
        """
        S      : grid size
        C      : num classes
        B      : slots per point type
        stride : img_size / S
        mode   : 'mse' for original paper, 'hybrid' for BCE/CE (recommended)
        """
        super().__init__()
        self.S       = S
        self.C       = C
        self.B       = B
        self.stride  = stride
        self.w_class = w_class
        self.w_coord = w_coord
        self.w_link  = w_link
        self.mode    = mode

    def forward(self, preds, gt_boxes_list, gt_labels_list):
        """
        preds          : model output dict {branch: {P,Q,xy,Lx,Ly}} or
                         {branch: {P,Q,xy,Lx,Ly,P_logit,Q_logit,Lx_logit,Ly_logit}}
        gt_boxes_list  : list of (G,4) tensors  [x1,y1,x2,y2] pixels
        gt_labels_list : list of (G,) long tensors

        Returns:
            total_loss : scalar
            log_dict   : dict of sub-losses for logging
        """
        device = preds['lt']['P'].device
        N = preds['lt']['P'].shape[0]

        tgts = build_targets(
            gt_boxes_list, gt_labels_list,
            self.S, self.C, self.B, self.stride,
        )

        total = torch.tensor(0., device=device)
        logs  = {}

        for branch in _BRANCH_NAMES:
            P  = preds[branch]['P']    # (N,S,S,2B)  sigmoid
            Q  = preds[branch]['Q']    # (N,S,S,2B,C) softmax
            xy = preds[branch]['xy']   # (N,S,S,2B,2) sigmoid
            Lx = preds[branch]['Lx']   # (N,S,S,2B,S) softmax
            Ly = preds[branch]['Ly']

            # Logits for hybrid mode
            P_logit  = preds[branch].get('P_logit')
            Q_logit  = preds[branch].get('Q_logit')
            Lx_logit = preds[branch].get('Lx_logit')
            Ly_logit = preds[branch].get('Ly_logit')

            tgt    = tgts[branch]
            P_hat  = tgt['P_hat'].to(device)
            Q_hat  = tgt['Q_hat'].to(device)
            xy_hat = tgt['xy_hat'].to(device)
            Lx_hat = tgt['Lx_hat'].to(device)
            Ly_hat = tgt['Ly_hat'].to(device)
            mask   = tgt['mask'].to(device)

            pos = mask
            neg = ~mask
            n_pos = pos.float().sum().clamp(min=1.)
            n_neg = neg.float().sum().clamp(min=1.)

            if self.mode == 'hybrid' and P_logit is not None:
                # ── Hybrid mode: BCE for P, CE for Q/Lx/Ly ──────────────

                # Eq.(3): negative slots — BCE on P logit
                l_neg = F.binary_cross_entropy_with_logits(
                    P_logit[neg], torch.zeros_like(P_logit[neg]),
                    reduction='sum'
                ) / n_neg

                if pos.any():
                    # P existence: BCE
                    l_exist = F.binary_cross_entropy_with_logits(
                        P_logit[pos], torch.ones_like(P_logit[pos]),
                        reduction='sum'
                    ) / n_pos

                    # Q class: cross-entropy on logits
                    # Q_hat is one-hot (N,S,S,2B,C), convert to class indices
                    Q_logit_pos = Q_logit[pos]       # (n_pos, C)
                    Q_hat_pos   = Q_hat[pos]         # (n_pos, C) one-hot
                    Q_target    = Q_hat_pos.argmax(dim=-1)  # (n_pos,)
                    l_cls = self.w_class * F.cross_entropy(
                        Q_logit_pos, Q_target, reduction='sum'
                    ) / n_pos

                    # xy coord: MSE (sigmoid output, same as paper)
                    l_coord = self.w_coord * (
                        (xy[pos] - xy_hat[pos]) ** 2
                    ).sum() / n_pos

                    # Lx, Ly link: cross-entropy on logits
                    Lx_logit_pos = Lx_logit[pos]     # (n_pos, S)
                    Ly_logit_pos = Ly_logit[pos]     # (n_pos, S)
                    Lx_target    = Lx_hat[pos].argmax(dim=-1)  # (n_pos,)
                    Ly_target    = Ly_hat[pos].argmax(dim=-1)

                    l_link = self.w_link * (
                        F.cross_entropy(Lx_logit_pos, Lx_target, reduction='sum') +
                        F.cross_entropy(Ly_logit_pos, Ly_target, reduction='sum')
                    ) / n_pos

                    l_branch = l_exist + l_cls + l_coord + l_link
                else:
                    l_exist = l_cls = l_coord = l_link = torch.tensor(0., device=device)
                    l_branch = torch.tensor(0., device=device)

            else:
                # ── Original MSE mode (paper formulation) ────────────────
                l_neg = (P[neg] ** 2).sum() / n_neg

                if pos.any():
                    l_exist = ((P[pos] - 1.0) ** 2).sum() / n_pos

                    l_cls = self.w_class * (
                        (Q[pos] - Q_hat[pos]) ** 2
                    ).sum() / n_pos

                    l_coord = self.w_coord * (
                        (xy[pos] - xy_hat[pos]) ** 2
                    ).sum() / n_pos

                    l_link = self.w_link * (
                        (Lx[pos] - Lx_hat[pos]) ** 2 +
                        (Ly[pos] - Ly_hat[pos]) ** 2
                    ).sum() / n_pos

                    l_branch = l_exist + l_cls + l_coord + l_link
                else:
                    l_exist = l_cls = l_coord = l_link = torch.tensor(0., device=device)
                    l_branch = torch.tensor(0., device=device)

            l_branch = l_branch + l_neg
            total    = total + l_branch

            logs[f'{branch}/exist']  = l_exist.item()
            logs[f'{branch}/cls']    = l_cls.item()
            logs[f'{branch}/coord']  = l_coord.item()
            logs[f'{branch}/link']   = l_link.item()
            logs[f'{branch}/neg']    = l_neg.item()

        logs['total'] = total.item()
        return total, logs
