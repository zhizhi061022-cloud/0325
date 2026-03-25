"""
model/decoder.py  —  PLN inference, faithful to paper Section 3.3

Paper Eq.(5): for a point pair (i,j) and (s,t) where |j-t|=B:

  P_obj_ijnst = P_ij * P_st * Q(n)_ij * Q(n)_st
                * (Lx(sx)_ij * Ly(sy)_ij + Lx(ix)_st * Ly(iy)_st) / 2

  where (ix,iy) are column,row of cell i,
        (sx,sy) are column,row of cell s.

Decoding pipeline per branch:
  1. For each center slot (i, j) with j in [0,B-1]:
       - candidate center position: cell_col*stride + xy[...,0]*stride
     For each corner slot (s, t) with t in [B, 2B-1]:
       - candidate corner position
  2. Pair constraint: |j - t| == B  (i.e. j=0↔t=B, j=1↔t=B+1)
     (we relax to: pair any center slot with any corner slot from the
      same "slot pair index" b: center_b=b, corner_b=B+b)
  3. Compute P_obj_ijnst for each (class n, center cell i, slot j,
     corner cell s, slot t=j+B)
  4. Recover box from (center_xy, corner_xy):
       lt: x1=kx, y1=ky, x2=2*cx-kx, y2=2*cy-ky
       rt: x1=2*cx-kx, y1=ky, x2=kx, y2=2*cy-ky
       lb: x1=kx, y1=2*cy-ky, x2=2*cx-kx, y2=ky
       rb: x1=2*cx-kx, y1=2*cy-ky, x2=kx, y2=ky
  5. Collect all boxes from 4 branches → NMS

Practical approximation for speed (paper Section 3.3):
  - Pre-filter: only keep center/corner candidates above conf_thresh
  - Only search corner in geometrically valid quadrant
"""

import torch
import torchvision.ops as ops

_BRANCH_NAMES = ('lt', 'rt', 'lb', 'rb')


def _decode_points(pred, stride, S, conf_thresh):
    """
    Extract surviving point candidates from one branch prediction
    for a SINGLE image (no batch dim).

    Returns:
        centers : list of dicts  {row, col, slot_b, px, py, P, Q(C,)}
        corners : list of dicts  {row, col, slot_b, px, py, P, Q(C,)}
    """
    P  = pred['P']    # (S, S, 2B)
    Q  = pred['Q']    # (S, S, 2B, C)
    xy = pred['xy']   # (S, S, 2B, 2)
    B2 = P.shape[-1]
    B  = B2 // 2

    centers, corners = [], []

    for slot in range(B2):
        is_center = slot < B
        point_list = centers if is_center else corners
        slot_b = slot if is_center else slot - B

        # find cells above threshold
        score_map = P[..., slot]          # (S, S)
        rows, cols = (score_map > conf_thresh).nonzero(as_tuple=True)

        for r, c in zip(rows.tolist(), cols.tolist()):
            ox = xy[r, c, slot, 0].item()
            oy = xy[r, c, slot, 1].item()
            px = (c + ox) * stride
            py = (r + oy) * stride
            point_list.append({
                'row': r, 'col': c, 'slot_b': slot_b,
                'px': px, 'py': py,
                'P': P[r, c, slot].item(),
                'Q': Q[r, c, slot],     # (C,) tensor
            })

    return centers, corners


def _box_from_pair(cx, cy, kx, ky, branch):
    """Recover [x1,y1,x2,y2] from center + corner."""
    if branch == 'lt':
        return kx, ky, 2*cx-kx, 2*cy-ky
    if branch == 'rt':
        return 2*cx-kx, ky, kx, 2*cy-ky
    if branch == 'lb':
        return kx, 2*cy-ky, 2*cx-kx, ky
    # rb
    return 2*cx-kx, 2*cy-ky, kx, ky


def _valid_quadrant(cx, cy, kx, ky, branch):
    """Geometric sanity check: corner should be in the right quadrant."""
    if branch == 'lt': return kx <= cx and ky <= cy
    if branch == 'rt': return kx >= cx and ky <= cy
    if branch == 'lb': return kx <= cx and ky >= cy
    return kx >= cx and ky >= cy   # rb


def decode_branch(pred, branch, stride, S, C, B, conf_thresh, top_k=50):
    """
    Decode one branch for one image using vectorized center-corner pairing.
    Returns (boxes, scores, labels) tensors (may be empty).
    """
    P = pred['P']
    Q = pred['Q']
    xy = pred['xy']
    Lx = pred['Lx']
    Ly = pred['Ly']

    all_boxes, all_scores, all_labels = [], [], []

    for b in range(B):
        ctr_mask = P[..., b] > conf_thresh
        cor_mask = P[..., B + b] > conf_thresh

        ctr_rows, ctr_cols = ctr_mask.nonzero(as_tuple=True)#找有点的坐标
        cor_rows, cor_cols = cor_mask.nonzero(as_tuple=True)

        if len(ctr_rows) == 0 or len(cor_rows) == 0:
            continue

        ctr_P = P[ctr_rows, ctr_cols, b]#概率
        cor_P = P[cor_rows, cor_cols, B + b]
        if len(ctr_P) > top_k:
            idx = ctr_P.topk(top_k).indices
            ctr_rows, ctr_cols, ctr_P = ctr_rows[idx], ctr_cols[idx], ctr_P[idx]
        if len(cor_P) > top_k:
            idx = cor_P.topk(top_k).indices
            cor_rows, cor_cols, cor_P = cor_rows[idx], cor_cols[idx], cor_P[idx]

        Nc, Nk = len(ctr_rows), len(cor_rows)

        ctr_px = (ctr_cols + xy[ctr_rows, ctr_cols, b, 0]) * stride
        ctr_py = (ctr_rows + xy[ctr_rows, ctr_cols, b, 1]) * stride
        cor_px = (cor_cols + xy[cor_rows, cor_cols, B + b, 0]) * stride
        cor_py = (cor_rows + xy[cor_rows, cor_cols, B + b, 1]) * stride

        cx = ctr_px[:, None]
        cy = ctr_py[:, None]
        kx = cor_px[None, :]
        ky = cor_py[None, :]

        if branch == 'lt':
            quad = (kx <= cx) & (ky <= cy)
        elif branch == 'rt':
            quad = (kx >= cx) & (ky <= cy)
        elif branch == 'lb':
            quad = (kx <= cx) & (ky >= cy)
        else:
            quad = (kx >= cx) & (ky >= cy)

        lx_ctr = Lx[ctr_rows, ctr_cols, b][:, cor_cols]
        ly_ctr = Ly[ctr_rows, ctr_cols, b][:, cor_rows]
        lx_cor = Lx[cor_rows, cor_cols, B + b][:, ctr_cols]
        ly_cor = Ly[cor_rows, cor_cols, B + b][:, ctr_rows]
        link = (lx_ctr * ly_ctr + lx_cor.T * ly_cor.T) / 2.0

        Q_ctr = Q[ctr_rows, ctr_cols, b]
        Q_cor = Q[cor_rows, cor_cols, B + b]
        pair_cls = Q_ctr[:, None, :] * Q_cor[None, :, :]
        cls_score, best_cls = pair_cls.max(dim=-1)

        score = ctr_P[:, None] * cor_P[None, :] * cls_score * link
        score = score * quad.float()

        valid = score > 1e-6
        if not valid.any():
            continue

        vi, vj = valid.nonzero(as_tuple=True)
        s = score[vi, vj]
        lbl = best_cls[vi, vj]

        _cx = cx.expand(Nc, Nk)[vi, vj]
        _cy = cy.expand(Nc, Nk)[vi, vj]
        _kx = kx.expand(Nc, Nk)[vi, vj]
        _ky = ky.expand(Nc, Nk)[vi, vj]

        if branch == 'lt':
            x1, y1, x2, y2 = _kx, _ky, 2 * _cx - _kx, 2 * _cy - _ky
        elif branch == 'rt':
            x1, y1, x2, y2 = 2 * _cx - _kx, _ky, _kx, 2 * _cy - _ky
        elif branch == 'lb':
            x1, y1, x2, y2 = _kx, 2 * _cy - _ky, 2 * _cx - _kx, _ky
        else:
            x1, y1, x2, y2 = 2 * _cx - _kx, 2 * _cy - _ky, _kx, _ky

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0)
        size_ok = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)

        if size_ok.any():
            all_boxes.append(boxes[size_ok])
            all_scores.append(s[size_ok])
            all_labels.append(lbl[size_ok])

    if not all_boxes:
        dev = pred['P'].device
        return (
            torch.zeros(0, 4, device=dev),
            torch.zeros(0, device=dev),
            torch.zeros(0, dtype=torch.long, device=dev),
        )

    return torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)



class PLNDecoder:
    """
    Full PLN decoder: 4 branches → merge → per-class NMS.
    """
    def __init__(self, img_size=448, C=20, B=2,
                 conf_thresh=0.1, nms_thresh=0.45, max_det=300):
        from model.pln import _IMG_TO_S
        self.S          = _IMG_TO_S[img_size]
        self.stride     = img_size // self.S
        self.C          = C
        self.B          = B
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.max_det     = max_det
        self.img_size    = img_size

    @torch.no_grad()
    def __call__(self, preds):
        """
        preds : model output dict (batch)
        Returns list of (boxes, scores, labels) per image.
        """
        N = preds['lt']['P'].shape[0]
        results = []

        for n in range(N):
            # extract single-image preds
            single = {
                br: {k: v[n] for k, v in preds[br].items()}
                for br in _BRANCH_NAMES
            }

            all_boxes, all_scores, all_labels = [], [], []

            for br in _BRANCH_NAMES:
                bxs, scs, lbs = decode_branch(
                    single[br], br,
                    self.stride, self.S, self.C, self.B,
                    self.conf_thresh,
                )
                if len(bxs):
                    all_boxes.append(bxs)
                    all_scores.append(scs)
                    all_labels.append(lbs)

            if not all_boxes:
                dev = preds['lt']['P'].device
                results.append((
                    torch.zeros(0, 4, device=dev),
                    torch.zeros(0, device=dev),
                    torch.zeros(0, dtype=torch.long, device=dev),
                ))
                continue

            boxes  = torch.cat(all_boxes)
            scores = torch.cat(all_scores)
            labels = torch.cat(all_labels)

            # clamp to image
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.img_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.img_size)

            # per-class NMS (paper Section 3.4)
            kb, ks, kl = [], [], []
            for c in range(self.C):
                m = labels == c
                if not m.any():
                    continue
                idx = ops.nms(boxes[m].float(), scores[m].float(), self.nms_thresh)
                kb.append(boxes[m][idx])
                ks.append(scores[m][idx])
                kl.append(labels[m][idx])

            if kb:
                boxes  = torch.cat(kb)
                scores = torch.cat(ks)
                labels = torch.cat(kl)
                if len(scores) > self.max_det:
                    tk = scores.topk(self.max_det).indices
                    boxes, scores, labels = boxes[tk], scores[tk], labels[tk]

            results.append((boxes, scores, labels))

        return results
