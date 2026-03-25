"""
model/target.py  —  Ground-truth encoding for faithful PLN

Paper Section 3.1 & 3.2:

For each GT box, for each of the 4 branches:
  - 5 key points: center O, lt C1, rt C2, lb C3, rb C4
  - Each branch uses one (center, corner) pair
  - Both points fall in some grid cell (i_center, i_corner)
  - Slot assignment: j=1..B for center, j=B+1..2B for corner
    (if multiple GT points fall in same cell+type, fill slots in order;
     overflow beyond B is ignored — paper doesn't specify, we use FIFO)

Per positive slot, targets are:
  P_hat  = 1
  Q_hat  = one-hot class vector
  x_hat  = (point_x - cell_left)  / cell_width   ∈ [0,1]
  y_hat  = (point_y - cell_top)   / cell_height  ∈ [0,1]
  Lx_hat = one-hot over columns: column of the LINKED point
  Ly_hat = one-hot over rows:    row    of the LINKED point

Per negative slot:
  P_hat = 0   (only P is supervised)

Link rule (paper Section 3.1):
  center slot j ∈ [1,B]    links to corner slot (j+B) at the linked cell
  corner slot j ∈ [B+1,2B] links to center slot (j-B) at the linked cell
  => Lx/Ly of center = column/row of the corner cell
  => Lx/Ly of corner = column/row of the center cell
"""

import torch


_BRANCH_NAMES = ('lt', 'rt', 'lb', 'rb')


def _five_points(x1, y1, x2, y2):#左上右下
    """Return (cx, cy, lt, rt, lb, rb) for a box."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2)


def _branch_points(x1, y1, x2, y2, branch):
    """
    Return (center_xy, corner_xy) for the given branch.
    center_xy : (cx, cy)
    corner_xy : (kx, ky)
    """
    (cx, cy), lt, rt, lb, rb = _five_points(x1, y1, x2, y2)
    corner = {'lt': lt, 'rt': rt, 'lb': lb, 'rb': rb}[branch]
    return (cx, cy), corner


def _cell_and_offset(px, py, stride, S):
    """
    Given pixel coords (px, py), return:
      col, row : cell indices (clamped to [0, S-1])
      off_x    : (px - col*stride) / stride  ∈ [0,1]
      off_y    : (py - row*stride) / stride  ∈ [0,1]
    """
    col = min(max(int(px / stride), 0), S - 1)
    row = min(max(int(py / stride), 0), S - 1)
    off_x = (px - col * stride) / stride
    off_y = (py - row * stride) / stride
    off_x = min(max(off_x, 0.0), 1.0)
    off_y = min(max(off_y, 0.0), 1.0)
    return col, row, off_x, off_y


def build_targets(gt_boxes_list, gt_labels_list, S, C, B, stride):
    """
    Build training targets for all 4 branches.

    Args:
        gt_boxes_list  : list of (G,4) tensors [x1,y1,x2,y2] in pixels
        gt_labels_list : list of (G,) long tensors
        S              : grid size
        C              : num classes
        B              : slots per point type (paper: 2)
        stride         : pixels per cell (= img_size / S)

    Returns:
        targets : dict  key=branch_name  value=dict with:
            P_hat  : (N, S, S, 2*B)       float  0 or 1
            Q_hat  : (N, S, S, 2*B, C)    float  one-hot
            xy_hat : (N, S, S, 2*B, 2)    float  [0,1]
            Lx_hat : (N, S, S, 2*B, S)    float  one-hot
            Ly_hat : (N, S, S, 2*B, S)    float  one-hot
            mask   : (N, S, S, 2*B)       bool   True=positive slot
    """
    N = len(gt_boxes_list)

    targets = {}
    for branch in _BRANCH_NAMES:
        P_hat  = torch.zeros(N, S, S, 2 * B)
        Q_hat  = torch.zeros(N, S, S, 2 * B, C)
        xy_hat = torch.zeros(N, S, S, 2 * B, 2)
        Lx_hat = torch.zeros(N, S, S, 2 * B, S)
        Ly_hat = torch.zeros(N, S, S, 2 * B, S)
        mask   = torch.zeros(N, S, S, 2 * B, dtype=torch.bool)

        for n in range(N):
            boxes  = gt_boxes_list[n]   # (G, 4)#一帧的所有GT框
            labels = gt_labels_list[n]  # (G,)

            # Track occupancy separately for center/corner cells, but assign
            # the SAME pair index b to both points of one object.
            center_used = {}
            corner_used = {}

            for g in range(len(boxes)):
                x1, y1, x2, y2 = boxes[g].tolist()
                label = int(labels[g].item())

                (cx, cy), (kx, ky) = _branch_points(x1, y1, x2, y2, branch)

                # center cell
                c_col, c_row, c_ox, c_oy = _cell_and_offset(cx, cy, stride, S)
                # corner cell
                k_col, k_row, k_ox, k_oy = _cell_and_offset(kx, ky, stride, S)

                key_c = (n, c_row, c_col)
                key_k = (n, k_row, k_col)

                center_flags = center_used.setdefault(key_c, [False] * B)
                corner_flags = corner_used.setdefault(key_k, [False] * B)

                pair_b = None
                for b in range(B):
                    if not center_flags[b] and not corner_flags[b]:
                        pair_b = b
                        break

                # Overflow: if no shared pair slot is free for this point pair,
                # drop the object in this branch.
                if pair_b is None:
                    continue

                center_flags[pair_b] = True
                corner_flags[pair_b] = True

                j_c = pair_b
                j_k = B + pair_b

                P_hat [n, c_row, c_col, j_c]    = 1.0
                Q_hat [n, c_row, c_col, j_c, label] = 1.0
                xy_hat[n, c_row, c_col, j_c, 0] = c_ox
                xy_hat[n, c_row, c_col, j_c, 1] = c_oy
                Lx_hat[n, c_row, c_col, j_c, k_col] = 1.0
                Ly_hat[n, c_row, c_col, j_c, k_row] = 1.0
                mask  [n, c_row, c_col, j_c]    = True

                P_hat [n, k_row, k_col, j_k]    = 1.0
                Q_hat [n, k_row, k_col, j_k, label] = 1.0
                xy_hat[n, k_row, k_col, j_k, 0] = k_ox
                xy_hat[n, k_row, k_col, j_k, 1] = k_oy
                Lx_hat[n, k_row, k_col, j_k, c_col] = 1.0
                Ly_hat[n, k_row, k_col, j_k, c_row] = 1.0
                mask  [n, k_row, k_col, j_k]    = True

        targets[branch] = dict(
            P_hat=P_hat, Q_hat=Q_hat, xy_hat=xy_hat,
            Lx_hat=Lx_hat, Ly_hat=Ly_hat, mask=mask,
        )

    return targets
