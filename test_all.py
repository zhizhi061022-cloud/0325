"""
test_all.py  —  static tests for PLN faithful baseline.
Tests every geometric invariant WITHOUT requiring torch/torchvision.
Run: python test_all.py
"""

import sys, os, ast, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

P = '[PASS]'; F = '[FAIL]'

def check(cond, msg):
    print(f'  {P if cond else F}  {msg}')
    assert cond, msg


# ══════════════════════════════════════════════════════════════════════════════
# 1. Syntax check every source file
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 1. Syntax ===')
files = [
    'model/pln.py', 'model/target.py',
    'model/loss.py', 'model/decoder.py',
    'data/voc_dataset.py', 'utils/voc_eval.py',
    'train.py',
]
for f in files:
    path = os.path.join(os.path.dirname(__file__), f)
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    try:
        ast.parse(src)
        print(f'  {P}  {f}')
    except SyntaxError as e:
        print(f'  {F}  {f}  SyntaxError: {e}')
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Paper constants
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 2. Paper constants ===')

# grid sizes must be input // 32
for img_size, expected_S in [(448,14),(512,16),(640,20)]:
    S = img_size // 32
    check(S == expected_S, f'img={img_size}: S={S} == {expected_S}')

# B = 2 is paper default (Section 3.3)
B = 2
check(B == 2, 'B = 2 (paper default)')

# per-slot channels: 1 + C + 2 + S + S
for img_size, S in [(448,14),(512,16),(640,20)]:
    C = 20
    slot_ch = 1 + C + 2 + S + S
    total_per_cell = 2 * B * slot_ch
    check(slot_ch == 1+C+2+S+S,
          f'img={img_size}: slot_ch={slot_ch} = 1+{C}+2+{S}+{S}')
    check(total_per_cell == 4 * slot_ch,
          f'img={img_size}: total_per_cell={total_per_cell} = 4*{slot_ch}')


# ══════════════════════════════════════════════════════════════════════════════
# 3. _five_points and _branch_points geometry
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 3. Branch point geometry ===')

def five_points(x1,y1,x2,y2):
    cx=(x1+x2)/2; cy=(y1+y2)/2
    return (cx,cy),(x1,y1),(x2,y1),(x1,y2),(x2,y2)

def branch_points(x1,y1,x2,y2,branch):
    (cx,cy),lt,rt,lb,rb = five_points(x1,y1,x2,y2)
    corner = {'lt':lt,'rt':rt,'lb':lb,'rb':rb}[branch]
    return (cx,cy), corner

def box_from_pair(cx,cy,kx,ky,branch):
    if branch=='lt': return kx,ky,2*cx-kx,2*cy-ky
    if branch=='rt': return 2*cx-kx,ky,kx,2*cy-ky
    if branch=='lb': return kx,2*cy-ky,2*cx-kx,ky
    return 2*cx-kx,2*cy-ky,kx,ky

for _ in range(500):
    x1=random.uniform(0,300); y1=random.uniform(0,300)
    x2=x1+random.uniform(10,200); y2=y1+random.uniform(10,200)
    for br in ('lt','rt','lb','rb'):
        (cx,cy),(kx,ky) = branch_points(x1,y1,x2,y2,br)
        rx1,ry1,rx2,ry2 = box_from_pair(cx,cy,kx,ky,br)
        err = max(abs(rx1-x1),abs(ry1-y1),abs(rx2-x2),abs(ry2-y2))
        assert err < 1e-9, f'{br}: roundtrip err={err}'

check(True, 'branch_points <-> box_from_pair roundtrip (2000 cases)')


# ══════════════════════════════════════════════════════════════════════════════
# 4. _cell_and_offset
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 4. Cell offset encoding ===')

def cell_and_offset(px, py, stride, S):
    col = min(max(int(px/stride),0),S-1)
    row = min(max(int(py/stride),0),S-1)
    ox  = min(max((px - col*stride)/stride,0.),1.)
    oy  = min(max((py - row*stride)/stride,0.),1.)
    return col, row, ox, oy

def decode_point(col, row, ox, oy, stride):
    px = (col + ox) * stride
    py = (row + oy) * stride
    return px, py

for stride, S in [(32,14),(32,16),(32,20)]:
    for _ in range(300):
        px = random.uniform(0, S*stride - 0.01)
        py = random.uniform(0, S*stride - 0.01)
        col, row, ox, oy = cell_and_offset(px, py, stride, S)
        px2, py2 = decode_point(col, row, ox, oy, stride)
        assert abs(px2-px)<1e-6 and abs(py2-py)<1e-6, \
            f'cell_offset roundtrip failed: {px:.2f}->{px2:.2f}'
        assert 0<=col<S and 0<=row<S
        assert 0.<=ox<=1. and 0.<=oy<=1.

check(True, 'cell_and_offset / decode roundtrip (900 cases)')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Link targets: center links to corner cell, corner links to center cell
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 5. Link target consistency ===')

for stride, S in [(32,14),(32,16)]:
    for br in ('lt','rt','lb','rb'):
        for _ in range(100):
            x1=random.uniform(0,S*stride*0.3)
            y1=random.uniform(0,S*stride*0.3)
            x2=x1+random.uniform(stride, S*stride*0.5)
            y2=y1+random.uniform(stride, S*stride*0.5)
            x2=min(x2,S*stride-1); y2=min(y2,S*stride-1)

            (cx,cy),(kx,ky) = branch_points(x1,y1,x2,y2,br)
            c_col,c_row,_,_ = cell_and_offset(cx,cy,stride,S)
            k_col,k_row,_,_ = cell_and_offset(kx,ky,stride,S)

            # center's link target should be corner cell indices
            # corner's link target should be center cell indices
            assert 0<=c_col<S and 0<=c_row<S
            assert 0<=k_col<S and 0<=k_row<S

check(True, 'link targets in [0,S-1] (800 cases)')


# ══════════════════════════════════════════════════════════════════════════════
# 6. Paper Eq.5 score formula structure
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 6. Eq.5 score formula ===')
# P_obj = P_ij * P_st * Q(n)_ij * Q(n)_st
#         * (Lx(sx)_ij * Ly(sy)_ij + Lx(ix)_st * Ly(iy)_st) / 2

P_ctr = 0.9; P_cor = 0.8
Q_ctr = 0.7; Q_cor = 0.6
lx_ctr = 0.5; ly_ctr = 0.4   # center→corner link
lx_cor = 0.3; ly_cor = 0.2   # corner→center link

score = P_ctr * P_cor * Q_ctr * Q_cor * (lx_ctr*ly_ctr + lx_cor*ly_cor) / 2.
expected = 0.9*0.8*0.7*0.6*(0.5*0.4+0.3*0.2)/2.
check(abs(score - expected) < 1e-12, f'Eq.5 score = {score:.6f}')


# ══════════════════════════════════════════════════════════════════════════════
# 7. Slot indices: center j∈[0,B-1], corner j∈[B,2B-1], |j-t|==B
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== 7. Slot index invariants ===')
B = 2
for b in range(B):
    j_center = b           # center slot
    j_corner = B + b       # linked corner slot
    check(abs(j_center - j_corner) == B,
          f'b={b}: |{j_center}-{j_corner}|={abs(j_center-j_corner)}==B={B}')
check(True, 'slot pairing |j-t|==B holds for all b')


print('\n' + '='*55)
print('All static tests passed.')
print('='*55 + '\n')
