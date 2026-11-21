"""left_ear_cheek -- precise line art for the viewer's left ear, cheek edge and beard edge.

Crop px frame (zoom 4, crop origin = original (130, 320)).
Owned: x 0-640 / y 80-300  and  x 0-400 / y 300-840.
Seams (from cand_base): face contour crosses y=80 at x~305, cheek/beard line crosses
x=400 at y~564, beard silhouette crosses y=840 at x~337.

Candidate sets used (same params as the CLI overlays):
  cand_valley_med : --kind valley --sigma 4 5 6 --low 0.12 --high 0.3 --min-len 25
  cand_light_med  : --kind light  --sigma 4 5 6 --low 0.2  --high 0.45 --min-len 40
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # .../dotted-sketch

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from zoomkit import Region

r = Region("left_ear_cheek")
canvas = r.blank()
photo = r.photo()
gray = r.gray().astype(np.float32)


# ---------------------------------------------------------------- helpers --
def densify(P, step=1.0):
    P = np.asarray(P, np.float32)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out, np.float32)


def smooth_path(P, s=3.0, closed=False):
    P = np.asarray(P, np.float32)
    mode = "wrap" if closed else "nearest"
    return np.stack([gaussian_filter1d(P[:, 0], s, mode=mode),
                     gaussian_filter1d(P[:, 1], s, mode=mode)], 1)


def snapped(pts, mode, radius, smooth=8.0):
    """kit snap (onto the real image line) with stronger offset smoothing than draw_polyline uses."""
    return r.snap(np.asarray(pts, np.float32), mode, radius, smooth=smooth)


def iso_x(field, y, x_lo, x_hi, level):
    """first x in [x_lo, x_hi) (scanning left->right) where field rises through level."""
    row = field[int(round(y)), x_lo:x_hi]
    idx = np.nonzero(row >= level)[0]
    if len(idx) == 0 or idx[0] == 0:
        return None
    i = idx[0]
    a, b = row[i - 1], row[i]
    return x_lo + i - 1 + (level - a) / max(b - a, 1e-6)


def nearest(P, q):
    return int(np.argmin(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1])))


# --------------------------------------------------------------------------
# 1. Ear outline -- one closed loop:
#  A outer rim: iso-contour (level 36) of the blurred photo -- ear vs. dark
#    backdrop / dark hair -- inside an ear zone that keeps hair strands out
#    (external contour, so the dark fossa can't pull it inward), edge-snapped.
#    From the back edge under the crus, over the helix, down the rim, round the
#    lobe bottom.
#  B lobe back + lower back edge (ear vs. sideburn hair), edge-snapped.
#  C concha pocket rim (lit concha floor vs. ear-hole shadow), edge-snapped.
# --------------------------------------------------------------------------
g3 = cv2.GaussianBlur(gray, (0, 0), 3)
zone = np.zeros(gray.shape, np.uint8)
cv2.fillPoly(zone, [np.array([
    (40, 60), (205, 60), (232, 96), (252, 140), (264, 190), (268, 250), (262, 272),
    (262, 330), (300, 330), (300, 700), (40, 720)], np.int32)], 1)
ear = ((g3 > 36) & (zone > 0)).astype(np.uint8)
ear = cv2.morphologyEx(ear, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
n, lab, st, _ = cv2.connectedComponentsWithStats(ear, connectivity=8)
ear = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
cs, _ = cv2.findContours(ear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
Cn = max(cs, key=len)[:, 0, :].astype(np.float32)
i0, i1 = nearest(Cn, (262, 300)), nearest(Cn, (282, 690))
iL = int(np.argmin(Cn[:, 0]))                                   # leftmost point must be on the path
if i0 < i1:
    rim = Cn[i0:i1 + 1] if i0 <= iL <= i1 else np.vstack([Cn[i1:], Cn[:i0 + 1]])[::-1]
else:
    rim = Cn[i1:i0 + 1][::-1] if i1 <= iL <= i0 else np.vstack([Cn[i0:], Cn[:i1 + 1]])
rim = snapped(smooth_path(rim[::3], 2.0), "edge", 6)

lobe = snapped([(284, 689), (293, 677), (298, 660), (302, 640), (304, 620), (304, 600),
                (300, 580), (294, 565), (283, 552), (273, 540), (268, 522), (265, 500),
                (264, 478), (265, 462)], "edge", 7)

concha = snapped([(265, 462), (254, 452), (238, 442), (223, 430), (215, 414), (212, 396),
                  (213, 378), (219, 360), (228, 345), (240, 333), (252, 320), (263, 306)], "edge", 7)

raw = densify(np.vstack([rim, lobe, concha]), 1.0)
# the soft back edge (dim ear vs. black hair, y 100-300) gets heavier smoothing
wb = ((raw[:, 0] > 222) & (raw[:, 1] > 100) & (raw[:, 1] < 300)).astype(np.float32)
wb = gaussian_filter1d(wb, 15, mode="wrap")[:, None]
loop = (1 - wb) * smooth_path(raw, 4.0, closed=True) + wb * smooth_path(raw, 14.0, closed=True)
r.draw_polyline(canvas, loop, snap=None, closed=True)

# --------------------------------------------------------------------------
# 2. Inner ear creases (valley candidates, medium scale)
#    20+21    inner helix line (scapha groove arcing over the top), stopped short
#             of the back edge so the two don't smear
#    25+47+78 antihelix front edge, down to the antitragus (25->47 gap bridged)
#    36+35    crus-of-helix crease under the bright bump, joined to the antihelix
# --------------------------------------------------------------------------
inner = r.blank()
vm = r.candidates("valley", sigma=(4.0, 5.0, 6.0), low=0.12, high=0.3, min_len=25)
r.draw_segments(inner, vm, ids=[20, 21, 25, 47, 78, 36, 35])
r.erase_rect(inner, 229, 200, 275, 255)          # 21 tail running into the back edge
r.erase_rect(inner, 249, 270, 280, 300)          # 35 tail: re-joined to the loop below
r.erase_rect(inner, 236, 530, 262, 552)          # 78 tail: keep clear of the lobe's back edge
r.draw_polyline(inner, [(178, 307), (178, 325), (176, 343)], snap="dark", radius=6)   # 25 -> 47
r.draw_polyline(inner, [(199, 309), (189, 312), (179, 316)], snap="dark", radius=4)   # 36 -> antihelix
ys, xs = np.nonzero(inner[270:300, 244:249])
c35 = (244 + xs[np.argmax(xs)], 270 + ys[np.argmax(xs)])        # crease end after trimming
j = loop[nearest(loop, (258, 288))]
r.draw_polyline(inner, [c35, tuple(j)], snap=None)              # crus crease meets the ear edge
canvas |= inner

# --------------------------------------------------------------------------
# 3. Face contour: skin vs. dark sideburn (gray iso-line) above, skin vs. grey
#    beard (saturation iso-line) below; bent at the ends onto the v1 seams.
# --------------------------------------------------------------------------
g5 = cv2.GaussianBlur(gray, (0, 0), 5)
sat = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32), (0, 0), 8)
face = []
for y in range(70, 451, 4):
    x = iso_x(g5, y, 300, 380, 72)
    if x is not None:
        face.append((x, y))
for y in range(454, 561, 4):
    x = iso_x(sat, y, 320, 420, 45)
    if x is not None:
        face.append((x, y))
face = np.array(face, np.float32)
face[face[:, 1] <= 70, 0] = 306                                 # seam: cross y=80 at x~306
top = (face[:, 1] > 70) & (face[:, 1] < 100)
w = (100 - face[top, 1]) / 30.0
face[top, 0] = w * 306 + (1 - w) * face[top, 0]
face = np.vstack([face, [(400, 564), (412, 570)]])              # seam: cross x=400 at y~564
face = smooth_path(densify(face, 1.0), 12.0)
r.draw_polyline(canvas, face, snap=None)

# --------------------------------------------------------------------------
# 4. Head / beard silhouette against the backdrop, from just below the lobe
# --------------------------------------------------------------------------
sil = []
for y in range(694, 853, 4):
    x = iso_x(g5, y, 290, 400, 50)
    if x is not None:
        sil.append((x, y))
sil = np.array(sil, np.float32)
wt = np.clip((sil[:, 1] - 790) / 50.0, 0, 1)                      # seam: ease onto x~338 at y=840
sil[:, 0] = (1 - wt) * sil[:, 0] + wt * 338
sil = smooth_path(densify(sil, 1.0), 9.0)
r.draw_polyline(canvas, sil, snap=None)

# --------------------------------------------------------------------------
# 5. Beard texture: one long grey strand down the middle of the owned beard
#    strip (light ridges 22+29), kept clear of the beard line and silhouette.
# --------------------------------------------------------------------------
tex = r.blank()
lm = r.candidates("light", sigma=(4.0, 5.0, 6.0), low=0.2, high=0.45, min_len=40)
r.draw_segments(tex, lm, ids=[22, 29])
r.erase_rect(tex, 330, 540, 400, 592)                           # keep clear of the beard line
r.draw_polyline(tex, [(368, 661), (368, 664), (369, 666)], snap=None)
# second strand: the grey lock curling right in the lower strip (light ridge, no candidate)
curl = snapped([(350, 773), (362, 778), (374, 785), (384, 795), (390, 806), (387, 818), (379, 828)],
               "light", 6, smooth=6.0)
r.draw_polyline(tex, smooth_path(densify(curl, 1.0), 3.0), snap=None)
canvas |= tex

r.save_lines(canvas)
