"""beard_body -- lower beard, neck, shoulders and cloth (zoom 2x, owned orig y 530-708).

Strokes:
  * left shoulder silhouette + left beard outline above it   (valley_big 6/17/37)
  * left beard edge against the neck skin, junction -> bottom (hand seeds, snapped on blurred V)
  * right beard edge against the background                  (hand seeds, matches base at the seam)
  * cloth silhouette on the right shoulder                    (edge3 40/71 + snapped edge polyline)
  * beard / cloth boundary from the cloth corner to the apex  (hand seeds snapped on blurred V) + valley_big 33
  * beard edge against the orange neck skin                   (hand seeds snapped on blurred S)
  * cloth inner edge against the neck skin                    (valley_big 41)
  * main cloth fold (deep crease)                             (valley_big 34)
  * beard texture: clump-scale dark gaps / light locks        (own beard-normalised sato, see beard_ridges)

Kit note: the kit's snap() works on grey at sigma 0.5*zoom with radius 3, which latches onto single
hairs here, and the beard-vs-skin boundaries are saturation / soft value steps.  So this script uses
its own snap2() on a chosen, more strongly blurred channel and then calls draw_polyline(snap=None).
Likewise the kit's valley/light candidates normalise by the whole figure, so beard clumps never pass
the thresholds; beard_ridges() re-normalises sato inside the beard only.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zoomkit import Region, Segments  # noqa: E402

r = Region("beard_body")
photo = r.photo()
hsv = cv2.cvtColor(photo, cv2.COLOR_RGB2HSV).astype(np.float32)
gray = r.gray().astype(np.float32)


def blur(img, s):
    return cv2.GaussianBlur(img, (0, 0), s)


V3 = blur(hsv[..., 2], 3.0)          # value, strand noise suppressed
S3 = blur(hsv[..., 1], 3.0)          # saturation (orange skin vs grey beard)
G15 = blur(gray, 1.5)                # silhouettes against the backdrop
G25 = blur(gray, 2.5)                # clump-scale gaps in the beard


# --------------------------------------------------------------------------- helpers
def densify(P, step=1.0):
    P = np.asarray(P, np.float32)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out, np.float32)


def snap2(pts, field, mode="edge", radius=6.0, smooth=4.0, penalty=0.15, post=1.5):
    """Move a seed polyline along its (smoothed) normal onto the strongest edge / darkest / lightest
    pixel of `field` within +/- radius, smooth the offsets, return the new polyline."""
    P = densify(pts, 1.0)
    t = np.gradient(P, axis=0)
    t = gaussian_filter1d(t, 4.0, axis=0, mode="nearest")
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-radius, radius + 0.01, 0.5, dtype=np.float32)
    if mode == "edge":
        cost = -np.hypot(cv2.Sobel(field, cv2.CV_32F, 1, 0), cv2.Sobel(field, cv2.CV_32F, 0, 1))
    elif mode == "dark":
        cost = field
    elif mode == "light":
        cost = -field
    else:
        raise ValueError(mode)
    Sm = P[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    vals = cv2.remap(cost.astype(np.float32), Sm[..., 0].astype(np.float32), Sm[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR)
    span = float(np.ptp(vals)) + 1e-6
    vals = vals + penalty * span * (np.abs(offs) / radius)[None, :]
    best = gaussian_filter1d(offs[np.argmin(vals, axis=1)], smooth, mode="nearest")
    Q = P + best[:, None] * nrm
    if post:
        Q = gaussian_filter1d(Q, post, axis=0, mode="nearest")
    return Q


def line(canvas, pts):
    r.draw_polyline(canvas, pts, snap=None)


def beard_ridges(dark: bool, lo=0.35, hi=0.6, min_len=40) -> Segments:
    """sato ridges at clump scale, normalised inside the beard only (deterministic ids)."""
    g = cv2.GaussianBlur(r.gray(), (0, 0), 1.0).astype(np.float32) / 255.0
    resp = sato(g, sigmas=[4, 5, 6], black_ridges=dark)
    beard = np.zeros(resp.shape, bool)
    beard[40:396, 440:1060] = True
    resp = resp / np.percentile(resp[beard], 99)
    resp[~beard] = 0
    return Segments(skeletonize(apply_hysteresis_threshold(resp, lo, hi)), min_len)


canvas = r.blank()

# --------------------------------------------------------------------------- candidates
vb = r.candidates("valley", sigma=(4, 6, 8), low=0.25, high=0.5, min_len=30)
e3 = r.candidates("edge", smooth=5, high_pct=80, low_ratio=0.3, min_len=40)

# 1. left beard outline (crosses y=40 at x~429 like base) + left shoulder silhouette
r.draw_segments(canvas, vb, ids=[6, 17, 37])
line(canvas, [(429, 32), (429, 26)])                 # reach ~10 px past the seam
line(canvas, [(435, 80), (435, 84)])                 # bridge 6 -> 17
line(canvas, [(468, 184), (466, 186)])               # bridge 17 -> 37

# 2. left beard edge against neck skin (v1 had it ~40-80 px too far right)
left_seed = [(467, 186), (474, 200), (479, 215), (484, 232), (491, 247), (498, 261), (505, 276),
             (510, 290), (516, 305), (527, 320), (536, 335), (545, 350), (551, 365), (557, 380),
             (562, 396), (564, 404)]
P = snap2(left_seed, V3, "edge", radius=7, smooth=5)
P[0] = (467, 186)
line(canvas, P)

# 3. right beard edge against the backdrop (base crosses y=40 at x~1141), down to the cloth line
line(canvas, [(1143, 28), (1141, 40), (1138, 50), (1135, 60), (1137, 70), (1139, 81), (1134, 90), (1128, 95)])

# 4. cloth silhouette over the right shoulder
r.draw_segments(canvas, e3, ids=[40, 71])
line(canvas, [(1200, 163), (1202, 164)])
sil_seed = [(1321, 206), (1350, 226), (1375, 248), (1400, 262), (1425, 273), (1450, 287), (1475, 296),
            (1500, 311), (1525, 318), (1550, 328), (1575, 340), (1596, 351), (1606, 356)]
P = snap2(sil_seed, G15, "edge", radius=6, smooth=3)
P[0] = (1321, 206)
line(canvas, P)

# 5. beard / cloth boundary: cloth corner -> apex, then valley_big 33 down to (1016, 207)
diag_seed = [(1108, 100), (1098, 118), (1088, 134), (1078, 150), (1069, 163)]
P = snap2(diag_seed, V3, "edge", radius=5, smooth=4)
line(canvas, np.vstack([[(1122, 93)], P, [(1067, 162)]]))
r.draw_segments(canvas, vb, ids=[33])

# 6. beard edge against the orange neck skin (saturation step)
skin_seed = [(1016, 207), (1009, 222), (1007, 236), (1005, 250), (1003, 265), (996, 280), (985, 295),
             (971, 310), (962, 325), (954, 340), (945, 355), (935, 370), (926, 385), (920, 396), (917, 404)]
P = snap2(skin_seed, S3, "edge", radius=7, smooth=8, post=3)
P[0] = (1016, 207)
line(canvas, P)

# 7. cloth inner edge against neck skin, 8. main cloth fold
r.draw_segments(canvas, vb, ids=[41, 34])

# --------------------------------------------------------------------------- beard texture
# 8-14 clump-scale strokes: dark gaps between the lit locks above, down-left flowing gaps below.
tex = r.blank()
dk = beard_ridges(dark=True)                                   # strict
dk2 = beard_ridges(dark=True, lo=0.25, hi=0.5, min_len=35)     # looser, for the sparser areas
r.draw_segments(tex, dk, ids=[1, 7, 8, 9, 11, 18, 19])
line(tex, [(815, 342), (818, 341)])                  # dark 19 -> 18: one diagonal gap
r.draw_segments(tex, dk2, ids=[27, 26, 30])
for seed in ([(797, 92), (792, 130), (793, 160), (797, 185)],          # channel below the mouth
             [(872, 108), (876, 130), (878, 150), (874, 172)],         # channel, right-centre
             [(952, 58), (962, 80), (970, 100), (978, 122)]):          # gap, upper right
    line(tex, snap2(seed, G25, "dark", radius=6, smooth=5, post=2))
line(tex, [(867, 304), (871, 303), (873, 301)])      # 18 -> dk2 27 (same diagonal, continues up-right)
r.erase_rect(tex, 540, 0, 600, 48)                  # keep texture off the seam (dark 1 top)
r.erase_rect(tex, 898, 333, 913, 345)                # curl at the foot of dk2 26
canvas |= tex

r.save_lines(canvas)
