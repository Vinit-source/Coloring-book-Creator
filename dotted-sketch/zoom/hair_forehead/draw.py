"""hair_forehead -- precise line art for the hair mass + forehead (zoom 3x).

Strokes:
  * outer silhouette  : iso-contour of the blurred photo against the black backdrop
  * hairline          : iso-contour of a skin map (R - B), i.e. exactly where skin stops
  * forehead creases  : valley candidates (left diagonal crease, centre-right crease)
  * hair texture      : bright-clump (light) / gap (valley) candidates inside the hair mass
Long contours are computed from image masks because no candidate set captures them as
one continuous stroke (edge/valley fragment on hair texture; base has loops).
Both long contours are eased into cand_base's seam crossings at y=696 (crop px).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zoomkit import Region, Segments  # noqa: E402
from skimage.filters import apply_hysteresis_threshold, sato  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

r = Region("hair_forehead")
SEAM_Y = 696          # owned boundary (orig y=232) in crop px
YMAX = SEAM_Y + 10    # extend strokes past the seam
BASE_X = dict(sil_l=64, hair_l=232, hair_r=1285, sil_r=1411)   # cand_base crossings at y=696

photo = r.photo().astype(np.float32)
gray = r.gray().astype(np.float32)


# --------------------------------------------------------------------------- helpers
def contour_arc(mask: np.ndarray, ymax: int) -> np.ndarray:
    """Longest run of the largest external contour lying above ymax and off the crop border."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    H, W = mask.shape
    keep = (c[:, 1] < ymax) & (c[:, 0] > 1) & (c[:, 0] < W - 2) & (c[:, 1] > 1)
    if keep.all():
        return c
    s = int(np.argmin(keep))
    k, c = np.roll(keep, -s), np.roll(c, -s, axis=0)
    runs, i = [], 0
    while i < len(k):
        if k[i]:
            j = i
            while j < len(k) and k[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    i, j = max(runs, key=lambda t: t[1] - t[0])
    return c[i:j]


def smooth_arc(P: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([gaussian_filter1d(P[:, 0], sigma, mode="nearest"),
                     gaussian_filter1d(P[:, 1], sigma, mode="nearest")], 1)


def x_at(P: np.ndarray, y: float, end: str) -> float:
    """x where the curve end ('start' or 'end') crosses row y."""
    Q = P if end == "start" else P[::-1]
    for a, b in zip(Q[:-1], Q[1:]):
        if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1]:
            t = (y - a[1]) / (b[1] - a[1])
            return float(a[0] + t * (b[0] - a[0]))
    raise ValueError("curve does not cross seam")


def ease_end(P: np.ndarray, end: str, x_target: float, L: float) -> np.ndarray:
    """Shift x of the curve end so it crosses SEAM_Y at x_target, blended over L px of height."""
    P = P.copy()
    dx = x_target - x_at(P, SEAM_Y, end)
    n = len(P)
    idx = range(n) if end == "start" else range(n - 1, -1, -1)
    for i in idx:
        t = (P[i, 1] - (SEAM_Y - L)) / L
        if t <= 0:
            break                       # left the blend zone (curve heads upward from here)
        t = min(1.0, t)
        P[i, 0] += dx * (t * t * (3 - 2 * t))
    return P


def clip_y(P: np.ndarray, ymax: float) -> np.ndarray:
    return P[P[:, 1] <= ymax]


canvas = r.blank()

# --------------------------------------------------------------------------- 1. outer silhouette
g_bl = cv2.GaussianBlur(gray, (0, 0), 5)
fig = g_bl > 40                      # backdrop ~18, outer hair >= ~76 -> mid-level edge
sil = smooth_arc(contour_arc(fig, YMAX + 30), 28)   # envelope over crown wisps
sil = ease_end(sil, "start", BASE_X["sil_l"] if sil[0, 0] < r.W / 2 else BASE_X["sil_r"], 60)
sil = ease_end(sil, "end", BASE_X["sil_r"] if sil[-1, 0] > r.W / 2 else BASE_X["sil_l"], 60)
sil = clip_y(sil, YMAX)
r.draw_polyline(canvas, sil, snap=None)

# --------------------------------------------------------------------------- 2. hairline (skin edge)
skin = cv2.GaussianBlur(photo[..., 0] - photo[..., 2], (0, 0), 7)
skin_m = skin > 30
hl = smooth_arc(contour_arc(skin_m, YMAX + 30), 10)
# Left end: eased onto cand_base's crossing (x=232). Right end: NOT eased -- it follows the true
# skin edge across the seam (agreed with right_eye: meet at orig (557, 232) = crop (~1251, 696)).
left_end = "start" if hl[0, 0] < hl[-1, 0] else "end"
right_end = "end" if left_end == "start" else "start"
hl = ease_end(hl, left_end, BASE_X["hair_l"], 90)
# true edge crosses at x~1247.8; nudge (+3 px, i.e. 1 orig px, over 90 px) onto the agreed meeting point
hl = ease_end(hl, right_end, r.to_crop(557, 232)[0], 90)
hl_right_x = x_at(hl, SEAM_Y, right_end)
print(f"right hairline crosses y={SEAM_Y} at crop x={hl_right_x:.1f} "
      f"(orig x={r.to_orig(hl_right_x, SEAM_Y)[0]:.1f}, y={r.to_orig(hl_right_x, SEAM_Y)[1]:.0f})")
hl = clip_y(hl, YMAX)
r.draw_polyline(canvas, hl, snap=None)

contours = canvas.copy()

# --------------------------------------------------------------------------- 3. forehead creases
vxl = r.candidates("valley", sigma=(4, 6, 8), low=0.25, high=0.5, min_len=60)
vmid = r.candidates("valley", sigma=(2, 3, 4.5), low=0.12, high=0.3, min_len=25)
creases = r.blank()
r.draw_segments(creases, vxl, ids=[18])          # left diagonal crease
r.draw_segments(creases, vmid, ids=[152, 145])   # its continuation + centre-right crease
r.erase_rect(creases, 0, 560, 428, 700)          # lower-left tail: kinks into the temple shadow
r.erase_rect(creases, 963, 520, 990, 540)        # junction tick at the right end of the crease
canvas |= creases

# --------------------------------------------------------------------------- 4. hair texture
# The kit's light/valley candidates shatter into short pieces on hair (every strand crossing is a
# junction). So: smooth the photo ALONG the local hair direction (structure-tensor orientation,
# line-integral convolution), then take bright (clump) / dark (gap) ridges of that image. The
# ridges are the real clumps, just without the cross-hatching noise -> long flowing strokes.
def flow_candidates(rho: float = 8, L: int = 18):
    gb = cv2.GaussianBlur(gray, (0, 0), 1.5)
    Ix, Iy = cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), rho)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), rho)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), rho)
    th = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2          # along-strand direction
    vx, vy = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
    yy, xx = np.mgrid[0:r.H, 0:r.W].astype(np.float32)
    acc, cnt = gb.copy(), np.ones_like(gb)
    for sgn in (1, -1):
        px, py, dx, dy = xx.copy(), yy.copy(), vx * sgn, vy * sgn
        for k in range(L):
            nx = cv2.remap(vx, px, py, cv2.INTER_LINEAR)
            ny = cv2.remap(vy, px, py, cv2.INTER_LINEAR)
            flip = (nx * dx + ny * dy) < 0
            nx, ny = np.where(flip, -nx, nx), np.where(flip, -ny, ny)
            px, py, dx, dy = px + nx, py + ny, nx, ny
            w = np.exp(-(k / (0.6 * L)) ** 2)
            acc += w * cv2.remap(gb, px, py, cv2.INTER_LINEAR)
            cnt += w
    lic = acc / cnt
    hair = r.figure_mask() & ~skin_m

    def ridge(dark: bool) -> Segments:
        R = sato(lic / 255, sigmas=[3, 4.5], black_ridges=dark) * hair
        R /= np.percentile(R[hair], 99.5)
        return Segments(skeletonize(apply_hysteresis_threshold(R, 0.25, 0.5)), 70)
    return ridge(False), ridge(True)


flow_L, flow_D = flow_candidates()
tex = r.blank()
r.draw_segments(tex, flow_L, ids=[18, 10, 13, 6, 3, 15, 4, 14, 17])
r.draw_segments(tex, flow_D, ids=[35, 6])
r.draw_polyline(tex, [(1183, 332), (1197, 337), (1211, 343)], snap="light", radius=6)   # bridge L14 -> L17
r.erase_rect(tex, 245, 335, 275, 362)     # L18: hook at its upper end
r.erase_rect(tex, 100, 603, 160, 640)     # L18: hook at its lower end
r.erase_rect(tex, 1290, 495, 1318, 530)   # D35: hook at its upper end
# keep texture clear of the two long contours (>= ~1.5 dot spacings)
dist = cv2.distanceTransform((~contours).astype(np.uint8), cv2.DIST_L2, 5)
tex &= dist > 20
canvas |= tex

r.save_lines(canvas)

# seam check
row = np.nonzero(canvas[SEAM_Y])[0]
print("columns at seam row:", row.tolist())
