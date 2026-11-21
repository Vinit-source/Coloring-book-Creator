"""left_eye -- precise line art for the viewer's left eye, brow, temple hair edge.

Owned area (crop px): x 0..848, y 80..512.

Every stroke is a guide polyline (several guides taken from valley-candidate
coordinates, e.g. valley sigma 3/4/6 strokes 40, 65/70/72, 75) refined by `track()`: a dynamic-programming snap that moves each
guide sample along its normal onto the photo feature (valley / signed edge)
while keeping the stroke smooth. This is more robust than the kit's
point-wise `snap` on the soft, upscaled photo (hair texture otherwise makes
the snapped line jitter), so it is used instead of `draw_polyline(snap=...)`;
the result is still drawn with `r.draw_polyline(..., snap=None)`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from zoomkit import Region, _densify  # noqa: E402

r = Region("left_eye")
PHOTO = r.photo().astype(np.float32)
GRAY = r.gray().astype(np.float32)
CHROMA = PHOTO[..., 0] - PHOTO[..., 2]          # R-B: skin high, grey hair / brow low

_blur_cache: dict = {}


def _blurred(src: str, sigma: float) -> np.ndarray:
    key = (src, sigma)
    if key not in _blur_cache:
        base = GRAY if src == "gray" else CHROMA
        _blur_cache[key] = cv2.GaussianBlur(base, (0, 0), sigma)
    return _blur_cache[key]


def track(guide, mode="dark", sigma=3.0, radius=10, lam=0.03, mu=0.05, smooth=2.5,
          dark_side=1, src="gray", step=2.0, pin_start=False, pin_end=False):
    """Refine a guide polyline onto the photo.

    mode 'dark'  -> follow the intensity valley (crease / lash line centre)
    mode 'edge'  -> follow the strongest edge whose dark side is +normal
                    (dark_side=+1) or -normal (dark_side=-1); for a guide drawn
                    left->right the normal points DOWN, top->bottom it points LEFT.
    DP over offsets along the normal with a quadratic smoothness penalty `lam`
    and a small pull `mu` toward the guide.  pin_* keeps an endpoint on the guide
    (used where a seam must meet the neighbour's crossing point).
    """
    P = _densify(np.asarray(guide, np.float32), step)
    Ps = gaussian_filter1d(P, 4, axis=0, mode="nearest")
    t = np.gradient(Ps, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-radius, radius + 1, 1.0, dtype=np.float32)
    S = P[:, None, :] + offs[None, :, None] * n[:, None, :]
    sx, sy = S[..., 0].astype(np.float32), S[..., 1].astype(np.float32)
    g = _blurred(src, sigma)
    if mode == "dark":
        C = cv2.remap(g, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    elif mode == "edge":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
        Cx = cv2.remap(gx, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Cy = cv2.remap(gy, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        C = dark_side * (Cx * n[:, None, 0] + Cy * n[:, None, 1])
    else:
        raise ValueError(mode)
    C = (C - C.min()) / (np.ptp(C) + 1e-6)
    C = C + mu * np.abs(offs)[None, :] / radius
    N, K = C.shape
    trans = lam * (offs[:, None] - offs[None, :]) ** 2          # [prev, cur]
    D = C[0].copy()
    if pin_start:
        D = D + 10 * (np.abs(offs) > 1)
    back = np.zeros((N, K), np.int32)
    for i in range(1, N):
        tot = D[:, None] + trans
        back[i] = np.argmin(tot, axis=0)
        D = tot[back[i], np.arange(K)] + C[i]
    if pin_end:
        D = D + 10 * (np.abs(offs) > 1)
    k = int(np.argmin(D))
    path = np.zeros(N, np.int32)
    for i in range(N - 1, -1, -1):
        path[i] = k
        k = back[i, k]
    o = gaussian_filter1d(offs[path], smooth, mode="nearest")
    if pin_start or pin_end:                      # fade offsets to 0 at pinned ends
        w = np.ones(N, np.float32)
        ramp = np.linspace(0, 1, min(N, 12), dtype=np.float32)
        if pin_start:
            w[:len(ramp)] = np.minimum(w[:len(ramp)], ramp)
        if pin_end:
            w[-len(ramp):] = np.minimum(w[-len(ramp):], ramp[::-1])
        o = o * w
    return P + o[:, None] * n


canvas = r.blank()
STROKES: dict[str, np.ndarray] = {}


def add(name, pts):
    STROKES[name] = np.asarray(pts, np.float32)
    r.draw_polyline(canvas, pts, snap=None)
    if "--debug" in sys.argv:
        P = STROKES[name]
        k = max(1, len(P) // 16)
        print(f"{name:12s}", " ".join(f"({x:.0f},{y:.0f})" for x, y in P[::k]))


# --------------------------------------------------------------------------- #
# 1. head silhouette: grey hair against the dark backdrop (bg on the left)
#    meets the hair_forehead seam at base crossing (85, 80); runs down to the
#    ear top (owned by left_ear_cheek just below y=512).
# --------------------------------------------------------------------------- #
add("silhouette", track(
    [(85, 68), (85, 80), (86, 120), (86, 160), (88, 200), (92, 240), (94, 280),
     (104, 320), (110, 360), (113, 400), (118, 430), (128, 460), (134, 490), (137, 524)],
    mode="edge", sigma=4, radius=12, dark_side=+1, pin_start=True))

# --------------------------------------------------------------------------- #
# 2. side-hair / temple-skin boundary (hair = low R-B chroma, on the left).
#    top pinned to the base crossing (309, 80), bottom near base crossing (265, 512)
# --------------------------------------------------------------------------- #
#    Near the top the temple skin is in deep shadow (dark brown, x~310-340):
#    the boundary follows where grey/black hair turns brown, not the shadow edge.
hs_top = track(
    [(308, 68), (309, 80), (311, 110), (312, 140), (310, 180), (304, 220), (300, 260)],
    mode="edge", src="chroma", sigma=4, radius=5, lam=0.08, dark_side=+1, pin_start=True)
hs_low = track(
    [(300, 260), (293, 300), (295, 340), (296, 380), (290, 420), (282, 460), (276, 490),
     (266, 512), (264, 524)],
    mode="edge", src="chroma", sigma=4, radius=12, dark_side=+1, pin_end=True)
add("hair_skin", np.vstack([hs_top, hs_low[1:]]))

# --------------------------------------------------------------------------- #
# 3. eyebrow outline: upper edge (dark below), lower edge (dark above), joined
#    at the tail; the brow head continues into the nose region past x=848 --
#    upper edge pinned to the nose agent's brow-head crossing (848, 128).
# --------------------------------------------------------------------------- #
brow_up = track(
    [(424, 312), (445, 282), (470, 255), (500, 228), (530, 205), (560, 178), (590, 155),
     (620, 143), (660, 141), (700, 144), (740, 144), (780, 138), (815, 130), (848, 128),
     (858, 128)],
    mode="edge", src="chroma", sigma=4, radius=14, lam=0.08, dark_side=+1, pin_end=True)
brow_lo = track(
    [(428, 334), (445, 330), (460, 320), (480, 307), (500, 290), (520, 270), (540, 253),
     (560, 245), (580, 232), (600, 227), (640, 220), (680, 224), (720, 230), (760, 236),
     (800, 240)],
    mode="edge", sigma=4, radius=12, lam=0.08, dark_side=-1)
add("brow_up", brow_up)
# medially (x > 712) the brow's lower edge dissolves into the lid-crease shadow
# and would run ~1 dot spacing above the crease -> stop it there
add("brow_lo", brow_lo[brow_lo[:, 0] <= 712])
add("brow_tail", [tuple(brow_up[0]), (418, 322), tuple(brow_lo[0])])

# --------------------------------------------------------------------------- #
# 4. eye
# --------------------------------------------------------------------------- #
# upper lid line: inner corner -> lash line over the iris -> merges with the
# hooded-lid crease line toward the outer corner (they coincide for x < 570)
upper_lid = track(
    [(824, 336), (810, 318), (790, 305), (760, 296), (720, 286), (680, 280), (640, 283),
     (620, 289), (600, 294), (585, 296), (570, 297), (555, 301), (540, 310), (515, 329),
     (494, 352), (478, 375), (462, 388), (446, 395)],
    mode="edge", sigma=2.5, radius=8, lam=0.05, dark_side=-1)
add("upper_lid", upper_lid)

# lower lid margin: bottom of the dark fissure, outer corner -> inner corner
lower_lid = track(
    [(446, 397), (460, 400), (480, 399), (500, 390), (520, 378), (540, 368), (560, 366),
     (580, 364), (600, 358), (620, 353), (640, 349), (660, 347), (680, 343), (700, 342),
     (720, 341), (750, 338), (785, 339), (810, 340), (824, 337)],
    mode="edge", sigma=2.5, radius=7, dark_side=-1)
OUTER = upper_lid[-1]                                        # outer canthus point
lower_lid = np.vstack([OUTER[None], lower_lid[lower_lid[:, 0] >= OUTER[0] + 14],
                       upper_lid[:1]])                       # join both canthi
add("lower_lid", lower_lid)

# upper-lid crease (valley), from where it separates from the lid line
add("crease", track(
    [(648, 263), (660, 261), (680, 258), (700, 257), (720, 255), (740, 254), (760, 258),
     (780, 266), (800, 276), (812, 285)],
    mode="dark", sigma=2.5, radius=7))

# iris: left and right limbus arcs between the lids (dark iris inside)
add("iris_l", track([(606, 296), (599, 310), (596, 326), (599, 340), (602, 352)],
                    mode="edge", sigma=2, radius=6, dark_side=-1))
add("iris_r", track([(712, 287), (717, 302), (718, 318), (713, 334), (707, 342)],
                    mode="edge", sigma=2, radius=6, dark_side=+1))

# --------------------------------------------------------------------------- #
# 5. outer-corner crow's foot + under-eye folds (valleys)
# --------------------------------------------------------------------------- #
crow = track([(450, 396), (430, 399), (410, 402), (392, 409), (375, 412), (356, 420)],
             mode="dark", sigma=3, radius=8)
add("crowsfoot", np.vstack([OUTER[None], crow[crow[:, 0] <= OUTER[0] - 10]]))
add("bag_fold", track(
    [(522, 417), (560, 422), (600, 428), (630, 430), (653, 425), (683, 412), (715, 403),
     (743, 393), (770, 384), (795, 373)],
    mode="dark", sigma=3, radius=8))
diag = track(
    [(439, 524), (463, 509), (487, 493), (511, 478), (535, 466), (559, 457), (583, 451),
     (607, 441)],
    mode="dark", sigma=3, radius=8)
add("fold_diag", diag[diag[:, 0] <= 588])        # stop before it runs into bag_fold (smear)

r.save_lines(canvas)
