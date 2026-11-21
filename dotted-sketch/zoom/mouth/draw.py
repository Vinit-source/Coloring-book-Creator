"""Mouth region (zoom 4x): mustache, lips, teeth, mouth corners, cheek/beard boundaries,
nasolabial folds and a few beard-flow strokes.

Most of the lip / beard boundaries here are *colour* edges (lip redness = LAB a*, skin vs
grey hair = HSV saturation) rather than luminance edges, so besides the kit's gray-only
`snap` this script has its own `snap_field` (same algorithm, any cost map, optional signed
gradient) and `iso_path` (a smoothed iso-contour of a blurred channel between two anchors).
Every stroke below is either one of those (guide polyline -> snapped onto the real edge)
or a kit `draw_polyline(..., snap=...)`.
"""
import sys

sys.path.insert(0, "/Users/Shared/Other/Coloring-book-Creator/dotted-sketch")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from zoomkit import Region, _densify  # noqa: E402

r = Region("mouth")
photo = r.photo()
_hsv = cv2.cvtColor(photo, cv2.COLOR_RGB2HSV).astype(np.float32)
_lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
CH = {"S": _hsv[..., 1], "L": _lab[..., 0], "A": _lab[..., 1], "B": _lab[..., 2]}
_cache: dict = {}


def F(ch: str, sig: float) -> np.ndarray:
    k = (ch, sig)
    if k not in _cache:
        _cache[k] = cv2.GaussianBlur(CH[ch], (0, 0), sig)
    return _cache[k]


def snap_field(pts, field, mode, radius=10.0, smooth=4.0, post=0.0):
    """Move each point of a guide along its normal (+-radius) onto the best pixel of
    `field`.  mode: dark | light | down | up | abs.  For the gradient modes the normal is
    n = (-t_y, t_x) (points DOWN for a left->right guide, LEFT for a top->bottom guide):
      down = field decreases along n, up = field increases along n, abs = strongest edge."""
    P = _densify(np.asarray(pts, np.float32), 1.0)
    t = np.gradient(gaussian_filter1d(P, 3, axis=0, mode="nearest"), axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-radius, radius + 1, 1.0, dtype=np.float32)
    S = P[:, None, :] + offs[None, :, None] * n[:, None, :]
    sx, sy = S[..., 0].astype(np.float32), S[..., 1].astype(np.float32)
    if mode in ("dark", "light"):
        v = cv2.remap(field, sx, sy, cv2.INTER_LINEAR)
        vals = v if mode == "dark" else -v
    else:
        gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        d = (cv2.remap(gx, sx, sy, cv2.INTER_LINEAR) * n[:, None, 0]
             + cv2.remap(gy, sx, sy, cv2.INTER_LINEAR) * n[:, None, 1])
        vals = {"down": d, "up": -d, "abs": -np.abs(d)}[mode]
    span = float(np.ptp(vals)) + 1e-6
    vals = vals + 0.15 * span * (np.abs(offs) / max(radius, 1))[None, :]
    best = gaussian_filter1d(offs[np.argmin(vals, axis=1)], smooth, mode="nearest")
    Q = P + best[:, None] * n
    if post:
        Q = gaussian_filter1d(Q, post, axis=0, mode="nearest")
    return Q


def iso_path(field, level, a, b, via, smooth=6.0):
    """Arc of the iso-contour {field = level} from the point nearest `a` to the point
    nearest `b`, taking the side of the closed contour that passes nearest `via`."""
    m = (field > level).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    a, b, via = (np.asarray(p, np.float32) for p in (a, b, via))
    best = None
    for c in cs:
        k = c[:, 0].astype(np.float32)
        da = np.linalg.norm(k - a, axis=1).min()
        db = np.linalg.norm(k - b, axis=1).min()
        if best is None or da + db < best[0]:
            best = (da + db, k)
    k = best[1]
    ia = int(np.argmin(np.linalg.norm(k - a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(k - b, axis=1)))
    n = len(k)
    fwd = k[[(ia + j) % n for j in range((ib - ia) % n + 1)]]
    bwd = k[[(ia - j) % n for j in range((ia - ib) % n + 1)]]
    dv = [np.linalg.norm(p - via, axis=1).min() for p in (fwd, bwd)]
    path = fwd if dv[0] <= dv[1] else bwd
    return gaussian_filter1d(path, smooth, axis=0, mode="nearest")


canvas = r.blank()
STROKES: dict[str, np.ndarray] = {}


def put(name, P):
    P = np.asarray(P, np.float32)
    STROKES[name] = P
    r.draw_polyline(canvas, P, snap=None)
    return P


def join(*parts):
    return np.concatenate([np.asarray(p, np.float32) for p in parts], 0)


# --------------------------------------------------------------------------- #
# 1. Cheek-skin / beard boundary ("U" on each side) + mustache top under the nose,
#    one continuous line.  U arms = saturation iso-line (skin is saturated, grey hair
#    is not); level chosen so the left arm meets the left_ear_cheek seam where v1 did
#    (x=80, y~344) and the right arm leaves near the top-right corner (x=1439, y~90).
# --------------------------------------------------------------------------- #
S10 = F("S", 10)
u_left = iso_path(S10, 57, a=(66, 330), b=(436, 201), via=(210, 445), smooth=20)
u_right = iso_path(S10, 57, a=(1094, 209), b=(1450, 60), via=(1250, 470), smooth=30)
# mustache top edge: skin above (light) -> hair below (dark); L gradient, smoothed hard
must_top = snap_field(
    [(436, 201), (462, 178), (495, 157), (530, 143), (565, 137), (600, 137), (640, 139),
     (672, 146), (705, 153), (740, 155), (775, 150), (810, 138), (850, 129), (895, 127),
     (940, 129), (975, 140), (1010, 160), (1050, 183), (1094, 209)],
    F("L", 5), "down", radius=12, smooth=12, post=4)
put("M_line", join(u_left, must_top, u_right))

# --------------------------------------------------------------------------- #
# 2. Nasolabial folds (dark crease from the nose wing down into the M line)
# --------------------------------------------------------------------------- #
M = STROKES["M_line"]


def nearest_on(P, q):
    P = np.asarray(P, np.float32)
    return P[int(np.argmin(np.linalg.norm(P - np.asarray(q, np.float32), axis=1)))]


# right fold: clear crease from the nose wing; stop where it starts to run alongside the
# M line (closer than ~1.5 dot spacings) instead of drawing two parallel lines
# Its top is eased onto v1's seam crossing (x=996 at y=80) so it meets the nose region's line.
fr = snap_field([(1022, 92), (1032, 106), (1042, 120), (1056, 140), (1066, 158)],
                F("L", 4), "dark", radius=10, smooth=8, post=3)
put("fold_R", gaussian_filter1d(join(_densify(np.float32([(972, 60), (985, 70), (996, 80),
                                                           (1010, 87)]), 1.0), fr),
                                3, axis=0, mode="nearest"))
# left fold: orange cheek -> paler band beside the mouth
# (it fades out where it would run alongside the U arm, ~1.5 dot spacings away)
put("fold_L", snap_field([(470, 70), (462, 82), (444, 100), (426, 120), (409, 140),
                          (391, 160), (377, 180), (368, 198)],
                         F("L", 5), "dark", radius=10, smooth=10, post=4))

# --------------------------------------------------------------------------- #
# 3. Nose-wing bottoms that dip below the nose seam (y=80)
# --------------------------------------------------------------------------- #
put("ala_L", snap_field([(519, 66), (522, 80), (528, 96), (540, 108), (553, 116),
                         (563, 110), (572, 98), (592, 93), (620, 95), (645, 90), (660, 84),
                         (664, 70)],
                        F("L", 2), "abs", radius=6, smooth=4, post=2))
# left nostril's outer rim: the nose region's nostril loop reaches the seam at ~(556, 80);
# carry it down the dark rim onto the nostril-sill line above
nl = snap_field([(555, 70), (556, 80), (559, 88), (564, 94)], F("L", 2), "abs", radius=4,
                smooth=3)
put("nostril_L", join(nl, [nearest_on(STROKES["ala_L"], (572, 97))]))
put("nostril_R", snap_field([(812, 68), (822, 82), (848, 84), (878, 88), (900, 91),
                             (915, 85), (921, 70)],
                            F("L", 2), "abs", radius=6, smooth=4, post=2))

# --------------------------------------------------------------------------- #
# 4. Lips
# --------------------------------------------------------------------------- #
CORNER_L, CORNER_R = (486, 400), (1036, 385)
# upper lip lower edge (top of the mouth opening; teeth hang from it)
UB = put("ulip_bot", snap_field(
    [CORNER_L, (510, 385), (540, 368), (570, 355), (600, 345), (630, 337), (660, 335),
     (700, 336), (740, 334), (770, 328), (800, 324), (830, 326), (860, 328), (890, 331),
     (920, 338), (950, 346), (980, 357), (1010, 370), CORNER_R],
    F("A", 3), "down", radius=8, smooth=10, post=4))
# upper lip top = mustache lower edge: grey hair -> red lip (a* rises inward)
ut = (snap_field(
    [(552, 352), (554, 320), (566, 280), (592, 252), (630, 236), (680, 224), (730, 220),
     (780, 214), (830, 214), (875, 221), (910, 234), (937, 254), (955, 280), (963, 310),
     (965, 338)],
    F("A", 4), "up", radius=12, smooth=10, post=4))
# the mustache arms come down onto the upper-lip line
put("ulip_top", join([nearest_on(UB, ut[0] + (0, 10))], ut, [nearest_on(UB, ut[-1] + (0, 14))]))
# lower lip top edge = lip parting line (red lip begins below the dark slit)
put("llip_top", snap_field(
    [CORNER_L, (505, 410), (525, 406), (550, 401), (575, 400), (600, 398), (625, 397),
     (650, 397), (675, 401), (700, 405), (725, 409), (750, 410), (775, 407), (800, 401),
     (825, 395), (850, 386), (875, 382), (900, 383), (925, 379), (950, 378), (975, 379),
     (1000, 382), (1020, 384), CORNER_R],
    F("A", 2), "up", radius=6, smooth=8, post=3))
# lower lip bottom edge (red lip -> beard)
put("llip_bot", snap_field(
    [CORNER_L, (494, 425), (505, 455), (522, 482), (545, 505), (570, 517), (595, 531),
     (620, 538), (670, 547), (720, 548), (770, 544), (820, 540), (870, 531), (920, 517),
     (960, 503), (992, 482), (1012, 455), (1020, 425), (1027, 400), CORNER_R],
    F("A", 3), "down", radius=10, smooth=8, post=3))

# --------------------------------------------------------------------------- #
# 5. Teeth (hang from ulip_bot): sides, bottoms, gaps between the main teeth
# --------------------------------------------------------------------------- #
L15 = F("L", 1.5)
def hang(P, first=True, last=False):
    P = np.asarray(P, np.float32).copy()
    if first:
        P = join([nearest_on(UB, P[0])], P)
    if last:
        P = join(P, [nearest_on(UB, P[-1])])
    return P


put("tooth12", hang(snap_field([(638, 336), (639, 352), (643, 368), (652, 377), (670, 378),
                           (690, 376), (705, 371), (725, 369), (740, 366), (750, 362)],
                          L15, "abs", radius=5, smooth=3, post=1.5)))
put("gap12", hang(snap_field([(694, 340), (694, 357), (695, 373)], L15, "dark", radius=4,
                             smooth=3)))
put("gap23", hang(snap_field([(751, 333), (751, 345), (752, 356)], L15, "dark", radius=5,
                             smooth=3)))
put("tooth34", hang(snap_field([(752, 352), (760, 357), (775, 360), (800, 362), (822, 360),
                           (835, 354), (840, 350), (852, 350), (864, 348), (869, 340),
                           (869, 330)],
                          L15, "abs", radius=5, smooth=3, post=1.5), first=False, last=True))

# --------------------------------------------------------------------------- #
# 6. Beard flow (texture hints, >= 1.5 dot spacings from every contour): hand guides
#    along the local hair direction, pulled onto a nearby bright strand.
# --------------------------------------------------------------------------- #
L3 = F("L", 3)
FLOW = {
    "flow_L_mustache": [(440, 252), (422, 300), (405, 350), (392, 400), (383, 445)],
    "flow_L_corner":   [(445, 455), (428, 505), (410, 555), (398, 600), (392, 632)],
    "flow_L_cheek":    [(262, 482), (240, 530), (218, 575), (202, 632)],
    "flow_L_far":      [(140, 472), (124, 520), (110, 570), (100, 632)],
    "flow_R_mustache": [(1074, 250), (1092, 305), (1106, 365), (1116, 425), (1122, 490)],
    "flow_R_low":      [(1060, 520), (1068, 570), (1075, 632)],
    "flow_R_cheek":    [(1228, 522), (1244, 570), (1258, 632)],
    "flow_R_far":      [(1338, 482), (1358, 535), (1374, 585), (1384, 632)],
}
for nm, g in FLOW.items():
    put(nm, snap_field(g, L3, "light", radius=8, smooth=10, post=6))

r.save_lines(canvas)
if len(sys.argv) > 1:                       # debug: dump stroke polylines for the smear check
    np.save(sys.argv[1], STROKES, allow_pickle=True)
