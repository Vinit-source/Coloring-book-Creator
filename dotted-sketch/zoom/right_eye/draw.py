"""right_eye -- precise line art for the viewer's right eye, brow, temple hair edge,
head silhouette (ear top left to right_ear_cheek).  Crop px coordinates (4x zoom, origin = orig (418, 212)).

Owned area in crop px: x 80..868, y 80..512.

Seam crossings (from cand_base): lower lid / inner canthus (80, 333) on the left seam;
silhouette (769, 80) and temple hairline (556, 80) on the top seam; temple-hair edge (621, 512)
and silhouette/ear (755, 512) on the bottom seam.  Ear top is right_ear_cheek's.

Workaround (kit unchanged): Region.snap() uses a fixed 2 px blur and magnitude-only edges,
which jitters on soft, hairy boundaries (brow, temple, silhouette).  snap2() below is the
same normal-search idea with a configurable blur, signed edge direction and smoothing.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zoomkit import Region, _densify  # noqa: E402

r = Region("right_eye")
GRAY = r.gray().astype(np.float32)
_blur_cache: dict[float, np.ndarray] = {}


def blurred(s: float) -> np.ndarray:
    if s not in _blur_cache:
        _blur_cache[s] = cv2.GaussianBlur(GRAY, (0, 0), s)
    return _blur_cache[s]


def snap2(pts, mode, radius=6.0, blur=2.0, smooth=4.0, pull=0.15):
    """Move each densified point along its normal onto the darkest pixel ('dark'),
    the strongest edge ('edge'), or a signed edge: 'edge+' = brightness rises along
    +normal, 'edge-' = falls along +normal.  For a left->right path +normal points
    down (+y); for a top->bottom path it points left (-x)."""
    P = _densify(np.asarray(pts, np.float32), 1.0)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-radius, radius + 1, 1.0, dtype=np.float32)
    g = blurred(blur)
    S = P[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    vals = cv2.remap(g, S[..., 0].astype(np.float32), S[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    if mode == "dark":
        cost = vals
    elif mode == "light":
        cost = -vals
    else:
        d = np.gradient(vals, axis=1)
        cost = {"edge": -np.abs(d), "edge+": -d, "edge-": d}[mode]
    span = float(np.ptp(cost)) + 1e-6
    cost = cost + pull * span * (np.abs(offs) / max(radius, 1))[None, :]
    best = offs[np.argmin(cost, axis=1)]
    best = gaussian_filter1d(best, smooth, mode="nearest")
    return P + best[:, None] * nrm


def line(canvas, pts, mode=None, **kw):
    P = snap2(pts, mode, **kw) if mode else np.asarray(pts, np.float32)
    r.draw_polyline(canvas, P, snap=None)
    return P


canvas = r.blank()

# --------------------------------------------------------------------------- #
# candidates
# --------------------------------------------------------------------------- #
edge = r.candidates("edge")
valley = r.candidates("valley")

# ---- upper lid (lash) line: top edge of the eye opening --------------------
r.draw_segments(canvas, edge, ids=[75, 71, 77])
line(canvas, [(70, 333), (62, 336)])                                   # past the seam to base crossing
line(canvas, [(105, 289), (115, 284), (125, 280), (135, 278)], "edge-", radius=5)
line(canvas, [(242, 282), (248, 285)], None)
# outer part: top edge of the opening down into the canthus ...
line(canvas, [(266, 291), (280, 300), (290, 303), (300, 310), (310, 317), (320, 322), (330, 328),
              (340, 334), (348, 340)], "edge-", radius=4, blur=2.5, smooth=5)
# ... then along the dark lash wing to its tip
line(canvas, [(348, 340), (356, 347), (364, 351), (372, 354), (380, 357), (390, 360), (400, 364),
              (410, 366), (420, 368)], "dark", radius=4, blur=2.5, smooth=5)

# ---- lower lid margin: dark lower lash line along the bottom of the opening --
r.draw_segments(canvas, valley, ids=[92, 89, 90, 93, 99])
line(canvas, [(62, 334), (70, 333), (78, 330)])                        # inner canthus to seam crossing
line(canvas, [(97, 329), (108, 329), (118, 328), (128, 327), (139, 325)], "dark", radius=4)
line(canvas, [(159, 323), (167, 320)], None)
line(canvas, [(197, 326), (204, 328), (210, 329)], None)
line(canvas, [(260, 332), (267, 334)], None)
line(canvas, [(297, 336), (308, 336), (318, 336), (328, 335), (336, 334), (340, 334)], "dark", radius=3)

# ---- upper-lid crease (valley centre line) ---------------------------------
r.draw_segments(canvas, valley, ids=[68, 63, 64, 69])
line(canvas, [(117, 250), (122, 247), (126, 244)], None)
line(canvas, [(156, 239), (163, 238)], None)
line(canvas, [(183, 240), (193, 243), (203, 247)], "dark", radius=4)
line(canvas, [(313, 289), (323, 295), (333, 301), (343, 309), (352, 317)],
     "dark", radius=4)

# ---- iris: left and right limbus arcs under the upper lid -------------------
line(canvas, [(161, 281), (159, 291), (157, 301), (157, 311), (158, 320), (161, 328)], "edge", radius=4)
line(canvas, [(278, 299), (280.5, 305), (281, 312), (280, 322), (276, 333)], "edge", radius=4)

# ---- eyebrow: upper and lower boundary of the brow mass --------------------
line(canvas, [(66, 124), (80, 121), (100, 114), (130, 106), (160, 105), (190, 109), (220, 112),
              (260, 122), (300, 135), (340, 162), (380, 186), (420, 214), (445, 245), (460, 278),
              (466, 305)],
     "edge-", radius=10, blur=6, smooth=15)
line(canvas, [(165, 226), (180, 218), (220, 203), (260, 193), (300, 197), (340, 211), (370, 230),
              (392, 255), (405, 280), (418, 302), (432, 312)],
     "edge+", radius=12, blur=5, smooth=10)
# tail tip: the brow mass ends at y ~310-320 for x 420-465 (bottom edge of the dark tail)
line(canvas, [(432, 312), (446, 317), (458, 317), (466, 305)], None)

# ---- under-eye fold (lid / cheek junction) ----------------------------------
# valley 111/115/119/122 lie on it but kink at their joins; one snapped stroke through them
line(canvas, [(83, 362), (94, 367), (104, 371), (114, 378), (124, 382), (134, 384), (152, 387),
              (162, 393), (175, 396), (187, 401), (200, 405), (215, 409), (235, 413), (255, 418),
              (268, 420)], "dark", radius=5, blur=3, smooth=6)

# ---- crow's feet from the outer corner --------------------------------------
r.draw_segments(canvas, valley, ids=[120])
line(canvas, [(404, 384), (414, 389), (424, 394), (427, 396)], "dark", radius=3)

# ---- temple skin / side-hair boundary ---------------------------------------
# Top (y 68..~200): the iso-intensity contour (blur 6) of the skin -> dark-hair edge through the
# agreed seam point (556, 80) = orig (557, 232), where hair_forehead's hairline arrives.  It
# follows the real notch of the edge (x~535 at y 100-125).  Below: the snapped edge, which
# meets the iso line at y~200.  Bottom: bends onto cand_base's crossing (621, 512).
temple = snap2([(540, 156), (542, 168), (545, 180), (546, 192), (550, 204), (552, 216), (557, 228),
                (561, 240), (564, 252), (570, 266), (572, 280), (573, 300), (577, 330), (578, 360),
                (582, 390), (588, 420), (598, 450), (606, 476)],
               "edge+", radius=10, blur=4, smooth=8)
SEAM = (556.0, 80.0)
g6 = blurred(6.0)
lev = float(g6[80, 556])
iso, xp = [], SEAM[0]
for y in range(68, 202, 2):
    row = g6[y, 515:600]
    xs = [515 + i + (lev - row[i]) / (row[i + 1] - row[i])
          for i in range(len(row) - 1) if (row[i] - lev) * (row[i + 1] - lev) <= 0 and row[i] != row[i + 1]]
    xp = min(xs, key=lambda x: abs(x - xp))
    iso.append((xp, float(y)))
iso = np.array(iso, np.float32)
iso[:, 0] = gaussian_filter1d(iso[:, 0], 2.0, mode="nearest")
# pin the seam crossing exactly; the (small) correction fades out by y=130
shift = SEAM[0] - np.interp(80.0, iso[:, 1], iso[:, 0])
iso[:, 0] += shift * np.clip((130.0 - iso[:, 1]) / 50.0, 0.0, 1.0)
# hand over to the snapped edge where the two lines are closest (y 170..200)
cand = [(abs(np.interp(ty, iso[:, 1], iso[:, 0]) - tx), i) for i, (tx, ty) in enumerate(temple) if 170 <= ty <= 200]
k = min(cand)[1]
line(canvas, np.vstack([iso[iso[:, 1] < temple[k, 1] - 3], temple[k:k + 1]]))
line(canvas, temple[k:])
line(canvas, [tuple(temple[-1]), (610, 490), (616, 502), (621, 512), (622, 522)])

# ---- head silhouette: hair vs dark background, down to the ear --------------
# right_ear_cheek owns the ear top (true edge just below the y=512 seam), so the silhouette
# simply runs down to the seam where it meets their ear outline at x~755.
sil = line(canvas, [(768, 68), (769, 80), (770, 90), (772, 120), (776, 150), (777, 180), (772, 210),
                    (766, 240), (762, 270), (763, 300), (763, 330), (760, 360), (755, 378), (755, 392),
                    (758, 406), (757, 420), (750, 436), (746, 450), (750, 466), (752, 482), (754, 497),
                    (755, 504)],
           "edge+", radius=8, blur=2.5, smooth=6)
line(canvas, [tuple(sil[-1]), (sil[-1][0], 518)])            # straight on past the seam

r.save_lines(canvas)
