"""Nose region (zoom 4x): precise line art for the dotted tracing portrait.

All coordinates are crop px (crop origin = original (270, 212), 4x).
Hand-drawn points come from measured intensity/gradient profiles of crop.png
and are snapped onto the real image line (dark = valley centre, edge = step).
"""
import sys

sys.path.insert(0, "/Users/Shared/Other/Coloring-book-Creator/dotted-sketch")
from zoomkit import Region  # noqa: E402

r = Region("nose")
canvas = r.blank()


def snapped(pts, mode, radius, smooth=8.0):
    """Snap with heavier smoothing of the normal offsets (kit default is 3),
    so strokes through hairy / textured areas stay smooth."""
    P = r.snap(pts, mode, radius=radius, smooth=smooth)
    r.draw_polyline(canvas, P)
    return P


# --------------------------------------------------------------------------
# right side of the nose: one continuous shadow valley from the brow head,
# down the nose side, into the alar crease and the nasolabial fold
# --------------------------------------------------------------------------
c6 = r.candidates("valley", sigma=(6, 8, 10), low=0.15, high=0.35, min_len=40)
r.draw_segments(canvas, c6, ids=[35, 57, 71, 79, 86])

# right brow head: upper edge coming in from the right_eye seam, turning down
# into the right side of the nose
P = snapped([(684, 121), (650, 119), (620, 118), (600, 125), (591, 145),
             (592, 175), (596, 205), (599, 228)], "edge", radius=6, smooth=10)
r.draw_polyline(canvas, [tuple(P[-1]), (603, 242)])          # join candidate 35's top end
# bridges between candidate pieces
r.draw_polyline(canvas, [(624, 425), (630, 429)])                      # 57 -> 71
r.draw_polyline(canvas, [(652, 514), (654, 530), (650, 545), (651, 553)])  # 71 -> 79 (valley x 648-656)
r.draw_polyline(canvas, [(710, 641), (712, 651), (714, 661)])              # 79 -> 86
# lower fold: candidate 86 drifts to the soft cheek side of the shadow band; below
# y 668 follow the sharp lip-side edge instead (measured: y720 x756, y732 x764) so the
# fold crosses the y=732 seam near base line 25 (x 756)
r.erase_rect(canvas, 720, 669, 800, 760)
r.draw_polyline(canvas, [(725, 668), (731, 678), (736, 687), (741, 696), (748, 708), (755, 720),
                         (760, 732), (766, 742)])

# --------------------------------------------------------------------------
# left brow head + left side of the nose bridge (shadow edge, fades ~y 350)
# --------------------------------------------------------------------------
snapped([(318, 127), (330, 126), (345, 134), (360, 140), (375, 150), (390, 162),
         (402, 182), (410, 205), (404, 235), (398, 265), (400, 295), (410, 322),
         (416, 345)], "edge", radius=6, smooth=10)

# --------------------------------------------------------------------------
# right eye inner canthus (closes the right_eye almond inside our stem)
# --------------------------------------------------------------------------
r.draw_polyline(canvas, [(684, 300), (670, 308), (660, 318), (653, 326), (665, 331), (684, 331)])

# --------------------------------------------------------------------------
# left alar crease -> splits into left nasolabial fold and left ala rim
# --------------------------------------------------------------------------
# nose-cheek junction (faint above y 570, seen as cheek-texture boundary), then alar crease
P = snapped([(320, 503), (319, 520), (314, 535), (308, 550), (306, 565), (308, 580), (307, 595),
             (305, 612), (302, 628), (296, 642), (287, 653)], "dark", radius=6, smooth=8)
r.draw_polyline(canvas, [tuple(P[-1]), (278, 662)])
r.draw_polyline(canvas, [(278, 662), (270, 675), (262, 688), (250, 705), (236, 716), (222, 728),
                         (215, 742)], snap="dark", radius=8)
P = snapped([(281, 672), (284, 686), (284, 700), (281, 712), (281, 724), (283, 734),
             (288, 743)], "edge", radius=6, smooth=10)
r.draw_polyline(canvas, [(278, 662), tuple(P[0])])

# --------------------------------------------------------------------------
# right ala rim (branches off the fold) and nostrils
# --------------------------------------------------------------------------
P = snapped([(705, 668), (702, 688), (696, 704), (691, 716), (684, 726),
             (676, 735), (665, 742)], "edge", radius=6, smooth=5)
r.draw_polyline(canvas, [(712, 651), tuple(P[0])])            # branch off the fold
r.draw_polyline(canvas, [(583, 727), (595, 714), (610, 705), (625, 697), (645, 692), (665, 692),
                         (680, 698), (692, 707)], snap="edge", radius=6)
r.draw_polyline(canvas, [(583, 727), (600, 733), (625, 736), (650, 738), (668, 737)], snap="edge", radius=5)

# left nostril: open at the bottom (bottom edge lies at y 740-748, mouth's area)
r.draw_polyline(canvas, [(320, 742), (316, 725), (316, 712), (322, 699), (335, 691), (350, 690),
                         (370, 693), (390, 698), (405, 706), (418, 718), (425, 730), (424, 742)],
                snap="edge", radius=6)

# --------------------------------------------------------------------------
# nose tip: lower edge of the lit tip (shading turn)
# --------------------------------------------------------------------------
r.draw_polyline(canvas, [(440, 648), (470, 640), (500, 640), (530, 643), (560, 648)], snap="edge", radius=10)

r.save_lines(canvas)
