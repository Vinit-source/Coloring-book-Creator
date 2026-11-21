"""right_ear_cheek -- viewer's right ear, sideburn/cheek boundary, neck silhouette (zoom 4x).

Crop px coordinates throughout (crop = orig x 490-660, y 320-550 at 4x).
Hand-drawn points come from gradient/intensity profiles measured on the crop; they are then
snapped (r.snap with extra smoothing) so each stroke sits on the real image edge.
"""
import sys

sys.path.insert(0, "/Users/Shared/Other/Coloring-book-Creator/dotted-sketch")
import numpy as np  # noqa: E402
from zoomkit import Region  # noqa: E402

r = Region("right_ear_cheek")
canvas = r.blank()


def snapped(pts, mode, radius=None, smooth=3.0):
    """r.snap with a stronger offset smoothing than draw_polyline's default (less jitter)."""
    return r.snap(np.asarray(pts, np.float32), mode, radius, smooth)


def line(P):
    return r.draw_polyline(canvas, P)


# ---------------------------------------------------------------- candidates
# broad valleys: the dark scapha horseshoe and the antihelix / notch grooves
vh = r.candidates("valley", sigma=(6, 9, 12), low=0.15, high=0.35, min_len=40)
# bright grey hair strands (beard texture hints)
lt = r.candidates("light", sigma=(3, 4, 6), min_len=40)

# scapha: 8 = top of the dark crescent, 9 = helix inner fold (groove B),
# 10 + 18 = antihelix edge (groove A), 22 = merged groove down to the notch,
# 26 = notch / antitragus edge
r.draw_segments(canvas, vh, ids=[8, 9, 10, 18, 22, 26])

# ---------------------------------------------------------------- ear contours
# outer helix rim against the background: helix root -> crest -> back -> lobe -> lobe front
rim = [(428, 142), (423, 128), (419, 112), (422, 95), (432, 86), (448, 81), (465, 79), (482, 81),
       (500, 86), (515, 99), (531, 117), (545, 140), (556, 162), (562, 190), (565, 225), (564, 265),
       (563, 300), (560, 322), (556, 342), (551, 362), (545, 381), (536, 402), (525, 420), (514, 440),
       (505, 460), (497, 480), (488, 500), (479, 520), (470, 540), (460, 560), (451, 580), (443, 600),
       (433, 620), (421, 640), (409, 656), (396, 666), (382, 672), (368, 679), (356, 684), (348, 682),
       (344, 670), (343, 650), (342, 630), (344, 610), (345, 590), (348, 572)]
RIM = snapped(rim, "edge", smooth=5)
# the crest's true edge (y~79) grazes the owned top seam (y=80); keep it >=1 px inside so it survives the merge
RIM[:, 1] = np.maximum(RIM[:, 1], 82)
line(RIM)

# front edge of the bright inner ear (helix root -> antihelix front -> concha rim), joins notch stroke 26
# (upper part: left edge of the bright crus/antihelix at x~388; below y~340 the bright ridge recedes
#  to x~420, so the line follows the concha bowl's rim diagonally, then curves into the notch)
front = [(420, 168), (413, 190), (411, 210), (402, 238), (391, 262), (387, 290), (387, 322), (395, 345),
         (410, 362), (420, 385), (424, 410), (418, 430), (404, 445)]
F = line(snapped(front, "edge", smooth=8))
line([tuple(RIM[0]), (425, 155), tuple(F[0])])              # helix root notch
r.erase_rect(canvas, 368, 436, 386, 457)                    # drop 26's stub above the junction
line([tuple(F[-1]), (394, 451), (388, 459)])

# ---------------------------------------------------------------- silhouettes
# neck / beard silhouette against the background below the lobe (ends on v1's seam crossing x~323 @ y=840)
# hair edge is low-contrast and strand-ridden: points measured at half-intensity, drawn unsnapped
corner = min(map(tuple, RIM), key=lambda q: (q[0] - 347) ** 2 + (q[1] - 685) ** 2)   # lobe/head corner
neck = [corner, (345, 700), (348, 715), (347, 732), (342, 752), (341, 770), (347, 790), (351, 806),
        (349, 821), (336, 834), (323, 841), (317, 850)]
line(neck)

# cheek skin / dark sideburn boundary; top bends onto v1's seam crossing (x 333 @ y=80)
cheek = [(320, 110), (320, 160), (318, 210), (317, 260), (314, 290), (308, 312)]
C = line(snapped(cheek, "edge", radius=4, smooth=8))
line([(334, 66), (331, 80), (325, 95), tuple(C[0])])

# ---------------------------------------------------------------- beard / sideburn texture
line(snapped([(350, 98), (354, 125), (354, 150), (351, 175), (347, 195)], "light", radius=4, smooth=6))
r.draw_segments(canvas, lt, ids=[8])

r.save_lines(canvas)
