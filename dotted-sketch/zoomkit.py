"""
Zoom kit -- high-precision, per-region line drawing for the dotted sketch (v2).

The portrait is partitioned into non-overlapping *owned* regions. For each
region we cut a zoomed crop (2-4x), compute candidate strokes from the real
image (so lines sit on true edges, not eyeballed positions), and let a
drawing script (`zoom/<region>/draw.py`) choose / erase / add strokes and
write `zoom/<region>/lines.png`. `merge` stitches every region onto a 4x
global canvas, thins it, and lays dots along each stroke.

CLI (run from repo root):
  K="uv run -q --with opencv-python-headless --with scikit-image --with pillow python dotted-sketch/zoomkit.py"
  $K prepare                         # crops, grids, baseline for all regions
  $K candidates <region> [--kind valley|edge|light|base] [--sigma ..] [--low ..] [--high ..] [--min-len ..]
  $K preview <region>                # overlay of lines.png on the zoomed photo + dotted preview
  $K selftest                        # lines.png := baseline everywhere, merge, compare to v1
  $K merge                           # final v2 outputs

Python API (inside zoom/<region>/draw.py):
  import sys; sys.path.insert(0, "<repo>/dotted-sketch"); from zoomkit import Region
  r = Region("left_eye")
  c = r.candidates("valley", sigma=(2, 3, 4), low=0.2, high=0.45, min_len=20)
  canvas = r.blank()
  r.draw_segments(canvas, c, ids=[3, 7, 12])          # or ids="all", exclude=[...]
  r.draw_polyline(canvas, [(x, y), ...], snap="dark")  # crop-pixel coords; snap: dark|light|edge|None
  r.erase_poly(canvas, [(x, y), ...])
  r.save_lines(canvas)                                 # writes lines.png + previews
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import make_dotted_sketch as v1  # noqa: E402  (reuse loaders, renderers, DFS ordering)

SRC = ROOT / "input" / "sri-ramakrishna.png"
ZOOM_DIR = ROOT / "zoom"
OUT_DIR = ROOT / "output"
BASELINE_SKEL = ZOOM_DIR / "_baseline" / "debug-4-skeleton.png"

G = 4                    # global merge canvas scale (x original pixels)
DOT_SPACING_1X = 4.5     # distance between dots along a stroke, in original px
DOT_R_1X = 1.05          # dot radius, in original px
SUPPRESS = 0.5           # dots from different strokes merge only if closer than this x spacing
STROKE = 3               # line thickness (px) written into lines.png at crop zoom

# owned: list of (x0, y0, x1, y1) in ORIGINAL pixels, half-open; together they tile the image.
# crop : (x0, y0, x1, y1) in ORIGINAL pixels -- owned area + context margin.
REGIONS: dict[str, dict] = {
    "hair_forehead":   dict(zoom=3, crop=(140, 0, 660, 252),  owned=[(0, 0, 798, 232)]),
    "left_eye":        dict(zoom=4, crop=(140, 212, 372, 360), owned=[(0, 232, 352, 340)]),
    "right_eye":       dict(zoom=4, crop=(418, 212, 635, 360), owned=[(438, 232, 798, 340)]),
    "nose":            dict(zoom=4, crop=(270, 212, 530, 415), owned=[(352, 232, 438, 340), (290, 340, 510, 395)]),
    "mouth":           dict(zoom=4, crop=(210, 375, 590, 550), owned=[(230, 395, 570, 530)]),
    "left_ear_cheek":  dict(zoom=4, crop=(130, 320, 310, 550), owned=[(0, 340, 290, 395), (0, 395, 230, 530)]),
    "right_ear_cheek": dict(zoom=4, crop=(490, 320, 660, 550), owned=[(510, 340, 798, 395), (570, 395, 798, 530)]),
    "beard_body":      dict(zoom=2, crop=(0, 510, 798, 708),  owned=[(0, 530, 798, 708)]),
}

_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_PALETTE = [(230, 25, 75), (60, 140, 50), (0, 90, 200), (245, 130, 48), (145, 30, 180),
            (0, 150, 150), (200, 0, 160), (120, 90, 0), (0, 0, 128), (170, 110, 40)]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
_SRC_CACHE: dict[int, np.ndarray] = {}


def source_rgb(zoom: int = 1) -> np.ndarray:
    if zoom not in _SRC_CACHE:
        base = v1.load_rgb(SRC, 1.0)
        _SRC_CACHE[zoom] = base if zoom == 1 else cv2.resize(
            base, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LANCZOS4)
    return _SRC_CACHE[zoom]


def image_size() -> tuple[int, int]:
    h, w = source_rgb(1).shape[:2]
    return w, h


def neighbor_count(sk: np.ndarray) -> np.ndarray:
    p = np.pad(sk.astype(np.uint8), 1)
    H, W = sk.shape
    return sum(p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in _NB8)


def owned_label_map(scale: int = 1) -> np.ndarray:
    """Label map (0 = unowned, i+1 = REGIONS index i) at `scale` x original."""
    w, h = image_size()
    lab = np.zeros((h * scale, w * scale), np.int32)
    for i, cfg in enumerate(REGIONS.values()):
        for x0, y0, x1, y1 in cfg["owned"]:
            lab[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = i + 1
    return lab


class Segments:
    """Skeleton broken at junctions into individually addressable strokes."""

    def __init__(self, skel: np.ndarray, min_len: int):
        deg = neighbor_count(skel)
        junction = skel & (deg >= 3)
        body = skel & ~junction
        n, lab, stats, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_len]
        remap = np.zeros(n, np.int32)
        for new_id, old in enumerate(keep, start=1):
            remap[old] = new_id
        self.label = remap[lab]                       # per-pixel segment id (0 = none)
        self.ids = list(range(1, len(keep) + 1))
        self.length = {i: int(stats[old, cv2.CC_STAT_AREA]) for i, old in zip(self.ids, keep)}
        # attach junction pixels to every segment that touches them so picked strokes stay joined
        self.junctions: dict[int, list[tuple[int, int]]] = {i: [] for i in self.ids}
        H, W = skel.shape
        for y, x in zip(*np.nonzero(junction)):
            touching = {int(self.label[y + dy, x + dx]) for dy, dx in _NB8
                        if 0 <= y + dy < H and 0 <= x + dx < W and self.label[y + dy, x + dx] > 0}
            for t in touching:
                self.junctions[t].append((y, x))

    def mask(self, ids="all", exclude=()) -> np.ndarray:
        chosen = set(self.ids if ids == "all" else ids) - set(exclude)
        unknown = chosen - set(self.ids)
        if unknown:
            raise ValueError(f"unknown segment ids: {sorted(unknown)}")
        m = np.isin(self.label, list(chosen))
        for i in chosen:
            for y, x in self.junctions[i]:
                m[y, x] = True
        return m


# --------------------------------------------------------------------------- #
# dots (v2): even arc-length spacing per stroke; cross-stroke merge only when
# dots nearly coincide, so close parallel details (lid + crease) both survive
# --------------------------------------------------------------------------- #
def place_dots_v2(sk: np.ndarray, spacing: float, suppress: float = SUPPRESS) -> list[tuple[float, float]]:
    rad = max(1.0, suppress * spacing)
    r2, cell = rad * rad, rad
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    dots: list[tuple[float, float]] = []

    def free(y: int, x: int) -> bool:
        cy, cx = int(y // cell), int(x // cell)
        for gy in (cy - 1, cy, cy + 1):
            for gx in (cx - 1, cx, cx + 1):
                for py, px in grid.get((gy, gx), ()):
                    if (py - y) ** 2 + (px - x) ** 2 < r2:
                        return False
        return True

    def add(y: int, x: int) -> None:
        grid.setdefault((int(y // cell), int(x // cell)), []).append((y, x))
        dots.append((float(x), float(y)))

    prev = None
    acc = 0.0
    for y, x in v1.ordered_pixels(sk):
        jump = prev is None or max(abs(y - prev[0]), abs(x - prev[1])) > 1
        if jump:
            if prev is not None and acc > 0.5 * spacing and free(*prev):
                add(*prev)                         # crisp stroke end
            acc = 0.0
            if free(y, x):
                add(y, x)                          # stroke start
        else:
            acc += float(np.hypot(y - prev[0], x - prev[1]))
            if acc >= spacing:
                if free(y, x):
                    add(y, x)
                acc = 0.0
        prev = (y, x)
    if prev is not None and acc > 0.5 * spacing and free(*prev):
        add(*prev)
    return dots


def render_dots(dots, w: int, h: int, r: float, bg: np.ndarray | None = None) -> np.ndarray:
    """RGB render; optional faded photo underlay."""
    img = np.full((h, w, 3), 255, np.uint8) if bg is None else bg.copy()
    for x, y in dots:
        cv2.circle(img, (int(round(x)), int(round(y))), max(1, int(round(r))), (0, 0, 0), -1, cv2.LINE_AA)
    return img


def faded(rgb: np.ndarray, keep: float = 0.45) -> np.ndarray:
    return (rgb.astype(np.float32) * keep + 255 * (1 - keep)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Region
# --------------------------------------------------------------------------- #
class Region:
    def __init__(self, name: str):
        if name not in REGIONS:
            raise SystemExit(f"unknown region {name!r}; choose from {list(REGIONS)}")
        cfg = REGIONS[name]
        self.name, self.z, self.crop, self.owned = name, cfg["zoom"], cfg["crop"], cfg["owned"]
        self.dir = ZOOM_DIR / name
        x0, y0, x1, y1 = self.crop
        self.W, self.H = (x1 - x0) * self.z, (y1 - y0) * self.z

    # ---- coordinates -------------------------------------------------------
    def to_crop(self, x: float, y: float) -> tuple[float, float]:
        """original px -> crop px"""
        return (x - self.crop[0]) * self.z, (y - self.crop[1]) * self.z

    def to_orig(self, x: float, y: float) -> tuple[float, float]:
        """crop px -> original px"""
        return x / self.z + self.crop[0], y / self.z + self.crop[1]

    # ---- images ------------------------------------------------------------
    def photo(self) -> np.ndarray:
        x0, y0, x1, y1 = self.crop
        z = self.z
        return source_rgb(z)[y0 * z:y1 * z, x0 * z:x1 * z].copy()

    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.photo(), cv2.COLOR_RGB2GRAY)

    def owned_mask(self) -> np.ndarray:
        m = np.zeros((self.H, self.W), bool)
        for ox0, oy0, ox1, oy1 in self.owned:
            ax0, ay0 = self.to_crop(ox0, oy0)
            ax1, ay1 = self.to_crop(ox1, oy1)
            m[max(0, int(ay0)):max(0, int(ay1)), max(0, int(ax0)):max(0, int(ax1))] = True
        return m

    def figure_mask(self) -> np.ndarray:
        """Where the person is (vs. the dark backdrop), in crop frame."""
        hsv = cv2.cvtColor(self.photo(), cv2.COLOR_RGB2HSV)
        v = cv2.GaussianBlur(hsv[..., 2].astype(np.float32), (0, 0), 2 * self.z)
        return v > 34

    def decorate(self, rgb: np.ndarray, grid: bool = True) -> np.ndarray:
        """Shade the not-owned area, outline the owned area, add a labelled crop-px grid."""
        out = rgb.copy()
        own = self.owned_mask()
        out[~own] = (out[~own].astype(np.float32) * 0.55 + np.array([150, 150, 150]) * 0.45).astype(np.uint8)
        cs, _ = cv2.findContours(own.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 90, 255), 2)
        if grid:
            step = 50
            for x in range(0, self.W, step):
                major = x % 100 == 0
                cv2.line(out, (x, 0), (x, self.H), (0, 200, 0) if major else (150, 220, 150), 1)
                if major:
                    _label(out, str(x), (x + 2, 12))
            for y in range(0, self.H, step):
                major = y % 100 == 0
                cv2.line(out, (0, y), (self.W, y), (0, 200, 0) if major else (150, 220, 150), 1)
                if major and y:
                    _label(out, str(y), (2, y - 3))
        return out

    # ---- candidates --------------------------------------------------------
    def candidates(self, kind: str = "valley", **p) -> Segments:
        z = self.z
        g = self.gray()
        fig = self.figure_mask()
        if kind in ("valley", "light"):
            sig = p.get("sigma") or (0.5 * z, 0.75 * z, 1.0 * z)
            g01 = cv2.GaussianBlur(g, (0, 0), 0.5 * z).astype(np.float32) / 255.0
            r = sato(g01, sigmas=list(sig), black_ridges=(kind == "valley"))
            r = r * fig
            r = r / (np.percentile(r[fig], 99.5) + 1e-9) if fig.any() else r
            m = apply_hysteresis_threshold(r, p.get("low", 0.2), p.get("high", 0.45))
            sk = skeletonize(m)
        elif kind == "edge":
            u = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
            for _ in range(int(p.get("smooth", 3))):
                u = cv2.bilateralFilter(u, d=9, sigmaColor=30, sigmaSpace=2 * z)
            gx = cv2.Sobel(u, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(u, cv2.CV_32F, 0, 1)
            mag = np.hypot(gx, gy)
            hi = float(np.percentile(mag[fig], p.get("high_pct", 90))) if fig.any() else 50.0
            lo = hi * p.get("low_ratio", 0.45)
            sk = skeletonize(cv2.Canny(u, lo, hi, L2gradient=True) > 0)
        elif kind == "base":
            if not BASELINE_SKEL.exists():
                raise SystemExit("baseline missing -- run `zoomkit.py prepare` first")
            base = cv2.imread(str(BASELINE_SKEL), cv2.IMREAD_GRAYSCALE) < 128
            x0, y0, x1, y1 = self.crop
            sub = base[y0:y1, x0:x1].astype(np.uint8)
            up = cv2.resize(sub, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            sk = skeletonize(cv2.dilate(up, v1.ellipse(3)) > 0)
        else:
            raise SystemExit(f"unknown kind {kind!r}")
        return Segments(sk, int(p.get("min_len", 5 * z)))

    def candidates_overlay(self, segs: Segments, path: Path, label_min: int | None = None) -> None:
        img = self.decorate(faded(self.photo(), 0.5), grid=False)
        label_min = label_min if label_min is not None else 3 * self.z
        for i in segs.ids:
            col = _PALETTE[i % len(_PALETTE)]
            m = segs.label == i
            img[cv2.dilate(m.astype(np.uint8), v1.ellipse(3)) > 0] = col
        for i in segs.ids:
            if segs.length[i] < label_min:
                continue
            ys, xs = np.nonzero(segs.label == i)
            k = len(ys) // 2
            _label(img, str(i), (int(xs[k]) + 3, int(ys[k]) - 3), _PALETTE[i % len(_PALETTE)])
        Image.fromarray(img).save(path)

    # ---- drawing -----------------------------------------------------------
    def blank(self) -> np.ndarray:
        return np.zeros((self.H, self.W), bool)

    def draw_segments(self, canvas: np.ndarray, segs: Segments, ids="all", exclude=()) -> None:
        canvas |= segs.mask(ids, exclude)

    def snap(self, pts, mode: str = "dark", radius: float | None = None, smooth: float = 3.0) -> np.ndarray:
        """Move each point of a hand-drawn polyline along its normal (within
        +/- radius crop px) onto the darkest / lightest / strongest-edge pixel."""
        radius = radius if radius is not None else 1.5 * self.z
        P = _densify(np.asarray(pts, np.float32), 1.0)
        if len(P) < 3:
            return P
        t = np.gradient(P, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        offs = np.arange(-radius, radius + 1, 1.0, dtype=np.float32)
        g = cv2.GaussianBlur(self.gray().astype(np.float32), (0, 0), 0.5 * self.z)
        if mode == "dark":
            cost = g
        elif mode == "light":
            cost = -g
        elif mode == "edge":
            cost = -np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        else:
            raise ValueError("snap mode must be dark|light|edge")
        S = P[:, None, :] + offs[None, :, None] * nrm[:, None, :]
        vals = cv2.remap(cost, S[..., 0].astype(np.float32), S[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        span = float(np.ptp(vals)) + 1e-6
        vals = vals + 0.15 * span * (np.abs(offs) / max(radius, 1))[None, :]   # prefer small moves
        best = offs[np.argmin(vals, axis=1)]
        best = gaussian_filter1d(best, smooth, mode="nearest")
        return P + best[:, None] * nrm

    def draw_polyline(self, canvas: np.ndarray, pts, snap: str | None = None, radius: float | None = None,
                      closed: bool = False) -> np.ndarray:
        P = np.asarray(pts, np.float32)
        if snap:
            P = self.snap(P, snap, radius)
        u = np.zeros(canvas.shape, np.uint8)
        cv2.polylines(u, [np.round(P).astype(np.int32)], closed, 1, 1, cv2.LINE_8)
        canvas |= u > 0
        return P

    def erase_poly(self, canvas: np.ndarray, pts) -> None:
        u = np.zeros(canvas.shape, np.uint8)
        cv2.fillPoly(u, [np.round(np.asarray(pts)).astype(np.int32)], 1)
        canvas &= u == 0

    def erase_rect(self, canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        canvas[max(0, y0):y1, max(0, x0):x1] = False

    def save_lines(self, canvas: np.ndarray) -> Path:
        thick = cv2.dilate(canvas.astype(np.uint8), v1.ellipse(STROKE)) > 0
        out = self.dir / "lines.png"
        cv2.imwrite(str(out), np.where(thick, 0, 255).astype(np.uint8))
        self.preview()
        return out

    # ---- preview -----------------------------------------------------------
    def preview(self) -> None:
        f = self.dir / "lines.png"
        if not f.exists():
            raise SystemExit(f"{f} not found")
        lines = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) < 128
        if lines.shape != (self.H, self.W):
            raise SystemExit(f"lines.png is {lines.shape[::-1]}, expected {(self.W, self.H)}")
        own = self.owned_mask()
        ov = self.decorate(faded(self.photo(), 0.55), grid=True)
        ov[lines & own] = (220, 0, 0)
        ov[lines & ~own] = (255, 150, 150)           # will be clipped away at merge
        Image.fromarray(ov).save(self.dir / "preview_overlay.png")
        sk = skeletonize(lines & own)
        dots = place_dots_v2(sk, DOT_SPACING_1X * self.z)
        dimg = render_dots(dots, self.W, self.H, DOT_R_1X * self.z)
        Image.fromarray(self.decorate(dimg, grid=False)).save(self.dir / "preview_dots.png")
        n_out = int((lines & ~own).sum())
        print(f"[{self.name}] {len(dots)} dots | line px outside owned area (clipped at merge): {n_out}")
        print(f"  -> {self.dir / 'preview_overlay.png'}\n  -> {self.dir / 'preview_dots.png'}")


def _densify(P: np.ndarray, step: float) -> np.ndarray:
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out, np.float32)


def _label(img: np.ndarray, text: str, org: tuple[int, int], color=(0, 0, 0)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_prepare(_a) -> None:
    lab = owned_label_map(1)
    assert (lab > 0).all(), "owned regions must tile the whole image"
    ZOOM_DIR.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "make_dotted_sketch.py"), "--outdir", str(ZOOM_DIR / "_baseline"),
                    "--name", "baseline", "--debug"], check=True)
    for name in REGIONS:
        r = Region(name)
        r.dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(r.photo()).save(r.dir / "crop.png")
        Image.fromarray(r.decorate(r.photo())).save(r.dir / "crop_grid.png")
        base = r.candidates("base")
        r.candidates_overlay(base, r.dir / "cand_base.png")
        (r.dir / "region.json").write_text(json.dumps(
            dict(name=name, zoom=r.z, crop_orig=r.crop, owned_orig=r.owned, crop_size=[r.W, r.H],
                 stroke_px=STROKE, dot_spacing_crop_px=DOT_SPACING_1X * r.z), indent=2))
        print(f"{name:16s} crop {r.W}x{r.H} @ {r.z}x")


def cmd_candidates(a) -> None:
    r = Region(a.region)
    p = {k: v for k, v in dict(sigma=a.sigma, low=a.low, high=a.high, min_len=a.min_len, smooth=a.smooth,
                                high_pct=a.high_pct, low_ratio=a.low_ratio).items() if v is not None}
    segs = r.candidates(a.kind, **p)
    tag = a.tag or a.kind
    out = r.dir / f"cand_{tag}.png"
    r.candidates_overlay(segs, out)
    (r.dir / f"cand_{tag}.json").write_text(json.dumps(
        dict(kind=a.kind, params=p, segments={i: segs.length[i] for i in segs.ids}), indent=1))
    print(f"[{r.name}] {a.kind} {p}: {len(segs.ids)} segments -> {out}")


def cmd_preview(a) -> None:
    Region(a.region).preview()


def merge(source: str, out_dir: Path, name: str, title: str | None) -> np.ndarray:
    w, h = image_size()
    canvas = np.zeros((h * G, w * G), bool)
    lab = owned_label_map(G)
    missing = []
    for idx, name_r in enumerate(REGIONS, start=1):
        r = Region(name_r)
        if source == "base":
            lines = cv2.dilate(r.candidates("base").mask().astype(np.uint8), v1.ellipse(STROKE)) > 0
        else:
            f = r.dir / "lines.png"
            if not f.exists():
                missing.append(name_r)
                continue
            lines = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) < 128
        if r.z != G:
            lines = cv2.resize(lines.astype(np.uint8) * 255, None, fx=G / r.z, fy=G / r.z,
                               interpolation=cv2.INTER_LINEAR) > 64
        x0, y0 = r.crop[0] * G, r.crop[1] * G
        hh, ww = min(lines.shape[0], h * G - y0), min(lines.shape[1], w * G - x0)
        sub = lines[:hh, :ww] & (lab[y0:y0 + hh, x0:x0 + ww] == idx)
        canvas[y0:y0 + hh, x0:x0 + ww] |= sub
    if missing:
        print("WARNING: no lines.png for:", ", ".join(missing))
    closed = cv2.morphologyEx(canvas.astype(np.uint8), cv2.MORPH_CLOSE, v1.ellipse(5)) > 0
    sk = v1.drop_small(skeletonize(closed), 6 * G)
    dots = place_dots_v2(sk, DOT_SPACING_1X * G)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / name
    W, H = w * G, h * G
    r_px = DOT_R_1X * G
    v1.write_svg(base.with_suffix(".svg"), dots, W, H, r_px, title)
    art = Image.fromarray(render_dots(dots, W, H, r_px)[..., 0])
    art.save(base.with_suffix(".png"))
    v1.render_a4(art, title).save(base.with_suffix(".pdf"), resolution=300)
    line_art = cv2.GaussianBlur((cv2.dilate(sk.astype(np.uint8), v1.ellipse(3)) * 255).astype(np.float32), (0, 0), 0.8)
    cv2.imwrite(str(out_dir / f"{name}-lineart.png"), (255 - np.clip(line_art, 0, 255)).astype(np.uint8))
    chk = faded(source_rgb(2), 0.5)
    sk2 = cv2.resize(cv2.dilate(sk.astype(np.uint8), v1.ellipse(3)), (w * 2, h * 2), interpolation=cv2.INTER_AREA) > 0
    chk[sk2] = (220, 0, 0)
    Image.fromarray(chk).save(out_dir / f"{name}-overlay-check.png")
    print(f"{len(dots)} dots -> {base}.{{svg,png,pdf}}, -lineart.png, -overlay-check.png")
    return sk


def cmd_selftest(_a) -> None:
    sk = merge("base", ZOOM_DIR / "_selftest", "selftest", None)
    w, h = image_size()
    v1sk = cv2.imread(str(BASELINE_SKEL), cv2.IMREAD_GRAYSCALE) < 128
    sk1 = sk.reshape(h, G, w, G).max(axis=(1, 3))          # max-pool 4x -> 1x
    near = cv2.dilate(sk1.astype(np.uint8), v1.ellipse(5)) > 0
    recall = (v1sk & near).sum() / max(1, v1sk.sum())
    v1near = cv2.dilate(v1sk.astype(np.uint8), v1.ellipse(5)) > 0
    precision = (sk1 & v1near).sum() / max(1, sk1.sum())
    print(f"selftest: v1 lines reproduced (recall) {recall:.1%}, merged lines on v1 (precision) {precision:.1%}")


def cmd_merge(a) -> None:
    merge("lines", OUT_DIR, a.name, a.title or None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)
    c = sub.add_parser("candidates")
    c.add_argument("region")
    c.add_argument("--kind", default="valley", choices=["valley", "edge", "light", "base"])
    c.add_argument("--tag", help="output name suffix (default: kind)")
    c.add_argument("--sigma", type=float, nargs="+")
    c.add_argument("--low", type=float)
    c.add_argument("--high", type=float)
    c.add_argument("--min-len", type=int)
    c.add_argument("--smooth", type=int)
    c.add_argument("--high-pct", type=float)
    c.add_argument("--low-ratio", type=float)
    c.set_defaults(fn=cmd_candidates)
    p = sub.add_parser("preview")
    p.add_argument("region")
    p.set_defaults(fn=cmd_preview)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    m = sub.add_parser("merge")
    m.add_argument("--name", default="sri-ramakrishna-dotted-v2")
    m.add_argument("--title", default="Sri Ramakrishna")
    m.set_defaults(fn=cmd_merge)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
