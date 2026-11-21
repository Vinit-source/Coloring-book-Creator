"""
Photo -> dotted-line tracing sketch (portrait on a dark background).

Raw edge detection on a bearded face gives thousands of hair edges, so the
big structural lines come from *region boundaries* instead:

  A. figure silhouette   : brightness vs. the dark backdrop   -> head, ears, shoulders
  B. skin region         : high HSV saturation (warm skin)    -> hairline, beard edge, mustache
  C. hair / beard region : low saturation + high local texture -> beard bottom, cloth edge
  D. facial features     : Canny on a heavily smoothed image, only inside the skin -> eyes, brows, nose, lips

The union of these lines is thinned to 1px, then evenly spaced dots are
walked along every stroke and exported as SVG (vector), PNG preview, and an
A4 PDF page.

Run (from repo root):
  uv run --with opencv-python-headless --with scikit-image --with pillow \
      python dotted-sketch/make_dotted_sketch.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_rgb(path: Path, work_scale: float) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    rgb = np.array(img.convert("RGB"))
    if work_scale != 1.0:
        h, w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (int(w * work_scale), int(h * work_scale)), interpolation=cv2.INTER_CUBIC)
    return rgb


def drop_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Keep only 8-connected components with at least `min_pixels` pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
    return keep[labels]


def fill_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    return ~drop_small(~mask, max_area)


def fill_from_bottom(mask: np.ndarray) -> np.ndarray:
    """A bust portrait runs off the bottom of the frame, so any background
    pocket that only reaches the *bottom* edge (e.g. the shadow under a beard)
    is really part of the figure. Keep only background reachable from the
    top/left/right edges."""
    n, labels = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=4)
    outside = np.zeros(n, dtype=bool)
    for edge in (labels[0, :], labels[:, 0], labels[:, -1]):
        outside[np.unique(edge)] = True
    outside[0] = False  # label 0 = figure pixels
    return ~outside[labels]


def masked_std(v: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Local std-dev of `v` using only pixels inside `m`, so the figure/backdrop
    step itself doesn't read as 'texture'."""
    mf = m.astype(np.float32)
    w = cv2.GaussianBlur(mf, (0, 0), sigma) + 1e-6
    mu = cv2.GaussianBlur(v * mf, (0, 0), sigma) / w
    ex2 = cv2.GaussianBlur(v * v * mf, (0, 0), sigma) / w
    return np.sqrt(np.maximum(ex2 - mu * mu, 0)) * m


def soften(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round off a binary mask's boundary (blur + re-threshold)."""
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) > 0.5


def ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def s(v: float, scale: float) -> float:
    """Scale a pixel-unit parameter with the working resolution."""
    return v * scale


# --------------------------------------------------------------------------- #
# A-C: region masks
# --------------------------------------------------------------------------- #
def region_masks(rgb: np.ndarray, a) -> dict[str, np.ndarray]:
    k = a.work_scale
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)

    # A. figure vs. dark backdrop
    fig = soften(cv2.GaussianBlur(V, (0, 0), s(2, k)) > a.bg_thresh, s(3, k))
    fig = fill_from_bottom(drop_small(fig, int(s(5000, k * k))))

    # B. warm, saturated skin
    S_s = cv2.GaussianBlur(S, (0, 0), s(3, k))
    skin = soften((S_s > a.skin_sat) & fig, s(5, k))
    skin = fill_holes(drop_small(skin, int(s(2000, k * k))), int(s(2500, k * k)))

    # C. hair / beard: unsaturated + textured (or deep shadow) inside the figure
    inner_fig = cv2.erode(fig.astype(np.uint8), ellipse(int(s(5, k)) | 1)) > 0
    std = masked_std(V, inner_fig, s(4, k))
    std = cv2.GaussianBlur(std, (0, 0), s(6, k)) / (cv2.GaussianBlur(inner_fig.astype(np.float32), (0, 0), s(6, k)) + 1e-6)
    f = inner_fig.astype(np.float32)
    v_mean = cv2.GaussianBlur(V * f, (0, 0), s(3, k)) / (cv2.GaussianBlur(f, (0, 0), s(3, k)) + 1e-6)
    hair = inner_fig & ~skin & ((std > a.hair_texture) | (v_mean < a.shadow_v))
    hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_OPEN, ellipse(int(s(21, k)) | 1)) > 0
    hair = soften(hair, s(8, k)) & fig
    hair = fill_holes(drop_small(hair, int(s(3000, k * k))), int(s(3000, k * k)))
    return {"figure": fig, "skin": skin, "hair": hair}


def boundary_lines(masks: dict[str, np.ndarray], min_len: float, border: int) -> np.ndarray:
    h, w = next(iter(masks.values())).shape
    out = np.zeros((h, w), np.uint8)
    for m in masks.values():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, [c for c in cs if cv2.arcLength(c, True) >= min_len], -1, 1, 1)
    # the photo's own frame edge is not a line in the drawing
    out[:border, :] = out[-border:, :] = 0
    out[:, :border] = out[:, -border:] = 0
    return out > 0


# --------------------------------------------------------------------------- #
# D: facial-feature lines inside the skin
# --------------------------------------------------------------------------- #
def feature_lines(rgb: np.ndarray, skin: np.ndarray, a) -> tuple[np.ndarray, np.ndarray]:
    k = a.work_scale
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    for _ in range(a.bilateral):
        g = cv2.bilateralFilter(g, d=9, sigmaColor=40, sigmaSpace=s(9, k))
    if a.median > 1:
        g = cv2.medianBlur(g, a.median)
    edges = cv2.Canny(g, a.canny_low, a.canny_high, L2gradient=True) > 0
    # face + ears only: skin regions that run off the bottom of the frame are chest/shoulders
    n, lab = cv2.connectedComponents(skin.astype(np.uint8), connectivity=8)
    off_bottom = np.zeros(n, dtype=bool)
    off_bottom[np.unique(lab[-1, :])] = True
    face = skin & ~off_bottom[lab]
    inner = cv2.erode(face.astype(np.uint8), ellipse(int(s(a.feature_inset, k)) | 1)) > 0
    return edges & inner, g


def to_skeleton(lines: np.ndarray, close_k: int, min_len: int) -> np.ndarray:
    m = lines.astype(np.uint8)
    if close_k > 1:
        # bridge 1-2px gaps so strokes stay continuous before thinning
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(close_k))
    sk = skeletonize(m > 0)
    return drop_small(sk, min_len)


# --------------------------------------------------------------------------- #
# 6: skeleton -> evenly spaced dots
# --------------------------------------------------------------------------- #
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def ordered_pixels(sk: np.ndarray) -> list[tuple[int, int]]:
    """Return skeleton pixels in line-following (DFS) order, starting each
    component at an endpoint, so dots are laid down *along* each stroke."""
    h, w = sk.shape
    pad = np.pad(sk.astype(np.uint8), 1)
    nbr = sum(np.roll(np.roll(pad, dy, 0), dx, 1) for dy, dx in _NB)[1:-1, 1:-1]
    ys, xs = np.nonzero(sk)
    endpoints = [(y, x) for y, x in zip(ys, xs) if nbr[y, x] == 1]
    others = list(zip(ys, xs))
    seen = np.zeros_like(sk, dtype=bool)
    order: list[tuple[int, int]] = []
    for start in endpoints + others:
        if seen[start]:
            continue
        stack = [start]
        while stack:
            y, x = stack.pop()
            if seen[y, x]:
                continue
            seen[y, x] = True
            order.append((y, x))
            for dy, dx in _NB:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] and not seen[ny, nx]:
                    stack.append((ny, nx))
    return order


def place_dots(sk: np.ndarray, spacing: float) -> list[tuple[float, float]]:
    """Greedy minimum-distance sampling with a grid hash."""
    cell = spacing
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    dots: list[tuple[float, float]] = []
    s2 = spacing * spacing
    for y, x in ordered_pixels(sk):
        cy, cx = int(y // cell), int(x // cell)
        ok = True
        for gy in (cy - 1, cy, cy + 1):
            for gx in (cx - 1, cx, cx + 1):
                for py, px in grid.get((gy, gx), ()):
                    if (py - y) ** 2 + (px - x) ** 2 < s2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((cy, cx), []).append((y, x))
            dots.append((float(x), float(y)))
    return dots


# --------------------------------------------------------------------------- #
# 7: exports
# --------------------------------------------------------------------------- #
def write_svg(path: Path, dots, w: int, h: int, r: float, title: str | None) -> None:
    pad_bottom = 0.08 * h if title else 0
    H = h + pad_bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {H:.0f}" width="{w}" height="{H:.0f}">',
        f'<rect width="{w}" height="{H:.0f}" fill="#fff"/>',
        '<g fill="#000">',
    ]
    parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}"/>' for x, y in dots]
    parts.append("</g>")
    if title:
        parts.append(
            f'<text x="{w / 2:.0f}" y="{h + pad_bottom * 0.65:.0f}" text-anchor="middle" '
            f'font-family="Georgia, serif" font-size="{pad_bottom * 0.5:.0f}" fill="#444">{title}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def render_png(dots, w: int, h: int, r: float, upscale: int) -> Image.Image:
    W, H = w * upscale, h * upscale
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    R = r * upscale
    for x, y in dots:
        X, Y = x * upscale, y * upscale
        d.ellipse((X - R, Y - R, X + R, Y + R), fill=0)
    return im


def _font(size: int) -> ImageFont.ImageFont:
    for f in ("/System/Library/Fonts/Supplemental/Georgia.ttf", "/Library/Fonts/Georgia.ttf",
              "/System/Library/Fonts/Times.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_a4(art: Image.Image, title: str | None, dpi: int = 300) -> Image.Image:
    pw, ph = int(8.27 * dpi), int(11.69 * dpi)
    page = Image.new("L", (pw, ph), 255)
    margin = int(0.6 * dpi)
    title_h = int(0.9 * dpi) if title else 0
    box_w, box_h = pw - 2 * margin, ph - 2 * margin - title_h
    s = min(box_w / art.width, box_h / art.height)
    a = art.resize((int(art.width * s), int(art.height * s)), Image.LANCZOS)
    x = (pw - a.width) // 2
    y = margin + (box_h - a.height) // 2
    page.paste(a, (x, y))
    if title:
        d = ImageDraw.Draw(page)
        f = _font(int(0.32 * dpi))
        tw = d.textlength(title, font=f)
        d.text(((pw - tw) / 2, ph - margin - title_h * 0.75), title, fill=60, font=f)
    return page


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=HERE / "input" / "sri-ramakrishna.png")
    ap.add_argument("--outdir", type=Path, default=HERE / "output")
    ap.add_argument("--name", default="sri-ramakrishna-dotted")
    ap.add_argument("--title", default="Sri Ramakrishna")
    ap.add_argument("--work-scale", type=float, default=1.0, help="resize factor before processing")
    # regions
    ap.add_argument("--bg-thresh", type=float, default=34, help="HSV value separating figure from dark backdrop")
    ap.add_argument("--skin-sat", type=float, default=55, help="HSV saturation above which a pixel is skin")
    ap.add_argument("--hair-texture", type=float, default=9, help="local brightness std-dev marking hair/beard")
    ap.add_argument("--shadow-v", type=float, default=55, help="deep shadow inside figure counts as beard/hair")
    ap.add_argument("--region-min-len", type=float, default=80, help="drop region outlines shorter than this")
    # facial features
    ap.add_argument("--bilateral", type=int, default=6, help="bilateral smoothing passes")
    ap.add_argument("--median", type=int, default=7, help="median blur kernel (odd, 0=off)")
    ap.add_argument("--canny-low", type=int, default=22)
    ap.add_argument("--canny-high", type=int, default=65)
    ap.add_argument("--feature-inset", type=float, default=14, help="keep features this far inside skin edge")
    # line clean-up + dots
    ap.add_argument("--close", type=int, default=3, help="gap-bridging kernel before thinning")
    ap.add_argument("--min-len", type=int, default=40, help="drop line fragments shorter than this (px)")
    ap.add_argument("--spacing", type=float, default=7.0, help="distance between dots (px, work scale)")
    ap.add_argument("--dot-r", type=float, default=1.6, help="dot radius (px, work scale)")
    ap.add_argument("--upscale", type=int, default=4, help="PNG preview upscale")
    ap.add_argument("--debug", action="store_true", help="also write intermediate images")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.input, args.work_scale)
    masks = region_masks(rgb, args)
    region = boundary_lines(masks, args.region_min_len, border=3)
    features, g = feature_lines(rgb, masks["skin"], args)
    sk = to_skeleton(region | features, args.close, args.min_len)
    dots = place_dots(sk, args.spacing)
    h, w = sk.shape

    out = args.outdir / args.name
    write_svg(out.with_suffix(".svg"), dots, w, h, args.dot_r, args.title or None)
    art = render_png(dots, w, h, args.dot_r, args.upscale)
    art.save(out.with_suffix(".png"))
    render_a4(art, args.title or None).save(out.with_suffix(".pdf"), resolution=300)

    if args.debug:
        tint = rgb.astype(np.float32)
        for name, col in (("hair", (40, 60, 255)), ("skin", (255, 160, 0))):
            m = masks[name]
            tint[m] = tint[m] * 0.45 + np.array(col) * 0.55
        cv2.imwrite(str(args.outdir / "debug-1-regions.png"), cv2.cvtColor(tint.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.outdir / "debug-2-smoothed.png"), g)
        cv2.imwrite(str(args.outdir / "debug-3-lines.png"),
                    255 - (region.astype(np.uint8) * 255 | features.astype(np.uint8) * 140))
        cv2.imwrite(str(args.outdir / "debug-4-skeleton.png"), 255 - sk.astype(np.uint8) * 255)

    print(f"{len(dots)} dots, {int(sk.sum())} skeleton px -> {out}.{{svg,png,pdf}}")


if __name__ == "__main__":
    main()
