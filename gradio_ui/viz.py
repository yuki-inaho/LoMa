from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _pad_canvas(img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, int, int]:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    off1 = (H - h0) // 2
    off2 = (H - h1) // 2
    W = w0 + w1
    canvas = np.zeros((H, W, 3), dtype=img0.dtype)
    canvas[off1 : off1 + h0, :w0] = img0
    canvas[off2 : off2 + h1, w0:] = img1
    return canvas, off1, off2


def draw_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray | None = None,
    mode: str = "all",
    inlier_mask: np.ndarray | None = None,
    seed: int = 12345,
) -> np.ndarray:
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    canvas, _, _ = _pad_canvas(img0, img1)
    h0, w0 = img0.shape[:2]
    n = len(pts0)
    scale = max(1, round(max(canvas.shape[:2]) / 1100))
    lw = max(1, 2 * scale // 2 + 1)
    r = max(2, 3 * scale)
    rng = np.random.default_rng(seed if n else 0)
    colors = rng.integers(60, 256, size=(max(n, 1), 3))
    for i in range(n):
        is_in = True
        if inlier_mask is not None and len(inlier_mask) == n:
            is_in = bool(inlier_mask[i])
        if mode == "inlier" and not is_in:
            continue
        if mode == "outlier" and is_in:
            continue
        x1, y1 = float(pts0[i][0]), float(pts0[i][1])
        x2, y2 = float(pts1[i][0]) + w0, float(pts1[i][1])
        color = tuple(int(c) for c in colors[i])
        thickness = lw if is_in else max(1, lw - 1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x2), int(y2)), r, color, -1, cv2.LINE_AA)
    return canvas


def confidence_histogram(conf: np.ndarray, thr: float, width: int = 720, height: int = 210) -> np.ndarray:
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    m_l, m_r, m_t, m_b = 46, 12, 14, 26
    plot_w = width - m_l - m_r
    plot_h = height - m_t - m_b
    bins = 40
    counts, _ = np.histogram(np.clip(conf, 0, 1), bins=bins, range=(0.0, 1.0))
    vmax = max(int(counts.max()), 1)
    bar_w = plot_w / bins
    for i, c in enumerate(counts):
        x0 = m_l + i * bar_w
        bh = int(plot_h * c / vmax)
        y0 = height - m_b - bh
        bin_thr = (i + 1) / bins
        fill = (150, 170, 210) if bin_thr <= thr else (70, 120, 200)
        d.rectangle([x0, y0, x0 + bar_w - 1, height - m_b], fill=fill)
    tx = m_l + min(max(float(thr), 0.0), 1.0) * plot_w
    for yy in range(m_t, height - m_b, 6):
        d.line([(tx, yy), (tx, min(yy + 4, height - m_b))], fill=(220, 30, 80), width=2)
    d.line([(m_l, height - m_b), (width - m_r, height - m_b)], fill=(60, 60, 60), width=1)
    d.text((m_l, m_t - 2), f"matches: {len(conf)}   threshold: {thr:.3f}", fill=(20, 20, 20))
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx = m_l + v * plot_w
        d.text((xx - 8, height - m_b + 5), f"{v:g}", fill=(90, 90, 90))
    return np.asarray(img)


def checker_mask(h: int, w: int, sq: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy // sq + xx // sq) % 2).astype(bool)


def warp_blend(
    img0: np.ndarray,
    img1: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.55,
    checker: bool = False,
) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    warped = cv2.warpPerspective(img1, np.asarray(H), (w0, h0))
    src = img0.astype(np.float32)
    dst = warped.astype(np.float32)
    if warped.shape != src.shape:
        return img0.copy()
    if checker:
        sq = max(16, min(h0, w0) // 10)
        cm = checker_mask(h0, w0, sq)[..., None]
        out = src * cm + dst * (1 - cm)
    else:
        a = float(np.clip(alpha, 0.0, 1.0))
        out = (1 - a) * src + a * dst
    return np.clip(out, 0, 255).astype(np.uint8)


def rectified_pair(img0: np.ndarray, img1: np.ndarray, H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    r0 = cv2.warpPerspective(img0, np.asarray(H1), (w0, h0))
    r1 = cv2.warpPerspective(img1, np.asarray(H2), (w1, h1))
    return _pad_canvas(r0, r1)[0]


def epipolar_overlay(
    img0: np.ndarray,
    img1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    F: np.ndarray,
    max_lines: int = 24,
) -> np.ndarray:
    n = min(len(pts0), max_lines)
    idx = np.linspace(0, len(pts0) - 1, n).astype(int) if len(pts0) > n else np.arange(len(pts0))
    p0 = np.ascontiguousarray(pts0[idx], dtype=np.float64)
    p1 = np.ascontiguousarray(pts1[idx], dtype=np.float64)
    lines1 = cv2.computeCorrespondEpilines(p0.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    Ft = F.T
    lines0 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, Ft).reshape(-1, 3)
    a0 = img0.astype(np.float32) * 0.55 + np.full_like(img0, 127, dtype=np.float32) * 0.45
    a1 = img1.astype(np.float32) * 0.55 + np.full_like(img1, 127, dtype=np.float32) * 0.45
    rng = np.random.default_rng(7)
    colors = rng.integers(40, 256, size=(n, 3))

    def draw_lines(base: np.ndarray, pts: np.ndarray, lines: np.ndarray):
        h, w = base.shape[:2]
        dd = base
        for i in range(n):
            l = lines[i]
            a, b, c = float(l[0]), float(l[1]), float(l[2])
            col = tuple(int(v) for v in colors[i])
            if abs(b) < 1e-8:
                xx = -c / a
                cv2.line(dd, (int(xx), 0), (int(xx), h - 1), col, 1, cv2.LINE_AA)
                continue
            xa, ya = -w, (-c - a * (-w)) / b
            xb, yb = 2 * w, (-c - a * (2 * w)) / b
            pta = (int(np.clip(xa, -w, 2 * w)), int(np.clip(ya, -h, 2 * h)))
            ptb = (int(np.clip(xb, -w, 2 * w)), int(np.clip(yb, -h, 2 * h)))
            cv2.line(dd, pta, ptb, col, 1, cv2.LINE_AA)
            cv2.circle(dd, (int(pts[i][0]), int(pts[i][1])), 4, col, -1)

    draw_lines(a0, p0, lines0)
    draw_lines(a1, p1, lines1)
    return _pad_canvas(np.clip(a0, 0, 255).astype(np.uint8),
                       np.clip(a1, 0, 255).astype(np.uint8))[0]


def keypoint_density(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    vis = img.copy()
    for p in pts[:5000]:
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (255, 90, 30), -1, cv2.LINE_AA)
    return vis
