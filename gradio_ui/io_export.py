from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from .engine import OUTPUTS_DIR


def make_npz(pts0, pts1, conf, M, inlier_full, geometry) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"matches_{ts}.npz"
    data = dict(
        pts0=pts0.astype(np.float32),
        pts1=pts1.astype(np.float32),
        conf=conf.astype(np.float32),
    )
    if inlier_full is not None:
        data["inliers"] = inlier_full.astype(bool)
    if M is not None:
        data[geometry.lower()] = M.astype(np.float64)
    np.savez_compressed(path, **data)
    return str(path)


def make_csv(pts0, pts1, conf, inlier_full) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"matches_{ts}.csv"
    buf = io.StringIO()
    buf.write("x0,y0,x1,y1,conf,inlier\n")
    for i in range(len(pts0)):
        inl = int(bool(inlier_full[i])) if inlier_full is not None else ""
        buf.write(f"{pts0[i][0]:.3f},{pts0[i][1]:.3f},{pts1[i][0]:.3f},{pts1[i][1]:.3f},{conf[i]:.5f},{inl}\n")
    path.write_text(buf.getvalue(), encoding="utf-8")
    return str(path)


def make_json(stats: dict) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"summary_{ts}.json"
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
