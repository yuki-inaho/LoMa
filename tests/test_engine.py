from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradio_ui.engine import ENGINE, MODELS, DEFAULT_MODEL_KEY, ransac_filter, select_matches
from gradio_ui.viz import draw_matches, confidence_histogram, warp_blend


def main():
    key = str(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL_KEY
    assert key in MODELS
    root = Path(__file__).resolve().parent.parent / "assets"
    a = cv2.cvtColor(cv2.imread(str(root / "toronto_A.jpg")), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imread(str(root / "toronto_B.jpg")), cv2.COLOR_BGR2RGB)

    print("[1] load_model:", ENGINE.load_model(key))
    p0, p1, md = ENGINE.run_pair(a, b, model_key=key, num_keypoints=1024, long_side=1024)
    print(f"[2] base mutual matches: {len(md.conf)}")
    assert 50 < len(md.conf) <= 1024 + 8

    thr = 0.1
    pts0, pts1, cf = select_matches(md, thr)
    assert len(pts0) == len(pts1) == len(cf)
    assert (cf > thr).all()
    print(f"[3] selected at thr={thr}: {len(pts0)}")

    F, maskF = ransac_filter(pts0, pts1, "Fundamental", thr=4.0)
    H, maskH = ransac_filter(pts0, pts1, "Homography", thr=4.0)
    assert F is not None and maskF is not None and int(maskF.sum()) > 20
    assert H is not None and maskH is not None and int(maskH.sum()) > 10
    print(f"[4] RANSAC F inliers={int(maskF.sum())} / H inliers={int(maskH.sum())}")

    rgb0 = ENGINE.load_rgb(p0); rgb1 = ENGINE.load_rgb(p1)
    c1 = draw_matches(rgb0, rgb1, pts0, pts1, cf, mode="inlier",
                      inlier_mask=maskF.astype(bool))
    assert c1.shape[0] >= max(rgb0.shape[0], rgb1.shape[0]) and c1.shape[1] > 1000
    hist = confidence_histogram(md.conf, thr)
    assert hist.shape[:2] == (210, 720)
    blended = warp_blend(rgb0, rgb1, np.asarray(H), alpha=0.5)
    assert blended.shape == rgb0.shape
    out = Path(__file__).parent.parent / "app_data"
    (out / "test_canvas.jpg").parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(out / "test_canvas.jpg"), cv2.cvtColor(c1, cv2.COLOR_RGB2BGR))
    print("[5] viz outputs ok")

    thr_mid = md.conf[len(md.conf) // 2]
    n_full, n_half = len(md.conf), int((md.conf > float(thr_mid)).sum())
    assert n_half < n_full
    print(f"[6] threshold monotonic slice ok: {n_full} -> {n_half}")

    del ENGINE._model
    ENGINE._model = None
    import torch
    torch.cuda.empty_cache()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
