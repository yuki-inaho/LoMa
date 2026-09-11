"""End-to-end LoMa B128 ONNX benchmark on the CUDA Execution Provider.

Tuned for an RTX 5090 (Blackwell, sm_120) x86_64 host with the pip
`onnxruntime-gpu` wheel and the exported `loma_*.onnx` stages. Uses the same
fixed-size preprocessing as `compare_onnx.py` (detector 752x1024,
descriptor 784x784) so latency is directly comparable to the PyTorch numbers
produced by `benchmark_gpu.py`.

Run:
  pixi run python bench_blackwell.py \
      --onnx-dir onnx --im-a assets/0015_A.jpg --im-b assets/0015_B.jpg \
      --iters 20 --out docs/blackwell_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from compare_onnx import DESC_HW, DET_H, DET_W, matches_pixels, preprocess


def timed(session, feeds, iters, warmup=3):
    for _ in range(warmup):
        out = session.run(None, feeds)
    start = time.perf_counter()
    for _ in range(iters):
        out = session.run(None, feeds)
    return (time.perf_counter() - start) / iters * 1000.0, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--im-a", default="assets/0015_A.jpg")
    ap.add_argument("--im-b", default="assets/0015_B.jpg")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--providers", nargs="+",
                    default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    available = ort.get_available_providers()
    providers = [p for p in args.providers if p in available]
    if not providers:
        raise SystemExit("none of %s available; have %s" % (args.providers, available))

    onnx_dir = Path(args.onnx_dir)
    det = ort.InferenceSession(str(onnx_dir / "loma_detector.onnx"), providers=providers)
    dsc = ort.InferenceSession(str(onnx_dir / "loma_descriptor_dedode_b.onnx"), providers=providers)
    mat = ort.InferenceSession(str(onnx_dir / "loma_matcher_B128.onnx"), providers=providers)

    det_a = preprocess(args.im_a, DET_H, DET_W, "cpu").numpy().astype(np.float32)
    det_b = preprocess(args.im_b, DET_H, DET_W, "cpu").numpy().astype(np.float32)
    dsc_a = preprocess(args.im_a, DESC_HW, DESC_HW, "cpu").numpy().astype(np.float32)
    dsc_b = preprocess(args.im_b, DESC_HW, DESC_HW, "cpu").numpy().astype(np.float32)

    det_ms_a, (k_a, p_a) = timed(det, {"image": det_a}, args.iters, args.warmup)
    det_ms_b, (k_b, p_b) = timed(det, {"image": det_b}, args.iters, args.warmup)
    dsc_ms_a, (d_a,) = timed(dsc, {"image": dsc_a, "keypoints": k_a}, args.iters, args.warmup)
    dsc_ms_b, (d_b,) = timed(dsc, {"image": dsc_b, "keypoints": k_b}, args.iters, args.warmup)
    mat_ms, (m0, m1, s0, s1) = timed(
        mat, {"kpts0": k_a, "kpts1": k_b, "desc0": d_a, "desc1": d_b}, args.iters, args.warmup)

    w_a, h_a = preprocess_path_size(args.im_a)
    w_b, h_b = preprocess_path_size(args.im_b)
    pairs = matches_pixels(m0, k_a, k_b, h_a, w_a, h_b, w_b)
    total = det_ms_a + det_ms_b + dsc_ms_a + dsc_ms_b + mat_ms

    summary = {
        "device": ort.get_device(),
        "providers_requested": args.providers,
        "providers_used": det.get_providers(),
        "onnx_dir": str(onnx_dir),
        "pairs": [args.im_a, args.im_b],
        "iters": args.iters,
        "detector_ms": {"a": round(det_ms_a, 2), "b": round(det_ms_b, 2)},
        "descriptor_ms": {"a": round(dsc_ms_a, 2), "b": round(dsc_ms_b, 2)},
        "matcher_ms": round(mat_ms, 2),
        "total_ms": round(total, 2),
        "fps": round(1000.0 / total, 2),
        "num_match_pairs": int(len(pairs)),
        "keypoints_per_image": int(k_a.shape[1]),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def preprocess_path_size(path):
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return w, h


if __name__ == "__main__":
    raise SystemExit(main())
