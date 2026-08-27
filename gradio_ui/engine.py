from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from loma import LoMa
from loma.loma import (
    LoMaB,
    LoMaB128,
    LoMaG,
    LoMaL,
    LoMaR,
    filter_matches,
    to_pixel_coords,
)

ROOT = Path(__file__).resolve().parent.parent
APP_DATA = ROOT / "app_data"
UPLOADS_DIR = APP_DATA / "uploads"
OUTPUTS_DIR = APP_DATA / "outputs"

MODELS: dict[str, object] = {
    "LoMa-B (標準・LightGlueサイズ)": LoMaB,
    "LoMa-B128 (軽量)": LoMaB128,
    "LoMa-L (大型)": LoMaL,
    "LoMa-G (最重量・最高精度)": LoMaG,
    "LoMa-R (回転不変)": LoMaR,
}
DEFAULT_MODEL_KEY = "LoMa-B (標準・LightGlueサイズ)"
RANSAC_METHODS = {
    "CV2_USAC_MAGSAC": cv2.USAC_MAGSAC,
    "CV2_USAC_ACCURATE": cv2.USAC_ACCURATE,
    "CV2_USAC_DEFAULT": cv2.USAC_DEFAULT,
    "CV2_RANSAC": cv2.RANSAC,
}


@dataclass
class Detection:
    kpts: torch.Tensor
    desc: torch.Tensor
    h: int
    w: int


@dataclass
class MatchData:
    pts0: np.ndarray
    pts1: np.ndarray
    conf: np.ndarray
    h0: int
    w0: int
    h1: int
    w1: int


def device_str() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA · {name} · {vram:.0f}GB"
    return "CPU"


class LomaEngine:
    def __init__(self, max_det_cache: int = 16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: LoMa | None = None
        self._model_key: str | None = None
        self._dets: OrderedDict[tuple, Detection] = OrderedDict()
        self.max_det_cache = max_det_cache
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_model(self, key: str) -> str:
        if key not in MODELS:
            raise ValueError(f"unknown model: {key}")
        if self._model is not None and self._model_key == key:
            return f"cached {key}"
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            self._dets.clear()
        t0 = time.time()
        self._model = LoMa(MODELS[key]())
        self._model.eval().to(self.device)
        self._model_key = key
        return f"{key} loaded in {time.time() - t0:.1f}s"

    @property
    def loaded_key(self) -> str | None:
        return self._model_key

    def _ensure_file(self, img_rgb: np.ndarray, long_side: int) -> Path:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if long_side and long_side > 0:
            h, w = bgr.shape[:2]
            s = long_side / max(h, w)
            if s < 1.0:
                bgr = cv2.resize(
                    bgr,
                    (int(round(w * s)), int(round(h * s))),
                    interpolation=cv2.INTER_AREA,
                )
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("failed to encode image")
        digest = hashlib.sha1(buf.tobytes()).hexdigest()[:16]
        path = UPLOADS_DIR / f"{digest}.jpg"
        if not path.exists():
            buf.tofile(path)
        return path

    def detect(self, img_rgb: np.ndarray, num_keypoints: int, long_side: int) -> Path:
        assert self._model is not None, "load_model() first"
        path = self._ensure_file(img_rgb, long_side)
        ck = (path.name, int(num_keypoints))
        if ck in self._dets:
            self._dets.move_to_end(ck)
            return path
        kpts, desc, h, w = self._model.detect_and_describe(
            str(path), num_keypoints=int(num_keypoints)
        )
        self._dets[ck] = Detection(
            kpts=kpts.detach().cpu(),
            desc=desc.detach().cpu().contiguous(),
            h=int(h),
            w=int(w),
        )
        while len(self._dets) > self.max_det_cache:
            self._dets.popitem(last=False)
        return path

    def load_rgb(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def match_scores(self, d0: Detection, d1: Detection) -> MatchData:
        assert self._model is not None
        k0 = d0.kpts.to(self.device)
        k1 = d1.kpts.to(self.device)
        x0 = d0.desc.to(self.device).contiguous()
        x1 = d1.desc.to(self.device).contiguous()
        with torch.inference_mode():
            scores = self._model(k0, k1, x0, x1)["scores"]
        m0, _, ms0, _ = filter_matches(scores, 0.0)
        valid = m0[0] > -1
        idx_a = torch.where(valid)[0]
        idx_b = m0[0][valid].clamp(min=0)
        conf = ms0[0][valid]
        pa = to_pixel_coords(k0[0][idx_a], d0.h, d0.w).cpu().numpy().astype(np.float32)
        pb = to_pixel_coords(k1[0][idx_b], d1.h, d1.w).cpu().numpy().astype(np.float32)
        return MatchData(
            pts0=pa.reshape(-1, 2),
            pts1=pb.reshape(-1, 2),
            conf=conf.cpu().numpy().astype(np.float32),
            h0=d0.h,
            w0=d0.w,
            h1=d1.h,
            w1=d1.w,
        )

    def run_pair(
        self,
        img0_rgb: np.ndarray,
        img1_rgb: np.ndarray,
        model_key: str,
        num_keypoints: int,
        long_side: int,
    ) -> tuple[Path, Path, MatchData]:
        status = self.load_model(model_key)
        p0 = self.detect(img0_rgb, num_keypoints, long_side)
        p1 = self.detect(img1_rgb, num_keypoints, long_side)
        md = self.match_scores(self._dets[(p0.name, int(num_keypoints))],
                               self._dets[(p1.name, int(num_keypoints))])
        return p0, p1, md


def select_matches(md: MatchData, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sel = md.conf > float(thr)
    return md.pts0[sel], md.pts1[sel], md.conf[sel]


def ransac_filter(
    pts0: np.ndarray,
    pts1: np.ndarray,
    geometry: str,
    method: str = "CV2_USAC_MAGSAC",
    thr: float = 4.0,
    confidence: float = 0.999,
    max_iter: int = 10000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if method not in RANSAC_METHODS:
        method = "CV2_USAC_MAGSAC"
    flag = RANSAC_METHODS[method]
    p0 = np.ascontiguousarray(pts0, dtype=np.float64)
    p1 = np.ascontiguousarray(pts1, dtype=np.float64)
    if len(p0) < 8:
        return None, None
    try:
        if geometry == "Fundamental":
            M, mask = cv2.findFundamentalMat(
                p0, p1, flag, ransacReprojThreshold=float(thr),
                confidence=float(confidence), maxIters=int(max_iter),
            )
        elif geometry == "Homography":
            M, mask = cv2.findHomography(
                p0, p1, flag, ransacReprojThreshold=float(thr),
                confidence=float(confidence), maxIters=int(max_iter),
            )
        else:
            return None, None
    except cv2.error:
        return None, None
    if M is None or mask is None or getattr(M, "size", 0) == 0:
        return None, None
    inl = mask.ravel().astype(bool)
    if inl.sum() == 0:
        return M, inl
    return M, inl


def rectify_pair(
    pts0: np.ndarray, pts1: np.ndarray, F: np.ndarray, size_wh: tuple[int, int]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            pts0.reshape(-1, 2), pts1.reshape(-1, 2), F, imgSize=size_wh
        )
        return H1, H2
    except cv2.error:
        return None, None


ENGINE = LomaEngine()
