from __future__ import annotations

import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .engine import (
    DEFAULT_MODEL_KEY,
    ENGINE,
    MODELS,
    MatchData,
    rectify_pair,
    ransac_filter,
    select_matches,
    device_str,
)
from . import viz
from .io_export import make_csv, make_json, make_npz

DESCRIPTION = """
<center><b><font size="8">LoMa <font color="#6589bf">Matcher</font></font></b></center>

<center><b>LoMa: Local Feature Matching Revisited</b> (ECCV 2026)</center>
"""

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LONG_MAP = {"オリジナル": 0, "2048": 2048, "1600": 1600, "1280": 1280, "1024": 1024}

EXAMPLES: dict[str, dict] = {
    "① toronto_A/B ・ 標準 (Fundamental)": dict(
        a="toronto_A.jpg", b="toronto_B.jpg", thr=0.10, nk=2048,
        model=DEFAULT_MODEL_KEY, rm="CV2_USAC_MAGSAC", rt=4.0, rc=0.999,
        ri=10000, geom="Fundamental", ls="オリジナル"),
    "② 0015_A/B ・ 難ペア (Fundamental・高精度RANSAC)": dict(
        a="0015_A.jpg", b="0015_B.jpg", thr=0.10, nk=2048,
        model=DEFAULT_MODEL_KEY, rm="CV2_USAC_MAGSAC", rt=0.5, rc=0.999999,
        ri=20000, geom="Fundamental", ls="オリジナル"),
    "③ 0022_A/B ・ Homography・長辺1600": dict(
        a="0022_A.jpg", b="0022_B.jpg", thr=0.10, nk=2048,
        model=DEFAULT_MODEL_KEY, rm="CV2_USAC_MAGSAC", rt=4.0, rc=0.999,
        ri=10000, geom="Homography", ls="1600"),
    "④ toronto_A/B ・ LoMa-G (最重量)": dict(
        a="toronto_A.jpg", b="toronto_B.jpg", thr=0.10, nk=2048,
        model="LoMa-G (最重量・最高精度)", rm="CV2_USAC_MAGSAC", rt=4.0,
        rc=0.999, ri=10000, geom="Fundamental", ls="オリジナル"),
}


def load_example(key: str):
    cfg = EXAMPLES[key]

    def rd(name: str) -> np.ndarray:
        img = cv2.imread(str(ASSETS_DIR / name))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return (rd(cfg["a"]), rd(cfg["b"]), cfg["thr"], cfg["nk"], cfg["model"],
            cfg["rm"], cfg["rt"], cfg["rc"], cfg["ri"], cfg["geom"], cfg["ls"])

RUN_OUTPUTS_KEYS = ["view", "summary", "hist", "geom", "stats", "gj", "f_npz", "f_csv", "f_json"]


def _full_inlier_mask(md: MatchData, sel_mask: np.ndarray,
                      sub: np.ndarray | None) -> np.ndarray | None:
    if sub is None:
        return None
    full = np.zeros(len(md.conf), dtype=bool)
    idx = np.where(sel_mask)[0]
    full[idx[sub.astype(bool)]] = True
    return full


def _render_summary(st: dict) -> str:
    if not st or st.get("md") is None:
        return ""
    md: MatchData = st["md"]
    sel = int(st.get("sel_count", 0))
    inl = int(st["inl_full"].sum()) if st.get("inl_full") is not None else 0
    lines = [
        f"**モデル**: {st['model']} ・ keypoints {int(st.get('num_kpts', 0))} / 枚",
        f"**threshold {st['thr']:.3f} ⇒ matches {sel} ｜ RANSAC inliers({st['geometry']}) {inl}**",
        f"detect {st['t_det']:.2f}s ・ match {st['t_match']:.2f}s ・ RANSAC {st['t_ran']:.3f}s",
        f"image: {md.w0}×{md.h0} ↔ {md.w1}×{md.h1}",
    ]
    note = st.get("geom_note") or ""
    if note:
        lines.append(f"_{note}_")
    elif st.get("M") is None and st.get("t_ran"):
        lines.append("_幾何推定できませんでした（inlier不足）_")
    return "\n\n".join(lines)


def _draw_view(st: dict, show_mode: str) -> np.ndarray | None:
    md: MatchData | None = st.get("md")
    if md is None:
        return None
    thr = float(st["thr"])
    sel = md.conf > thr
    p0, p1, cf = select_matches(md, thr)
    inl_full = st.get("inl_full")
    mode_map = {"全一致": "all", "inlierのみ": "inlier", "outlierのみ": "outlier"}
    sub_inl = inl_full[sel] if inl_full is not None else None
    return viz.draw_matches(st["rgb0"], st["rgb1"], p0, p1, cf,
                            mode=mode_map.get(show_mode, "all"),
                            inlier_mask=sub_inl)


def _render_hist(md: MatchData, thr: float) -> np.ndarray:
    return viz.confidence_histogram(md.conf, float(thr))


def _render_geom(st: dict, alpha: float, checker: bool) -> tuple[np.ndarray | None, str]:
    M = st.get("M")
    if M is None:
        return None, ""
    if st["geometry"] == "Homography":
        img = viz.warp_blend(st["rgb0"], st["rgb1"], M, alpha=float(alpha), checker=bool(checker))
        note = "Image1 を Homography warp → Image0 に重ね合わせ"
        return img, note
    h1h2 = st.get("H1H2")
    if h1h2 is not None and all(v is not None for v in h1h2):
        img = viz.rectified_pair(st["rgb0"], st["rgb1"], h1h2[0], h1h2[1])
        return img, "Fundamental によるステレオ平行化表示"
    md: MatchData = st["md"]
    sel = md.conf > float(st["thr"])
    p0, p1, _ = select_matches(md, st["thr"])
    inl = st.get("inl_full")
    if inl is not None:
        keep = inl[sel]
        p0, p1 = p0[keep], p1[keep]
    if len(p0) < 8:
        return None, "エピポーラ線には inlier ≥ 8 必要"
    img = viz.epipolar_overlay(st["rgb0"], st["rgb1"], p0, p1, M)
    return img, "エピポーラ線オーバーレイ（平行化が不安定だったため代替表示）"


def _build_stats(st: dict) -> dict:
    md: MatchData = st["md"]
    inl = int(st["inl_full"].sum()) if st.get("inl_full") is not None else 0
    return {
        "device": ENGINE.device,
        "model": st["model"],
        "raw_mutual_matches": len(md.conf),
        "matches_at_threshold": int(st.get("sel_count", 0)),
        "ransac_inliers": inl,
        "geometry": st["geometry"],
        "threshold": st["thr"],
        "num_keypoints_per_image": st.get("num_kpts"),
        "timings_sec": {
            "detect_total": round(st["t_det"], 3),
            "match": round(st["t_match"], 3),
            "ransac": round(st["t_ran"], 4),
        },
        "sizes": {"image0": [md.w0, md.h0], "image1": [md.w1, md.h1]},
    }


def _do_ransac(st: dict, rmethod, rthr, rconf, riter, geometry, alpha, checker) -> tuple:
    md: MatchData = st["md"]
    sel = md.conf > float(st["thr"])
    p0, p1, cf = select_matches(md, st["thr"])
    t0 = time.time()
    M, sub = ransac_filter(p0, p1, geometry, method=rmethod, thr=rthr,
                           confidence=rconf, max_iter=int(riter))
    st["t_ran"] = time.time() - t0
    st["M"] = M
    st["geometry"] = geometry
    st["inl_full"] = _full_inlier_mask(md, sel, sub)
    st["sel_count"] = int(sel.sum())
    st["H1H2"] = None
    if M is not None and geometry == "Fundamental" and sub is not None and sub.sum() >= 8:
        hh = rectify_pair(p0[sub], p1[sub], M, (md.w0, md.h0))
        st["H1H2"] = hh
    canvas = _draw_view(st, "全一致")
    gimg, gnote = _render_geom(st, alpha, checker)
    st["geom_note"] = gnote
    summary = _render_summary(st)
    stats = _build_stats(st)
    gj: dict = {}
    if M is not None:
        key = "F" if geometry == "Fundamental" else "H"
        gj[key] = np.asarray(M).tolist()
        if st.get("H1H2") and all(v is not None for v in st["H1H2"]):
            gj["H1"] = np.asarray(st["H1H2"][0]).tolist()
            gj["H2"] = np.asarray(st["H1H2"][1]).tolist()
    f_npz = make_npz(p0, p1, cf, M, st.get("inl_full"), geometry)
    f_csv = make_csv(p0, p1, cf, st.get("inl_full"))
    f_json = make_json(stats)
    return canvas, summary, gimg, stats, gj, f_npz, f_csv, f_json


def run_matching(image0, image1, thr, num_kpts, model_key, ransac_method,
                 ransac_thr, ransac_conf, ransac_iters, geometry, long_side_label,
                 progress=gr.Progress(track_tqdm=True)):
    return _run_core(image0, image1, thr, num_kpts, model_key, ransac_method,
                     ransac_thr, ransac_conf, ransac_iters, geometry,
                     long_side_label)


def run_example(key: str, progress=gr.Progress(track_tqdm=True)):
    cfg = EXAMPLES.get(key)
    if cfg is None:
        raise gr.skip()
    img0 = _load_asset(cfg["a"])
    img1 = _load_asset(cfg["b"])
    out = _run_core(img0, img1, cfg["thr"], cfg["nk"], cfg["model"], cfg["rm"],
                    cfg["rt"], cfg["rc"], cfg["ri"], cfg["geom"], cfg["ls"])
    fills = (img0, img1, cfg["thr"], cfg["nk"], cfg["model"], cfg["rm"],
             cfg["rt"], cfg["rc"], cfg["ri"], cfg["geom"], cfg["ls"])
    return tuple(out) + fills


def _load_asset(name: str) -> np.ndarray:
    img = cv2.imread(str(ASSETS_DIR / name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _run_core(image0, image1, thr, num_kpts, model_key, ransac_method,
              ransac_thr, ransac_conf, ransac_iters, geometry, long_side_label):
    if image0 is None or image1 is None:
        raise gr.Error("両方の画像をアップロード（または例を選択）してください")
    try:
        long_side = LONG_MAP.get(str(long_side_label), int(long_side_label or 0)) \
            if not isinstance(long_side_label, str) else LONG_MAP.get(long_side_label, 0)
        t0 = time.time()
        load_status = ENGINE.load_model(model_key)
        print(f"[loma-app] {load_status}")
        p0, p1, md = ENGINE.run_pair(
            np.asarray(image0), np.asarray(image1),
            model_key=model_key, num_keypoints=int(num_kpts),
            long_side=int(long_side),
        )
        t_det = time.time() - t0
        t0 = time.time()
        _ = float(md.conf.mean())
        t_match = time.time() - t0
        st = {
            "md": md,
            "p0": str(p0), "p1": str(p1),
            "rgb0": ENGINE.load_rgb(p0), "rgb1": ENGINE.load_rgb(p1),
            "thr": float(thr), "sel_count": int((md.conf > float(thr)).sum()),
            "model": model_key, "num_kpts": int(num_kpts), "long_side": int(long_side),
            "rmethod": ransac_method, "rthr": float(ransac_thr),
            "rconf": float(ransac_conf), "riter": int(ransac_iters),
            "geometry": geometry, "alpha": 0.55, "checker": False,
            "M": None, "H1H2": None, "inl_full": None,
            "t_det": t_det, "t_match": t_match, "t_ran": 0.0,
        }
        view, summary, gimg, stats, gj, f1, f2, f3 = _do_ransac(
            st, st["rmethod"], st["rthr"], st["rconf"], st["riter"],
            geometry, 0.55, False)
        hist = _render_hist(md, float(thr))
        return view, summary, hist, gimg, stats, gj, f1, f2, f3, st
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            import torch
            torch.cuda.empty_cache()
            raise gr.Error("GPUメモリ不足です。長辺リサイズ短縮・キー点削減・軽量モデルを試してください") from e
        raise gr.Error(f"処理中にエラーが発生しました: {msg[:300]}") from e


def on_threshold_change(state, show_mode, thr):
    if not isinstance(state, dict) or state.get("md") is None:
        return None, "", None, state
    state["thr"] = float(thr)
    state["sel_count"] = int((state["md"].conf > float(thr)).sum())
    canvas = _draw_view(state, show_mode)
    hist = _render_hist(state["md"], float(thr))
    return canvas, _render_summary(state), hist, state


def on_show_change(state, show_mode):
    if not isinstance(state, dict) or state.get("md") is None:
        return None, state
    return _draw_view(state, show_mode), state


def on_rerun_ransac(state, show_mode_unused, ransac_method, ransac_thr, ransac_conf,
                    ransac_iters, geometry, alpha, checker):
    if not isinstance(state, dict) or state.get("md") is None:
        gr.Warning("先に Run Match を実行してください")
        return None, "", None, {}, {}, None, None, None, state
    state.update({"rmethod": ransac_method, "rthr": float(ransac_thr),
                  "rconf": float(ransac_conf), "riter": int(ransac_iters)})
    view, summary, gimg, stats, gj, f1, f2, f3 = _do_ransac(
        state, ransac_method, ransac_thr, ransac_conf, ransac_iters,
        geometry, alpha, checker)
    return view, summary, gimg, stats, gj, f1, f2, f3, state


def on_geom_controls(state, alpha, checker):
    if not isinstance(state, dict) or state.get("md") is None:
        return None, state
    state["alpha"] = float(alpha)
    state["checker"] = bool(checker)
    img, _ = _render_geom(state, float(alpha), bool(checker))
    return img, state


RESET_VALUES = (
    None, None,
    0.10, 2048, DEFAULT_MODEL_KEY, "CV2_USAC_MAGSAC",
    4.0, 0.999, 10000, "Fundamental", "オリジナル",
    None, "", None, None, {}, {}, None, None, None, {},
)


CUSTOM_CSS = """
#main-view img {min-height: 480px; object-fit: contain;}
#match-summary {padding: 4px 10px;}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LoMa Matcher") as demo:
        gr.Markdown(DESCRIPTION)
        gr.Markdown(f"Device: **{device_str()}** ・ 初回Run時に選択モデルの重みを自動DLします")

        state = gr.State({})
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                example_dd = gr.Dropdown(list(EXAMPLES.keys()),
                                         label="動作確認用 Example（選択で自動実行＋パラメータ反映）",
                                         interactive=True, type="value")
                model_dd = gr.Dropdown(list(MODELS.keys()), value=DEFAULT_MODEL_KEY,
                                       label="Matching Model", interactive=True)
                with gr.Row():
                    image0 = gr.Image(label="Image 0", type="numpy", image_mode="RGB",
                                      height=300, sources=["upload", "webcam", "clipboard"])
                    image1 = gr.Image(label="Image 1", type="numpy", image_mode="RGB",
                                      height=300, sources=["upload", "webcam", "clipboard"])
                with gr.Row():
                    btn_run = gr.Button("Run Match", variant="primary", scale=3)
                    btn_reset = gr.Button("Reset", scale=1)

                with gr.Accordion("Matching 設定", open=True):
                    thr = gr.Slider(0.0, 0.99, value=0.10, step=0.001,
                                    label="Match threshold（Run後ライブ反映）")
                    with gr.Row():
                        num_kpts = gr.Dropdown([512, 1024, 2048, 4096, 8192],
                                               value=2048, label="Max keypoints",
                                               allow_custom_value=False)
                        long_side = gr.Dropdown(list(LONG_MAP.keys()), value="オリジナル",
                                                label="入力長辺リサイズ", interactive=True)

                with gr.Accordion("RANSAC 設定", open=True):
                    with gr.Row():
                        ransac_method = gr.Dropdown(["CV2_USAC_MAGSAC", "CV2_USAC_ACCURATE",
                                                     "CV2_USAC_DEFAULT", "CV2_RANSAC"],
                                                    value="CV2_USAC_MAGSAC",
                                                    label="RANSAC method")
                        geometry = gr.Radio(["Fundamental", "Homography"],
                                            value="Fundamental", label="Geometry")
                    with gr.Row():
                        ransac_thr = gr.Slider(0.1, 20.0, value=4.0, step=0.1,
                                               label="Reproj threshold [px]")
                        ransac_conf = gr.Slider(0.5, 0.999999, value=0.999, step=1e-6,
                                                label="Confidence")
                    ransac_iters = gr.Number(value=10000, precision=0, label="Max iterations")
                    btn_ransac = gr.Button("RANSAC のみ再実行", variant="secondary")

            with gr.Column(scale=7):
                summary_md = gr.Markdown(elem_id="match-summary")
                with gr.Tabs():
                    with gr.Tab("Matches"):
                        view_img = gr.Image(label="Matches", type="numpy", interactive=False,
                                            elem_id="main-view")
                        show_mode = gr.Radio(["全一致", "inlierのみ", "outlierのみ"],
                                             value="全一致", label="表示切替",
                                             interactive=True)
                    with gr.Tab("Threshold Explorer"):
                        hist_img = gr.Image(label="Match confidence histogram",
                                            type="numpy", interactive=False)
                        gr.Markdown("**濃青**: 採用 (≥ threshold) ／ **淡色**: 除外。"
                                    "左の threshold スライダーで即時更新（モデル再計算なし）")
                    with gr.Tab("Geometry"):
                        geom_img = gr.Image(label="Geometry view", type="numpy",
                                            interactive=False, elem_id="main-view")
                        with gr.Row():
                            alpha_s = gr.Slider(0.0, 1.0, value=0.55, step=0.01,
                                                label="Blend alpha (Homography)")
                            checker_cb = gr.Checkbox(False, label="Checkerboard blend")
                with gr.Row():
                    stats_json = gr.JSON(label="Matches Statistics")
                    geom_json = gr.JSON(label="Geometry (F / H)")
                with gr.Row():
                    f_npz = gr.File(label="⬇ npz")
                    f_csv = gr.File(label="⬇ csv")
                    f_json = gr.File(label="⬇ json")

        RUN_INPUTS = [image0, image1, thr, num_kpts, model_dd, ransac_method,
                      ransac_thr, ransac_conf, ransac_iters, geometry, long_side]
        RUN_OUTPUTS = [view_img, summary_md, hist_img, geom_img, stats_json,
                       geom_json, f_npz, f_csv, f_json, state]

        btn_run.click(run_matching, inputs=RUN_INPUTS, outputs=RUN_OUTPUTS,
                      api_name="run_match")

        thr.release(on_threshold_change,
                    inputs=[state, show_mode, thr],
                    outputs=[view_img, summary_md, hist_img, state],
                    show_progress="minimal")
        thr.change(on_threshold_change,
                   inputs=[state, show_mode, thr],
                   outputs=[view_img, summary_md, hist_img, state],
                   show_progress="minimal")
        show_mode.change(on_show_change, inputs=[state, show_mode],
                         outputs=[view_img, state])

        ransac_reinputs = [state, show_mode, ransac_method, ransac_thr,
                           ransac_conf, ransac_iters, geometry, alpha_s, checker_cb]
        ransac_reoutputs = [view_img, summary_md, geom_img, stats_json,
                            geom_json, f_npz, f_csv, f_json, state]
        btn_ransac.click(on_rerun_ransac, inputs=ransac_reinputs, outputs=ransac_reoutputs)
        geometry.change(on_rerun_ransac, inputs=ransac_reinputs, outputs=ransac_reoutputs)

        for comp in (alpha_s, checker_cb):
            comp.change(on_geom_controls, inputs=[state, alpha_s, checker_cb],
                        outputs=[geom_img, state])

        RESET_OUTPUTS = [image0, image1, thr, num_kpts, model_dd, ransac_method,
                         ransac_thr, ransac_conf, ransac_iters, geometry, long_side,
                         view_img, summary_md, hist_img, geom_img, stats_json,
                         geom_json, f_npz, f_csv, f_json, state]
        assert len(RESET_OUTPUTS) == len(RESET_VALUES)
        btn_reset.click(lambda: RESET_VALUES, inputs=[], outputs=RESET_OUTPUTS,
                        api_name="reset")

        example_dd.select(run_example, inputs=example_dd,
                          outputs=RUN_OUTPUTS + RUN_INPUTS,
                          api_name="run_example")
        return demo
