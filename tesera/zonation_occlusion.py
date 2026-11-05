"""
Sliding-window occlusion for the TESERA zonation models, rendered as RGB maps.

Two linear (glmnet gaussian) models on the same 99 PCs used by TESERA-MI:
    absolute zonation (A), relative zonation (R)
Composites, identical to the R script:
    Unzonated = -A        -> green
    Zone 1    =  A - R    -> blue
    Zone 3    =  A + R    -> red

Per-tile attribution = prediction(full slide) - prediction(window removed),
so a positive value means the tiles in that window push the slide toward that
dimension. Composite attributions follow directly from linearity.

Required files in config.PARAMS_DIR:
    pca_center_scale.csv, pca_pc_loadings.csv   (already present for TESERA)
    zonation_coefs.csv                          (from export_zonation_params.R)
    zonation_reference.csv                      (optional, for validation)

Usage (from the repo root):
    python -m tesera.zonation_occlusion --work-dir W --tumor-predictions T \
        --out-dir O --slides S1 S2 ... [--overlay]
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .slide_embedding import _get_slide_encoder, _prepare_tensors

ZONATION_COEFS_FILE = os.path.join(config.PARAMS_DIR, "zonation_coefs.csv")
TARGETS = ("absolute", "relative")
CHANNELS = ("zone3", "unzonated", "zone1")  # order = R, G, B (matches R script)


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class ZonationParams:
    A: np.ndarray  # (EMBED_DIM_SLIDE, 2): slide embedding -> [absolute, relative]
    c: np.ndarray  # (2,)
    n_pcs: int


def load_zonation_params(n_pcs: int = config.N_PCS,
                         params_dir: str = config.PARAMS_DIR) -> ZonationParams:
    cs = pd.read_csv(os.path.join(params_dir, config.PARAM_FILES["pca_center_scale"]))
    feature_order = cs["feature"].tolist()
    center = cs["center"].to_numpy(dtype=float)
    scale = cs["scale"].to_numpy(dtype=float)

    load = pd.read_csv(os.path.join(params_dir, config.PARAM_FILES["pca_pc_loadings"]))
    load = load.set_index("feature").loc[feature_order]
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    missing = [p for p in pc_cols if p not in load.columns]
    if missing:
        raise ValueError(f"Loadings file lacks columns: {missing[:5]}")
    L = load[pc_cols].to_numpy(dtype=float)

    coefs = pd.read_csv(os.path.join(params_dir, ZONATION_COEFS_FILE)).set_index("feature")
    b = coefs.loc["(Intercept)", list(TARGETS)].to_numpy(dtype=float)
    W = coefs.loc[pc_cols, list(TARGETS)].to_numpy(dtype=float)

    # Fold standardize -> PCA -> linear model into one affine map
    A = (L / scale[:, None]) @ W
    c = b - ((center / scale) @ L) @ W
    return ZonationParams(A=A, c=c, n_pcs=n_pcs)


def composites(d_abs: np.ndarray, d_rel: np.ndarray) -> dict:
    return {"unzonated": -d_abs, "zone1": d_abs - d_rel, "zone3": d_abs + d_rel}


def _lattice_step(slide_id: str, work_dir: str) -> tuple[float, float]:
    """Exact tile pitch in thumbnail pixels, recomputed as in tile_embedding.embed_slide."""
    m = pd.read_csv(os.path.join(work_dir, slide_id, config.MPP_NAME)).iloc[0]
    native_px_x = int(config.PATCH_SIZE_MICRONS / float(m["mpp_x"]))
    native_px_y = int(config.PATCH_SIZE_MICRONS / float(m["mpp_y"]))
    return (native_px_x * float(m["thumb_width"]) / float(m["slide_width"]),
            native_px_y * float(m["thumb_height"]) / float(m["slide_height"]))


def _thumbnail_path(slide_id: str, work_dir: str) -> str:
    return os.path.join(work_dir, slide_id, f"{slide_id}{config.THUMBNAIL_SUFFIX}")


# -----------------------------------------------------------------------------
# Occlusion
# -----------------------------------------------------------------------------
def occlude_slide_zonation(slide_id, work_dir, tumor_predictions_csv, out_dir,
                           params: ZonationParams | None = None, device=None,
                           batch_size: int = 8, window_tiles: int = 5,
                           params_dir: str = config.PARAMS_DIR) -> str | None:
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params is None:
        params = load_zonation_params(params_dir=params_dir)
    os.makedirs(out_dir, exist_ok=True)

    prep = _prepare_tensors(slide_id, work_dir, tumor_predictions_csv)
    if prep is None:
        print(f"[zon] {slide_id}: no tumor tiles, skipping")
        return None
    tile_embed, coords_tl, names = prep
    tile_embed = tile_embed.to(device).float()
    coords_tl = coords_tl.to(device)

    slide_dir = os.path.join(work_dir, slide_id)
    coords_df = pd.read_csv(os.path.join(slide_dir, config.COORDS_NAME),
                            usecols=[0, 1, 2], names=["tile_name", "x", "y"])
    lookup = coords_df.set_index("tile_name")[["x", "y"]].to_dict("index")
    coords_raw = np.array([[lookup[n]["x"], lookup[n]["y"]] for n in names],
                          dtype=np.float32)

    encoder = _get_slide_encoder(device)
    A = torch.tensor(params.A, dtype=torch.float32, device=device)
    c = torch.tensor(params.c, dtype=torch.float32, device=device)

    def predict_batched(emb_list, coord_list):
        outs = []
        for emb, coord in zip(emb_list, coord_list):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = encoder(emb, coord)
            outs.append(out[-1].float() @ A + c)  # (1, 2)
        return torch.cat(outs, dim=0)

    n_tiles = tile_embed.shape[1]
    step_x, step_y = _lattice_step(slide_id, work_dir)
    half_x = (window_tiles / 2.0) * step_x
    half_y = (window_tiles / 2.0) * step_y

    with torch.no_grad():
        pred_full = predict_batched([tile_embed], [coords_tl]).cpu().numpy()[0]

    attr = np.zeros((n_tiles, 2), dtype=np.float32)
    n_removed = np.zeros(n_tiles, dtype=np.int32)

    for start in range(0, n_tiles, batch_size):
        end = min(start + batch_size, n_tiles)
        emb_variants, coord_variants = [], []
        for i in range(start, end):
            cx, cy = coords_raw[i]
            in_window = ((np.abs(coords_raw[:, 0] - cx) <= half_x) &
                         (np.abs(coords_raw[:, 1] - cy) <= half_y))
            n_removed[i] = int(in_window.sum())
            keep = ~in_window
            if not keep.any():
                emb_variants.append(tile_embed)
                coord_variants.append(coords_tl)
                continue
            keep_t = torch.from_numpy(np.where(keep)[0]).long().to(device)
            emb_variants.append(tile_embed[:, keep_t, :])
            coord_variants.append(coords_tl[:, keep_t, :])
        with torch.no_grad():
            pred_occl = predict_batched(emb_variants, coord_variants).cpu().numpy()
        for k, i in enumerate(range(start, end)):
            attr[i] = 0.0 if n_removed[i] >= n_tiles else pred_full - pred_occl[k]

    comp = composites(attr[:, 0], attr[:, 1])
    out_df = pd.DataFrame({
        "tile_name": names,
        "x": coords_raw[:, 0],
        "y": coords_raw[:, 1],
        "n_removed": n_removed,
        "attr_absolute": attr[:, 0],
        "attr_relative": attr[:, 1],
        "attr_unzonated": comp["unzonated"],
        "attr_zone1": comp["zone1"],
        "attr_zone3": comp["zone3"],
        "absolute_full": pred_full[0],
        "relative_full": pred_full[1],
    })
    out_path = os.path.join(out_dir, f"{slide_id}_zonation_occlusion.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[zon] {slide_id}: absolute {pred_full[0]:+.4f}, relative {pred_full[1]:+.4f}, "
          f"{n_tiles} tiles -> {out_path}")
    del tile_embed, coords_tl
    return out_path


# -----------------------------------------------------------------------------
# RGB encoding and rendering (no legend, no axes)
# -----------------------------------------------------------------------------
def attributions_to_rgb(df: pd.DataFrame, scaling: str = "slide",
                        clip_pct=(1.0, 99.0), fixed_ranges: dict | None = None) -> np.ndarray:
    """
    scaling:
      "slide"  - each channel min-max rescaled within the slide (like
                 scales::rescale in the R script), after percentile clipping.
      "shared" - one common range across all three channels, so channel
                 brightness reflects relative magnitude between dimensions.
      "fixed"  - user ranges {"zone3": (lo, hi), ...} for cross-slide comparison.
    """
    V = np.stack([df[f"attr_{ch}"].to_numpy(dtype=float) for ch in CHANNELS], axis=1)
    valid = (df["n_removed"].to_numpy() < len(df)) if "n_removed" in df else np.ones(len(df), bool)
    if not valid.any():
        valid[:] = True
    lo_p, hi_p = clip_pct
    if scaling == "slide":
        lo = np.percentile(V[valid], lo_p, axis=0)
        hi = np.percentile(V[valid], hi_p, axis=0)
    elif scaling == "shared":
        lo = np.full(3, np.percentile(V[valid], lo_p))
        hi = np.full(3, np.percentile(V[valid], hi_p))
    elif scaling == "fixed":
        if fixed_ranges is None:
            raise ValueError("scaling='fixed' requires fixed_ranges")
        lo = np.array([fixed_ranges[ch][0] for ch in CHANNELS], dtype=float)
        hi = np.array([fixed_ranges[ch][1] for ch in CHANNELS], dtype=float)
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    span = np.where(hi > lo, hi - lo, 1.0)
    rgb = np.clip((V - lo) / span, 0.0, 1.0)
    rgb[:, hi <= lo] = 0.5
    return rgb


def render_grid(df: pd.DataFrame, rgb: np.ndarray, out_png: str,
                step_xy: tuple[float, float], px_per_tile: int = 8) -> str:
    """One block per tile on the tile lattice, transparent background."""
    from PIL import Image
    xy = df[["x", "y"]].to_numpy(dtype=float)
    col = np.rint((xy[:, 0] - xy[:, 0].min()) / step_xy[0]).astype(int)
    row = np.rint((xy[:, 1] - xy[:, 1].min()) / step_xy[1]).astype(int)
    img = np.zeros((row.max() + 1, col.max() + 1, 4), dtype=np.uint8)
    img[row, col, :3] = np.rint(rgb * 255).astype(np.uint8)
    img[row, col, 3] = 255
    img = np.repeat(np.repeat(img, px_per_tile, axis=0), px_per_tile, axis=1)
    Image.fromarray(img, "RGBA").save(out_png)
    return out_png


def render_overlay(df: pd.DataFrame, rgb: np.ndarray, slide_id: str, work_dir: str,
                   out_png: str, step_xy: tuple[float, float],
                   alpha: float = 0.6) -> str | None:
    """Blend onto the slide thumbnail. Coords are tile centers in thumbnail pixels."""
    from PIL import Image
    path = _thumbnail_path(slide_id, work_dir)
    if not os.path.exists(path):
        print(f"[zon] {slide_id}: thumbnail not found at {path}, skipping overlay")
        return None
    thumb = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0
    H, W = thumb.shape[:2]
    hx, hy = step_xy[0] / 2.0, step_xy[1] / 2.0

    color = np.zeros_like(thumb)
    mask = np.zeros((H, W), dtype=float)
    for (x, y), c in zip(df[["x", "y"]].to_numpy(dtype=float), rgb):
        # int() truncation in embed_slide puts the true center in [x, x+1)
        cx, cy = x + 0.5, y + 0.5
        x0, x1 = max(int(round(cx - hx)), 0), min(int(round(cx + hx)), W)
        y0, y1 = max(int(round(cy - hy)), 0), min(int(round(cy + hy)), H)
        if x1 > x0 and y1 > y0:
            color[y0:y1, x0:x1] = c
            mask[y0:y1, x0:x1] = 1.0
    m = (alpha * mask)[..., None]
    out = thumb * (1 - m) + color * m
    Image.fromarray(np.rint(out * 255).astype(np.uint8), "RGB").save(out_png)
    return out_png


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def process_slide(slide_id, work_dir, tumor_predictions_csv, out_dir, params=None,
                  device=None, batch_size=8, window_tiles=5, scaling="slide",
                  clip_pct=(1.0, 99.0), px_per_tile=8, overlay=False, alpha=0.6,
                  render_only=False) -> dict:
    csv_path = os.path.join(out_dir, f"{slide_id}_zonation_occlusion.csv")
    if not (render_only and os.path.exists(csv_path)):
        if "{slide_id}" in str(tumor_predictions_csv):
            tumor_predictions_csv = str(tumor_predictions_csv).format(slide_id=slide_id)
        csv_path = occlude_slide_zonation(slide_id, work_dir, tumor_predictions_csv, out_dir,
                                          params=params, device=device,
                                          batch_size=batch_size, window_tiles=window_tiles)
        if csv_path is None:
            return {}
    df = pd.read_csv(csv_path)
    rgb = attributions_to_rgb(df, scaling=scaling, clip_pct=clip_pct)
    step_xy = _lattice_step(slide_id, work_dir)
    out = {"csv": csv_path,
           "grid": render_grid(df, rgb, os.path.join(out_dir, f"{slide_id}_zonation_rgb.png"),
                               step_xy, px_per_tile=px_per_tile)}
    if overlay:
        out["overlay"] = render_overlay(
            df, rgb, slide_id, work_dir,
            os.path.join(out_dir, f"{slide_id}_zonation_rgb_overlay.png"),
            step_xy, alpha=alpha)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Zonation occlusion RGB heatmaps")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--tumor-predictions", required=True,
                    help="Path to tumor predictions CSV; may contain {slide_id}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--slides", nargs="*", default=None)
    ap.add_argument("--slides-file", default=None, help="Text file, one slide id per line")
    ap.add_argument("--window-tiles", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--scaling", choices=["slide", "shared"], default="slide")
    ap.add_argument("--clip-low", type=float, default=1.0)
    ap.add_argument("--clip-high", type=float, default=99.0)
    ap.add_argument("--px-per-tile", type=int, default=8)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--render-only", action="store_true",
                    help="Re-render from existing occlusion CSVs")
    args = ap.parse_args(argv)

    slides = list(args.slides or [])
    if args.slides_file:
        with open(args.slides_file) as f:
            slides += [ln.strip() for ln in f if ln.strip()]
    if not slides:
        ap.error("no slides given")

    params = None if args.render_only else load_zonation_params()
    for sid in slides:
        process_slide(sid, args.work_dir, args.tumor_predictions, args.out_dir,
                      params=params, batch_size=args.batch_size,
                      window_tiles=args.window_tiles, scaling=args.scaling,
                      clip_pct=(args.clip_low, args.clip_high),
                      px_per_tile=args.px_per_tile, overlay=args.overlay,
                      alpha=args.alpha, render_only=args.render_only)


if __name__ == "__main__":
    main()
