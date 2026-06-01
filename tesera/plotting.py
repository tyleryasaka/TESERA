from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config

HEATMAP_ALPHA = 0.6


def _make_heatmap(thumbnail, xs, ys, attributions, cmap_name, vmin, vmax,
                  smoothing_factor):
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.ndimage import gaussian_filter, median_filter

    cmap = matplotlib.colormaps[cmap_name]
    norm = Normalize(vmin=vmin, vmax=vmax)
    H, W = thumbnail.shape[:2]

    sx = np.sort(np.unique(xs))
    sy = np.sort(np.unique(ys))
    xsp, ysp = np.diff(sx), np.diff(sy)
    spacing = np.median(np.concatenate([xsp[xsp > 0], ysp[ysp > 0]]))
    paint = int(np.ceil(spacing)) + 1
    half = paint // 2
    sigma = smoothing_factor * paint / 4

    grid_attr = np.zeros((H, W), dtype=np.float64)
    grid_count = np.zeros((H, W), dtype=np.float64)
    attrs = np.clip(attributions, vmin, vmax)
    for xc, yc, a in zip(xs, ys, attrs):
        x0, y0 = int(round(xc)) - half, int(round(yc)) - half
        x1, y1 = x0 + paint, y0 + paint
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        grid_attr[y0:y1, x0:x1] += a
        grid_count[y0:y1, x0:x1] += 1.0

    mask = grid_count > 0
    avg = np.where(mask, grid_attr / np.maximum(grid_count, 1), 0).astype(np.float32)
    filt = median_filter(avg, size=max(3, paint // 2))
    sm = gaussian_filter(filt, sigma=sigma)
    sm_mask = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    norm_sm = np.where(sm_mask > 0.05, sm / np.maximum(sm_mask, 1e-6), 0)
    display = np.where(mask, norm_sm, np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(thumbnail)
    ax.imshow(np.ma.masked_invalid(display), cmap=cmap, norm=norm,
              alpha=HEATMAP_ALPHA)
    ax.axis("off")
    sm_obj = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_obj.set_array([])
    fig.colorbar(sm_obj, ax=[ax], fraction=0.015, pad=0.02, shrink=0.5)
    return fig


def plot_occlusion_heatmaps(slide_ids, occlusion_dir, work_dir, out_dir,
                            percentile=90.0, window_tiles=5,
                            cmaps=("magma",), thumbnail_suffix=None,
                            thumb_patch_size_name=None):
    import matplotlib.pyplot as plt
    from PIL import Image

    if thumbnail_suffix is None:
        thumbnail_suffix = config.THUMBNAIL_SUFFIX
    if thumb_patch_size_name is None:
        thumb_patch_size_name = config.THUMB_PATCH_SIZE_NAME

    os.makedirs(out_dir, exist_ok=True)
    info, pooled = [], []
    for sid in slide_ids:
        occ_csv = os.path.join(occlusion_dir, f"{sid}_window_occlusion.csv")
        thumb = os.path.join(work_dir, sid, f"{sid}{thumbnail_suffix}")
        psz = os.path.join(work_dir, sid, thumb_patch_size_name)
        if not all(os.path.exists(p) for p in (occ_csv, thumb, psz)):
            print(f"[plot] {sid}: skipped (missing files)")
            continue
        occ = pd.read_csv(occ_csv)
        raw = occ["attribution"].to_numpy(dtype=np.float64)
        centered = raw - float(np.mean(raw))
        info.append({"sample": sid, "occ": occ, "centered": centered,
                     "thumb": thumb})
        pooled.append(centered)

    if not info:
        print("[plot] no valid occlusion slides")
        return []

    pooled = np.concatenate(pooled)
    lo = (100 - percentile) / 2
    vmin = float(np.percentile(pooled, lo))
    vmax = float(np.percentile(pooled, 100 - lo))
    if vmin == vmax:
        print("[plot] no variation in attributions; aborting heatmaps")
        return []

    written = []
    for it in info:
        thumbnail = np.array(Image.open(it["thumb"]).convert("RGB"))
        xs = it["occ"]["x"].to_numpy()
        ys = it["occ"]["y"].to_numpy()
        for cmap_name in cmaps:
            fig = _make_heatmap(thumbnail, xs, ys, it["centered"], cmap_name,
                                vmin, vmax, smoothing_factor=window_tiles)
            out = os.path.join(out_dir,
                               f"{it['sample']}_occlusion.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
            print(f"[plot] wrote {out}")
    return written


def plot_pca(results, reference_csv, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ref = pd.read_csv(reference_csv)
    tes_colors = {"TES1": "#F08080", "TES2": "#20B2AA"}

    pc1 = np.array([r.pc_scores[0] for r in results])
    pc2 = np.array([r.pc_scores[1] for r in results])
    clusters = np.array([r.tes_cluster for r in results])

    fig, ax = plt.subplots(figsize=(6, 6))
    for tes, color in tes_colors.items():
        m = clusters == tes
        if m.any():
            ax.scatter(pc1[m], pc2[m], s=22, c=color, alpha=0.8,
                       edgecolors="none")

    def _limits(ref_vals, user_vals):
        lo, hi = float(ref_vals.min()), float(ref_vals.max())
        pad = 0.03 * (hi - lo)
        umin, umax = float(user_vals.min()), float(user_vals.max())
        if umin < lo:
            lo = umin - pad
        if umax > hi:
            hi = umax + pad
        return lo, hi

    ax.set_xlim(*_limits(ref["PC1"], pc1))
    ax.set_ylim(*_limits(ref["PC2"], pc2))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("TESERA Subtype")

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=c, markeredgecolor="none", label=t)
               for t, c in tes_colors.items()]
    ax.legend(handles=handles, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path
