from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config
from .model import TESERAModel
from .slide_embedding import _get_slide_encoder, _prepare_tensors


def _risk_params_as_torch(model: TESERAModel, device):
    import torch
    center = torch.tensor(model.center, dtype=torch.float32, device=device)
    scale = torch.tensor(model.scale, dtype=torch.float32, device=device)
    loadings = torch.tensor(model.loadings, dtype=torch.float32, device=device)
    cox = torch.tensor(model.cox_coefs, dtype=torch.float32, device=device)
    return center, scale, loadings, cox


def occlude_slide(slide_id, work_dir, tumor_predictions_csv, model: TESERAModel,
                  out_dir, device=None, batch_size: int = 8,
                  window_tiles: int = 5) -> str | None:
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    prep = _prepare_tensors(slide_id, work_dir, tumor_predictions_csv)
    if prep is None:
        print(f"[occl] {slide_id}: no tumor tiles, skipping")
        return None
    tile_embed, coords_tl, names = prep
    tile_embed = tile_embed.to(device).float()
    coords_tl = coords_tl.to(device)


    slide_dir = os.path.join(work_dir, slide_id)
    coords_df = pd.read_csv(os.path.join(slide_dir, config.COORDS_NAME),
                            usecols=[0, 1, 2], names=["tile_name", "x", "y"])
    coords_lookup = coords_df.set_index("tile_name")[["x", "y"]].to_dict("index")
    coords_raw = np.array([[coords_lookup[n]["x"], coords_lookup[n]["y"]]
                           for n in names], dtype=np.float32)

    encoder = _get_slide_encoder(device)
    center, scale, loadings, cox = _risk_params_as_torch(model, device)

    def risk_batched(emb_list, coord_list):
        risks = []
        for emb, coord in zip(emb_list, coord_list):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = encoder(emb, coord)
            slide_emb = out[-1].float()
            standardized = (slide_emb - center) / scale
            pc = standardized @ loadings
            risks.append((pc * cox).sum(dim=-1))
        return torch.cat(risks, dim=0)

    n_tiles = tile_embed.shape[1]
    sx = np.sort(np.unique(coords_raw[:, 0]))
    sy = np.sort(np.unique(coords_raw[:, 1]))
    dx, dy = np.diff(sx), np.diff(sy)
    spacing = float(np.median(np.concatenate([dx[dx > 0], dy[dy > 0]])))
    half_window = (window_tiles / 2.0) * spacing

    with torch.no_grad():
        risk_full = float(risk_batched([tile_embed], [coords_tl]).item())

    attributions = np.zeros(n_tiles, dtype=np.float32)
    n_removed = np.zeros(n_tiles, dtype=np.int32)

    for start in range(0, n_tiles, batch_size):
        end = min(start + batch_size, n_tiles)
        emb_variants, coord_variants = [], []
        for i in range(start, end):
            cx, cy = coords_raw[i]
            in_window = ((np.abs(coords_raw[:, 0] - cx) <= half_window) &
                         (np.abs(coords_raw[:, 1] - cy) <= half_window))
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
            risk_occl = risk_batched(emb_variants, coord_variants).cpu().numpy()
        for k, i in enumerate(range(start, end)):
            attributions[i] = 0.0 if n_removed[i] >= n_tiles \
                else risk_full - risk_occl[k]

    out_df = pd.DataFrame({
        "tile_name": names,
        "x": coords_raw[:, 0],
        "y": coords_raw[:, 1],
        "attribution": attributions,
        "n_removed": n_removed,
    })
    out_path = os.path.join(out_dir, f"{slide_id}_window_occlusion.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[occl] {slide_id}: full risk {risk_full:+.4f}, "
          f"{n_tiles} tiles -> {out_path}")
    del tile_embed, coords_tl
    return out_path
