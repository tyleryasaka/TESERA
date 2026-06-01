from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config
from .embedding_loader import load_slide_embeddings
from .tumor_filter import tumor_tile_ids

_SLIDE_ENCODER = None


def _get_slide_encoder(device):
    global _SLIDE_ENCODER
    if _SLIDE_ENCODER is None:
        config.load_env()
        from gigapath.slide_encoder import create_model
        enc = create_model(config.SLIDE_ENCODER_HF, config.SLIDE_ENCODER_NAME,
                           config.SLIDE_ENCODER_IN_DIM, global_pool=True)
        _SLIDE_ENCODER = enc.to(device).eval()
        for p in _SLIDE_ENCODER.parameters():
            p.requires_grad_(False)
    return _SLIDE_ENCODER


def _prepare_tensors(slide_id, work_dir, tumor_predictions_csv=None):
    import torch

    slide_dir = os.path.join(work_dir, slide_id)
    embeddings = load_slide_embeddings(slide_dir)
    if embeddings is None:
        return None

    if tumor_predictions_csv is not None and os.path.exists(tumor_predictions_csv):
        keep = tumor_tile_ids(tumor_predictions_csv)
        embeddings = {k: v for k, v in embeddings.items() if k in keep}
    if len(embeddings) == 0:
        return None

    coords_path = os.path.join(slide_dir, config.COORDS_NAME)
    patch_size_path = os.path.join(slide_dir, config.THUMB_PATCH_SIZE_NAME)
    coords_df = pd.read_csv(coords_path, usecols=[0, 1, 2],
                            names=["tile_name", "x", "y"])
    coords_dict = coords_df.set_index("tile_name")[["x", "y"]].to_dict("index")
    ps = pd.read_csv(patch_size_path, usecols=[0, 1], names=["x", "y"])
    tile_size_thumb = float(ps.loc[0, "x"])

    emb_list, xy_list, names = [], [], []
    for name in embeddings:
        if name in coords_dict:
            emb_list.append(embeddings[name])
            xy_list.append([coords_dict[name]["x"], coords_dict[name]["y"]])
            names.append(name)
    if len(emb_list) == 0:
        return None

    tile_embed = torch.stack(emb_list).unsqueeze(0)
    coords = torch.tensor(xy_list, dtype=torch.float).unsqueeze(0)
    coords = coords * (256.0 / tile_size_thumb) - 128.0
    return tile_embed, coords, names


def embed_slide_level(slide_id, work_dir, tumor_predictions_csv=None,
                      device=None) -> np.ndarray | None:
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prep = _prepare_tensors(slide_id, work_dir, tumor_predictions_csv)
    if prep is None:
        print(f"[slide] {slide_id}: no tumor tiles / coords, skipping")
        return None
    tile_embed, coords, _ = prep
    encoder = _get_slide_encoder(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        out = encoder(tile_embed.to(device), coords.to(device))
    slide_emb = out[-1].squeeze().float().cpu().numpy()
    print(f"[slide] {slide_id}: {tile_embed.shape[1]} tiles -> dim {slide_emb.shape[0]}")
    return slide_emb
