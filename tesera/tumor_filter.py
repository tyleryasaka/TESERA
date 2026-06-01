from __future__ import annotations

import os

import pandas as pd

from . import config
from .embedding_loader import load_slide_embeddings, embeddings_to_matrix
from .spatial_smoothing import read_coords, smooth_slide


def load_classifier(model_path: str):
    import joblib
    model = joblib.load(model_path)
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = -1
        except Exception:
            pass
    return model


def classify_slide(slide_id: str, work_dir: str, model, out_dir: str,
                   smooth: bool = True, apply_cleanup: bool = config.APPLY_CLEANUP
                   ) -> str | None:
    slide_dir = os.path.join(work_dir, slide_id)
    embeddings = load_slide_embeddings(slide_dir)
    if embeddings is None:
        print(f"[tumor] {slide_id}: no embeddings, skipping")
        return None

    patch_ids, X = embeddings_to_matrix(embeddings)
    p_tumor = model.predict_proba(X)[:, 1]
    pred_df = pd.DataFrame({"patch_id": [str(p) for p in patch_ids],
                            config.P_TUMOR_COL: p_tumor})

    out_csv = os.path.join(out_dir, f"{slide_id}_tile_predictions.csv")
    os.makedirs(out_dir, exist_ok=True)

    coords_csv = os.path.join(slide_dir, config.COORDS_NAME)
    if not smooth or not os.path.exists(coords_csv):
        if smooth:
            print(f"[tumor] {slide_id}: missing coords; writing unsmoothed")
        pred_df.to_csv(out_csv, index=False, float_format="%.6f")
        return out_csv

    coords = read_coords(coords_csv)
    smoothed = smooth_slide(pred_df, coords, apply_cleanup=apply_cleanup)
    smoothed.to_csv(out_csv, index=False, float_format="%.6f")
    n_pos = int(((smoothed[config.TUMOR_STATUS_COL] == 1) &
                 (~smoothed[config.REMOVED_COL])).sum())
    print(f"[tumor] {slide_id}: {len(pred_df)} tiles, {n_pos} tumor after smoothing"
          + (" (+cleanup)" if apply_cleanup else ""))
    return out_csv


def tumor_tile_ids(predictions_csv: str) -> set:
    df = pd.read_csv(predictions_csv)
    df["patch_id"] = df["patch_id"].astype(str)
    if config.TUMOR_STATUS_COL not in df.columns:
        mask = df[config.P_TUMOR_COL] >= config.SMOOTH_THRESHOLD
        return set(df.loc[mask, "patch_id"])
    removed = df.get(config.REMOVED_COL, False)
    if hasattr(removed, "fillna"):
        removed = removed.fillna(False)
    mask = (df[config.TUMOR_STATUS_COL] == 1) & (~removed)
    return set(df.loc[mask, "patch_id"])
