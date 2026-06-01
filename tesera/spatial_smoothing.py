from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from . import config


def read_coords(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["patch_id", "thumb_cx", "thumb_cy"])
    df["patch_id"] = df["patch_id"].astype(str)
    return df.drop_duplicates("patch_id")


def masked_gaussian_prob(xs, ys, probs, sigma_tiles=config.SMOOTH_SIGMA_TILES):
    from scipy.ndimage import gaussian_filter

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    probs = np.asarray(probs, dtype=float)

    sx = np.sort(np.unique(xs))
    sy = np.sort(np.unique(ys))
    dx = np.diff(sx)
    dy = np.diff(sy)
    gaps = np.concatenate([dx[dx > 0], dy[dy > 0]])
    if gaps.size == 0:
        return probs.copy()
    spacing = float(np.median(gaps))

    cols = np.round((xs - xs.min()) / spacing).astype(int)
    rows = np.round((ys - ys.min()) / spacing).astype(int)
    n_cols = int(cols.max()) + 1
    n_rows = int(rows.max()) + 1

    grid_prob = np.zeros((n_rows, n_cols), dtype=np.float32)
    grid_mask = np.zeros((n_rows, n_cols), dtype=np.float32)
    grid_prob[rows, cols] = probs
    grid_mask[rows, cols] = 1.0

    sm = gaussian_filter(grid_prob, sigma=sigma_tiles)
    smm = gaussian_filter(grid_mask, sigma=sigma_tiles)
    smn = np.where(smm > 1e-3, sm / np.maximum(smm, 1e-6), 0.0)
    return smn[rows, cols]


def snap_to_grid(vals: np.ndarray, step: float) -> np.ndarray:
    sorted_unique = np.sort(np.unique(vals))
    grid_idx = np.zeros(len(sorted_unique), dtype=np.int64)
    for k in range(1, len(sorted_unique)):
        if sorted_unique[k] - sorted_unique[k - 1] < step * 0.5:
            grid_idx[k] = grid_idx[k - 1]
        else:
            grid_idx[k] = grid_idx[k - 1] + 1
    lookup = {v: i for v, i in zip(sorted_unique, grid_idx)}
    return np.array([lookup[v] for v in vals], dtype=np.int64)


def infer_step(vals: np.ndarray) -> float:
    sorted_unique = np.sort(np.unique(vals))
    if len(sorted_unique) < 2:
        return 1.0
    diffs = np.diff(sorted_unique)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    vals_u, counts = np.unique(np.round(diffs, 6), return_counts=True)
    return float(vals_u[np.argmax(counts)])


def connected_components_4(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    n = len(gx)
    cell_to_idx = {(int(gx[i]), int(gy[i])): i for i in range(n)}
    comp = np.full(n, -1, dtype=np.int64)
    current = -1
    for start in range(n):
        if comp[start] != -1:
            continue
        current += 1
        comp[start] = current
        dq = deque([start])
        while dq:
            i = dq.popleft()
            cx, cy = int(gx[i]), int(gy[i])
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = cell_to_idx.get((cx + ddx, cy + ddy))
                if j is not None and comp[j] == -1:
                    comp[j] = current
                    dq.append(j)
    return comp


def _debris_min_tiles(debris_area_mm2: float) -> int:
    tile_mm = config.PATCH_SIZE_MICRONS / 1000.0
    return max(1, int(np.floor(debris_area_mm2 / (tile_mm * tile_mm))))


def remove_debris(df, min_component_tiles):
    n = len(df)
    if n == 0:
        return df
    x = df["thumb_cx"].to_numpy(dtype=float)
    y = df["thumb_cy"].to_numpy(dtype=float)
    gx = snap_to_grid(x, infer_step(x))
    gy = snap_to_grid(y, infer_step(y))
    comp = connected_components_4(gx, gy)
    sizes = np.bincount(comp)
    return df.iloc[sizes[comp] >= min_component_tiles].reset_index(drop=True)


def prune_thin_structures(df, min_neighbors=config.MIN_NEIGHBORS):
    n = len(df)
    if n == 0:
        return df
    x = df["thumb_cx"].to_numpy(dtype=float)
    y = df["thumb_cy"].to_numpy(dtype=float)
    gx = snap_to_grid(x, infer_step(x))
    gy = snap_to_grid(y, infer_step(y))

    keep = np.ones(n, dtype=bool)
    while True:
        active = np.where(keep)[0]
        if len(active) == 0:
            break
        gxa = gx[active].tolist()
        gya = gy[active].tolist()
        aset = set(zip(gxa, gya))
        m = len(active)

        def count(dx, dy):
            return np.fromiter(
                ((gxa[k] + dx, gya[k] + dy) in aset for k in range(m)),
                dtype=np.int8, count=m)

        nb = count(1, 0) + count(-1, 0) + count(0, 1) + count(0, -1)
        to_remove = nb < min_neighbors
        if not to_remove.any():
            break
        keep[active[to_remove]] = False
    return df.iloc[keep].reset_index(drop=True)


def smooth_slide(pred_df, coords_df,
                 sigma_tiles=config.SMOOTH_SIGMA_TILES,
                 threshold=config.SMOOTH_THRESHOLD,
                 apply_cleanup=config.APPLY_CLEANUP,
                 debris_area_mm2=config.DEBRIS_AREA_MM2,
                 min_neighbors=config.MIN_NEIGHBORS):
    pred_df = pred_df.copy()
    pred_df["patch_id"] = pred_df["patch_id"].astype(str)
    coords_df = coords_df.copy()
    coords_df["patch_id"] = coords_df["patch_id"].astype(str)

    df = pred_df.merge(coords_df, on="patch_id", how="inner")
    if len(df) == 0:
        out = pred_df.copy()
        out[config.TUMOR_STATUS_COL] = 0
        out[config.REMOVED_COL] = True
        return out[["patch_id", config.P_TUMOR_COL,
                    config.TUMOR_STATUS_COL, config.REMOVED_COL]]


    smp = masked_gaussian_prob(df["thumb_cx"], df["thumb_cy"],
                               df[config.P_TUMOR_COL], sigma_tiles)
    df[config.TUMOR_STATUS_COL] = (smp >= threshold).astype(int)


    if apply_cleanup:
        min_tiles = _debris_min_tiles(debris_area_mm2)
        clean = remove_debris(df, min_tiles)
        clean = prune_thin_structures(clean, min_neighbors)
        clean = remove_debris(clean, min_tiles)
        surviving = set(clean["patch_id"].tolist())
        df[config.REMOVED_COL] = ~df["patch_id"].isin(surviving)
    else:
        df[config.REMOVED_COL] = False

    out = pred_df.merge(
        df[["patch_id", config.TUMOR_STATUS_COL, config.REMOVED_COL]],
        on="patch_id", how="left")
    out[config.TUMOR_STATUS_COL] = out[config.TUMOR_STATUS_COL].fillna(0).astype(int)

    out[config.REMOVED_COL] = out[config.REMOVED_COL].fillna(True)
    return out[["patch_id", config.P_TUMOR_COL,
                config.TUMOR_STATUS_COL, config.REMOVED_COL]]
