from __future__ import annotations

import os

from . import config


def load_slide_embeddings(slide_dir: str):
    import torch

    path = os.path.join(slide_dir, config.EMBED_CACHE_NAME)
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = data.get("embeddings")
    if not embeddings:
        return None
    return embeddings


def embeddings_to_matrix(embeddings):
    import numpy as np
    import torch

    patch_ids = list(embeddings.keys())
    first = next(iter(embeddings.values()))
    if isinstance(first, torch.Tensor):
        X = torch.stack([
            (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).reshape(-1)
            for v in embeddings.values()
        ]).numpy()
    else:
        X = np.stack([np.asarray(v).reshape(-1) for v in embeddings.values()])
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return patch_ids, X
