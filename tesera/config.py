from __future__ import annotations

import os


TARGET_MPP = 0.5
TARGET_TILE_PX = 256
PATCH_SIZE_MICRONS = TARGET_TILE_PX * TARGET_MPP
BLANK_STD_THRESHOLD = 10.0
TILE_ENCODER_HF = "hf_hub:prov-gigapath/prov-gigapath"
SLIDE_ENCODER_NAME = "gigapath_slide_enc12l768d"
SLIDE_ENCODER_IN_DIM = 1536

SLIDE_ENCODER_HF = "hf_hub:prov-gigapath/prov-gigapath"
EMBED_DIM_TILE = 1536
EMBED_DIM_SLIDE = 768

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
PARAMS_DIR = os.path.join(_PKG_DIR, "params")


SMOOTH_SIGMA_TILES = 3.0
SMOOTH_THRESHOLD = 0.50

APPLY_CLEANUP = True
DEBRIS_AREA_MM2 = 2.0
MIN_NEIGHBORS = 2


TUMOR_STATUS_COL = "smoothened_tumor_status"
REMOVED_COL = "removed_by_cleanup"
P_TUMOR_COL = "p_tumor"


N_PCS = 99
SURVIVAL_HORIZONS_MONTHS = (12.0, 36.0, 60.0)
DAYS_PER_MONTH = 30.4


PARAM_FILES = {
    "tumor_classifier": "tumor_classifier.joblib",
    "pca_center_scale": "pca_center_scale.csv",
    "pca_pc_loadings":  "pca_pc_loadings.csv",
    "cox_coefs":        "cox_coefs.csv",
    "kmeans_centers":   "kmeans_centers.csv",
    "baseline_hazard":  "baseline_hazard.csv",
    "pca_reference":    "pca_reference.csv",
}


EMBED_CACHE_NAME = "tile_embeddings.pt"
COORDS_NAME = "tile_coords.csv"
THUMB_PATCH_SIZE_NAME = "thumb_patch_size.csv"
THUMBNAIL_SUFFIX = "_thumbnail.png"
MPP_NAME = "mpp.csv"


def param_path(key: str) -> str:
    return os.path.join(PARAMS_DIR, PARAM_FILES[key])


def load_env():
    for base in (os.getcwd(), _REPO_ROOT):
        path = os.path.join(base, ".env")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
