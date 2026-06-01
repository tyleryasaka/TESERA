from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def breslow_baseline(time, event, lp):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    w = np.exp(np.asarray(lp, dtype=float))

    uniq_ev = np.unique(time[event == 1])
    H0 = np.empty(len(uniq_ev), dtype=float)
    cum = 0.0
    for k, tk in enumerate(uniq_ev):
        d = int(((time == tk) & (event == 1)).sum())
        risk = w[time >= tk].sum()
        cum += d / risk if risk > 0 else 0.0
        H0[k] = cum
    return uniq_ev, H0


def survival_at(lp, event_times, H0, horizons):
    lp = np.atleast_1d(np.asarray(lp, dtype=float))
    horizons = np.asarray(horizons, dtype=float)
    out = np.empty((lp.shape[0], horizons.shape[0]), dtype=float)
    for j, h in enumerate(horizons):
        idx = int(np.searchsorted(event_times, h, side="right")) - 1
        H0h = H0[idx] if idx >= 0 else 0.0
        out[:, j] = np.exp(-H0h * np.exp(lp))
    return out


@dataclass
class SlideResult:
    sample: str
    tes_cluster: str
    risk_score: float
    os_1yr: float
    os_3yr: float
    os_5yr: float
    pc_scores: np.ndarray


class TESERAModel:
    def __init__(self, n_pcs: int = config.N_PCS):
        self.n_pcs = n_pcs


        cs = pd.read_csv(config.param_path("pca_center_scale"))
        self.feature_order = cs["feature"].tolist()
        self.center = cs["center"].to_numpy(dtype=float)
        self.scale = cs["scale"].to_numpy(dtype=float)


        load = pd.read_csv(config.param_path("pca_pc_loadings"))

        load = load.set_index("feature").loc[self.feature_order].reset_index()
        pc_cols = [c for c in load.columns if c.startswith("PC")]
        pc_cols = pc_cols[: self.n_pcs]
        self.pc_cols = pc_cols
        self.loadings = load[pc_cols].to_numpy(dtype=float)


        cox = pd.read_csv(config.param_path("cox_coefs"))
        cox = cox.set_index("feature").loc[pc_cols].reset_index()
        self.cox_coefs = cox["coef"].to_numpy(dtype=float)


        km = pd.read_csv(config.param_path("kmeans_centers"))
        self.tes_labels = km["TES"].tolist()
        self.kmeans_centers = km[pc_cols].to_numpy(dtype=float)


        base = pd.read_csv(config.param_path("baseline_hazard"))
        self.base_event_times = base["time_months"].to_numpy(dtype=float)
        self.base_H0 = base["cumulative_hazard"].to_numpy(dtype=float)


    def pc_scores(self, slide_embedding: np.ndarray) -> np.ndarray:
        x = np.asarray(slide_embedding, dtype=float).reshape(-1)
        standardized = (x - self.center) / self.scale
        return standardized @ self.loadings

    def assign_cluster(self, pc: np.ndarray) -> str:
        d = np.linalg.norm(self.kmeans_centers - pc[None, :], axis=1)
        return self.tes_labels[int(np.argmin(d))]

    def risk_score(self, pc: np.ndarray) -> float:
        return float(pc @ self.cox_coefs)

    def predict_survival(self, lp) -> np.ndarray:
        return survival_at(lp, self.base_event_times, self.base_H0,
                           config.SURVIVAL_HORIZONS_MONTHS)


    def score_slide(self, sample: str, slide_embedding: np.ndarray) -> SlideResult:
        pc = self.pc_scores(slide_embedding)
        lp = self.risk_score(pc)
        os1, os3, os5 = self.predict_survival(lp)[0]
        return SlideResult(
            sample=sample,
            tes_cluster=self.assign_cluster(pc),
            risk_score=lp,
            os_1yr=float(os1),
            os_3yr=float(os3),
            os_5yr=float(os5),
            pc_scores=pc,
        )
