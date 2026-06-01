from __future__ import annotations

import os

from . import config


def _resolve_work_dir(out_dir, work_dir):
    if work_dir is None:
        work_dir = os.path.join(out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def _default_device(device):
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def run_tile(svs_files, out_dir, work_dir=None, device=None):
    from .tile_embedding import embed_slide

    config.load_env()
    os.makedirs(out_dir, exist_ok=True)
    work_dir = _resolve_work_dir(out_dir, work_dir)
    device = _default_device(device)

    done = []
    for svs in svs_files:
        sid = embed_slide(svs, work_dir, device=device)
        if sid is not None:
            done.append(sid)
    print(f"[pipeline] tile stage: embedded {len(done)}/{len(svs_files)} "
          f"slide(s) into {work_dir}")
    return {"work_dir": work_dir, "slides": done}


def run_analyze(svs_files, out_dir, work_dir=None,
                do_occlusion=False, smooth=True, apply_cleanup=True,
                window_tiles=5, occlusion_batch_size=8, device=None):
    import pandas as pd
    from .model import TESERAModel
    from .tile_embedding import _extract_slide_id
    from .tumor_filter import load_classifier, classify_slide
    from .slide_embedding import embed_slide_level

    config.load_env()
    os.makedirs(out_dir, exist_ok=True)
    work_dir = _resolve_work_dir(out_dir, work_dir)
    device = _default_device(device)
    tumor_dir = os.path.join(out_dir, "tumor_predictions")
    os.makedirs(tumor_dir, exist_ok=True)

    model = TESERAModel()
    classifier = load_classifier(config.param_path("tumor_classifier"))

    results = []
    for svs in svs_files:
        sid = _extract_slide_id(os.path.basename(svs))
        cache = os.path.join(work_dir, sid, config.EMBED_CACHE_NAME)
        if not os.path.exists(cache):
            print(f"[pipeline] {sid}: no tile embeddings under {work_dir} "
                  "(run the 'tile' stage first); skipping")
            continue

        pred_csv = classify_slide(sid, work_dir, classifier, tumor_dir,
                                  smooth=smooth, apply_cleanup=apply_cleanup)
        if pred_csv is None:
            continue

        slide_emb = embed_slide_level(sid, work_dir, pred_csv, device=device)
        if slide_emb is None:
            continue

        results.append(model.score_slide(sid, slide_emb))

    if not results:
        print("[pipeline] no slides scored.")
        return None

    res_df = pd.DataFrame([{
        "sample": r.sample,
        "TES_cluster": r.tes_cluster,
        "risk_score": r.risk_score,
        "OS_1yr": r.os_1yr,
        "OS_3yr": r.os_3yr,
        "OS_5yr": r.os_5yr,
    } for r in results])
    res_csv = os.path.join(out_dir, "tesera_results.csv")
    res_df.to_csv(res_csv, index=False, float_format="%.6f")
    print(f"[pipeline] wrote {res_csv}")

    pca_png = None
    try:
        from .plotting import plot_pca
        ref_csv = config.param_path("pca_reference")
        if os.path.exists(ref_csv):
            pca_png = plot_pca(results, ref_csv,
                               os.path.join(out_dir, "pca.png"))
        else:
            print(f"[pipeline] {ref_csv} not found; skipping PCA plot")
    except Exception as e:
        print(f"[pipeline] PCA plot skipped ({e})")

    occ_outputs = []
    if do_occlusion:
        try:
            from .occlusion import occlude_slide
            from .plotting import plot_occlusion_heatmaps
            occ_dir = os.path.join(out_dir, "occlusion")
            scored_ids = [r.sample for r in results]
            for sid in scored_ids:
                pred_csv = os.path.join(tumor_dir, f"{sid}_tile_predictions.csv")
                occlude_slide(sid, work_dir, pred_csv, model, occ_dir,
                              device=device, batch_size=occlusion_batch_size,
                              window_tiles=window_tiles)
            occ_outputs = plot_occlusion_heatmaps(
                scored_ids, occ_dir, work_dir,
                os.path.join(out_dir, "occlusion_heatmaps"),
                window_tiles=window_tiles)
        except Exception as e:
            print(f"[pipeline] occlusion step failed ({e})")

    return {"results_csv": res_csv, "pca_png": pca_png,
            "occlusion": occ_outputs, "work_dir": work_dir}


def run(svs_files, out_dir, work_dir=None,
        do_occlusion=False, smooth=True, apply_cleanup=True, window_tiles=5,
        occlusion_batch_size=8, device=None):
    work_dir = _resolve_work_dir(out_dir, work_dir)
    run_tile(svs_files, out_dir, work_dir=work_dir, device=device)
    return run_analyze(svs_files, out_dir, work_dir=work_dir,
                       do_occlusion=do_occlusion, smooth=smooth,
                       apply_cleanup=apply_cleanup, window_tiles=window_tiles,
                       occlusion_batch_size=occlusion_batch_size, device=device)
