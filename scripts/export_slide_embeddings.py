#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tesera import config
from tesera.slide_embedding import embed_slide_level


def to_numpy(x):
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu().numpy()
	return np.asarray(x, dtype=np.float32).ravel()


def main():
	ap = argparse.ArgumentParser(description="Export TESERA slide-level embeddings to CSV.")
	ap.add_argument("--out_dir", required=True, help="Output directory used for run_tesera.py.")
	ap.add_argument("--work_dir", default=None, help="Intermediate cache directory (default: <out_dir>/work).")
	ap.add_argument("--no_tumor_filter", action="store_true", help="Use all tiles instead of tumor-filtered tiles.")
	ap.add_argument("--csv", default=None, help="Output CSV path (default: <out_dir>/slide_embeddings.csv).")
	ap.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
	args = ap.parse_args()

	config.load_env()
	work_dir = args.work_dir or os.path.join(args.out_dir, "work")
	tumor_dir = os.path.join(args.out_dir, "tumor_predictions")
	out_csv = args.csv or os.path.join(args.out_dir, "slide_embeddings.csv")
	device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

	sids = sorted(d for d in os.listdir(work_dir) if os.path.exists(os.path.join(work_dir, d, config.EMBED_CACHE_NAME)))
	if not sids:
		sys.exit(f"No cached tile embeddings found under {work_dir}")

	rows = []
	for sid in sids:
		pred_csv = None
		if not args.no_tumor_filter:
			pred_csv = os.path.join(tumor_dir, f"{sid}_tile_predictions.csv")
			if not os.path.exists(pred_csv):
				print(f"[export] {sid}: no tumor predictions at {pred_csv}; skipping")
				continue
		emb = embed_slide_level(sid, work_dir, pred_csv, device=device)
		if emb is None:
			print(f"[export] {sid}: slide embedding failed; skipping")
			continue
		rows.append((sid, to_numpy(emb)))
		print(f"[export] {sid}: done")

	if not rows:
		sys.exit("No slide embeddings produced.")

	dim = len(rows[0][1])
	df = pd.DataFrame(np.vstack([r[1] for r in rows]), columns=[f"emb_{i}" for i in range(dim)])
	df.insert(0, "sample", [r[0] for r in rows])
	df.to_csv(out_csv, index=False, float_format="%.6f")
	print(f"[export] wrote {out_csv} ({len(rows)} slides x {dim} dims)")


if __name__ == "__main__":
	main()
