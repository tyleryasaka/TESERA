#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tesera import pipeline


def main():
    ap = argparse.ArgumentParser(
        description="TESERA: morphologic subtyping and prognostic risk "
                    "scoring of HCC from H&E whole-slide images.")
    ap.add_argument("svs", nargs="*", help="Input SVS files.")
    ap.add_argument("--slides_file",
                    help="Text file of SVS paths, one per line.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--stage", choices=["tile", "analyze", "all"], default="all",
                    help="Stage to run (default: all).")
    ap.add_argument("--work_dir", default=None,
                    help="Intermediate cache directory (default: <out_dir>/work).")
    ap.add_argument("--occlusion", action="store_true",
                    help="Also compute occlusion heatmaps.")
    ap.add_argument("--no_smooth", action="store_true",
                    help="Disable spatial smoothing of tumor predictions "
                         "(enabled by default).")
    ap.add_argument("--no_tumor_cleanup", action="store_true",
                    help="Disable debris/thin-structure cleanup of tumor tiles "
                         "(enabled by default).")
    ap.add_argument("--window_tiles", type=int, default=5,
                    help="Occlusion window size in tiles (default: 5).")
    ap.add_argument("--occlusion_batch_size", type=int, default=8,
                    help="Occlusion batch size (default: 8).")
    ap.add_argument("--device", default=None,
                    help="Torch device (default: cuda if available).")
    args = ap.parse_args()

    svs_files = list(args.svs)
    if args.slides_file:
        with open(args.slides_file) as f:
            svs_files += [ln.strip() for ln in f if ln.strip()]
    if not svs_files:
        ap.error("No SVS files given (positional args or --slides_file).")

    if args.stage != "analyze":
        missing = [p for p in svs_files if not os.path.exists(p)]
        if missing:
            ap.error("These SVS files do not exist:\n  " + "\n  ".join(missing))

    device = None
    if args.device:
        import torch
        device = torch.device(args.device)

    if args.stage == "tile":
        out = pipeline.run_tile(
            svs_files=svs_files, out_dir=args.out_dir,
            work_dir=args.work_dir, device=device)
    elif args.stage == "analyze":
        out = pipeline.run_analyze(
            svs_files=svs_files, out_dir=args.out_dir, work_dir=args.work_dir,
            do_occlusion=args.occlusion, smooth=not args.no_smooth,
            apply_cleanup=not args.no_tumor_cleanup,
            window_tiles=args.window_tiles,
            occlusion_batch_size=args.occlusion_batch_size, device=device)
    else:
        out = pipeline.run(
            svs_files=svs_files, out_dir=args.out_dir, work_dir=args.work_dir,
            do_occlusion=args.occlusion, smooth=not args.no_smooth,
            apply_cleanup=not args.no_tumor_cleanup,
            window_tiles=args.window_tiles,
            occlusion_batch_size=args.occlusion_batch_size, device=device)

    if out:
        print("\nDone. Outputs:")
        for k, v in out.items():
            if v:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
