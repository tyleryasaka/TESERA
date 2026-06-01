# TESERA

TESERA (Tissue Encoded Subtype Elucidation and Risk Assessment) is a computational
pathology pipeline for hepatocellular carcinoma (HCC) that operates on routine
hematoxylin and eosin (H&E) whole-slide images. For each slide it assigns an
unsupervised morphologic subtype (TES1 or TES2), computes a continuous prognostic
risk score, and reports model-estimated 1-, 3-, and 5-year overall survival. This
repository provides the inference pipeline and accompanying per-slide visualizations.

## Requirements and installation

TESERA requires Python 3.10 or later and a CUDA-capable GPU for whole-slide
encoding. Install the Python dependencies:

```
pip install -r requirements.txt
```

The whole-slide reader depends on the OpenSlide C library, which must be installed
separately (e.g. `apt-get install openslide-tools` or `brew install openslide`).

## Authentication

Tile and slide encoding use the Prov-GigaPath foundation model, which is gated on
Hugging Face. Request access to `prov-gigapath/prov-gigapath`, then supply a
Hugging Face access token via the `HF_TOKEN` environment variable. The token may be
exported in the shell or placed in a `.env` file at the repository root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The foundation-model weights are downloaded and cached automatically on first use.

## Usage

Apply the pipeline to one or more whole-slide images:

```
python scripts/run_tesera.py SLIDE1.svs SLIDE2.svs --out_dir results/
```

Slides may instead be enumerated in a text file, one path per line:

```
python scripts/run_tesera.py --slides_file slides.txt --out_dir results/
```

Common options:

```
--out_dir DIR     output directory (required)
--occlusion       additionally compute per-slide occlusion risk heatmaps
--device DEVICE   compute device, e.g. cuda or cpu (default: cuda if available)
```

The full set of options is available via `python scripts/run_tesera.py --help`.

## Output

All results are written to the directory given by `--out_dir`:

- `tesera_results.csv` — one row per slide, reporting the assigned subtype
  (TES1 or TES2), the continuous risk score, and estimated 1-, 3-, and 5-year
  overall survival.
- `pca.png` — projection of the analyzed slides onto the reference
  cohort, shown by morphologic subtype and by risk score.
- `occlusion_heatmaps/` — per-slide risk-attribution heatmaps, produced when
  `--occlusion` is specified.
