from __future__ import annotations

import csv
import os

import numpy as np

from . import config

_TILE_ENCODER = None
_TRANSFORM = None


def _extract_slide_id(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def _get_tile_encoder(device):
    global _TILE_ENCODER, _TRANSFORM
    if _TILE_ENCODER is None:
        config.load_env()
        import timm
        from torchvision import transforms
        _TILE_ENCODER = timm.create_model(config.TILE_ENCODER_HF,
                                           pretrained=True)
        _TILE_ENCODER = _TILE_ENCODER.to(device).eval()
        _TRANSFORM = transforms.Compose([
            transforms.Resize(256,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return _TILE_ENCODER, _TRANSFORM


def _is_blank(image) -> bool:
    data = np.asarray(image).reshape(-1, 1)
    return float(np.std(data)) < config.BLANK_STD_THRESHOLD


def _get_mpp(slide):
    import openslide
    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "0"))
    mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "0"))
    return mpp_x, mpp_y


def embed_slide(svs_file: str, work_dir: str, device=None,
                batch_size: int = 256) -> str | None:
    import torch
    from PIL import Image
    import openslide

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slide_id = _extract_slide_id(os.path.basename(svs_file))
    out_subdir = os.path.join(work_dir, slide_id)
    os.makedirs(out_subdir, exist_ok=True)
    cache_path = os.path.join(out_subdir, config.EMBED_CACHE_NAME)
    if os.path.exists(cache_path):
        print(f"[tile] {slide_id}: cache exists, skipping")
        return slide_id

    slide = openslide.OpenSlide(svs_file)
    mpp_x, mpp_y = _get_mpp(slide)
    if mpp_x == 0 or mpp_y == 0:
        print(f"[tile] {slide_id}: MPP unavailable, skipping")
        return None

    native_px_x = int(config.PATCH_SIZE_MICRONS / mpp_x)
    native_px_y = int(config.PATCH_SIZE_MICRONS / mpp_y)
    slide_w, slide_h = slide.dimensions

    thumbnail = slide.get_thumbnail((2000, 2000))
    thumb_path = os.path.join(out_subdir, f"{slide_id}{config.THUMBNAIL_SUFFIX}")
    thumbnail.save(thumb_path)
    thumb_w, thumb_h = thumbnail.size


    tps_x = int(native_px_x * thumb_w / slide_w)
    tps_y = int(native_px_y * thumb_h / slide_h)
    with open(os.path.join(out_subdir, config.THUMB_PATCH_SIZE_NAME),
              "w", newline="") as f:
        csv.writer(f).writerow([tps_x, tps_y])


    with open(os.path.join(out_subdir, config.MPP_NAME), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mpp_x", "mpp_y", "slide_width", "slide_height",
                    "thumb_width", "thumb_height"])
        w.writerow([mpp_x, mpp_y, slide_w, slide_h, thumb_w, thumb_h])

    encoder, transform = _get_tile_encoder(device)

    coords, patches, patch_ids = [], [], []
    counter = 0
    for x in range(0, slide_w, native_px_x):
        if x + native_px_x > slide_w:
            continue
        for y in range(0, slide_h, native_px_y):
            if y + native_px_y > slide_h:
                continue
            patch = slide.read_region((x, y), 0,
                                      (native_px_x, native_px_y)).convert("RGB")
            if _is_blank(patch):
                continue
            patch = patch.resize((config.TARGET_TILE_PX, config.TARGET_TILE_PX),
                                 Image.BICUBIC)
            patches.append(transform(patch).unsqueeze(0))
            cx, cy = x + native_px_x // 2, y + native_px_y // 2
            coords.append([f"{slide_id}_{counter}",
                           int(cx * thumb_w / slide_w),
                           int(cy * thumb_h / slide_h)])
            patch_ids.append(f"{slide_id}_{counter}")
            counter += 1

    if counter == 0:
        print(f"[tile] {slide_id}: no non-blank tiles, skipping")
        return None

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.cat(patches, dim=0)),
                        batch_size=batch_size, shuffle=False)
    embeddings = {}
    with torch.no_grad():
        for i, (batch,) in enumerate(loader):
            out = encoder(batch.to(device)).cpu()
            for j, emb in enumerate(out):
                embeddings[patch_ids[i * batch_size + j]] = emb
    torch.save({"embeddings": embeddings}, cache_path)

    with open(os.path.join(out_subdir, config.COORDS_NAME),
              "w", newline="") as f:
        csv.writer(f).writerows(coords)

    print(f"[tile] {slide_id}: {counter} tiles embedded")
    return slide_id
