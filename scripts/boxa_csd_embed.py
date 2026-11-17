#!/usr/bin/env python3
"""Run ON BOXA with the ComfyUI venv python. CSD style and content embeddings for one
directory of images, written as JSON for scripts/csd_style.py on the Mac.

    python.exe boxa_csd_embed.py --in "/opt/loramake/csd/in/<batch>" --out "/opt/loramake/csd/out/<batch>.json"

Model: yuxi-liu-wired/CSD (contrastive style descriptor on a CLIP ViT-L/14 backbone), the
torchscript export at /opt/models/csd/csd_clip_model_torchscript.pt. The export holds fp16
weights, so inputs are fed as fp16; preprocessing follows CLIP (bicubic resize of the short
side to 224, center crop, CLIP mean/std). Both output vectors are L2-normalized by the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

MODEL_PATH = "/opt/models/csd/csd_clip_model_torchscript.pt"
MODEL_ID = "yuxi-liu-wired/CSD torchscript (csd-vit-l)"
SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
BATCH = 16
DECIMALS = 5


def load_image(path: Path, torch, Image):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = SIZE / min(width, height)
    image = image.resize((max(SIZE, round(width * scale)), max(SIZE, round(height * scale))), Image.BICUBIC)
    width, height = image.size
    left = (width - SIZE) // 2
    top = (height - SIZE) // 2
    image = image.crop((left, top, left + SIZE, top + SIZE))
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(SIZE, SIZE, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSD embeddings for a directory of images")
    parser.add_argument("--in", dest="src", required=True, help="directory of images")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args(argv)
    import torch
    from PIL import Image

    src = Path(args.src)
    files = sorted(path for path in src.iterdir() if path.suffix.lower() in SUFFIXES)
    if not files:
        print(f"no images in {src}", flush=True)
        return 2
    model = torch.jit.load(args.model, map_location="cuda").eval()
    items = []
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(files), BATCH):
            chunk = files[start:start + BATCH]
            batch = torch.stack([load_image(path, torch, Image) for path in chunk]).half().cuda()
            _feature, content, style = model(batch)
            for path, content_vec, style_vec in zip(chunk, content.float().cpu(), style.float().cpu(), strict=False):
                items.append({
                    "name": path.name,
                    "style": [round(value, DECIMALS) for value in style_vec.tolist()],
                    "content": [round(value, DECIMALS) for value in content_vec.tolist()],
                })
            print(f"embedded {len(items)}/{len(files)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"v": 1, "model": MODEL_ID, "model_path": args.model, "count": len(items), "items": items}
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out} ({len(items)} items, {time.time() - started:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
