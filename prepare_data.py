import argparse
import json
from pathlib import Path

import imagehash
from PIL import Image

import config

ROOT = config.ROOT
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
RESAMPLE = {
    "nearest": Image.NEAREST,
    "box": Image.BOX,
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
}


def load_captions(raw_dir):
    path = Path(raw_dir) / "captions.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): v for k, v in data.items()}
    return {}


def resize_square(img, size, resample):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), resample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(config.DATA_RAW))
    parser.add_argument("--out", default=str(config.DATA_PROCESSED))
    parser.add_argument("--res", type=int, default=config.resolution)
    parser.add_argument("--resample", default="nearest", choices=list(RESAMPLE))
    parser.add_argument("--quantize-colors", type=int, default=config.quantize_colors)
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--metadata", default=str(config.METADATA))
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resample = RESAMPLE[args.resample]
    captions = load_captions(raw)

    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTS)
    lines = []
    seen_hashes = set()
    skipped = 0
    for path in files:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        if min(img.size) < args.min_size:
            skipped += 1
            continue
        phash = str(imagehash.phash(img))
        if phash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(phash)

        img = resize_square(img, args.res, resample)
        if args.quantize_colors and args.quantize_colors > 0:
            img = img.quantize(colors=args.quantize_colors, method=Image.MEDIANCUT).convert("RGB")

        dest = out / (path.stem + ".png")
        img.save(dest, "PNG")

        tags = captions.get(path.stem) or captions.get(path.name) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        lines.append({"image": str(dest.relative_to(ROOT)), "tags": tags})

    metadata = Path(args.metadata)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"total={len(files)} kept={len(lines)} skipped={skipped}")
    print(f"metadata -> {metadata}")


if __name__ == "__main__":
    main()
