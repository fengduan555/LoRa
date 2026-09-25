import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
RESAMPLE = {"nearest": Image.NEAREST, "box": Image.BOX, "lanczos": Image.LANCZOS}


def find_raw(img_id):
    for p in (ROOT / "data" / "raw").glob(img_id + ".*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def resize_square(img, size, resample):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), resample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(ROOT / "data2" / "metadata2.jsonl"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--resample", default="nearest", choices=list(RESAMPLE))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if l
    ]
    if args.limit:
        rows = rows[: args.limit]

    resample = RESAMPLE[args.resample]
    done = 0
    missing = 0
    for n, r in enumerate(rows):
        dest = ROOT / r["image"]
        img_id = dest.stem
        raw = find_raw(img_id)
        if raw is None:
            missing += 1
            continue
        try:
            img = Image.open(raw).convert("RGB")
        except Exception:
            missing += 1
            continue
        img = resize_square(img, args.res, resample)
        if args.colors and args.colors > 0:
            img = img.quantize(colors=args.colors, method=Image.MEDIANCUT).convert("RGB")
        img.save(dest, "PNG")
        done += 1
        if (n + 1) % 1000 == 0:
            print(f"  {n+1}/{len(rows)} done={done} missing={missing}")
    print(f"total={len(rows)} resized={done} missing_raw={missing} res={args.res} colors={args.colors}")


if __name__ == "__main__":
    main()
