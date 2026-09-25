import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(ROOT / "data2" / "metadata2.jsonl"))
    ap.add_argument("--min-std", type=float, default=8.0)
    ap.add_argument("--black-thresh", type=int, default=12)
    ap.add_argument("--white-thresh", type=int, default=243)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "data2_outliers.txt"))
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if l
    ]

    flags = []
    stds = []
    empties = 0
    for r in rows:
        if not r.get("tags"):
            empties += 1
        p = Path(r["image"])
        if not p.is_absolute():
            p = ROOT / p
        try:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        except Exception as exc:
            flags.append((r["image"], "open_error", str(exc)))
            continue
        gray = arr.mean(2)
        std = float(gray.std())
        mean = float(gray.mean())
        n_unique = int(np.unique(arr.reshape(-1, 3), axis=0).shape[0])
        stds.append(std)
        if std < args.min_std:
            flags.append((r["image"], f"flat(std={std:.1f})", f"mean={mean:.0f} unique={n_unique}"))
        elif mean < args.black_thresh:
            flags.append((r["image"], f"near_black(mean={mean:.1f})", f"std={std:.1f}"))
        elif mean > args.white_thresh:
            flags.append((r["image"], f"near_white(mean={mean:.1f})", f"std={std:.1f}"))
        elif n_unique <= 1:
            flags.append((r["image"], "single_color", f"unique={n_unique}"))

    stds = np.array(stds)
    print(f"total={len(rows)} flagged={len(flags)} empty_tags={empties}")
    print(
        f"std 分布: min={stds.min():.1f} p1={np.percentile(stds,1):.1f} "
        f"median={np.median(stds):.1f} max={stds.max():.1f}"
    )
    counts = {}
    for _, kind, _ in flags:
        key = kind.split("(")[0]
        counts[key] = counts.get(key, 0) + 1
    print("flagged 类型:", counts)
    for f in flags[:40]:
        print("  ", f[0], "|", f[1], "|", f[2])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{a}\t{b}\t{c}" for a, b, c in flags), encoding="utf-8")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
