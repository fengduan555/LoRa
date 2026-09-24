import argparse
import json
from collections import Counter
from pathlib import Path

import config
from build_vocab import build_vocab

DROP_ANY = {
    "fake_screenshot",
    "comic",
    "text",
    "english_text",
    "speech_bubble",
    "dialogue_box",
    "multiple_girls",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple_boys",
}
KEEP_ANY = {"1girl", "1boy", "cat", "animal", "dog"}


def keep(tags):
    ts = set(tags)
    if ts & DROP_ANY:
        return False
    return bool(ts & KEEP_ANY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(config.METADATA))
    parser.add_argument("--out-dir", default=str(config.ROOT / "data2"))
    parser.add_argument("--limit", type=int, default=config.max_vocab)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if line
    ]
    kept = [r for r in rows if keep(r.get("tags", []))]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta2 = out_dir / "metadata2.jsonl"
    with meta2.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tokens = build_vocab([r.get("tags", []) for r in kept], args.limit)
    vocab2 = out_dir / "vocab2.json"
    vocab2.write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")

    ratio = len(kept) / max(1, len(rows))
    print(f"total={len(rows)} kept={len(kept)} ({ratio:.1%}) vocab={len(tokens)}")
    print(f"metadata2 -> {meta2}")
    print(f"vocab2    -> {vocab2}")


if __name__ == "__main__":
    main()
