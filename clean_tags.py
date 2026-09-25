import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAD = "[PAD]"
NULL = "[NULL]"

STOPLIST = {
    "absurd_res", "highres", "lowres", "bad_id", "bad_twitter_id", "bad_pixiv_id",
    "bad_link", "commentary", "commentary_request", "english_commentary", "translated",
    "watermark", "signature", "artist_name", "twitter_username", "web_address",
    "dated", "username", "logo", "copyright_request", "artist_request",
    "text", "english_text", "speech_bubble", "dialogue_box", "comic",
    "source_request", "duplicate", "third-party_edit", "commission",
}
WHITELIST = {"pixel_art"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-metadata", default=str(ROOT / "data2" / "metadata2_raw.jsonl"))
    ap.add_argument("--out-metadata", default=str(ROOT / "data2" / "metadata2.jsonl"))
    ap.add_argument("--vocab", default=str(ROOT / "data2" / "vocab2.json"))
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--limit", type=int, default=8000)
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in Path(args.in_metadata).read_text(encoding="utf-8").splitlines()
        if l
    ]

    before = [len(r.get("tags", [])) for r in rows]

    freq = Counter()
    for r in rows:
        for t in set(r.get("tags", [])):
            if t in STOPLIST:
                continue
            freq[t] += 1

    kept = {t for t, c in freq.items() if c >= args.min_count}
    kept |= WHITELIST
    ordered = [t for t in WHITELIST if t in freq]
    rest = sorted((t for t in kept if t not in WHITELIST), key=lambda t: (-freq[t], t))
    ordered += rest[: max(0, args.limit - len(ordered))]
    kept_set = set(ordered)

    for r in rows:
        seen = set()
        out = []
        for t in r.get("tags", []):
            if t in kept_set and t not in seen:
                out.append(t)
                seen.add(t)
        r["tags"] = out

    after = [len(r.get("tags", [])) for r in rows]

    out_meta = Path(args.out_metadata)
    with out_meta.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tokens = [PAD, NULL] + ordered
    Path(args.vocab).write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")

    n = len(rows)
    print(f"images={n}")
    print(f"tags/img before: {sum(before)/n:.1f}  after: {sum(after)/n:.1f}")
    print(f"unique tags (>= {args.min_count}): {len(kept)}  vocab tokens: {len(tokens)}")
    print(f"metadata -> {out_meta}")
    print(f"vocab    -> {args.vocab}")


if __name__ == "__main__":
    main()
