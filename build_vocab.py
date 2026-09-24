import argparse
import json
from pathlib import Path

import config

ROOT = config.ROOT
PAD_TOKEN = "[PAD]"
NULL_TOKEN = "[NULL]"


def build_vocab(tag_lists, limit):
    freq = {}
    for tags in tag_lists:
        for tag in tags:
            freq[tag] = freq.get(tag, 0) + 1
    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [PAD_TOKEN, NULL_TOKEN] + [t for t, _ in ordered[:limit]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(config.METADATA))
    parser.add_argument("--vocab", default=str(config.VOCAB_PATH))
    parser.add_argument("--limit", type=int, default=config.max_vocab)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if line
    ]
    tokens = build_vocab([r.get("tags", []) for r in rows], args.limit)
    Path(args.vocab).write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    print(f"images={len(rows)} vocab_size={len(tokens)} -> {args.vocab}")


if __name__ == "__main__":
    main()
