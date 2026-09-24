import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CAPTIONS = RAW / "captions.json"
LOG = RAW / "downloaded.jsonl"
UA = "LoRa-pixel-dataset/1.0 (personal research)"
KEEP = ("tag_string_general", "tag_string_character")


def post_tags(post):
    tags = []
    for field in KEEP:
        tags.extend(t for t in (post.get(field) or "").split() if t)
    return tags


def main():
    captions = json.loads(CAPTIONS.read_text(encoding="utf-8"))
    ids = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line:
            ids.append(str(json.loads(line)["id"]))
    missing = [i for i in ids if i not in captions]
    print(f"total logged={len(ids)} missing={len(missing)}")

    session = requests.Session()
    session.headers["User-Agent"] = UA
    fixed = 0
    for n, pid in enumerate(missing, 1):
        for attempt in range(4):
            try:
                resp = session.get(f"https://danbooru.donmai.us/posts/{pid}.json", timeout=30)
                if resp.status_code == 200:
                    captions[pid] = post_tags(resp.json())
                    fixed += 1
                    break
                if resp.status_code in (429, 503):
                    time.sleep(2 ** attempt)
                    continue
                print(f"[{pid}] http {resp.status_code}")
                break
            except requests.RequestException:
                time.sleep(2 ** attempt)
        if n % 50 == 0:
            CAPTIONS.write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")
            print(f"  {n}/{len(missing)} fixed={fixed}")
        time.sleep(0.2)

    CAPTIONS.write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")
    print(f"done fixed={fixed} captions={len(captions)}")


if __name__ == "__main__":
    main()
