import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API = "https://danbooru.donmai.us/posts.json"
UA = "LoRa-pixel-dataset/1.0 (personal research)"
EXTS = {"png", "jpg", "jpeg", "webp", "gif"}
KEEP_TAG_FIELDS = ("tag_string_general", "tag_string_character")

_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        session = requests.Session()
        session.headers["User-Agent"] = UA
        _local.session = session
    return _local.session


def fetch_page(tags, page, limit, auth):
    params = {"tags": tags, "limit": limit, "page": page}
    params.update(auth)
    for attempt in range(6):
        try:
            resp = get_session().get(API, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 503):
                wait = 2 ** attempt * 2
                print(f"  rate limited ({resp.status_code}), wait {wait}s")
                time.sleep(wait)
                continue
            print(f"  http {resp.status_code}: {resp.text[:160]}")
            return []
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  request error: {exc}, retry in {wait}s")
            time.sleep(wait)
    return []


def choose_url(post):
    return post.get("large_file_url") or post.get("file_url")


def is_candidate(post, done_ids, min_side):
    pid = post.get("id")
    if not pid or pid in done_ids or not choose_url(post):
        return False
    general = (post.get("tag_string_general") or "").split()
    if "animated" in general:
        return False
    if (post.get("file_ext") or "").lower() not in EXTS:
        return False
    if post.get("image_width", 0) < min_side or post.get("image_height", 0) < min_side:
        return False
    return True


def post_tags(post):
    tags = []
    for field in KEEP_TAG_FIELDS:
        tags.extend(t for t in (post.get(field) or "").split() if t)
    return tags


def download_image(post):
    for attempt in range(4):
        try:
            resp = get_session().get(choose_url(post), timeout=60)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--limit-per-page", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=30000)
    parser.add_argument("--user", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-side", type=int, default=64)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "downloaded.jsonl"
    captions_path = out / "captions.json"

    done_ids = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line:
                done_ids.add(json.loads(line)["id"])
    captions = {}
    if captions_path.exists():
        captions = json.loads(captions_path.read_text(encoding="utf-8"))

    auth = {}
    if args.user and args.api_key:
        auth = {"login": args.user, "api_key": args.api_key}

    queries = ["pixel_art rating:general", "pixel_art rating:sensitive"]

    count = len(done_ids)
    log_file = log_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for query in queries:
                page = 1
                while count < args.max_images and page <= 1000:
                    print(f"[query] {query} | page {page} | total {count}")
                    posts = fetch_page(query, page, args.limit_per_page, auth)
                    if not posts:
                        break
                    candidates = [p for p in posts if is_candidate(p, done_ids, args.min_side)]
                    for post, content in zip(candidates, pool.map(download_image, candidates)):
                        if count >= args.max_images:
                            break
                        if content is None:
                            continue
                        pid = post["id"]
                        ext = (post.get("file_ext") or "").lower()
                        name = f"{pid}.{ext}"
                        (out / name).write_bytes(content)
                        captions[str(pid)] = post_tags(post)
                        done_ids.add(pid)
                        log_file.write(json.dumps({"id": pid, "file": name}) + "\n")
                        count += 1
                        if count % 500 == 0:
                            log_file.flush()
                            captions_path.write_text(
                                json.dumps(captions, ensure_ascii=False), encoding="utf-8"
                            )
                            print(f"  saved {count}")
                    page += 1
                    time.sleep(1.0)
    finally:
        log_file.close()
        captions_path.write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")
    print(f"done total={len(done_ids)}")


if __name__ == "__main__":
    main()
