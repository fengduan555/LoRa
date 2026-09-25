import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SCORE_TH = 0.35

# COCO-17
NOSE, L_SHO, R_SHO = 0, 5, 6
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16


def find_raw(img_id):
    raw = ROOT / "data" / "raw"
    for p in raw.glob(img_id + ".*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def _vis(kpts, scores, idx):
    return scores[idx] >= SCORE_TH


def pose_tags(kpts, scores, h):
    tags = set()
    vis = scores >= SCORE_TH
    if vis.sum() < 4:
        return tags

    def p(i):
        return kpts[i]

    sh = [i for i in (L_SHO, R_SHO) if vis[i]]
    hp = [i for i in (L_HIP, R_HIP) if vis[i]]
    sh_mid = np.mean([p(i) for i in sh], axis=0) if sh else None
    hp_mid = np.mean([p(i) for i in hp], axis=0) if hp else None
    torso = 1.0
    if sh_mid is not None and hp_mid is not None:
        torso = max(1.0, abs(sh_mid[1] - hp_mid[1]))

    if sh_mid is not None and hp_mid is not None:
        axis = sh_mid - hp_mid
        if abs(axis[1]) < 0.4 * abs(axis[0]) + 1e-6:
            tags.add("lying")

    kn = [i for i in (L_KNE, R_KNE) if vis[i]]
    an = [i for i in (L_ANK, R_ANK) if vis[i]]
    if hp and kn and "lying" not in tags:
        knee_y = np.mean([p(i)[1] for i in kn])
        hip_y = np.mean([p(i)[1] for i in hp])
        if knee_y - hip_y < 0.6 * torso:
            tags.add("sitting")
        elif an:
            ank_y = np.mean([p(i)[1] for i in an])
            if ank_y > hip_y + 0.6 * torso:
                tags.add("standing")

    for w, s in ((L_WRI, L_SHO), (R_WRI, R_SHO)):
        if vis[w] and vis[s] and p(w)[1] < p(s)[1] - 0.3 * torso:
            tags.add("arms_up")
    return tags


def composition_tags(kpts, scores, h):
    tags = set()
    vis = scores >= SCORE_TH
    if vis.sum() < 3:
        return tags
    ys = kpts[vis][:, 1]
    span = (ys.max() - ys.min()) / max(1.0, h)
    if vis[L_ANK] or vis[R_ANK]:
        if span > 0.7:
            tags.add("full_body")
    elif (vis[L_HIP] or vis[R_HIP]) is False and (vis[NOSE] or vis[L_SHO] or vis[R_SHO]):
        if span > 0.3:
            tags.add("portrait")
    return tags


def orientation_tags(kpts, scores):
    tags = set()
    vis = scores >= SCORE_TH
    sho = [i for i in (L_SHO, R_SHO) if vis[i]]
    if vis[NOSE] and len(sho) == 2:
        mid_x = (kpts[L_SHO][0] + kpts[R_SHO][0]) / 2
        width = abs(kpts[L_SHO][0] - kpts[R_SHO][0]) + 1e-6
        off = kpts[NOSE][0] - mid_x
        if abs(off) > 0.35 * width:
            tags.add("from_side")
            tags.add("facing_right" if off > 0 else "facing_left")
    elif len(sho) == 1:
        tags.add("from_side")
        tags.add("facing_right" if sho[0] == L_SHO else "facing_left")
    return tags


def color_tags(img):
    arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    flat = arr.reshape(-1, 3)
    k = 4
    idx = np.linspace(0, len(flat) - 1, k).astype(int)
    centers = flat[idx].copy()
    assign = np.zeros(len(flat), dtype=int)
    for _ in range(10):
        d = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        assign = d.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = flat[m].mean(0)
    sizes = np.bincount(assign, minlength=k).astype(float)
    tags = set()
    order = np.argsort(-sizes)
    for j in order:
        r, g, b = centers[j] / 255.0
        hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
        if vv < 0.15 or (vv > 0.9 and ss < 0.15) or ss < 0.12:
            continue
        deg = hh * 360
        if deg < 15 or deg >= 345:
            name = "red"
        elif deg < 45:
            name = "orange"
        elif deg < 70:
            name = "yellow"
        elif deg < 160:
            name = "green"
        elif deg < 200:
            name = "cyan"
        elif deg < 260:
            name = "blue"
        elif deg < 300:
            name = "purple"
        else:
            name = "pink"
        tags.add(f"dominant_{name}")
        tags.add("warm_palette" if (deg < 70 or deg >= 300) else "cool_palette")
        break
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(ROOT / "data2" / "metadata2.jsonl"))
    ap.add_argument("--vocab", default=str(ROOT / "data2" / "vocab2.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true", help="don't write metadata/vocab")
    args = ap.parse_args()

    from rtmlib import Body

    body = Body(mode="lightweight", backend="onnxruntime", device=args.device)

    rows = [
        json.loads(l)
        for l in Path(args.metadata).read_text(encoding="utf-8").splitlines()
        if l
    ]
    if args.limit:
        rows = rows[: args.limit]

    structured = set()
    added_count = {}
    skipped_pose = 0
    POSE_EXIST = {"sitting", "standing", "lying", "kneeling", "on_back", "on_stomach"}
    ORIENT_EXIST = {"from_side", "facing_left", "facing_right", "from_behind", "profile",
                    "from_above", "from_below", "looking_at_viewer", "looking_away"}
    COMP_EXIST = {"full_body", "portrait", "close-up", "upper_body", "cowboy_shot", "lower_body"}
    for n, r in enumerate(rows):
        img_id = Path(r["image"]).stem
        orig = set(r.get("tags", []))
        tags = set(orig)
        proc = ROOT / r["image"]
        try:
            if proc.exists():
                tags |= color_tags(Image.open(proc))
        except Exception:
            pass
        raw = find_raw(img_id)
        if raw is not None:
            try:
                import cv2

                im = cv2.imread(str(raw))
                if im is not None:
                    kpts, scores = body(im)
                    if len(kpts):
                        k = int(np.argmax(scores.mean(1)))
                        kp, sc = kpts[k], scores[k]
                        if not (orig & POSE_EXIST):
                            tags |= pose_tags(kp, sc, im.shape[0])
                        if not (orig & COMP_EXIST):
                            tags |= composition_tags(kp, sc, im.shape[0])
                        if not (orig & ORIENT_EXIST):
                            tags |= orientation_tags(kp, sc)
                    else:
                        skipped_pose += 1
            except Exception:
                skipped_pose += 1
        new = tags - orig
        for t in new:
            added_count[t] = added_count.get(t, 0) + 1
        structured |= new
        r["tags"] = sorted(tags)
        if (n + 1) % 200 == 0:
            print(f"  {n+1}/{len(rows)} pose_skipped={skipped_pose}")

    print(f"processed={len(rows)} pose_skipped={skipped_pose}")
    print("structured tags added:", dict(sorted(added_count.items(), key=lambda kv: -kv[1])))

    if args.dry_run:
        print("dry-run: not writing")
        return

    with Path(args.metadata).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    vocab_set = set(vocab)
    for t in sorted(structured):
        if t not in vocab_set:
            vocab.append(t)
    Path(args.vocab).write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"vocab tokens: {len(vocab)} -> {args.vocab}")


if __name__ == "__main__":
    main()
