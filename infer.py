import argparse
import time
from pathlib import Path

import torch
from PIL import Image

import config
import flow
from model.dit import DiT
from text_encoder import TagTextEncoder
from vocab import TagTokenizer

ROOT = config.ROOT


def load_ema_into(module, prefix, shadow):
    state = module.state_dict()
    dump = {k: shadow[f"{prefix}.{k}"] for k in state}
    module.load_state_dict({k: dump[k].to(state[k].dtype) for k in state})


def tensor_to_pil(x):
    arr = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0)
    return Image.fromarray(arr.cpu().numpy(), "RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=config.sample_steps)
    parser.add_argument("--cfg", type=float, default=config.sample_cfg)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upscale", type=int, default=config.upscale)
    parser.add_argument("--palette", type=int, default=config.palette)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device)

    tokenizer = TagTokenizer(ckpt["itos"], hash_buckets=ckpt.get("hash_buckets", 2048))
    dit = DiT(tokenizer.vocab_size, **ckpt["model_cfg"]).to(device).eval()
    text_encoder = TagTextEncoder(tokenizer.vocab_size, **ckpt["text_cfg"]).to(device).eval()

    if ckpt.get("ema") and not args.no_ema:
        load_ema_into(dit, "dit", ckpt["ema"])
        load_ema_into(text_encoder, "text", ckpt["ema"])
        print("using EMA weights")
    else:
        dit.load_state_dict(ckpt["model"])
        text_encoder.load_state_dict(ckpt["text_encoder"])

    tags = [t.strip() for t in args.prompt.split(",") if t.strip()]
    known = [t for t in tags if t.lower() in tokenizer.stoi]
    unknown = [t for t in tags if t.lower() not in tokenizer.stoi]
    if unknown:
        print(f"OOV tags (handled by n-gram hashing): {unknown}")
    print(f"known tags: {known}")

    ids, mask = tokenizer.encode_batch([tags], device=device)
    cond = flow.encode_text(text_encoder, ids, mask)
    nids, nmask = tokenizer.null_batch(1, device=device)
    uncond = flow.encode_text(text_encoder, nids, nmask)

    res = ckpt["model_cfg"]["resolution"]
    t0 = time.time()
    img = flow.euler_sample(
        dit, cond, uncond, (1, 3, res, res),
        steps=args.steps, cfg_scale=args.cfg, device=device,
    )
    print(f"sampled {args.steps} steps in {time.time() - t0:.2f}s")

    pil = tensor_to_pil(img[0])
    if args.upscale and args.upscale > 1:
        pil = pil.resize((res * args.upscale, res * args.upscale), Image.NEAREST)
    if args.palette and args.palette > 0:
        pil = pil.quantize(colors=args.palette, method=Image.MEDIANCUT).convert("RGB")

    out = Path(args.out) if args.out else config.SAMPLE_DIR / "infer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
