import argparse
from pathlib import Path

import torch
from PIL import Image

import config
import flow
from model.dit import DiT
from text_encoder import TagTextEncoder
from vocab import TagTokenizer

PROMPTS = [
    ("single", "pixel_art, 1girl, simple_background"),
    ("animal", "pixel_art, cat, simple_background"),
    ("scenery", "pixel_art, landscape, sky, cloud"),
    ("multi", "pixel_art, 2girls, dress, simple_background"),
]


def tensor_to_pil(x):
    arr = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0)
    return Image.fromarray(arr.cpu().numpy(), "RGB")


def load_ema_into(module, prefix, shadow):
    state = module.state_dict()
    module.load_state_dict({k: shadow[f"{prefix}.{k}"].to(state[k].dtype) for k in state})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=str(config.SAMPLE_DIR / "baseline"))
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--cfgs", default="1,3,5,7")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--palette", type=int, default=32)
    parser.add_argument("--upscale", type=int, default=8)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    tokenizer = TagTokenizer(ckpt["itos"], hash_buckets=ckpt.get("hash_buckets", 2048))
    dit = DiT(tokenizer.vocab_size, **ckpt["model_cfg"]).to(device).eval()
    text_encoder = TagTextEncoder(tokenizer.vocab_size, **ckpt["text_cfg"]).to(device).eval()
    if ckpt.get("ema") and not args.no_ema:
        load_ema_into(dit, "dit", ckpt["ema"])
        load_ema_into(text_encoder, "text", ckpt["ema"])

    res = ckpt["model_cfg"]["resolution"]
    seeds = [int(s) for s in args.seeds.split(",")]
    cfgs = [float(c) for c in args.cfgs.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nids, nmask = tokenizer.null_batch(1, device=device)
    uncond = flow.encode_text(text_encoder, nids, nmask)
    cell = res * args.upscale

    for name, prompt in PROMPTS:
        tags = [t.strip() for t in prompt.split(",") if t.strip()]
        ids, mask = tokenizer.encode_batch([tags], device=device)
        cond = flow.encode_text(text_encoder, ids, mask)
        grid = Image.new("RGB", (len(seeds) * cell, len(cfgs) * cell), (255, 255, 255))
        for r, cfg in enumerate(cfgs):
            for c, seed in enumerate(seeds):
                torch.manual_seed(seed)
                img = flow.euler_sample(
                    dit, cond, uncond, (1, 3, res, res),
                    steps=args.steps, cfg_scale=cfg, device=device,
                )
                pil = tensor_to_pil(img[0]).resize((cell, cell), Image.NEAREST)
                if args.palette > 0:
                    pil = pil.quantize(colors=args.palette, method=Image.MEDIANCUT).convert("RGB")
                grid.paste(pil, (c * cell, r * cell))
        path = out / f"{name}.png"
        grid.save(path)
        print(f"{name:8s} | prompt='{prompt}' | cfg(top->down)={cfgs} | seed(left->right)={seeds} -> {path}")

    print(f"baseline eval done -> {out}")


if __name__ == "__main__":
    main()
