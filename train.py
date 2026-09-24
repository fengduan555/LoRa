import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
import flow
from model.dit import DiT
from text_encoder import TagTextEncoder
from vocab import TagTokenizer

ROOT = config.ROOT
SAMPLE_PROMPTS = config.sample_prompts


class PixelDataset(Dataset):
    def __init__(self, metadata_path, augment=False):
        self.augment = augment
        self.items = [
            json.loads(line)
            for line in Path(metadata_path).read_text(encoding="utf-8").splitlines()
            if line
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = Path(item["image"])
        if not path.is_absolute():
            path = ROOT / path
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, item.get("tags", [])


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tags = [b[1] for b in batch]
    return imgs, tags


class EMA:
    def __init__(self, modules, decay):
        self.decay = decay
        self.shadow = {}
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                self.shadow[f"{name}.{key}"] = value.detach().clone().float()

    @torch.no_grad()
    def update(self, modules):
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                shadow = self.shadow[f"{name}.{key}"]
                if value.dtype.is_floating_point:
                    shadow.mul_(self.decay).add_(value.detach().float(), alpha=1.0 - self.decay)
                else:
                    shadow.copy_(value)

    @torch.no_grad()
    def copy_to(self, modules):
        for name, module in modules.items():
            state = module.state_dict()
            module.load_state_dict(
                {k: self.shadow[f"{name}.{k}"].to(state[k].dtype) for k in state}
            )


def tensor_to_pil(x):
    arr = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0)
    return Image.fromarray(arr.cpu().numpy(), "RGB")


def save_grid(images, path, ncols=4):
    w, h = images[0].size
    rows = (len(images) + ncols - 1) // ncols
    canvas = Image.new("RGB", (ncols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(images):
        canvas.paste(img, ((i % ncols) * w, (i // ncols) * h))
    canvas.save(path)


def lr_lambda_factory(warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def split_params(module):
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(config.METADATA))
    parser.add_argument("--vocab", default=str(config.VOCAB_PATH))
    parser.add_argument("--out", default=str(config.OUTPUTS))
    parser.add_argument("--sample-out", default=None)
    parser.add_argument("--res", type=int, default=config.resolution)
    parser.add_argument("--dim", type=int, default=config.dim)
    parser.add_argument("--depth", type=int, default=config.depth)
    parser.add_argument("--heads", type=int, default=config.heads)
    parser.add_argument("--text-dim", type=int, default=config.text_dim)
    parser.add_argument("--text-depth", type=int, default=config.text_depth)
    parser.add_argument("--text-heads", type=int, default=config.text_heads)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--text-lr", type=float, default=config.text_lr)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup", type=int, default=config.warmup)
    parser.add_argument("--steps", type=int, default=config.steps)
    parser.add_argument("--grad-clip", type=float, default=config.grad_clip)
    parser.add_argument("--ema-decay", type=float, default=config.ema_decay)
    parser.add_argument("--null-prob", type=float, default=config.null_prob)
    parser.add_argument("--grad-checkpoint", action="store_true", default=config.grad_checkpoint)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--log-every", type=int, default=config.log_every)
    parser.add_argument("--sample-every", type=int, default=config.sample_every)
    parser.add_argument("--save-every", type=int, default=config.save_every)
    parser.add_argument("--keep-last", type=int, default=config.keep_last)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=config.sample_steps)
    parser.add_argument("--sample-cfg", type=float, default=config.sample_cfg)
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    ckpt_dir = out_dir / "ckpt"
    sample_dir = Path(args.sample_out) if args.sample_out else out_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = TagTokenizer.load(args.vocab)
    dataset = PixelDataset(args.metadata, augment=args.augment)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    print(f"device={device} images={len(dataset)} vocab={tokenizer.vocab_size}")

    dit = DiT(
        tokenizer.vocab_size,
        resolution=args.res,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        text_dim=args.text_dim,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    text_encoder = TagTextEncoder(
        tokenizer.vocab_size, dim=args.text_dim, depth=args.text_depth, heads=args.text_heads
    ).to(device)

    dit_decay, dit_nodecay = split_params(dit)
    text_decay, text_nodecay = split_params(text_encoder)
    optimizer = torch.optim.AdamW(
        [
            {"params": dit_decay, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": dit_nodecay, "lr": args.lr, "weight_decay": 0.0},
            {"params": text_decay, "lr": args.text_lr, "weight_decay": args.weight_decay},
            {"params": text_nodecay, "lr": args.text_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(args.warmup, args.steps)
    )
    ema = EMA({"dit": dit, "text": text_encoder}, args.ema_decay)
    sample_dit = copy.deepcopy(dit).eval()
    sample_text = copy.deepcopy(text_encoder).eval()

    model_cfg = dict(
        resolution=args.res,
        patch_size=4,
        in_channels=3,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        text_dim=args.text_dim,
    )
    text_cfg = dict(dim=args.text_dim, depth=args.text_depth, heads=args.text_heads)

    def run_sample(step):
        ema.copy_to({"dit": sample_dit, "text": sample_text})
        sample_dit.eval()
        sample_text.eval()
        tag_lists = [[t.strip() for t in p.split(",") if t.strip()] for p in SAMPLE_PROMPTS]
        ids, mask = tokenizer.encode_batch(tag_lists, device=device)
        cond = flow.encode_text(sample_text, ids, mask)
        nids, nmask = tokenizer.null_batch(len(tag_lists), device=device)
        uncond = flow.encode_text(sample_text, nids, nmask)
        shape = (len(tag_lists), 3, args.res, args.res)
        img = flow.euler_sample(
            sample_dit, cond, uncond, shape,
            steps=args.sample_steps, cfg_scale=args.sample_cfg, device=device,
        )
        save_grid(
            [tensor_to_pil(img[i]) for i in range(img.shape[0])],
            sample_dir / f"step_{step:07d}.png",
        )

    def save_ckpt(step):
        path = ckpt_dir / f"ckpt_{step:07d}.pt"
        torch.save(
            {
                "step": step,
                "model": dit.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "itos": tokenizer.tokens,
                "hash_buckets": tokenizer.hash_buckets,
                "model_cfg": model_cfg,
                "text_cfg": text_cfg,
            },
            path,
        )
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for old in ckpts[: max(0, len(ckpts) - args.keep_last)]:
            old.unlink()
        print(f"saved {path.name}")

    dit.train()
    text_encoder.train()
    step = 0
    if args.resume:
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            ck = torch.load(ckpts[-1], map_location=device)
            dit.load_state_dict(ck["model"])
            text_encoder.load_state_dict(ck["text_encoder"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck:
                scheduler.load_state_dict(ck["scheduler"])
            if ck.get("ema"):
                ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
            step = ck.get("step", 0)
            print(f"resumed {ckpts[-1].name} at step {step}")
    t0 = time.time()
    running = None
    while step < args.steps:
        for imgs, tags in loader:
            x0 = imgs.to(device, non_blocking=True)
            ids, mask = tokenizer.encode_batch(tags, device=device)
            if args.null_prob > 0:
                drop = torch.rand(x0.shape[0], device=device) < args.null_prob
                if drop.any():
                    ids[drop] = tokenizer.pad_id
                    mask[drop] = False
                    ids[drop, 0] = tokenizer.null_id
                    mask[drop, 0] = True

            noise = torch.randn_like(x0)
            t = flow.sample_timesteps(x0.shape[0], device)
            xt = flow.add_noise(x0, noise, t)
            target = flow.velocity_target(x0, noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                tokens, pooled = text_encoder(ids, mask)
                pred = dit(xt, t, tokens, mask, pooled)
                loss = F.mse_loss(pred.float(), target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(dit.parameters()) + list(text_encoder.parameters()), args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            ema.update({"dit": dit, "text": text_encoder})

            step += 1
            running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
            if step % args.log_every == 0:
                sec = (time.time() - t0) / args.log_every
                print(
                    f"step {step} loss {running:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"{sec:.2f}s/it"
                )
                t0 = time.time()
            if step % args.sample_every == 0:
                run_sample(step)
            if step % args.save_every == 0:
                save_ckpt(step)
            if step >= args.steps:
                break

    save_ckpt(step)
    run_sample(step)
    print("training done")


if __name__ == "__main__":
    main()
