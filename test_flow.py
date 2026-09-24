import torch
import torch.nn.functional as F

import flow
from model.dit import DiT
from text_encoder import TagTextEncoder
from vocab import TagTokenizer


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TagTokenizer(["[PAD]", "[NULL]", "pixel_art", "1girl", "smile"])
    res = 16
    enc = TagTextEncoder(tok.vocab_size, dim=128, depth=2, heads=4).to(device)
    dit = DiT(
        tok.vocab_size, resolution=res, patch_size=4, dim=96, depth=3, heads=3, text_dim=128
    ).to(device)

    x0 = torch.randn(2, 3, res, res, device=device)
    noise = torch.randn_like(x0)
    zeros = torch.zeros(2, device=device)
    ones = torch.ones(2, device=device)
    assert torch.allclose(flow.add_noise(x0, noise, zeros), noise)
    assert torch.allclose(flow.add_noise(x0, noise, ones), x0)
    assert torch.allclose(flow.velocity_target(x0, noise), x0 - noise)
    print("test1 interpolant endpoints ok (t=0 -> noise, t=1 -> data)")

    ids, mask = tok.encode_batch([["pixel_art", "1girl"]], device=device)
    cond = flow.encode_text(enc, ids, mask)
    uncond = flow.encode_text(enc, *tok.null_batch(1, device=device))
    out = flow.euler_sample(dit, cond, uncond, (1, 3, res, res), steps=10, cfg_scale=4.0, device=device)
    assert out.shape == (1, 3, res, res) and not torch.isnan(out).any()
    out_cfg1 = flow.euler_sample(dit, cond, None, (1, 3, res, res), steps=10, cfg_scale=1.0, device=device)
    assert out_cfg1.shape == (1, 3, res, res)
    print(f"test2 sampling ok (cfg=4 and cfg=1), range=[{out.min():.2f},{out.max():.2f}]")

    target_img = torch.rand(1, 3, res, res, device=device) * 2 - 1
    tags = [["pixel_art", "1girl", "smile"]]
    ids, mask = tok.encode_batch(tags, device=device)
    opt = torch.optim.AdamW(list(dit.parameters()) + list(enc.parameters()), lr=5e-3)
    dit.train()
    enc.train()
    with torch.no_grad():
        dit.final_linear.weight.normal_(0.0, 0.02)
        dit.final_linear.bias.normal_(0.0, 0.02)

    loss = None
    for step in range(600):
        noise = torch.randn_like(target_img)
        t = flow.sample_timesteps(1, device)
        xt = flow.add_noise(target_img, noise, t)
        v_target = flow.velocity_target(target_img, noise)
        toks, pool = enc(ids, mask)
        pred = dit(xt, t, toks, mask, pool)
        loss = F.mse_loss(pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"test3 overfit single image: final_loss={loss.item():.5f}")

    dit.eval()
    enc.eval()
    cond = flow.encode_text(enc, ids, mask)
    sample = flow.euler_sample(dit, cond, None, (1, 3, res, res), steps=30, cfg_scale=1.0, device=device)
    err_sample = (sample - target_img).abs().mean().item()
    err_noise = (torch.randn_like(target_img) - target_img).abs().mean().item()
    print(f"           sample_err={err_sample:.4f}  vs  pure_noise_err={err_noise:.4f}")
    assert err_sample < 0.5 * err_noise, "flow+DiT did not learn to reconstruct"
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
