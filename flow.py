import torch


def sample_timesteps(batch_size, device):
    return torch.rand(batch_size, device=device)


def add_noise(x0, noise, t):
    t = t.reshape(-1, 1, 1, 1)
    return (1.0 - t) * noise + t * x0


def velocity_target(x0, noise):
    return x0 - noise


def encode_text(text_encoder, ids, mask):
    tokens, pooled = text_encoder(ids, mask)
    return (tokens, mask, pooled)


@torch.no_grad()
def euler_sample(model, cond, uncond, shape, steps=30, cfg_scale=4.0, device="cuda"):
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    tok, mask, pool = cond
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t, tok, mask, pool)
        if cfg_scale != 1.0 and uncond is not None:
            utok, umask, upool = uncond
            v_uncond = model(x, t, utok, umask, upool)
            v = v_uncond + cfg_scale * (v - v_uncond)
        x = x + v * dt
    return x.clamp(-1.0, 1.0)
