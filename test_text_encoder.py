import torch

from model.dit import DiT
from text_encoder import TagTextEncoder
from vocab import TagTokenizer


def main():
    tokens = ["[PAD]", "[NULL]", "1girl", "long_hair", "smile", "pixel_art", "blue_eyes"]
    tok = TagTokenizer(tokens)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} vocab_size={tok.vocab_size} bucket_start={tok.bucket_start}")

    enc = TagTextEncoder(tok.vocab_size).to(device).eval()

    ids, mask = tok.encode_batch(
        [["1girl"], ["1girl", "long_hair", "smile", "blue_eyes"], []], device=device
    )
    with torch.no_grad():
        seq, pooled = enc(ids, mask)
    assert seq.shape[0] == ids.shape[0] and seq.shape[1] == ids.shape[1] and seq.shape[2] == 256
    assert pooled.shape == (ids.shape[0], 256)
    assert not torch.isnan(seq).any() and not torch.isnan(pooled).any()
    print(f"test1 variable-length ok: seq={tuple(seq.shape)} pooled={tuple(pooled.shape)}")

    a_ids, a_mask = tok.encode_batch([["1girl", "smile"]], device=device)
    b_ids, b_mask = tok.encode_batch(
        [["1girl", "smile"], ["1girl", "long_hair", "blue_eyes", "pixel_art"]], device=device
    )
    with torch.no_grad():
        _, pa = enc(a_ids, a_mask)
        _, pb = enc(b_ids, b_mask)
    diff = (pa[0] - pb[0]).abs().max().item()
    assert torch.allclose(pa[0], pb[0], atol=1e-5), diff
    print(f"test2 padding invariance ok: max_diff={diff:.3e}")

    assert tok.encode([]) == [tok.null_id]
    print("test3 empty -> [NULL] ok")

    unk = tok.encode(["totally_unknown_tag_xyz"])
    assert unk and all(i >= tok.bucket_start for i in unk)
    assert all(i < tok.vocab_size for i in unk)
    print(f"test4 OOV n-gram ok: {len(unk)} tokens in [{tok.bucket_start},{tok.vocab_size})")

    dit = DiT(tok.vocab_size).to(device).eval()
    x = torch.randn(2, 3, 64, 64, device=device)
    t = torch.rand(2, device=device)
    with torch.no_grad():
        out = dit(x, t, seq[:2], mask[:2], pooled[:2])
    assert out.shape == x.shape, tuple(out.shape)
    print(f"test5 DiT bridge ok: out={tuple(out.shape)}")

    dit.train()
    enc.train()
    with torch.no_grad():
        dit.final_linear.weight.normal_(0.0, 0.02)
    toks, pool = enc(ids[:2], mask[:2])
    out = dit(x, t, toks, mask[:2], pool)
    out.square().mean().backward()
    grad = enc.embed.weight.grad
    assert grad is not None and grad.abs().sum().item() > 0
    print(f"test6 end-to-end backward ok: embed_grad_sum={grad.abs().sum().item():.4f}")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
