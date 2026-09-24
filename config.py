from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data2" / "metadata2.jsonl"
VOCAB_PATH = ROOT / "data2" / "vocab2.json"
OUTPUTS = ROOT / "outputs"
CKPT_DIR = OUTPUTS / "ckpt"
SAMPLE_DIR = OUTPUTS / "samples"
TRAIN_OUT = OUTPUTS / "v2"
TRAIN_SAMPLE_OUT = OUTPUTS / "samples" / "v2"

resolution = 64
quantize_colors = 32
hash_buckets = 2048
max_tags = 64
max_vocab = 6000

patch_size = 4
in_channels = 3
dim = 512
depth = 16
heads = 8
mlp_ratio = 4.0
grad_checkpoint = True
text_dim = 256
text_depth = 3
text_heads = 4

batch_size = 24
lr = 2e-4
text_lr = 3e-4
weight_decay = 0.01
warmup = 4000
steps = 60000
grad_clip = 1.0
ema_decay = 0.9999
null_prob = 0.1
augment = True
num_workers = 4
log_every = 50
sample_every = 2000
save_every = 5000
keep_last = 3
seed = 42

sample_steps = 30
sample_cfg = 4.0
sample_prompts = [
    "pixel_art, 1girl, solo, long_hair, smile",
    "pixel_art, 1boy, sword, armor",
    "pixel_art, landscape, tree, sky, cloud",
    "pixel_art, cat, animal, simple_background",
]

upscale = 8
palette = 32
