import hashlib
import json
from pathlib import Path

import torch

import config

PAD_TOKEN = "[PAD]"
NULL_TOKEN = "[NULL]"
UNK_TOKEN = "[UNK]"
HASH_BUCKETS = config.hash_buckets
MAX_LEN = config.max_tags


def _ngram_bucket(gram, buckets):
    digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


class TagTokenizer:
    def __init__(self, tokens, hash_buckets=HASH_BUCKETS):
        base = [t for t in tokens if t not in (PAD_TOKEN, NULL_TOKEN, UNK_TOKEN)]
        self.tokens = [PAD_TOKEN, NULL_TOKEN, UNK_TOKEN] + base
        self.hash_buckets = hash_buckets
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.null_id = self.stoi[NULL_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.bucket_start = len(self.tokens)

    @property
    def vocab_size(self):
        return len(self.tokens) + self.hash_buckets

    @classmethod
    def load(cls, path, hash_buckets=HASH_BUCKETS):
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tokens, hash_buckets=hash_buckets)

    def save(self, path):
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    def _oov_ids(self, tag):
        marked = "^" + tag + "$"
        grams = [marked[i:i + 3] for i in range(len(marked) - 2)] or [marked]
        return [self.bucket_start + _ngram_bucket(g, self.hash_buckets) for g in grams]

    def encode(self, tags, max_len=MAX_LEN):
        ids = []
        for tag in tags:
            tag = tag.strip().lower()
            if not tag:
                continue
            if tag in self.stoi:
                ids.append(self.stoi[tag])
            else:
                ids.extend(self._oov_ids(tag))
            if len(ids) >= max_len:
                break
        return ids[:max_len] or [self.null_id]

    def encode_batch(self, batch_tags, device=None, max_len=MAX_LEN):
        seqs = [self.encode(tags, max_len) for tags in batch_tags]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.bool)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = True
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask

    def null_batch(self, batch_size, device=None):
        ids = torch.full((batch_size, 1), self.null_id, dtype=torch.long)
        mask = torch.ones((batch_size, 1), dtype=torch.bool)
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask
