"""
Embedding demo — Raschka Ch. 2 §2.7–2.8

Tokenize sample text → token IDs → token embeddings + positional embeddings
→ input embeddings ready for the transformer (Ch. 3+).

Run:  python embedding_demo.py
"""

import os
import urllib.request

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

VOCAB_SIZE = 50257  # GPT-2 BPE vocabulary
EMBED_DIM = 256
MAX_LENGTH = 4
BATCH_SIZE = 8


def load_sample_text() -> str:
    if not os.path.exists("the-verdict.txt"):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        urllib.request.urlretrieve(url, "the-verdict.txt")
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        return f.read()


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(
                torch.tensor(token_ids[i + 1 : i + max_length + 1])
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size, max_length, stride):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def demo_toy_embedding():
    """§2.7 — embedding layer as a lookup table (small vocab for illustration)."""
    print("=" * 60)
    print("2.7  Token embeddings (toy example)")
    print("=" * 60)

    input_ids = torch.tensor([2, 3, 5, 1])
    torch.manual_seed(123)
    layer = nn.Embedding(6, 3)

    print("Token IDs:     ", input_ids.tolist())
    print("Weight matrix:\n", layer.weight)
    print("One token (id=3):\n", layer(torch.tensor([3])))
    print("All tokens:\n", layer(input_ids))
    print()


def demo_full_pipeline(raw_text: str):
    """§2.8 — token + positional embeddings on book sample text."""
    print("=" * 60)
    print("2.8  Token + positional embeddings (The Verdict)")
    print("=" * 60)

    tokenizer = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader(
        raw_text, BATCH_SIZE, MAX_LENGTH, stride=MAX_LENGTH
    )
    token_ids, targets = next(iter(dataloader))

    print(f"Sample text: {len(raw_text):,} chars, "
          f"{len(tokenizer.encode(raw_text)):,} BPE tokens")
    print(f"Batch token IDs ({BATCH_SIZE} sequences x {MAX_LENGTH} tokens):\n",
          token_ids)
    print("Decoded first sequence:", repr(tokenizer.decode(token_ids[0].tolist())))

    token_emb_layer = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    pos_emb_layer = nn.Embedding(MAX_LENGTH, EMBED_DIM)

    token_embeddings = token_emb_layer(token_ids)
    pos_embeddings = pos_emb_layer(torch.arange(MAX_LENGTH))
    input_embeddings = token_embeddings + pos_embeddings

    print(f"\nToken embeddings:  {tuple(token_embeddings.shape)}")
    print(f"Pos embeddings:    {tuple(pos_embeddings.shape)}")
    print(f"Input embeddings:  {tuple(input_embeddings.shape)}")
    print("  -> [batch, seq_len, embed_dim], ready for attention / transformer")
    print(f"\nFirst token vector (first 8 dims): "
          f"{input_embeddings[0, 0, :8].tolist()}")
    print()


def main():
    raw_text = load_sample_text()
    demo_toy_embedding()
    demo_full_pipeline(raw_text)
    print("Done.")


if __name__ == "__main__":
    main()
