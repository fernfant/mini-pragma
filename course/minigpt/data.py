"""
Download a corpus and turn it into a stream of BPE token ids saved as uint16
.bin files (train.bin / val.bin) that train.py memory-maps.

Two datasets:
  shakespeare  — ~1 MB of Shakespeare. Tiny, downloads in a second. For sanity checks.
  tinystories  — simple kids' stories, engineered so a small model writes fluent
                 English. The real run. Use --mb to cap how much you download.

Usage:
  python data.py shakespeare
  python data.py tinystories --mb 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
import tiktoken

HERE = Path(__file__).parent
ENC = tiktoken.get_encoding("gpt2")

URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "tinystories_train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "tinystories_val": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def download(url: str, dest: Path, cap_bytes: int | None = None):
    if dest.exists() and (cap_bytes is None or dest.stat().st_size >= cap_bytes * 0.95):
        print(f"  have {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    print(f"  downloading {url}")
    got = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if cap_bytes and got >= cap_bytes:
                    break
                print(f"\r  {got/1e6:7.0f} MB", end="", flush=True)
    print(f"\r  saved {dest.name} ({got/1e6:.0f} MB)        ")


def encode_to_bin(text: str, out: Path):
    ids = ENC.encode_ordinary(text)
    arr = np.array(ids, dtype=np.uint16)  # gpt2 vocab < 65536, fits in uint16
    arr.tofile(out)
    print(f"  wrote {out.name}: {len(arr):,} tokens")


def prep_shakespeare(d: Path):
    raw = d / "input.txt"
    download(URLS["shakespeare"], raw)
    text = raw.read_text(encoding="utf-8")
    n = int(len(text) * 0.9)
    encode_to_bin(text[:n], d / "train.bin")
    encode_to_bin(text[n:], d / "val.bin")


def prep_tinystories(d: Path, mb: int):
    train_txt = d / "train.txt"
    val_txt = d / "val.txt"
    download(URLS["tinystories_train"], train_txt, cap_bytes=mb * (1 << 20))
    download(URLS["tinystories_val"], val_txt, cap_bytes=20 * (1 << 20))
    # read text, drop a possibly-truncated final story on the capped train file
    train_text = train_txt.read_text(encoding="utf-8", errors="ignore")
    train_text = train_text.rsplit("<|endoftext|>", 1)[0] + "<|endoftext|>"
    encode_to_bin(train_text, d / "train.bin")
    encode_to_bin(val_txt.read_text(encoding="utf-8", errors="ignore"), d / "val.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["shakespeare", "tinystories"])
    ap.add_argument("--mb", type=int, default=200,
                    help="tinystories: how many MB of training text to download")
    args = ap.parse_args()

    d = HERE / "data" / args.dataset
    d.mkdir(parents=True, exist_ok=True)
    print(f"preparing {args.dataset} -> {d}")
    if args.dataset == "shakespeare":
        prep_shakespeare(d)
    else:
        prep_tinystories(d, args.mb)
    print("done.")


if __name__ == "__main__":
    main()
