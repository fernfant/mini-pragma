"""
Generate text from a trained checkpoint.

  python sample.py --ckpt ckpt_tinystories.pt --prompt "Once upon a time"
  python sample.py --ckpt ckpt_shakespeare.pt --prompt "ROMEO:" --tokens 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch

from model import GPT, GPTConfig

ENC = tiktoken.get_encoding("gpt2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    ck = torch.load(Path(args.ckpt), map_location=device)
    model = GPT(GPTConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (iter {ck.get('iter','?')}, val loss {ck.get('val_loss',0):.3f})\n")

    ids = ENC.encode_ordinary(args.prompt) or ENC.encode_ordinary("\n")
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=args.tokens,
                             temperature=args.temperature, top_k=args.top_k)
    print(ENC.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
