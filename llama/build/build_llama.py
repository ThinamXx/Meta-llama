import time
import torch
from pathlib import Path
import json
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

import sys

sys.path.append("/home/ubuntu/bin/Meta-llama/llama")

from llama2 import Transformer, ModelArgs


class Llama(nn.Module):
    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"Checkpoints not found in {checkpoints_dir}"
            checkpoint_path = checkpoints[0]
            print(f"Loading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return Llama(model, tokenizer, model_args)


if __name__ == "__main__":
    torch.manual_seed(2024)
    allow_cuda = True
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"

    checkpoints = "/home/ubuntu/bin/Meta-llama/assets"

    llama = Llama.build(
        checkpoints_dir=f"{checkpoints}/llama-2-7b",
        tokenizer_path=f"{checkpoints}/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device,
    )
    print("Model build successful")
