import torch
from typing import Optional
from llama2 import Transformer, ModelArgs
from build.build_llama import Llama
from sentencepiece import SentencePieceProcessor


class LlamaInference(Llama):
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

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # converting prompts to tokens
        prompts_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        batch_size = len(prompts_tokens)
        # making sure batch size is not too big
        assert (
            batch_size <= self.args.max_batch_size
        ), f"Batch size {batch_size} exceeds max batch size {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompts_tokens)
        # making sure prompts are not too long than sequence length
        assert (
            max_prompt_len <= self.args.max_seq_len
        ), f"Prompt length {max_prompt_len} exceeds max sequence length {self.args.max_seq_len}"
        
        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)
        
        
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