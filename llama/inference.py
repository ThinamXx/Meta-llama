import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from llama2 import Transformer, ModelArgs
from build.build_llama import Llama
from sentencepiece import SentencePieceProcessor


class LlamaInference(nn.Module):
    def __init__(
        self,
        model: Llama,
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

        # list that contains the generated tokens along with the prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device
        )
        for k, tok in enumerate(prompts_tokens):
            tokens[k, : len(tok)] = torch.tensor(
                tok, dtype=torch.long, device=self.args.device
            )

        # eos token
        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_token_mask = (
            tokens != pad_id
        )  # true for prompt tokens, false for pad tokens.
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")

        for cur_pos in cur_iterator:
            with torch.no_grad():
                if self.args.use_cache:
                    # use KV cache for faster inference.
                    logits = self.model.forward(
                        tokens[:, cur_pos - 1 : cur_pos], cur_pos, None
                    )
                else:
                    # the current position is optional in this case.
                    # we will use positional encoding to the entire sequence.
                    logits = self.model.forward(tokens[:, :cur_pos], cur_pos, None)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # use greedy method
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # replace the pad tokens with the generated tokens
            next_token = torch.where(
                prompt_token_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # eos token reached
            eos_reached = (~prompt_token_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id()
            )
            if all(eos_reached):
                break

        # convert tokens to text
        output_tokens = []
        output_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # if eos is reached, cut off the generated tokens
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            output_tokens.append(current_prompt_tokens)
            output_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (output_tokens, output_text)

    def _sample_top_p(self, probs, top_p):
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(sorted_probs, dim=-1)
        mask = probs_sum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        # redistribute the probability so that the sum is 1
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        # sample from the modified distribution
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, dim=-1, index=next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(2024)
    allow_cuda = True
    device = "cuda" if allow_cuda and torch.cuda.is_available() else "cpu"

    checkpoints = "/home/ubuntu/bin/Meta-llama/assets"

    prompts = [
        "I believe",
    ]

    llama = Llama.build(
        checkpoints_dir=f"{checkpoints}/llama-2-7b",
        tokenizer_path=f"{checkpoints}/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )
    print("Model build successful")

    model = LlamaInference(llama.model, llama.tokenizer, llama.args)

    out_tokens, out_text = model.text_completion(prompts, max_gen_len=20, temperature=0)
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(out_text[i])
        print("Generated by model completed")
