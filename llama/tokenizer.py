# https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py
# learning about SentencePiece tokenizer:

import os
from logging import getLogger
from typing import List 

from sentencepiece import SentencePieceTrainer 


logger = getLogger()


class Tokenizer:
    "SentencePiece tokenizer for encoding and decoding text"
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        
        self.sp_model = SentencePieceTrainer(model_file=model_path)
        logger.info(f"Loaded tokenizer from {model_path}")
        
        # BOS/EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )
        
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """Encode a string into a list of token IDs:

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sentence token.
            eos (bool): Whether to append the end-of-sentence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Decode a list of token IDs into a string:

        Args:
            t (List[int]): A list of token IDs.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)