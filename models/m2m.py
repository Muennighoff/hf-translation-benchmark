from typing import List

import torch
from transformers import AutoTokenizer
from transformers import M2M100ForConditionalGeneration


class M2M:
    """
    M2M model wrapper.
    See below for the generation settings of M2M (5 beams):
    https://huggingface.co/facebook/m2m100_418M/blob/main/config.json

    Args:
        device: Where torch should move the model
    """

    def __init__(self, device=torch.device("cuda"), weights: str = "facebook/m2m100_418M") -> None:

        self.device = device
        self.model = M2M100ForConditionalGeneration.from_pretrained(weights).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(weights)

    def generate(self, texts: List, src_lang: str, tar_lang: str, batch_size: int = 1) -> List[str]:
        """
        Generates translations of texts from source to target using the models configured generation settings.

        Args:
            texts: Texts to translate
            src_lang: Language code of source language
            tar_lang: Language code of target language

        Returns:
            texts: Translated texts
        """
        self.tokenizer.src_lang = src_lang
        forced_bos_token_id = self.tokenizer.get_lang_id(tar_lang)

        generations = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                gen = self.model.generate(**batch, forced_bos_token_id=forced_bos_token_id).cpu()
                generations.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))

        return generations
