from typing import List

import torch
from transformers import AutoTokenizer
from transformers import M2M100ForConditionalGeneration


class M2M:
    """
    M2M model wrapper.

    Args:
        device: Where torch should move the model
    """

    def __init__(self, device=torch.device("cuda"), weights: str = "facebook/m2m100_418M") -> None:

        self.device = device
        self.model = M2M100ForConditionalGeneration.from_pretrained(weights).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(weights)

    def greedy_until(self, texts: List, src_lang: str, tar_lang: str) -> List[str]:
        """
        Greedily generates translation of texts from source to target.

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
            for txt in texts:
                txt_tensor = self.tokenizer(txt, truncation=True, return_tensors="pt")[
                    "input_ids"
                ].to(self.device)
                gen = self.model.generate(txt_tensor, forced_bos_token_id=forced_bos_token_id).cpu()
                generations.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0])

        return generations
