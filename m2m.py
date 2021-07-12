from typing import List
from typing import List

from transformers import AutoTokenizer
from transformers import M2M100ForConditionalGeneration


class M2M:
    def __init__(self, device="cuda", weights="facebook/m2m100_418M") -> None:

        self.device = device
        self.model = M2M100ForConditionalGeneration.from_pretrained(weights).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("weights")

    def greedy_until(self, texts: List, src_lang: str, tar_lang: str) -> str:
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

        for txt in texts:
            encoded_txt = self.tokenizer(txt, return_tensors="pt")
            gen = self.model.generate(**encoded_txt, forced_bos_token_id=forced_bos_token_id)
            generations.append(gen)

        return generations
