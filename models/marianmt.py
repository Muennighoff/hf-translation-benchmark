from typing import List

import torch
from transformers import AutoTokenizer
from transformers import MarianMTModel


### DEFINITIONS ###

# The codes are inconsistent across the models unfortunately, so need to be defined by weight basis

MARIANMT_CODES = {
    "Helsinki-NLP/opus-mt-en-roa": {
        "es": ">>spa<< ",
        "fr": ">>fra<< ",
        "it": ">>ita<< ",
        "ro": ">>ron<< ",
        "ca": ">>cat<< ",
        "id": ">>ind<< ",
    }
}

### MODEL ###


class MARIANMT:
    """
    MARIANMT model wrapper.

    The models are never many-to-many, so either target or source language are predefined.
    If target language predefined, no need for codes.
    If source language predefined, codes are prepended.

    Args:
        device: Where torch should move the model
        weights: Weights to download from huggingface
    """

    def __init__(
        self, device=torch.device("cuda"), weights: str = "Helsinki-NLP/opus-mt-en-roa"
    ) -> None:

        self.device = device
        self.weights = weights
        self.model = MarianMTModel.from_pretrained(weights).to(self.device)
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
        if self.weights in MARIANMT_CODES:
            prepend = MARIANMT_CODES[self.weights][tar_lang]
        else:
            prepend = ""

        generations = []

        with torch.no_grad():
            for txt in texts:
                txt = prepend + txt
                txt_tensor = self.tokenizer(txt, truncation=True, return_tensors="pt")[
                    "input_ids"
                ].to(self.device)
                gen = self.model.generate(txt_tensor).cpu()
                generations.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0])

        return generations
