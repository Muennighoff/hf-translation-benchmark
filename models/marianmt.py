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
        if self.weights in MARIANMT_CODES:
            prepend = MARIANMT_CODES[self.weights][tar_lang]
        else:
            prepend = ""

        generations = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = [prepend + txt for txt in batch]
                batch = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                gen = self.model.generate(**batch).cpu()
                generations.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))

        return generations
