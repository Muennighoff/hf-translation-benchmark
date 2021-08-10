from typing import List

import torch
from transformers import AutoTokenizer
from transformers import MBartForConditionalGeneration

### DEFINITIONS ###

# See https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
MBART_CODES = {
    "af": "af_ZA",
    "az": "az_AZ",
    "bn": "bn_IN",
    "cs": "cs_CZ",
    "de": "de_DE",
    "en": "en_XX",
    "es": "es_XX",
    "et": "et_EE",
    "fa": "fa_IR",
    "fi": "fi_FI",
    "fr": "fr_XX",
    "gl": "gl_ES",
    "gu": "gu_IN",
    "he": "he_IL",
    "hi": "hi_IN",
    "hr": "hr_HR",
    "id": "id_ID",
    "it": "it_IT",
    "ja": "ja_XX",
    "ka": "ka_GE",
    "kk": "kk_KZ",
    "km": "km_KH",
    "ko": "ko_KR",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "mk": "mk_MK",
    "ml": "ml_IN",
    "mn": "mn_MN",
    "mr": "mr_IN",
    "my": "my_MM",
    "ne": "ne_NP",
    "nl": "nl_XX",
    "pl": "pl_PL",
    "ps": "ps_AF",
    "pt": "pt_XX",
    "ro": "ro_RO",
    "ru": "ru_RU",
    "si": "si_LK",
    "sv": "sv_SE",
    "sw": "sw_KE",
    "ta": "ta_IN",
    "te": "te_IN",
    "th": "th_TH",
    "tl": "tl_XX",
    "tr": "tr_TR",
    "uk": "uk_UA",
    "ur": "ur_PK",
    "vi": "vi_VN",
    "xh": "xh_ZA",
    "zh": "zh_CN",
}

### MODEL ###


class MBART:
    """
    MBART model wrapper.
    See below for the generation settings of MBART (5 beams):
    https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/blob/main/config.json

    Args:
        device: Where torch should move the model
    """

    def __init__(
        self, device=torch.device("cuda"), weights: str = "facebook/mbart-large-50-many-to-many-mmt"
    ) -> None:

        self.device = device
        self.model = MBartForConditionalGeneration.from_pretrained(weights).to(self.device)
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
        self.tokenizer.src_lang = MBART_CODES[src_lang]
        forced_bos_token_id = self.tokenizer.lang_code_to_id[MBART_CODES[tar_lang]]

        generations = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                gen = self.model.generate(**batch, forced_bos_token_id=forced_bos_token_id).cpu()
                generations.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))

        return generations
