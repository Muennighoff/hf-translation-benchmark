from collections import defaultdict
import logging
from typing import List

import pysbd
import torch
from transformers import AutoTokenizer
from transformers import M2M100ForConditionalGeneration


LANGUAGE_CODE_LABELS = {
    "am": "Amharic",
    "ar": "Arabic",
    "ast": "Asturian",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan/Valencian",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "ff": "Fulah",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic/Scottish Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian/Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "ilo": "Iloko",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lb": "Luxembourgish/Letzeburgesch",
    "lg": "Ganda",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch/Flemish",
    "no": "Norwegian",
    "ns": "Northern Sotho",
    "oc": "Occitan (post 1500)",
    "or": "Oriya",
    "pa": "Panjabi/Punjabi",
    "pl": "Polish",
    "ps": "Pushto/Pashto",
    "pt": "Portuguese",
    "ro": "Romanian/Moldavian/Moldovan",
    "ru": "Russian",
    "sd": "Sindhi",
    "si": "Sinhala/Sinhalese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
}

# Where available, map languages to the same pysbd language
SRC_TO_PYSBD_DIRECT = {
    "am": "am",
    "ar": "ar",
    "bg": "bg",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "es": "es",
    "fa": "fa",
    "fr": "fr",
    "hi": "hi",
    "hy": "hy",
    "it": "it",
    "ja": "ja",
    "kk": "kk",
    "mr": "mr",
    "my": "my",
    "nl": "nl",
    "pl": "pl",
    "ru": "ru",
    "sk": "sk",
    "ur": "ur",
    "zh": "zh",
}

# Else map languages by their family or script to pysbd compatible languages
SRC_TO_PYSBD_INDIRECT = {
    "ast": "es",
    "az": "kk",
    "ba": "kk",
    "be": "ru",
    "bn": "hi",
    "br": "fr",
    "bs": "bg",
    "ca": "es",
    "ceb": "en",  # Not same family, but Latin script
    "cz": "sk",
    "cy": "en",
    "et": "de",
    "ff": "en",  # Not same family, but Latin script
    "fi": "de",
    "fy": "en",
    "ga": "en",
    "gd": "en",
    "gl": "es",
    "gu": "hi",
    "ha": "ar",
    "he": "ar",
    "hr": "bg",
    "ht": "fr",
    "hu": "de",
    "id": "nl",
    "ig": "ar",
    "ilo": "en",  # Not same family, but Latin script
    "is": "da",
    "jv": "zh",  # Not same family, but related scripts
    "ka": "ru",
    "km": "zh",  # Not same family, but related scripts
    "kn": "hi",
    "ko": "ja",  # Not same family, but empirically good results
    "lb": "de",
    "lg": "en",  # Not same family, but Latin script
    "ln": "en",  # Not same family, but Latin-like script
    "lo": "zh",  # Not same family, but related scripts
    "lt": "pl",
    "lv": "pl",
    "mg": "en",  # Not same family, but Latin script
    "mk": "bg",
    "ml": "hi",
    "mn": "ru",  # Not same family, but Cyrillic script
    "ms": "en",  # Not same family, but Latin script
    "ne": "hi",
    "no": "da",
    "ns": "en",  # Not same family, but Latin script
    "oc": "fr",
    "or": "hi",
    "pa": "hi",
    "ps": "fa",
    "pt": "es",
    "ro": "es",
    "sd": "hi",
    "si": "mr",
    "sl": "sk",
    "so": "en",  # Shares lang family with Arabic, but is mostly using Latin script
    "sq": "pl",
    "sr": "bg",
    "ss": "en",  # Not same family, but Latin script
    "su": "en",  # Not same family, but Latin script
    "sv": "da",
    "sw": "en",  # Not same family, but Latin script
    "ta": "hi",
    "th": "zh",  # Not same family, but related scripts
    "tl": "en",  # Not same family, but Latin script
    "tn": "en",  # Not same family, but Latin script
    "tr": "kk",
    "uk": "ru",
    "uz": "kk",
    "vi": "en",  # Not same family, but Latin script
    "wo": "en",  # Not same family, but Latin script
    "xh": "en",  # Not same family, but Latin script
    "yi": "en",  # Not same family, but related scripts
    "yo": "en",  # Not same family, but Latin script
}

SRC_TO_PYSBD = {**SRC_TO_PYSBD_DIRECT, **SRC_TO_PYSBD_INDIRECT}

MAX_INPUT_TOKENS = 300


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

    def generate(
        self,
        input_series,
        tar_lang: str,
        src_lang: str,
        split_sentences: bool = True,
        batch_size: int = 1,
        num_beams: int = 5,
        **kwargs,
    ) -> List[str]:
        """
        Generates translations of texts from source to target language.

        Args:
            input_series: Series with texts to process [LIST in this case]
            tar_lang: Language code of target language
            src_lang: Language code of source language
            split_sentences: Whether the model should split sentences before translating
            batch_size: Num texts to process at once
            num_beams: Number of beams for beam search, 1 means no beam search,
                the default of 5 is used by e.g. M2M100
        Returns:
            translated_texts: Translated texts
        """
        self.tokenizer.src_lang = src_lang
        if split_sentences:
            seg = pysbd.Segmenter(language=SRC_TO_PYSBD[src_lang], clean=False)
            logging.info(f"Mapped {src_lang} to {SRC_TO_PYSBD[src_lang]}")

        translated_texts = []
        success_count = 0
        with torch.no_grad():
            for i in range(0, len(input_series), batch_size):
                # Subselect batch_size items
                batch = input_series[i : i + batch_size]
                # Turn into List[List[str]] with each str being one sentence
                if split_sentences:
                    batch = [seg.segment(txt) for txt in batch]
                else:
                    batch = [[txt] for txt in batch]
                # Prepare the model inputs
                batch_tokens = defaultdict(list)
                batch_ix = []
                for txt in batch:
                    for sentence in txt:
                        # Convert string to list of integers according to tokenizer's vocabulary
                        tokens = self.tokenizer.tokenize(sentence)
                        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                        # Enforce a maximum length in case of incorrect splitting or too long sentences
                        for i in range(0, len(tokens), MAX_INPUT_TOKENS):
                            input_dict = self.tokenizer.prepare_for_model(
                                tokens[i : i + MAX_INPUT_TOKENS], add_special_tokens=True
                            )
                            # input_ids: Same as tokens, but with model-specific beginning and end tokens
                            # attention_mask: List of 1s for each input_id, i.e. the tokens it should attend to
                            batch_tokens["input_ids"].append(input_dict["input_ids"])
                            batch_tokens["attention_mask"].append(input_dict["attention_mask"])
                        if len(tokens) > MAX_INPUT_TOKENS:
                            logging.warning(
                                f"Sentence is too long ({len(tokens)} > {MAX_INPUT_TOKENS}), and will be translated in pieces, which might degrade performance. Check the source language and/or consider using the 'Split Sentences' option."
                            )
                    # Store the new length with each new sub_batch to discern what batch each text belongs to
                    batch_ix.append(len(batch_tokens["input_ids"]))
                # No need for truncation, as all inputs are now trimmed to less than the models seq length
                batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
                # Move to CPU/GPU
                batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
                translated_batch = self.model.generate(
                    **batch_tokens,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tar_lang),
                    num_beams=num_beams,
                    **kwargs,
                ).cpu()
                # Decode back to strings
                translated_batch = self.tokenizer.batch_decode(
                    translated_batch, skip_special_tokens=True
                )
                # Stitch back together by iterating through start & end indices, e.g. (0,1), (1,3)..
                translated_batch = [
                    " ".join(translated_batch[ix_s:ix_e])
                    for ix_s, ix_e in zip([0] + batch_ix, batch_ix)
                ]
                translated_texts.extend(translated_batch)
                success_count += 1

        return translated_texts
