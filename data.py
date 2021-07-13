from typing import Dict, List

import sacrebleu


class Translation:
    def __init__(self, sacrebleu_dataset, sacrebleu_language_pair=None) -> None:
        """
        sacrebleu_dataset: sacrebleu_dataset
        """
        self.sacrebleu_dataset = sacrebleu_dataset

        self.sacrebleu_language_pair = sacrebleu_language_pair
        self.lang_codes = self.sacrebleu_language_pair.split("-")

    def download(self):
        """
        Download datasets.
        """
        # This caches in the users home dir automatically
        self.src_file, self.ref_file = sacrebleu.download_test_set(
            self.sacrebleu_dataset, self.sacrebleu_language_pair
        )
        self.src_data, self.ref_data = [
            [line.rstrip() for line in sacrebleu.smart_open(file)]
            for file in (self.src_file, self.ref_file)
        ]

    def get_src_ref(self, sample=None) -> List[Dict]:
        """
        Returns:
            A iterable of any object, that doc_to_text can handle
            sample: How much to sample - Note that a None slice returns the entire list
        """
        return self.src_data[:sample], self.ref_data[:sample]

    def doc_to_text(self, doc):
        return doc["src"], self.lang_codes

    # def format_bleu(preds: List, refs: List) -> tuple(List[str], List[List[str]]):

    def score_bleu(self, preds: List, refs: List) -> float:
        """
        Calculates BLEU score.
        """
        # Format into
        # refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_bleu(preds, [refs]).score


TRANSLATION_BENCHMARKS = {}

for ds_name in sacrebleu.get_available_testsets():
    TRANSLATION_BENCHMARKS[ds_name] = []
    for lang_pair in sacrebleu.get_langpairs_for_testset(ds_name):
        TRANSLATION_BENCHMARKS[ds_name].append((ds_name, lang_pair))
        TRANSLATION_BENCHMARKS[f"{ds_name}-{lang_pair}"] = [(ds_name, lang_pair)]


TRANSLATION_BENCHMARKS["cs-en"] = [("wmt20", "cs-en")]
TRANSLATION_BENCHMARKS["de-en"] = [("wmt20", "de-en")]
TRANSLATION_BENCHMARKS["de-fr"] = [("wmt20", "de-fr")]
TRANSLATION_BENCHMARKS["en-cs"] = [("wmt20", "en-cs")]
TRANSLATION_BENCHMARKS["en-de"] = [("wmt20", "en-de")]
TRANSLATION_BENCHMARKS["en-iu"] = [("wmt20", "en-iu")]
TRANSLATION_BENCHMARKS["en-ja"] = [("wmt20", "en-ja")]
TRANSLATION_BENCHMARKS["en-km"] = [("wmt20", "en-km")]
TRANSLATION_BENCHMARKS["en-pl"] = [("wmt20", "en-pl")]
TRANSLATION_BENCHMARKS["en-ps"] = [("wmt20", "en-ps")]
TRANSLATION_BENCHMARKS["en-ru"] = [("wmt20", "en-ru")]
TRANSLATION_BENCHMARKS["en-ta"] = [("wmt20", "en-ta")]
TRANSLATION_BENCHMARKS["en-zh"] = [("wmt20", "en-zh")]
TRANSLATION_BENCHMARKS["fr-de"] = [("wmt20", "fr-de")]
TRANSLATION_BENCHMARKS["iu-en"] = [("wmt20", "iu-en")]
TRANSLATION_BENCHMARKS["ja-en"] = [("wmt20", "ja-en")]
TRANSLATION_BENCHMARKS["km-en"] = [("wmt20", "km-en")]
TRANSLATION_BENCHMARKS["pl-en"] = [("wmt20", "pl-en")]
TRANSLATION_BENCHMARKS["ps-en"] = [("wmt20", "ps-en")]
TRANSLATION_BENCHMARKS["ru-en"] = [("wmt20", "ru-en")]
TRANSLATION_BENCHMARKS["ta-en"] = [("wmt20", "ta-en")]
TRANSLATION_BENCHMARKS["zh-en"] = [("wmt20", "zh-en")]
