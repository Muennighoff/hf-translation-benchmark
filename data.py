from typing import Dict, List

import sacrebleu


TRANSLATION_BENCHMARKS = {
    ds_name: sacrebleu.get_langpairs_for_testset(ds_name)
    for ds_name in sacrebleu.get_available_testsets()
}


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

    def get_src_ref(self) -> List[Dict]:
        """
        Returns:
            A iterable of any object, that doc_to_text can handle
        """
        return self.src_data, self.ref_data

    def doc_to_text(self, doc):
        return doc["src"], self.lang_codes

    # def format_bleu(preds: List, refs: List) -> tuple(List[str], List[List[str]]):

    def score_bleu(preds: List, refs: List) -> float:
        """
        Calculates BLEU score.
        """
        # Format into
        # refs, preds = _sacreformat(refs, preds)
        return sacrebleu.corpus_bleu(preds, [refs]).score
