from typing import Dict, List

import sacrebleu

### Language Specifics ###

import jieba
import MeCab
import mecab_ko_dic
import nagisa

def zh_split(zh_text: List[str]) -> List[str]:
    return [" ".join(jieba.cut(zh_text[0]))]


def ja_split(ja_text: List[str]) -> List[str]:
    return [" ".join(nagisa.tagging(ja_text[0]).words)]


def ko_split(ko_text: List[str]) -> List[str]:
    tagger = MeCab.Tagger(mecab_ko_dic.MECAB_ARGS)
    # Skip the tags ::2 & the final EOS token :-1
    return [" ".join(tagger.parse(ko_text[0]).split()[::2][:-1])]

NO_SPACE_LANG = {"zh": zh_split, "ja": ja_split, "ko": ko_split}


### Core logic ###


class Translation:
    def __init__(self, sacrebleu_dataset, sacrebleu_language_pair=None) -> None:
        """
        sacrebleu_dataset: sacrebleu_dataset
        """
        self.sacrebleu_dataset = sacrebleu_dataset

        self.sacrebleu_language_pair = sacrebleu_language_pair
        self.lang_codes = self.sacrebleu_language_pair.split("-")

    def download(self) -> None:
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

    def score_bleu(self, preds: List, refs: List) -> float:
        """
        Calculates BLEU score.
        """
        # TODO (Muennighoff): Format when there are multiple refs
        # refs, preds = _sacreformat(refs, preds)

        # If target lang is e.g. chinese, add spaces
        if self.lang_codes[-1] in NO_SPACE_LANG:
            preds = NO_SPACE_LANG[self.lang_codes[-1]](preds)
            refs = NO_SPACE_LANG[self.lang_codes[-1]](refs)

        return sacrebleu.corpus_bleu(preds, [refs]).score


### Data Presets ###

TRANSLATION_BENCHMARKS = {}

# Allowing indexing via wmt20 or wmt20-de-fr or de-fr
for ds_name in sacrebleu.get_available_testsets():
    TRANSLATION_BENCHMARKS[ds_name] = []
    for lang_pair in sacrebleu.get_langpairs_for_testset(ds_name):
        TRANSLATION_BENCHMARKS[ds_name].append((ds_name, lang_pair))
        TRANSLATION_BENCHMARKS[f"{ds_name}-{lang_pair}"] = [(ds_name, lang_pair)]
         # Setting a default for each lang pair
        TRANSLATION_BENCHMARKS[lang_pair] = [(ds_name, lang_pair)]
        
# Overwriting some default languages with most recent / commonly used versions
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

TRANSLATION_BENCHMARKS["en-hi"] = [("wmt14", "en-hi")]
TRANSLATION_BENCHMARKS["hi-en"] = [("wmt14", "hi-en")]

TRANSLATION_BENCHMARKS["en-es"] = [("wmt13", "en-es")]
TRANSLATION_BENCHMARKS["es-en"] = [("wmt13", "es-en")]

TRANSLATION_BENCHMARKS["en-ko"] = [("iwslt17", "en-ko")]
TRANSLATION_BENCHMARKS["ko-en"] = [("iwslt17", "ko-en")]
TRANSLATION_BENCHMARKS["en-fr"] = [("iwslt17", "en-fr")]
TRANSLATION_BENCHMARKS["fr-en"] = [("iwslt17", "fr-en")]

TRANSLATION_BENCHMARKS["en-it"] = [("wmt09", "en-it")]
TRANSLATION_BENCHMARKS["it-en"] = [("wmt09", "it-en")]


# Allowing indexing via e.g. "gdp-top10" by pre-selecting lang pairs

# https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)
lang_pairs = ["en-zh", "en-ja", "en-de", "de-fr", "en-hi", "en-fr", "en-it", "en-ko"]
TRANSLATION_BENCHMARKS["gdp-top10"] = []
for pair in lang_pairs:
    TRANSLATION_BENCHMARKS["gdp-top10"].extend(TRANSLATION_BENCHMARKS[pair])
    src, tar = pair.split("-")
    inv_pair = "-".join([tar, src])
    TRANSLATION_BENCHMARKS["gdp-top10"].extend(TRANSLATION_BENCHMARKS[inv_pair])

# https://huggingface.co/Helsinki-NLP/opus-mt-en-roa
# From english to some romance languages available in MarianMT
lang_pairs = ["en-es", "en-fr", "en-it"]
TRANSLATION_BENCHMARKS["mt-en-roa"] = []
for pair in lang_pairs:
    TRANSLATION_BENCHMARKS["mt-en-roa"].extend(TRANSLATION_BENCHMARKS[pair])

# https://huggingface.co/Helsinki-NLP/opus-mt-roa-en
# From romance languages to english available in MarianMT
lang_pairs = ["es-en", "fr-en", "it-en"]
TRANSLATION_BENCHMARKS["mt-roa-en"] = []
for pair in lang_pairs:
    TRANSLATION_BENCHMARKS["mt-roa-en"].extend(TRANSLATION_BENCHMARKS[pair])

# https://en.wikipedia.org/wiki/List_of_languages_by_number_of_native_speakers
