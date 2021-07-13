import argparse
import random
import logging

import torch
import numpy as np
from tqdm import tqdm

from m2m import M2M
from data import TRANSLATION_BENCHMARKS, Translation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="m2m")
    parser.add_argument("--weights", type=str, default="facebook/m2m100_418M")

    # TODO: Lateron users should be able to just choose the language & the best data is picked
    # No one cares about wmt20 or wmt19
    parser.add_argument("--data", type=str, default="wmt20")
    parser.add_argument("--sample", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


MODELS = {"m2m": M2M}


def main():

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MODELS[args.model](device=device, weights=args.weights)

    # Iterate through tasks
    results = {}
    for lang_pair in TRANSLATION_BENCHMARKS[args.data]:

        # Data preparation
        task = Translation(args.data, lang_pair)
        task.download()
        src, ref = task.get_src_ref(args.sample)

        # Predict on all data
        src_lang, tar_lang = lang_pair.split("-")
        preds = []
        for txt in tqdm(src):
            preds.append(model.greedy_until(txt, src_lang, tar_lang))

        # Score on all data
        score = task.score_bleu(preds, ref)

        results[lang_pair] = score

    out = "\n-----\n".join([f"{lang_pair} - BLEU: {score}" for lang_pair, score in results.items])
    print(out)
    with open("./out.txt") as f:
        f.write(out)


if __name__ == "__main__":
    main()
