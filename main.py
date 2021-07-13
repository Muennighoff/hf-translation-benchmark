import argparse
import random
import logging
import time

import torch
import numpy as np
from tqdm import tqdm
from pytorch_memlab import LineProfiler


from m2m import M2M
from mbart import MBART
from marianmt import MARIANMT

from data import TRANSLATION_BENCHMARKS, Translation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="m2m")
    parser.add_argument("--weights", type=str, default="facebook/m2m100_418M")
    parser.add_argument("--data", help="E.g. de-fr, wmt20, top10_gdp", type=str, default="en-fr")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="out.txt")
    return parser.parse_args()


MODELS = {"m2m": M2M, "mbart": MBART, "marianmt": MARIANMT}


def main():

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MODELS[args.model](device=device, weights=args.weights)

    # Iterate through tasks
    results = {}
    for dataset, lang_pair in tqdm(TRANSLATION_BENCHMARKS[args.data]):

        # Data preparation
        task = Translation(dataset, lang_pair)
        task.download()
        src, ref = task.get_src_ref(args.sample)

        # Predict on all data
        src_lang, tar_lang = lang_pair.split("-")

        with LineProfiler(model.greedy_until) as prof:
            x = time.time()
            preds = model.greedy_until(src, src_lang, tar_lang)
            avg_speed = (time.time() - x) / len(src)
        mem_report = prof.display()

        # Score on all data
        score = task.score_bleu(preds, ref)

        results[lang_pair] = {"score": score, "avg_speed": avg_speed, "mem_report": mem_report}

        logging.info(f"Scored {score} on language pair {lang_pair}.")

    out_string = f"MODEL REPORT: {args.weights}"
    for lang_pair, info in results.items():
        out_string += f'\n---------------------------\n{lang_pair}\n\nBLEU\n{info["score"]}\n\nAVG SPEED\n{info["avg_speed"]}\n\nMEM REPORT\n{info["mem_report"]}'

    print(out_string)
    with open(args.out, "w") as f:
        f.write(out_string)


if __name__ == "__main__":
    main()
