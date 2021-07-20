import argparse
import random
import logging
import time

import torch
from torch.profiler import profile, ProfilerActivity
import numpy as np
from tqdm import tqdm

from models import M2M, MBART, MARIANMT
from data.data import TRANSLATION_BENCHMARKS, Translation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="m2m")
    parser.add_argument("--weights", type=str, default="facebook/m2m100_418M")
    parser.add_argument("--data", help="E.g. de-fr, wmt20, gdp-top10", type=str, default="en-fr")
    parser.add_argument("--bs", type=int, help="Batch size to use", default=1)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="out.txt")
    return parser.parse_args()


MODELS = {"m2m": M2M, "mbart": MBART, "marianmt": MARIANMT}


def main():

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        prof_acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        device = torch.device("cuda")
    else:
        prof_acts = [ProfilerActivity.CPU]
        device = torch.device("cpu")

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

        x = time.time()
        preds = model.greedy_until(src, src_lang, tar_lang, args.bs)
        avg_speed = (time.time() - x) / len(src)

        # Profile CPU / CUDA usage on just one sample
        with profile(activities=prof_acts, record_shapes=True) as prof:
            model.greedy_until(src[:1], src_lang, tar_lang)

        # Score on all data
        score = task.score_bleu(preds, ref)

        results[lang_pair] = {
            "score": score,
            "avg_speed": avg_speed,
            "mem_report": prof.key_averages().table(sort_by="cpu_time_total", row_limit=10),
            "example": f"Src: {src[0]}\nPred: {preds[0]}\nRef: {ref[0]}",
        }

        logging.info(f"Scored {score} on language pair {lang_pair}.")

    out_string = f"MODEL REPORT: {args.weights}"
    for lang_pair, info in results.items():
        out_string += f'\n---------------------------\n{lang_pair}\n\nBLEU\n{info["score"]}\n\nAVG SPEED\n{info["avg_speed"]}\n\nMEM REPORT\n{info["mem_report"]}\n\nEXAMPLE\n{info["example"]}'

    print(out_string)
    with open(args.out, "w") as f:
        f.write(out_string)


if __name__ == "__main__":
    main()
