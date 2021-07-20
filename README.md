
# Huggingface 🤗 Translation Benchmark

This simple repository is for quickly benchmarking multilingual transformer models.

# Usage

## Running benchmarking

```
python main.py --model m2m --weights facebook/m2m100_418M --data en-ja
```

Data could be:
- Language Pair: en-zh
- WMT dataset: wmt20
- WMT data + language pair: wmt20-en-zh
- Predefined set: gdp-top10

Feel free to add new definitions~


### Benchmark Example: Languages of Top-10 GDP Countries

#### Commands used

```
python main.py --model m2m --weights facebook/m2m100_418M --data gdp-top10 --sample 500 --out m2m418-gdp.txt
```

```
python main.py --model m2m --weights facebook/m2m100_1.2B --data gdp-top10 --sample 500 --out m2m1200-gdp.txt
```

```
python main.py --model mbart --weights facebook/mbart-large-50-many-to-many-mmt --data gdp-top10 --sample 500 --out mbart-gdp.txt
```

```
bash models/marian-gdp-top10.sh 500 50
```

Note that the code always samples the same first 500 examples, so all models predicted on the same data.

#### Specs

GPU: 1x Tesla P100, CUDA 11.0

CPU (when also GPU): Intel Xeon 2.00 GHz, 1 Core

CPU (no GPU): Intel Xeon 2.20 GHz, 2 Cores

#### Results

SPEED is reported as **average seconds** per sample.


|             |             | m2m-418    | m2m-1200  | mbart     | marianmt  | gpt-2* |
| ----------- | ----------- |----------- |-----------|-----------|-----------|-----------|
| en-zh       |  BLEU       | **0.402**  | **0.276** | **0.235** | **0.350** | **0.021** |
|             |  GPU-SPEED  | **0.284**  | **0.583** | **0.326** | **0.067** | **X** |
|             |  CPU-SPEED  | **4.888**  | **12.199**| **5.601** | **1.216** | **X** |
| zh-en       |  BLEU       | **20.495** | **25.952**| **22.890**| **21.379**| **0.280** |
|             |  GPU-SPEED  | **0.410**  | **0.802** | **0.431** | **0.120** | **X** |
|             |  CPU-SPEED  | **6.861**  | **14.463**| **7.657** | **2.562** | **X** |
| en-ja       |  BLEU       | **0.910**  | **2.396** | **2.242** | **0.002** | **X** |
|             |  GPU-SPEED  | **0.290**  | **0.627** | **0.294** | **0.071** | **X** |
|             |  CPU-SPEED  | **4.729**  | **11.359**| **5.359** | **1.095** | **X** |
| ja-en       |  BLEU       | **11.375** | **13.150**| **17.588**| **9.059** | **X** |
|             |  GPU-SPEED  | **0.318**  | **0.666** | **0.344** | **0.302** | **X** |
|             |  CPU-SPEED  | **5.393**  | **12.469**| **6.239** | **7.348** | **X** |
| en-de       |  BLEU       | **21.951** | **27.846**| **27.677**| **29.932**| **0.458** |
|             |  GPU-SPEED  | **0.376**  | **0.802** | **0.418** | **0.078** | **X** |
|             |  CPU-SPEED  | **6.177**  | **15.320**| **8.396** | **1.418** | **X** |
| de-en       |  BLEU       | **29.729** | **32.457**| **38.043**| **38.910**| **0.626** |
|             |  GPU-SPEED  | **0.519**  | **1.048** | **0.549** | **0.091** | **X** |
|             |  CPU-SPEED  | **8.571**  | **19.416**| **11.189**| **1.635** | **X** |
| de-fr       |  BLEU       | **26.232** | **30.204**| **17.587**| **29.179**| **X** |
|             |  GPU-SPEED  | **0.263**  | **0.527** | **0.313** | **0.055** | **X** |
|             |  CPU-SPEED  | **3.843**  | **9.953** | **6.170** | **1.040** | **X** |
| fr-de       |  BLEU       | **19.767** | **24.830**| **15.877**| **23.511**| **X** |
|             |  GPU-SPEED  | **0.259**  | **0.520** | **0.260** | **0.045** | **X** |
|             |  CPU-SPEED  | **3.630**  | **9.696** | **4.820** | **0.869** | **X** |
| en-hi       |  BLEU       | **19.637** | **20.701**| **18.350**| **11.974**| **X** |
|             |  GPU-SPEED  | **0.276**  | **0.533** | **0.261** | **0.064** | **X** |
|             |  CPU-SPEED  | **5.750**  | **11.537**| **4.729** | **1.036** | **X** |
| hi-en       |  BLEU       | **22.044** | **24.534**| **24.275**| **13.086**| **X** |
|             |  GPU-SPEED  | **0.241**  | **0.478** | **0.239** | **0.059** | **X** |
|             |  CPU-SPEED  | **4.851**  | **10.455**| **4.389** | **1.105** | **X** |
| en-fr       |  BLEU       | **31.116** | **34.476**| **32.826**| **38.482**| **X** |
|             |  GPU-SPEED  | **0.230**  | **0.468** | **0.271** | **0.041** | **X** |
|             |  CPU-SPEED  | **3.738**  | **8.485** | **5.112** | **0.648** | **X** |
| fr-en       |  BLEU       | **31.713** | **34.347**| **37.543**| **38.694**| **X** |
|             |  GPU-SPEED  | **0.175**  | **0.353** | **0.203** | **0.033** | **X** |
|             |  CPU-SPEED  | **2.838**  | **6.364** | **3.499** | **0.438** | **X** |
| en-it       |  BLEU       | **21.336** | **23.061**| **21.365**| **24.244**| **X** |
|             |  GPU-SPEED  | **0.235**  | **0.496** | **0.254** | **0.047** | **X** |
|             |  CPU-SPEED  | **3.511**  | **9.096** | **5.143** | **0.709** | **X** |
| it-en       |  BLEU       | **27.634** | **30.382**| **26.987**| **29.892**| **X** |
|             |  GPU-SPEED  | **0.205**  | **0.424** | **0.239** | **0.056** | **X** |
|             |  CPU-SPEED  | **2.960**  | **7.679** | **4.835** | **0.974** | **X** |
| en-ko       |  BLEU       | **2.932**  | **3.475** | **6.005** | **X** | **X** |
|             |  GPU-SPEED  | **0.229**  | **0.432** | **0.262** | **X** | **X** |
|             |  CPU-SPEED  | **4.400**  | **9.451** | **4.564** | **X** | **X** |
| ko-en       |  BLEU       | **11.862** | **12.902**| **20.109**| **17.443**| **X** |
|             |  GPU-SPEED  | **0.176**  | **0.341** | **0.229** | **0.043** | **X** |
|             |  CPU-SPEED  | **3.312**  | **7.259** | **4.143** | **0.649** | **X** |
| Average     |  BLEU       | **18.695** | **21.312**| **20.600**| **21.742**| **0.346** |
|             |  GPU-SPEED  | **0.280**  | **0.569** | **0.306** | **0.078** | **X** |
|             |  CPU-SPEED  | **4.716**  | **10.950** | **5.7403** | **1.516** | **X** |

*Run with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).


**Possible** reasons for different results from what's reported in the papers:
- Different models, e.g. M2M 12B parameter model
- Different data, e.g. EN-ZH from WMT19, not WMT20 as above
- Original <-> Huggingface differences (The above use HF)
- Different BLEU score calculation (e.g. different N-Grams, above uses 1-4 N-Grams)
- Overstated results


## Adding models

Just copy one of the existing models (e.g. m2m.py) & implement the greedy_until method. 
Then add the necessary import statements to `models/__init__.py` & `main.py`.

Ideas to add:
- GPT-J
- GPT-3


## Adding data

Would require some structural changes, but could definitely be worth it.

Ideas to add:
- Tatoeba
- More non EN pairs

# TODO

- Train transformer models to detokenize no-space langs like KO / CN / JP; We don't want to use those additional packages with GPL-license
- As soon as PyTorch has fixed memory profiling, include memory profiling again, see [1](https://github.com/pytorch/kineto/issues/308), [2](https://github.com/pytorch/pytorch/pull/60432)



