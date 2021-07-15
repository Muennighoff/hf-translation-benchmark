
## Huggingface 🤗 Translation Benchmark


This simple repository is for quickly benchmarking multilingual transformer models.


## Usage

### Running benchmarking

```
python main.py --model m2m --weights facebook/m2m100_418M --data en-ja --sample 100 --out m2m-en-ja.txt
```

See the table below for a BLEU Score benchmark on the Top 10 languages by country GDP:

|       |  m2m-418  | m2m-1200  | mbart  | marianmt  | GPT-2* |
| ----------- | ----------- |-----------|-----------|-----------|
| en-zh   |  **X/3**  |  **X/3** |  **X/3** | **X/3** | **X/3** |
| zh-en   |  **X/3**  |  **X/3** |  **X/3** | **X/3** | **X/3** |
| en-ja   |  **X/3**  |  **X/3** |  **X/3** | **X/3** | **X/3** |
| ja-en   |  **X/3**  |  **X/3** |  **X/3** |  **X/3**| **X/3** |
| Memory   |  **X/3**  |  **X/3** |  **X/3** | **X/3**| **X/3** |



*Run with (lm-evaluation-harness)[https://github.com/EleutherAI/lm-evaluation-harness]




### Adding models

Just copy one of the existing models (e.g. m2m.py) & implement the greedy_until method. 
Then add the necessary import statements to `models/__init__.py` & `main.py`.

### Adding data

Would require some structural changes, but could definitely be worth it.

## TODO

- As soon as PyTorch has fixed memory profiling, include memory profiling again, see (1)[https://github.com/pytorch/kineto/issues/308], (2)[https://github.com/pytorch/pytorch/pull/60432]



