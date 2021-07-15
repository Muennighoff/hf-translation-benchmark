
## Huggingface 🤗 Translation Benchmark


This simple repository is for quickly benchmarking multilingual transformer models.


## Usage

### Running benchmarking

```
python main.py
```

### Adding models

Just copy one of the existing models (e.g. m2m.py) & implement the greedy_until method. 
Then add the necessary import statements to `models/__init__.py` & `main.py`.

### Adding data

Would require some structural changes, but could definitely be worth it.

## TODO

- As soon as PyTorch has fixed memory profiling, include memory profiling again, see (1)[https://github.com/pytorch/kineto/issues/308], (2)[https://github.com/pytorch/pytorch/pull/60432]



