#!/bin/bash

# Allows for quick test runs - Set sample to e.g. 1
sample=${1:-100}
bs=${2:-1}

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-en-zh --data en-zh --out marianmt-en-zh-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-zh-en --data zh-en --out marianmt-zh-en-gdp.txt

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-en-jap --data en-ja --out marianmt-en-jap-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-jap-en --data ja-en --out marianmt-ja-en-gdp.txt

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-en-de --data en-de --out marianmt-en-de-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-de-en --data de-en --out marianmt-de-en-gdp.txt

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-de-fr --data de-fr --out marianmt-de-fr-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-fr-de --data fr-de --out marianmt-fr-de-gdp.txt

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-en-hi --data en-hi --out marianmt-en-hi-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-hi-en --data hi-en --out marianmt-hi-en-gdp.txt

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-en-fr --data en-fr --out marianmt-en-fr-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-fr-en --data fr-en --out marianmt-fr-en-gdp.txt

python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-en-it --data en-it --out marianmt-en-it-gdp.txt
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-it-en --data it-en --out marianmt-it-en-gdp.txt

# No English > Korean model
python main.py --sample $sample --bs $bs --model marianmt --weights Helsinki-NLP/opus-mt-ko-en --data ko-en --out marianmt-ko-en-gdp.txt

# Cat all into one
( for i in marianmt*gdp.txt ; do cat $i ; printf '\n\n\n\n----****----\n\n\n\n' ; done ) >> marianmt-gdp.txt
