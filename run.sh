#!/bin/bash

mkdir -p ./data
python ./utils/fake_data.py --n 1000 > ./data/train.tsv
python ./utils/fake_data.py --n 1000 > ./data/test.tsv

python -m pmet --attention --bidirectional