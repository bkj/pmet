#!/bin/bash

mkdir -p ./data
python ./utils/fake-data.py --n 1000 > ./data/train.jl
python ./utils/fake-data.py --n 1000 > ./data/test.jl