#!/bin/bash

mkdir -p ./data
python ./utils/fake-data.py --n 1000 > ./datatrain.jl
python ./utils/fake-data.py --n 1000 > ./data/test.jl

python models/selfatt-model.py