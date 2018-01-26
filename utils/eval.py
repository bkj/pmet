#!/usr/bin/env python

from __future__ import print_function

import sys
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    labels = open(args.labels).read().splitlines()
    predictions = open(args.predictions).read().splitlines()
    
    labels, predictions = np.array(labels), np.array(predictions)
    
    acc = (labels == predictions).mean()
    print('acc=%f' % acc, file=sys.stderr)
    
    print(pd.crosstab(labels, predictions))