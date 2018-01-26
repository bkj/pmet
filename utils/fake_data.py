#!/usr/bin/env python

"""
    fake-data.py
    
    Generate fake data
"""

import os
import sys
import random
import argparse
import ujson as json
import pandas as pd
from faker import Faker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--types', type=str, default="sha1,user_name,ssn,unix_time,street_suffix,safe_email,name_male")
    parser.add_argument('--dataframe', action="store_true")
    parser.add_argument('--list', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    faker = Faker()
    args = parse_args()
    if args.list:
        print('\n'.join(dir(faker)))
        os._exit(0)
        
    types = args.types.split(',')
    
    if not args.dataframe:
        for i in range(args.n):
            c = random.choice(types)
            print '\t'.join((c, str(getattr(faker, c)())))
    else:
        data = []
        for i in range(args.n):
            tmp = {}
            for type_ in types:
                tmp[type_] = str(getattr(faker, type_)())
            data.append(tmp)
            
        pd.DataFrame(data).to_csv(sys.stdout, sep='\t', index=False)
