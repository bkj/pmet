#!/usr/bin/env python

"""
    fake-data.py
    
    Generate fake data, for testing `pit` models
"""

import os
import random
import argparse
import ujson as json
from faker import Faker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--types', type=str, default="sha1,user_name,ssn,unix_time,street_suffix,safe_email,name_male")
    parser.add_argument('--list', action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    faker = Faker()
    args = parse_args()
    if args.list:
        print('\n'.join(dir(faker)))
        os._exit(0)
    
    types = args.types.split(',')
    
    for i in xrange(args.n):
        c = random.choice(types)
        print json.dumps({
            "label" : c,
            "value" : getattr(faker, c)()
        })