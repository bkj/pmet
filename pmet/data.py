
"""
    data.py
"""

import os
import dill as pickle
from torchtext import data

def make_train_dataset(indir, train='train', test='test', device=0):
    
    LABS = data.Field()
    VALS = data.Field(tokenize=list)
    
    datasets = data.TabularDataset.splits(
        path=indir,
        train=train,
        test=test,
        validation=None,
        format='tsv', 
        fields=[
            ("lab", LABS),
            ("val", VALS),
        ])
    
    LABS.build_vocab(datasets[0])
    VALS.build_vocab(datasets[0])
    
    return {
        "iterator" : data.BucketIterator.splits(
            datasets,
            batch_size=1,
            shuffle=True,
            repeat=False,
            device=device,
            sort_key=lambda x: len(x.val),
        ),
        "n_chars" : len(VALS.vocab),
        "n_classes" : len(LABS.vocab),
        "vocabs" : (LABS, VALS)
    }


def load_test_dataset(data_path, VALS, device=0):
    
    data_path = os.path.abspath(data_path)
    
    dataset = data.TabularDataset(
        path=data_path,
        format='tsv',
        fields=[
            ("val", VALS),
        ])
    
    return data.Iterator(
        dataset=dataset,
        batch_size=1,
        device=device,
        shuffle=False,
        sort=False,
        repeat=False,
    )

