
"""
    data.py
"""

from torchtext import data

def make_iter(indir, train='train.tsv', test='test.tsv', validation=None, batch_size=32):
    
    LABS = data.Field()
    VALS = data.Field(tokenize=list)
    
    datasets = data.TabularDataset.splits(path=indir, 
        train=train,
        test=test,
        validation=validation,
        format='tsv', 
        fields=[
            ("lab", LABS),
            ("val", VALS),
        ])
    
    LABS.build_vocab(datasets[0])
    VALS.build_vocab(datasets[0])
    
    return data.BucketIterator.splits(
        datasets,
        batch_size=1,
        shuffle=True,
        repeat=False,
    ), (len(VALS.vocab), len(LABS.vocab))

