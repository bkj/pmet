'''

    bkj-babi-mnets.py
    
'''

from keras.utils.data_utils import get_file
import tarfile
import numpy as np
import re

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            
            # original
            # substory = [x for x in story if x]
            
            # !! jhoward
            # substory = [[str(i)+":"] + x for i,x in enumerate(story) if x] 
            
            # bkj -- maybe slightly better than jhoward, just in case there's
            # leakage in the placement of null lines
            substory = [x for x in story if x] # tweak
            substory = [[str(i)+":"] + x for i,x in enumerate(substory) if x]
            
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise

tar = tarfile.open(path)

challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
train_stories = parse_stories(tar.extractfile(challenge.format('train')).readlines())
test_stories = parse_stories(tar.extractfile(challenge.format('test')).readlines())
all_stories = train_stories + test_stories

vocab = set([])
for story, q, answer in all_stories:
    vocab = vocab.union(set(reduce(lambda a,b: a + b, story) + q + [answer]))

vocab = sorted(vocab)

word_lookup = dict(zip(vocab, range(1, len(vocab) + 1)))

max_n_sentences = max(map(lambda x: len(x[0]), all_stories))
max_sentence_length = np.hstack(map(lambda x: map(len, x[0]), all_stories)).max()
max_query_length = max(map(lambda x: len(x[1]), all_stories))
vocab_size = len(word_lookup) + 1

# --
# Vectorize data

# Train
X_train = np.zeros((len(train_stories), max_n_sentences, max_sentence_length)).astype('int')
Q_train = np.zeros((len(train_stories), max_query_length)).astype('int')
y_train = np.zeros(len(train_stories)).astype('int')
for i, (s, q, a) in enumerate(train_stories):
    for j, ss in enumerate(s):
        X_train[i,j,:len(ss)] = [word_lookup[w] for w in ss]
    
    Q_train[i,:len(q)] = [word_lookup[w] for w in q]
    y_train[i] = word_lookup[a]

# Test
X_test = np.zeros((len(test_stories), max_n_sentences, max_sentence_length)).astype('int')
Q_test = np.zeros((len(test_stories), max_query_length)).astype('int')
y_test = np.zeros(len(test_stories)).astype('int')
for i, (s, q, a) in enumerate(test_stories):
    for j, ss in enumerate(s):
        X_test[i,j,:len(ss)] = [word_lookup[w] for w in ss]
    
    Q_test[i,:len(q)] = [word_lookup[w] for w in q]
    y_test[i] = word_lookup[a]


# --
# Define model

from keras.models import Model
from keras.layers import *

emb_dim = 20

# Computing attention mask
inp_q = Input(shape=Q_train.shape[1:])
emb_q = Embedding(input_dim=vocab_size, output_dim=emb_dim, name='emb_q')(inp_q)
emb_q = Lambda(lambda x: K.sum(x, axis=1), output_shape=(emb_dim, ))(emb_q)
emb_q = Reshape((1, emb_dim))(emb_q)

inp_x  = Input(shape=X_train.shape[1:])
emb_x1 = TimeDistributed(Embedding(input_dim=vocab_size, output_dim=emb_dim), name='emb_x1')(inp_x)
emb_x1 = Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_n_sentences, emb_dim))(emb_x1)

att_mask = dot([emb_x1, emb_q], axes=2)
att_mask = Reshape((max_n_sentences,))(att_mask)
att_mask = Activation('softmax')(att_mask)
att_mask = Reshape((max_n_sentences, 1))(att_mask)

# Applying attention mask
emb_x2 = TimeDistributed(Embedding(input_dim=vocab_size, output_dim=emb_dim), name='emb_x2')(inp_x)
emb_x2 = Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_n_sentences, emb_dim))(emb_x2)

att_emb = dot([att_mask, emb_x2], axes=1)
att_emb = Reshape((emb_dim, ))(att_emb)
out = Dense(vocab_size, activation='softmax')(att_emb)

model = Model(inputs=[inp_x, inp_q], outputs=out)
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.summary()

K.set_value(model.optimizer.lr, 1e-2)

fitist = model.fit(
    [X_train, Q_train], y_train,
    batch_size=32,
    epochs=4,
    validation_data=([X_test, Q_test], y_test)
)

# --
# So what's going on here?
# !! If we append sequence IDs to the beginning of the sentences, it fits perfectly

ws = sorted(word_lookup.items(), key=lambda x: x[1])

wq = model.get_layer(name='emb_q').get_weights()[0]
wq = np.vstack(wq)
pprint(sorted(zip(ws, (wq[1:] ** 2).sum(axis=1)), key=lambda x: x[1]))

wx1 = model.get_layer(name='emb_x1').get_weights()[0]
wx1 = np.vstack(wx1)
pprint(sorted(zip(ws, (wx1[1:] ** 2).sum(axis=1)), key=lambda x: x[1]))

wx2 = model.get_layer(name='emb_x2').get_weights()[0]
wx2 = np.vstack(wx2)
pprint(sorted(zip(ws, (wx2[1:] ** 2).sum(axis=1)), key=lambda x: x[1]))
