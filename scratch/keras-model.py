from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras import backend as K

import ujson as json
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# --
# Setup

seed = 123
epochs = 10
batch_size = 32

# --
# IO

data = map(json.loads, open('./data.jl'))
X, y = zip(*[d.values() for d in data])

uchars = set(reduce(lambda a,b: a+b, X))
char_lookup = dict(zip(uchars, range(len(uchars))))

ulabels = set(y)
label_lookup = dict(zip(ulabels, range(len(ulabels))))

max_len = max(map(len, X))
X_tmp = np.zeros((len(X), max_len))
for i,x in enumerate(X):
    X_tmp[i][-len(x):] = [char_lookup[xx] for xx in x]

X = X_tmp.astype('int')
y = np.array(pd.get_dummies(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

# --
# Define model

lstm_params = {
    "n_chars" : len(char_lookup),
    "n_classes" : len(label_lookup),
    "emb_dim" : 16,
    "hidden_dim" : 32,
    "max_len" : max_len
}

model = Sequential()
model.add(Embedding(len(char_lookup), lstm_params['hidden_dim'], input_length=lstm_params['max_len'], mask_zero=True))
model.add(LSTM(lstm_params['hidden_dim']))
model.add(Dense(lstm_params['n_classes'], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fitist = model.fit(
    X_train, y_train, 
    epochs=epochs * 2, 
    batch_size=1, 
    verbose=True,
    validation_data=(X_test, y_test)
)

_ = plt.plot(fitist.history['loss'])
_ = plt.plot(fitist.history['val_loss'])
plt.show()


preds = model.predict(X_train).argmax(1)
pd.crosstab(preds, y_train.argmax(1))
