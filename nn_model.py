#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda
from keras.layers.merge import multiply, concatenate, Concatenate, Dot
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from processing import getProcessedData


embeddings, X_vectors, y_vectors = getProcessedData('data/devset.csv')

embeddings_weights = embeddings.syn0
Tx = X_vectors.shape[1]
Ty = y_vectors.shape[1]
ref_words_size = y_vectors.shape[2]
m = X_vectors.shape[0]



def model(n_a, n_s):
    """
    n_a : hidden state size of the Pre-LSTM
    n_s : hidden state size of the post-attention LSTM
    embeddings_weights: embeddings weights
    """
    
    X = Input(shape=(Tx, embeddings_weights.shape[1]))
    
        
    #Pre-attention LSTM (encoder)
    encoder = Bidirectional(LSTM(n_a, dropout=0.2, return_sequences=True))(X)
    
    #Attention model
    flattened = Flatten()(encoder)
    attention_outputs = []
    
    for t in range(Ty):
        weighted = Dense(Tx, activation='softmax')(flattened)
        unfolded = Permute([2, 1])(RepeatVector(n_a * 2)(weighted))
        multiplied = multiply([encoder, unfolded])
        summed = Lambda(lambda x: K.sum(x, axis=-2))(multiplied)
        attention_outputs.append(Reshape((1, n_a * 2))(summed))
    
    attention_out = concatenate(attention_outputs, axis=-2)
    
    
    # Post-attention LSTM (decoder)
    decoder = LSTM(n_s, return_sequences = True)(attention_out)
        
    #Dense layer
    decoder = Dense(ref_words_size, activation='softmax')(decoder)
            
    model = Model(inputs = X, outputs = decoder)
        
    return model


n_a = 64
n_s = 128
model = model(n_a, n_s)

model.summary()


model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

outputs = list(y_vectors.swapaxes(0,1))

model.fit(X_vectors, y_vectors, epochs=4, batch_size=64)

X_sample = X_vectors[:10,:,:]

y_sample = model.predict(X_sample)


t = np.argmax(y_sample, axis=2)

t1 = y_sample[0,:,:]










