#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import LSTM, RepeatVector, Dense, Input, Flatten, Reshape, Permute, Lambda
from keras.layers.merge import multiply, concatenate
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

Tx = 20 #Size of the input
Ty = 40 #Size of the output
n_a = 64 #Number of layer in the encoder
n_s = 128 #Number of layer in the decoder
embeddings_weights_shape = 300


def ecoder_decoder_model(ref_words_size):
    """

    """
    
    X = Input(shape=(Tx, embeddings_weights_shape))
    
        
    #Pre-attention LSTM (encoder)
    encoder = Bidirectional(LSTM(n_a, dropout=0.2, return_sequences=True))(X)
    
    #Attention model
    flat_layer = Flatten()(encoder)
    attention_outputs = []
    
    for t in range(Ty):
        A = Dense(Tx, activation='softmax')(flat_layer)
        B = Permute([2, 1])(RepeatVector(n_a * 2)(A))
        C = multiply([encoder, B])
        D = Lambda(lambda x: K.sum(x, axis=-2))(C)
        attention_outputs.append(Reshape((1, n_a * 2))(D))
    
    attention_out = concatenate(attention_outputs, axis=-2)
    
    
    # Post-attention LSTM (decoder)
    decoder = LSTM(n_s, return_sequences = True)(attention_out)
        
    #Dense layer
    decoder = Dense(ref_words_size, activation='softmax')(decoder)
            
    model = Model(inputs = X, outputs = decoder)
        
    return model

def run_model(X_vectors, y_vectors, y_id2word, model_path, print_model):
    
    ref_words_size = y_vectors.shape[2]
    
    #Create the model
    model = ecoder_decoder_model(ref_words_size)
    
    if(print_model):
        print(model.summary())

    #Compile the model then fit
    model.compile(optimizer=Adam(lr=0.01), metrics=['accuracy'], loss='categorical_crossentropy')
    model.fit(X_vectors, y_vectors, epochs=1, batch_size=64)
    
    model_json = model.to_json()
    with open("ressources/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("ressources/model.h5")
    #model.save_weights(model_path)
    
    
    
    
    


def predict_model(X_vectors):
    
    #Loading the model
    print('Loading model...')
    json_file = open('ressources/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    
    model.load_weights("ressources/model.h5")
    model.compile(optimizer=Adam(lr=0.01), metrics=['accuracy'], loss='categorical_crossentropy')
    
    print('Predicting...')
    y_predict = model.predict(X_vectors)
    
    return y_predict
    
    
    





    

    

