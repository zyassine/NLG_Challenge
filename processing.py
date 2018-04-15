#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

import gensim
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter
from utils import saveToPickle


VOCAB_SIZE = 1000
REF_MAX_LEN = 40
MR_MAX_LEN = 20 
END = 'END'
NOTIN = 'NOTIN'
PAD = 'PAD'

def getIdDict(mr):
    mr = mr.replace("\"", "")
    mr = mr.split(",")
    
    values= [re.search(r'\[(.*)\]', e).group(1) for e in mr]
    keys = [re.search(r'(.*)\[.*\]', e).group(1) for e in mr]
    
    mr_dict = {}
    for i in range(0, len(mr)):
        mr_dict[keys[i]] = values[i]
    
    return mr_dict


def readData(filename):
    df = pd.read_csv(filename)
    X = []
    y = []
    for i in range(len(df)):
        mr = df.iloc[i,0]
        ref = df.iloc[i,1]
        mr_dict = getIdDict(mr)
        X.append(mr_dict)
        y.append(ref)
    return X, y


def processRef(refs):
    new_refs = []
    for ref in refs:
        ref = text_to_word_sequence(ref, filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~.')
        new_refs.append(ref)
    all_words = []
    for ref in new_refs:
        for word in ref:
            all_words.append(word)
    cnt = Counter(all_words)
    n = len(cnt)
    common_words = cnt.most_common(min(n,VOCAB_SIZE))
    y_id2word = {}
    y_word2id = {}
    i = 0
    for word in common_words:
        y_id2word[i] = word[0]
        y_word2id[word[0]] = i
        i += 1
        
    y_id2word[i] = END
    y_word2id[END] = i 
    i += 1
    y_id2word[i] = NOTIN
    y_word2id[NOTIN] = i
    i += 1
    y_id2word[i] = PAD
    y_word2id[PAD] = i
        
    return new_refs, y_id2word, y_word2id
    
def getWord2Vec():
    file = 'ressources/GoogleNews-vectors-negative300.bin'
    return gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)


            
def transformRefToVectors(y, y_id2word, y_word2id):
    y_vectors = np.zeros((len(y), REF_MAX_LEN, len(y_id2word)))
    for i, ref in enumerate(y):
        for j, word in enumerate(ref):
            if(j >= REF_MAX_LEN-1):
                break
            if word in y_word2id:
                y_vectors[i][j][y_word2id[word]] = 1
            else:
                y_vectors[i][j][y_word2id[NOTIN]] = 1
                
        if(j<REF_MAX_LEN-1):
            j += 1
            y_vectors[i][j][y_word2id[END]] = 1
        
        for j in range(len(ref)+1,REF_MAX_LEN):
            y_vectors[i][j][y_word2id[PAD]] = 1
    
    return y_vectors

def transformMrToVectors(mrs, embeddings):
    X_vectors = []
    for mr in mrs:
        vect = []
        for key in mr.keys():
            keys = key.split()
            for k in keys:
                if(k in embeddings.vocab):
                    vect.append(embeddings[k])
            values = mr[key].split()
            for v in values:
                if(v in embeddings.vocab):
                    vect.append(embeddings[v])
        
        if(len(vect) > MR_MAX_LEN):
            vect = vect[:MR_MAX_LEN]  
        else:
            for i in range(len(vect), MR_MAX_LEN):
                vect.append(np.zeros((300,)))
            
        X_vectors.append(vect)
        
    return X_vectors
        
            
def getProcessedDataTrain(file):
    X, y = readData(file)


    y, y_id2word, y_word2id = processRef(y)

    y_vectors = transformRefToVectors(y, y_id2word, y_word2id)

    embeddings = getWord2Vec()

    X_vectors = transformMrToVectors(X, embeddings)
    
    X_vectors = np.array(X_vectors)
    y_vectors = np.array(y_vectors)
    
    saveToPickle(y_id2word)
    
    return X_vectors, y_vectors, y_id2word


def getProcessedDataTest(test_path):
    df = pd.read_csv(test_path)
    X = []
    for i in range(len(df)):
        mr = df.iloc[i,0]
        mr_dict = getIdDict(mr)
        X.append(mr_dict)
    
    embeddings = getWord2Vec()
    X_vectors = transformMrToVectors(X, embeddings)
    X_vectors = np.array(X_vectors)
    
    return X_vectors


    
    


                                 

