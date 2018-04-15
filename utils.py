#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import pickle

PICKLE_FILE = 'ressources/ID2WORD.pickle'

def reconstructRef(ref_vectors, y_id2word):
    refs = []
    for i in range(ref_vectors.shape[0]):
        ref = []
        vect = np.argmax(ref_vectors[i,:,:], axis=1)
        for j in range(vect.shape[0]):
            word = y_id2word[vect[j]]
            if(word == 'END' or word =='PAD'):
                break
            ref.append(word)
        ref = ' '.join(ref)
        ref += '.'
        refs.append(ref)
    
    return refs


def writeTofile(refs, filename):
    with open(filename, 'a') as f:
        for ref in refs:
            f.write(ref)
            f.write('\n')
            
def readFromFile(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    return lines



def bleu_score(real_refs, predicted_ref):
    score = 0
    for i in range(len(real_refs)):
        score += sentence_bleu([real_refs[i]], predicted_ref[i])
    score = score/len(real_refs)
    return score

def bleu_score_from_file(file_real, file_predicted):
    real = readFromFile(file_real)
    predicted = readFromFile(file_predicted)
    return bleu_score(real, predicted)



def saveToPickle(y_id2word):
    with open(PICKLE_FILE, 'wb') as handle:
        pickle.dump(y_id2word, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def getId2Word():
    with open(PICKLE_FILE, 'rb') as handle:
        y_id2word = pickle.load(handle)
        
    return y_id2word
    
    
    




