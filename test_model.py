#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from processing import *
from utils import *
from nn_model import *


def run(test_path, result_path):
    print("Preprocessing data ...")
    X_vectors = getProcessedDataTest(test_path)
    y_predict = predict_model(X_vectors)
    y_id2word = getId2Word()
    
    refs = reconstructRef(y_predict, y_id2word)
    
    writeTofile(refs, result_path)
    
   
if __name__ == "__main__":
   test_path = sys.argv[2]
   result_path = sys.argv[4]
   print("We are in main")
   run(test_path, result_path)

