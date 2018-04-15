#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from processing import *
from utils import *
from nn_model import *


def run(train_path, model_path):
    print("Preprocessing data ...")
    X_vectors, y_vectors, y_id2word = getProcessedDataTrain(train_path)
    print("Running model...")
    run_model(X_vectors, y_vectors, y_id2word, model_path, False)
    
   
if __name__ == "__main__":
   train_path = sys.argv[2]
   model_path = sys.argv[4]
   print("We are in main")
   run(train_path, model_path)