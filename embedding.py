#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import smart_open
import gensim

"Source: https://github.com/jroakes/glove-to-word2vec/blob/master/convert.py"

def prepend_slow(infile, outfile, line):
	"""
	Slower way to prepend the line by re-creating the inputfile.
	"""
	with open(infile, 'r') as fin:
		with open(outfile, 'w') as fout:
			fout.write(line + "\n")
			for line in fin:
				fout.write(line)


def get_lines(glove_file_name):
    """Return the number of vectors in a file in GloVe format."""
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_lines = sum(1 for line in f)
    return num_lines
	


def getWord2Vec():
    glove_file="ressources/glove.840B.300d.txt"

    #num_lines = get_lines(glove_file)

    gensim_file='ressources/glove_model2.txt'
    #gensim_first_line = "{} {}".format(num_lines,300)
    
    #prepend_slow(glove_file, gensim_file, gensim_first_line)
    file = 'ressources/GoogleNews-vectors-negative300.bin'
    return gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
