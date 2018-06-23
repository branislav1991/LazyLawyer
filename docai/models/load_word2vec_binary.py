from __future__ import division

import os
import numpy as np
import struct
import sys

def load_word2vec_binary(path, max_vectors=200000):
    word_vecs = {} 
    with open(path, "rb") as f: 
        header = f.readline() 
        vocab_size, layer1_size = map(int, header.split()) 
        binary_len = np.dtype('float32').itemsize * layer1_size 

        vocab_size = min(vocab_size, max_vectors)
        for line in range(vocab_size): 
            word = [] 
            while True: 
                ch = f.read(1).decode(errors='ignore')
                if ch == ' ': 
                    word = ''.join(word) 
                    break 
                if ch != '\n': 
                    word.append(ch)   
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')   
    return word_vecs

if __name__ == '__main__':
    path = os.path.join('trained_models', 'GoogleNews-vectors-negative300.bin') 
    load_word2vec_binary(path)