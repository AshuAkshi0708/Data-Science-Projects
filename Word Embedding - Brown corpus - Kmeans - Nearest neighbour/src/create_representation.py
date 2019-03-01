#!/usr/bin/env python

import nltk
from nltk.corpus import brown
import numpy as np
from math import log

from config import *

"""
convert word list file to a map from word to id
"""
def word2map(filename):
    word2idx = {};
    with open(filename) as f:
        for line in f:
            word2idx[line.strip('\n')] = len(word2idx);
    return word2idx;

if __name__ == "__main__":
    # add nltk serach path
    nltk.data.path.append(DATA_HOME);

    # get brown text stream
    print ("getting text stream...")
    brown_text = list(filter(lambda x: x.isalpha(), map(lambda x: x.lower(), brown.words())));
    M = len(brown_text);

    # mapping word to index
    print ("generating word map...")
    V2id = word2map(DATA_HOME + "V.txt");
    C2id = word2map(DATA_HOME + "C.txt");
    print (V2id);
    print (C2id);

    # prepare for the calculation of Pr(c) and Pr(c|w)
    # use ones to apply laplace smoothing
    print ("counting context appearance...");
    window_count = np.ones((V_SIZE, C_SIZE));
    core_count = np.ones((1, C_SIZE));
    for i in range(M):
        w = brown_text[i];
        if w not in V2id:#has_key(w):
            continue;
        wid = V2id[w];
        for j in range(i - HALF_WINDOW, i + HALF_WINDOW + 1):
            if j < 0 or j >= M or j == i:
                continue;
            c = brown_text[j];
            if c not in C2id:
                continue;
            cid = C2id[c];
            window_count[wid][cid] += 1;
            core_count[0][cid] += 1;
    #print (window_count)
    #print (core_count)
    # calculate Pr(c) and Pr(c|w)
    print ("calculating probability...");
    pcw, pc = window_count, core_count;
    for i in range(len(pcw)):
        pcw[i] = pcw[i] / pcw[i].sum();
    pc = pc / pc.sum();

    # calculate pointwise mutual information
    phi = np.zeros((V_SIZE, C_SIZE));
    for i in range(V_SIZE):
        for j in range(C_SIZE):
            phi[i][j] = max(0, log(pcw[i][j] /  pc[0][j]));

    # save representation matrix to file
    print ("saving representation...");
    np.save("representation-" + str(C_SIZE) + ".npy", phi);
