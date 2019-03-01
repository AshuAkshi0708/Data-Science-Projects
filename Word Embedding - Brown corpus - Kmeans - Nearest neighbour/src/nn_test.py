#!/usr/bin/env python

import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_similarity

from config import *

if __name__ == "__main__":
    word2id, id2word = {}, {};
    with open(DATA_HOME + "V.txt") as f:
        count = 0;
        for line in f:
            word = line.strip('\n');
            word2id[word], id2word[count] = count, word;
            count += 1;

    # use cosine distance
    tr = 1 - cosine_similarity(np.load("representation-" + str(U_SIZE) + ".npy"));

    neigh = KNeighborsClassifier(n_neighbors = 3, metric = "precomputed");
    neigh.fit(tr, np.zeros((V_SIZE)));

    # find NN for 25 random words
    rand_inds = random.sample([i for i in range(V_SIZE)], 25);
    for i in rand_inds:
        w = id2word[i];
        dist, ind = neigh.kneighbors(tr[i].reshape(1, -1));
        uid = ind[0][1];
        u = id2word[uid];
        print ("%-15s %-15s %f" %(w, u, dist[0][1]));
