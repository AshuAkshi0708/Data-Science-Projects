#!/usr/bin/env python

import random
import numpy as np
from numpy.linalg import norm

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from queue import PriorityQueue

from config import *

if __name__ == "__main__":
    word2id, id2word = {}, {};
    with open(DATA_HOME + "V.txt") as f:
        count = 0;
        for line in f:
            word = line.strip('\n');
            word2id[word], id2word[count] = count, word;
            count += 1;

    # cluster words
    X = np.load("representation-" + str(U_SIZE) + ".npy");
    kmeans = KMeans(n_clusters = CLUSTER_NUMBER).fit(X);
    print (kmeans)
    # sort words by distance to center
    word_groups = {i:PriorityQueue() for i in range(100)};
    print(word_groups)
    for i in range(len(X)):
        representation = X[i];
        word = id2word[i];
        center_id = kmeans.predict(X[i].reshape(1, -1))[0];
        dist = norm(representation - kmeans.cluster_centers_[center_id]);
        word_groups[center_id].put((float(dist), word));

    # print only relatively large groups
    for i in range(CLUSTER_NUMBER):
        if word_groups[i].qsize() < CLUSTER_THRESHOLD:
            continue;
        count = 0;
        for j in range(CLUSTER_THRESHOLD):
            if word_groups[i].empty():
                break;
            print (word_groups[i].get()[1])
            count += 1;
            if count % 10 == 0:
                print;
        print ("\n******************************");
