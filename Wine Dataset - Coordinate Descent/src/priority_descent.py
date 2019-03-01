#!/usr/bin/env python

import random
import numpy as np

from numpy.linalg import norm
from sklearn.metrics import log_loss

from config import TRAIN_NUMBER
from wine_io import loadDataset
from softmax import *

if __name__ == "__main__":
    train_samples, train_labels, test_samples, test_labels = loadDataset("shuffle_wine.data", TRAIN_NUMBER);

    theta = np.load("theta.npy");
    log = open("prior-log", "w");

    for i in range(270):
        loss = entropy_loss(theta, train_samples, train_labels);
        if i % 5 == 0:
            error = np.mean(predict(theta, test_samples) != test_labels);
            l = str(i) + "\t" + str(loss) + "\t" + str(error) + "\n"
            print l,
            log.write(l);

        max_gap, d = 0.0, -1;
        for j in range(42):
            x, y = j % 14, d / 14;
            origin_val = theta[x][y];
            update_coordinate(theta, j, train_samples, train_labels);
            curr_gap = loss - entropy_loss(theta, train_samples, train_labels);
            if curr_gap > max_gap:
                max_gap, d = curr_gap, j;
        update_coordinate(theta, d, train_samples, train_labels)
    log.close();
    np.save("prior-theta.npy", theta);
