#!/usr/bin/env python

import random
import numpy as np

from sklearn.metrics import log_loss

from config import TRAIN_NUMBER
from wine_io import loadDataset
from softmax import *

if __name__ == "__main__":
    train_samples, train_labels, test_samples, test_labels = loadDataset("shuffle_wine.data", TRAIN_NUMBER);

    theta = np.random.rand(14, 3);
    theta = theta / (10.0 * norm(theta));
    #theta = np.zeros((14, 3));
    #theta = np.load("theta.npy");
    np.save("theta.npy", theta);
    log = open("random.log", "w");

    for i in range(3000):
        loss = entropy_loss(theta, train_samples, train_labels);
        if i % 10 == 0:
            error = np.mean(predict(theta, test_samples) != test_labels);
            l = str(i) + "\t" + str(loss) + "\t" + str(error) + "\n"
            print l,
            log.write(l);

        d = random.randint(0, 41);
        update_coordinate(theta, d, train_samples, train_labels);
    log.close();
    np.save("random-theta.npy", theta);
