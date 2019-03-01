#!/usr/bin/env python

import numpy as np
from sklearn import linear_model
from sklearn.metrics import log_loss
from numpy.linalg import norm

from softmax import *
from config import TRAIN_NUMBER
from wine_io import loadDataset

if __name__ == "__main__":
    MAX_ITER = 5000;
    C = 0.5;
    train_samples, train_labels, test_samples, test_labels = loadDataset("shuffle_wine.data", TRAIN_NUMBER);
    logreg = linear_model.LogisticRegression(C = C, max_iter = MAX_ITER, multi_class = "multinomial", solver = 'newton-cg');
    logreg.fit(train_samples, train_labels);

    # cross entropy loss on training set
    print C * log_loss(train_labels, logreg.predict_proba(train_samples), normalize = False) + 0.5 * norm(logreg.coef_) ** 2;
    # error rate on testing set
    print "error rate = ", np.mean(logreg.predict(test_samples) != test_labels);
