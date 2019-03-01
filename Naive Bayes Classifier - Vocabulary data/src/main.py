#!/usr/bin/env python
import math
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from scipy import stats

from nb_classifier import *
from vocabulary_io import getVocabulary

TEST_TIMES = 50;
TEST_METHOD = "random";
CONFIDENCE_LEVEL = 0.95;
M = 50000;

twenty_train = fetch_20newsgroups(data_home = DATA_HOME, subset = "train", categories = None, shuffle = True);
twenty_test  = fetch_20newsgroups(data_home = DATA_HOME, subset = "test",  categories = None, shuffle = True);

error = [];
for i in range(TEST_TIMES):
    vocabulary = getVocabulary(TEST_METHOD, M);
    clf = MultinomialNB();
    trainNB(clf, twenty_train, vocabulary, True);
    e = testNB(clf, twenty_test, vocabulary, True);
    error.append(e);

s = np.array(error);
mu, sigma = np.mean(s), np.std(s, ddof = 1);
i =  stats.norm.interval(CONFIDENCE_LEVEL, loc = mu, scale = sigma);

print "m = ", mu;
print "A = ", i[1] - np.mean(s);
print "confidence interval = ", i;
