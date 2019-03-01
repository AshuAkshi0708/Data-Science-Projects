#!/usr/bin/env python

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

DATA_HOME = "../data/";

"""
train naive bayes classifier with training set and given vocabulary list
    @clf: multinomial NB classifier to be trained
    @log_tf: use log term frequency
"""
def trainNB(clf, train_set, vocabulary, log_tf = True):
    count_vect = CountVectorizer(vocabulary = vocabulary);
    X_train_counts = count_vect.fit_transform(train_set.data);
    if log_tf:
        X_train_counts = X_train_counts.log1p();
    clf.fit(X_train_counts, train_set.target);

"""
test naive bayes classifier with testing set and given vocabulary list
    @clf: multinomial NB classifier to be tested
    @log_tf: use log term frequency
"""
def testNB(clf, test_set, vocabulary, log_tf = True):
    count_vect = CountVectorizer(vocabulary = vocabulary);
    X_test_counts = count_vect.transform(test_set.data);
    if log_tf:
        X_test_counts = X_test_counts.log1p();
    predicted = clf.predict(X_test_counts);

    error = np.mean(predicted != test_set.target);
    #report = metrics.classification_report(test_set.target, predicted, target_names = test_set.target_names);

    print "error rate = ", error * 100, "%";
    return error;

if __name__ == "__main__":
    #categories = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"];

    # load training and testing dataset as twenty_train and twenty_test
    twenty_train = fetch_20newsgroups(data_home = DATA_HOME, subset = "train", categories = None, shuffle = True);
    twenty_test  = fetch_20newsgroups(data_home = DATA_HOME, subset = "test",  categories = None, shuffle = True);

    import vocabulary_io
    vocabulary = vocabulary_io.getVocabulary("chosen", 5000);

    clf = MultinomialNB();
    trainNB(clf, twenty_train, vocabulary, True);
    testNB(clf, twenty_test, vocabulary, True);
