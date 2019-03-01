#!/usr/bin/env python

import random
import numpy as np

from config import DATA_HOME

"""
preprocess wine dataset
    @source: original dataset file
    @target: processed dataset file
    @shuffle: whether to shuffle original dataset
    @augment: whether to augment original dataset
"""
def preprocessDataset(source, target, shuffle = True, augment = True):
    with open(DATA_HOME + source, "r") as f:
        data = [line.strip('\n') for line in f];
    if shuffle:
        random.shuffle(data);
    if augment:
        for i in range(len(data)):
            data[i] = data[i] + ",1";
    with open(DATA_HOME + target, "w") as f:
        for line in data:
            f.write(line + "\n");

"""
load dataset
input:
    @source: dataset file
    @train_number: number of samples in training set
output:
    @train_samples, train_labels:
        the first (train_number) samples and their labels
    @test_samples, test_labels:
        rest samples and their labels
"""
def loadDataset(source, train_number):
    with open(DATA_HOME + source, "r") as f:
        data = [line.strip('\n').split(',') for line in f];
    train_samples = np.matrix([map(float, data[i][1:]) for i in range(0, train_number)]);
    train_labels = np.array(map(int, [data[i][0] for i in range(0, train_number)]));
    test_samples = np.matrix([map(float, data[i][1:]) for i in range(train_number, len(data))]);
    test_labels = np.array(map(int, [data[i][0] for i in range(train_number, len(data))]));
    return train_samples, train_labels, test_samples, test_labels;

if __name__ == "__main__":
    preprocessDataset("wine.data", "shuffle_wine.data");
    train_samples, train_labels, test_samples, test_labels = loadDataset("shuffle_wine.data", 128);
    print len(train_samples);
    print len(test_samples);
