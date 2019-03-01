#!/usr/bin/env python

import mnist
import numpy as np
import sys

"""
a simple nn classifier using L1 distance
"""
class nn_classifier:
    def __init__(self, dataset):
        self.images, self.labels = dataset[0], dataset[1];
    
    """
    predict label of the given image
    """
    def predict(self, image):
        min_dist, label = sys.maxint, -1;
        for (timage, tlabel) in zip(self.images, self.labels):
            diff = timage.astype(int) - image.astype(int);
            dist = sum(map(lambda x: x ** 2, diff));
            if dist < min_dist:
                min_dist, label = dist, tlabel;
        return label;
    
    """
    add sample to training set -- for prototype selector
    """
    def addSample(self, image, label):
        self.images.append(image);
        self.labels.append(label);


if __name__ == "__main__":
    train_dataset = mnist.read_dataset("train");
    nc = nn_classifier(train_dataset);
    images, labels = mnist.read_dataset("test");
    print labels[10], nc.predict(images[10]);
