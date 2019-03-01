#!/usr/bin/env python

import nn_classifier
import pickle

"""
save training set to files
"""
def save_dataset(dataset, images_path, labels_path):
    images, labels = dataset[0], dataset[1];
    images_file = open(images_path, "w");
    labels_file = open(labels_path, "w");
    pickle.dump(images, images_file);
    pickle.dump(labels, labels_file);

"""
load training set from files provided
"""
def load_dataset(images_path, labels_path):
    images_file = open(images_path, "r");
    labels_file = open(labels_path, "r");
    images = pickle.load(images_file);
    labels = pickle.load(labels_file);
    return images, labels;
