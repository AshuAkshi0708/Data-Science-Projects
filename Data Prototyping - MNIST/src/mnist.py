#!/usr/bin/env python

import mlxtend.data

import random
import pickle

MNIST_PATH = "../data/mnist/";

train_images = MNIST_PATH + "train-images-idx3-ubyte";
train_labels = MNIST_PATH + "train-labels-idx1-ubyte";
test_images = MNIST_PATH + "t10k-images-idx3-ubyte";
test_labels = MNIST_PATH + "t10k-labels-idx1-ubyte";

sampled_test_images = MNIST_PATH + "sampled-test-images";
sampled_test_labels = MNIST_PATH + "sampled-test-labels";

"""
read original MNIST database to numpy form
"""
def read_dataset(dataset = "train"):
    if dataset == "train":
        return mlxtend.data.loadlocal_mnist(train_images, train_labels);
    elif dataset == "test":
        return mlxtend.data.loadlocal_mnist(test_images, test_labels);
    else:
        raise ValueError("dataset must be train or test");

"""
read sampled test dataset
"""
def read_sampled_dataset():
    images_file = open(sampled_test_images, "r");
    labels_file = open(sampled_test_labels, "r");

    images = pickle.load(images_file);
    labels = pickle.load(labels_file);
    images_file.close();
    labels_file.close();
    return (images, labels);

if __name__ == "__main__":
    """
    dataset = read_dataset("test");
    images, labels = dataset;
    print len(images), len(images[0]);
    print len(labels);
    """
    dataset = read_dataset("test");
    images, labels = dataset[0], dataset[1];
    simages, slabels = [], [];
    for (image, label) in zip(images, labels):
        if random.random() <= 0.05:
            simages.append(image);
            slabels.append(label);
    images_file = open(sampled_test_images, "w");
    labels_file = open(sampled_test_labels, "w");

    pickle.dump(simages, images_file);
    pickle.dump(slabels, labels_file);
