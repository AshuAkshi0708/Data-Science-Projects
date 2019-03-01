#!/usr/bin/env python

import mnist
import nn_classifier
import dataset_io

import random

HART_PATH = "../data/hart/";

hart_train_images = HART_PATH + "hart-train-images";
hart_train_labels = HART_PATH + "hart-train-labels";

def hart_prototype_selector(dataset, M):
    images, labels = dataset[0], dataset[1];
    nc = nn_classifier.nn_classifier([[], []]);

    c = list(zip(images, labels));
    random.shuffle(c);
    images, labels = zip(*c);

    round, count = 0, 0;
    while True:
        for (image, label) in zip(images, labels):
            if label != nc.predict(image):
                nc.addSample(image, label);
                count = count + 1;
                print round, count;
                if len(nc.images) >= M:
                    break;
        if len(nc.images) >= M or count == 0:
            break;
        round = round + 1;

    return [nc.images, nc.labels];

def test_hart_prototypes(N):
    dataset = dataset_io.load_dataset(hart_train_images, hart_train_labels);
    print len(dataset[0]);
    nc = nn_classifier.nn_classifier(dataset);
    images, labels = mnist.read_sampled_dataset();
    print len(nc.images);

    error = 0;
    N = len(images);
    for i in range(N):
        print i, error;
        guess = nc.predict(images[i]);
        if labels[i] != guess:
            error =  error + 1;
    print float(error) / N * 100, "%";


if __name__ == "__main__":
    """
    dataset = mnist.read_dataset("train");
    hart_prototypes = hart_prototype_selector(dataset, 1000);
    dataset_io.save_dataset(hart_prototypes, hart_train_images, hart_train_labels);
    print len(hart_prototypes[0]);
    """
    test_hart_prototypes(200);
