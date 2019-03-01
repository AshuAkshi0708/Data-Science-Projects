#!/usr/bin/env python

import mnist
import nn_classifier
import dataset_io

import random

RANDOM_PATH = "../data/random/";

random_train_images = RANDOM_PATH + "random-train-images";
random_train_labels = RANDOM_PATH + "random-train-labels";

def random_prototype_selector(dataset, M):
    images, labels = dataset[0], dataset[1];
    pimages, plabels = [], [];
    N = len(images);
    p = float(M) / N;   # probability to keep a sample in subset
    for (image, label) in zip(images, labels):
        if random.random() <= p:
            pimages.append(image);
            plabels.append(label);
        if len(pimages) == M:
            break;
    return [pimages, plabels];

def test_random_prototypes(N):
    dataset = dataset_io.load_dataset(random_train_images, random_train_labels);
    nc = nn_classifier.nn_classifier(dataset);
    images, labels = mnist.read_sampled_dataset();
    error = 0;
    N = len(images);;
    for i in range(N):
        if labels[i] != nc.predict(images[i]):
            error = error + 1;
        print i, error;
    print float(error) / N * 100, "%";


if __name__ == "__main__":
   dataset = mnist.read_dataset("train");
   random_prototypes = random_prototype_selector(dataset, 1000);
   dataset_io.save_dataset(random_prototypes, random_train_images, random_train_labels);
   test_random_prototypes(200);
