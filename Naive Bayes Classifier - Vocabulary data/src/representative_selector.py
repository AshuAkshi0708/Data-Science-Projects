#!/usr/bin/env python

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import vocabulary_io

DATA_HOME = "../data/";

def findRepresentative(vocabulary, categories, N = 20):
    # calculate times a word appears in all categories
    twenty_train = fetch_20newsgroups(data_home = DATA_HOME, subset = "train", categories = None, shuffle = True);
    count_vect = CountVectorizer(vocabulary = vocabulary);
    total_train_counts = count_vect.transform(twenty_train.data);

    # generate file word count vector for each category
    representative = {};
    for category in categories:
        twenty_train = fetch_20newsgroups(data_home = DATA_HOME, subset = "train", categories = [category], shuffle = True);
        count_vect = CountVectorizer(vocabulary = vocabulary);
        X_train_counts = count_vect.transform(twenty_train.data);

        temp = [];
        for i in range(len(vocabulary)):
            pc = float(X_train_counts.getcol(i).count_nonzero()) / float(total_train_counts.getcol(i).count_nonzero());
            temp.append((vocabulary[i], pc));
        temp = sorted(temp, key = lambda x: x[1], reverse = True);
        representative[category] = temp[0:min(N, len(temp))];

    return representative;

if __name__ == "__main__":
    part_categories = ["alt.atheism", "comp.graphics"];

    full_categories = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
                  "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos",
                  "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
                  "sci.electronics", "sci.med", "sci.space", "soc.religion.christian",
                  "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"];

    vocabulary = vocabulary_io.getVocabulary("chosen", 1000);
    representative = findRepresentative(vocabulary, full_categories, 10);
    f = open("r.txt", "w")
    for (k, v) in representative.items():
        f.write(k + "\n");
        for e in v:
            f.write(e[0] + ", ")
        f.write("***********\n\n")
