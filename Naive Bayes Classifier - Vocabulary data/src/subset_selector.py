#!/usr/bin/env python

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy

import vocabulary_io

DATA_HOME = "../data/";

def calculateWordInformation(vocabulary, categories):
    # generate file word count vector for each category
    counts_map = {};
    for category in categories:
        twenty_train = fetch_20newsgroups(data_home = DATA_HOME, subset = "train", categories = [category], shuffle = True);
        count_vect = CountVectorizer(vocabulary = vocabulary);
        X_train_counts = count_vect.transform(twenty_train.data);
        counts_map[category] = X_train_counts;

    # calculate conditional entropy of each word
    # +1 is for laplace smoothing
    word_entropy = [];
    for i in range(len(vocabulary)):
        print i;
        p, q = [], [];
        for (category, counts) in counts_map.items():
            p.append(counts.getcol(i).count_nonzero() + 1);
            q.append(counts.get_shape()[0] - counts.getcol(i).count_nonzero() + 1);
        sp, sq = sum(p) + 1, sum(q) + 1;
        wp, wq = float(sp) / (sp + sq), float(sq) / (sp + sq);
        word_entropy.append((vocabulary[i], 1 * entropy(p) + 0 * entropy(q)));

    # let front words have smaller conditional entropy, therefore provide more information
    return sorted(word_entropy, key = lambda x: x[1]);

if __name__ == "__main__":
    part_categories = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"];

    full_categories = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
                  "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos",
                  "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt",
                  "sci.electronics", "sci.med", "sci.space", "soc.religion.christian",
                  "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"];

    vocabulary = vocabulary_io.getVocabulary("full", None);
    word_entropy = calculateWordInformation(vocabulary, full_categories);
    chosen_vocabulary = map(lambda x: x[0], word_entropy);
    vocabulary_io.saveVocabulary(chosen_vocabulary);
