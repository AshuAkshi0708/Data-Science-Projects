#!/usr/bin/env python

import nltk
nltk.download('stopwords')
nltk.download('brown')
from nltk.corpus import brown, stopwords
import collections

from config import *

"""
create most commonly-occuring word lists V and C
"""
if __name__ == "__main__":
    # add nltk serach path
    nltk.data.path.append(DATA_HOME);
    english_stopwords = stopwords.words('english');

    # remove stopwords and punctuation, making everything lowercase
    print ("precessing brwon...");
    brown_words = filter(lambda x: x.isalpha() and x not in english_stopwords and len(x) > 1,
        map(lambda x: x.lower(), brown.words())
    );

    # generating most commonly-occuring words
    print ("generating V and C...")
    brown_count = collections.Counter(brown_words);
    print (brown_count);
    V = brown_count.most_common(V_SIZE);
    C = brown_count.most_common(C_SIZE);

    # saving two word lists
    print ("saving to file...");
    with open(DATA_HOME + "V.txt", "w+") as f:
        for word in V:
            f.write(word[0] + "\n");
    with open(DATA_HOME + "C.txt", "w+") as f:
        for word in C:
            f.write(word[0] + "\n");
