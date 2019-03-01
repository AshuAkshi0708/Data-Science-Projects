#!/usr/bin/env python
import random

VOCABULARY_HOME = "../data/";

"""
get vocabulary list
    @type:
        full: original vocabulary
        random: randomly pick M words from original vocabulary
        chosen: pick top M from chosen subset
"""
def getVocabulary(type = "full", M = None):
    if type == "full":
        vocabulary = file2list(VOCABULARY_HOME + "vocabulary.txt");
        return vocabulary;
    elif type == "random":
        vocabulary = file2list(VOCABULARY_HOME + "vocabulary.txt");
        return random.sample(vocabulary, min(M, len(vocabulary)));
    elif type == "chosen":
        vocabulary = file2list(VOCABULARY_HOME + "chosen_vocabulary.txt");
        return vocabulary[0: min(M, len(vocabulary))];
    else:
        print "wrong parameter for type(full, random, chosen)";
        return None;

"""
save vocabulary list (always to chosen_vocabulary.txt)
"""
def saveVocabulary(vocabulary):
    list2file(VOCABULARY_HOME + "chosen_vocabulary.txt", vocabulary);

"""
read vocabulary file to list
"""
def file2list(filename):
    f = open(filename, "r");
    l = [line.strip('\n') for line in f];
    f.close();
    return l;

"""
save vocabulary list to file
"""
def list2file(filename, l):
    f = open(filename, "w");
    for w in l:
        f.write(w + "\n");
    f.close();
