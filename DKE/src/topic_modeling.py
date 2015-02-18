from __future__ import print_function

__author__ = 'moiseeva'

import numpy as np
from gensim import corpora, models, similarities
from utils import print_dict
from sklearn.lda import LDA
import cPickle as pickle
range = xrange


with open('../data/transforms', 'rb') as f:
    transforms = pickle.load(f)
with open('../data/vocabulary', 'rb') as f:
    vocabulary = pickle.load(f)
    vocabulary = {v: k for k, v in vocabulary.items()}

X = transforms.toarray()

tfidf = models.TfidfModel(X)
corpus_tfidf = tfidf[transforms]

n_topics = 6
lda = models.LdaModel(corpus_tfidf, id2word=vocabulary, num_topics=n_topics)

for i in range(0, n_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[1])
    print("Top 10 terms for topic #" + str(i) + ": "+ ", ".join(terms))

