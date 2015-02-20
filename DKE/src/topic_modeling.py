from __future__ import print_function
import os
import sys
import numpy as np
import cPickle as pickle
from utils import open_write
from sklearn import decomposition
from zlabelLDA import zlabelLDA
range = xrange

__author__ = 'annie'


def topic_model_on_nmf(dtm, vocab, num_topics=5, num_top_words=10, num_top_topics=3, file_out=sys.stdout):
    """
    See https://de.dariah.eu/tatom/topic_model_python.html
    :param dtm:
    :param vocab:
    :param num_topics:
    :param num_top_words:
    :param num_top_topics:
    :param file_out:
    :return: Phi - P(w|z), Theta - P(z|d)
    """
    clf = decomposition.NMF(n_components=num_topics, random_state=1)

    doctopic = clf.fit_transform(dtm)
    topic_words = []
    for topic in clf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append([vocab[i] for i in word_idx])

    doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    novel_names = []

    for fn in filenames:
        basename = os.path.basename(fn)
        name, ext = os.path.splitext(basename)
        novel_names.append(name)

    novel_names = np.asarray(novel_names)
    num_groups = len(set(novel_names))
    doctopic_grouped = np.zeros((num_groups, num_topics))
    for i, name in enumerate(sorted(set(novel_names))):
        doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

    doctopic = doctopic_grouped

    print('doctopic...\n', doctopic, file=file_out)

    novels = sorted(set(novel_names))
    print("\nTop NMF topics in...", file=file_out)
    for i in range(len(doctopic)):
        top_topics = np.argsort(doctopic[i,:])[::-1][0:num_top_topics]
        top_topics_str = ' '.join(str(t) for t in top_topics)
        print("{}: {}".format(novels[i], top_topics_str), file=file_out)

    print("\nshow the top 10 words...", file=file_out)
    for t in range(len(topic_words)):
        print(u"Topic {}: {}".format(t, u' '.join(topic_words[t][:num_top_words])), file=file_out)

    print("\nshow phi...", np.array_str(clf.components_, precision=2), file=file_out)

    return clf.components_, doctopic


def topic_model_on_zlda(docs, vocab, num_topics=5, zlabels=None, eta=0.95, file_out=None):
    """
    See http://pages.cs.wisc.edu/~andrzeje/research/zl_lda.html
    :param docs:
    :param vocab:
    :param num_topics:
    :param zlabels:
    :param eta: confidence in the our labels. If eta = 0 --> don't use z-labels, if eta = 1 --> "hard" z-labels.
    :param file_out:
    :return: Phi - P(w|z), Theta - P(z|d)
    """
    alpha = .1 * np.ones((1, num_topics))
    beta = .1 * np.ones((num_topics, len(vocab)))
    numsamp = 100
    randseed = 194582

    if not zlabels:
        zlabels = [[0]*len(text) for text in docs]

    phi, theta, sample = zlabelLDA(docs, zlabels, eta, alpha, beta, numsamp, randseed)
    if file_out:
        print('\nTheta - P(z|d)\n', np.array_str(theta, precision=2), file=file_out)
        print('\n\nPhi - P(w|z)\n', np.array_str(phi,precision=2), file=file_out)
        print('\n\nsample', file=file_out)
        for doc in range(len(docs)):
            print(sample[doc], file=file_out)

    return phi, theta


def save_phi_theta(file, phi, theta):
    with open(file, 'wb') as f:
        pickle.dump((phi, theta), f)


if __name__ == '__main__':
    with open('../data/transforms', 'rb') as f:
        dtm = pickle.load(f)
    with open('../data/docs', 'rb') as f:
        docs = pickle.load(f)
    with open('../data/vocabulary', 'rb') as f:
        vocab = pickle.load(f)
        vocab = {v: k for k, v in vocab.items()}

    path_to_dir = '/Users/annie/SELabs/practice/txt'
    filenames = sorted(os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir) if file[-4:] == '.txt')

    num_topics = 5
    num_top_words = 10
    num_top_topics = 3
    zlabels = None
    eta = 0.95

    # with open_write('../data/out.txt') as file_out:
    #     topic_model_on_nmf(dtm, vocab, num_topics, num_top_words, num_top_topics, file_out)
    with open_write('../data/out1.txt') as file_out:
        theta, phi = topic_model_on_zlda(docs, vocab, num_topics, zlabels, eta, file_out)


