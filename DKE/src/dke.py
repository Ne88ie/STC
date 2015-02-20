from __future__ import division, print_function
import os
import numpy as np
import cPickle as pickle
from topic_modeling import topic_model_on_zlda
from utils import open_write
range = xrange

__author__ = 'annie'


class Text:
    def __init__(self, text_number, text, dke):
        self.text = set(text)
        self.text_number = text_number
        self.dke = dke
        self.keywords = []
        self.rewards = []
        self.rewards_buffer = []
        self.r_s_z = np.zeros(dke.num_topics)
        self.r_s_z_buffer = []

    def get_r_w_z(self, topic, word):
        r_w_z = self.get_p_z_w(topic, word) * self.dke.theta[self.text_number, topic]
        return r_w_z

    def get_p_z_w(self, topic, word):
        p_w_z = self.dke.phi[topic, word]
        p_w = sum(self.dke.phi[:, word])
        return p_w_z/p_w

    def reward_function(self, word):
        rewards = 0
        r_s_z_buffer = self.r_s_z
        for topic in range(self.dke.num_topics):
            r_w_z = self.get_r_w_z(topic, word)
            r_s_z_buffer[topic] += r_w_z
            rewards += self.dke.theta[self.text_number, topic] * r_s_z_buffer[topic] ** self.dke.lambda_
        self.rewards_buffer.append(rewards)
        self.r_s_z_buffer.append(r_s_z_buffer)
        return rewards

    def keywords_extract(self):
        while len(self.keywords) < self.dke.number_of_keywords:
            words = list(self.text - set(self.keywords))
            word = max(words, key=self.reward_function)
            self.keywords.append(word)
            ind = words.index(word)
            self.rewards.append(self.rewards_buffer[ind])
            self.r_s_z += self.r_s_z_buffer[ind]
            self.r_s_z_buffer = []
            self.rewards_buffer = []
        return self.keywords, self.rewards

class DKE:
    """
    See http://infoscience.epfl.ch/record/192441/files/Habibi_ACL_2013.pdf
    """
    def __init__(self, docs, vocab, num_topics=5, number_of_keywords=10, zlabels=None, eta=0.95, lambda_=0.75):
        self.docs = docs
        self.vocab = vocab
        self.num_topics = num_topics
        self.number_of_keywords = number_of_keywords
        self.phi, self.theta = topic_model_on_zlda(docs, vocab, num_topics, zlabels, eta)
        self.keywords = []
        self.rewards = []
        self.lambda_ = lambda_

    def keywords_ind_extract(self):
        for i, doc in enumerate(self.docs):
            keywords, rewards = Text(i, doc, self).keywords_extract()
            self.keywords.append(keywords)
            self.rewards.append(rewards)
        return self.keywords

    def keywords_extract(self):
        self.keywords_ind_extract()
        return [[self.vocab[word] for word in doc] for doc in self.keywords]


def save_keywords(keywords, filenames, path_to_demonstrative_file, path_to_results_dir=None):
    with open_write(path_to_demonstrative_file) as f:
        for i, file in enumerate(filenames):
            f.write(u'{0}: {1}\n'.format(os.path.split(file)[-1], u', '.join(keywords[i])))
    if path_to_results_dir:
        if not os.path.exists(path_to_results_dir):
                    os.mkdir(path_to_results_dir)
        for i, file in enumerate(filenames):
            with open_write(os.path.join(path_to_results_dir, os.path.split(file)[-1])) as f:
                f.write(u'\n'.join(keywords[i]))



if __name__ == "__main__":
    with open('../data/docs', 'rb') as f:
        docs = pickle.load(f)
    with open('../data/vocabulary', 'rb') as f:
        vocab = pickle.load(f)
        vocab = {v: k for k, v in vocab.items()}

    dke = DKE(docs, vocab)
    keywords = dke.keywords_extract()
    path_to_demonstrative_file = '../data/keywords.txt'
    path_to_dir = '/Users/annie/SELabs/data/utf_new_RGD/txt/validFiles'
    filenames = sorted(os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir))
    path_to_results_dir = '../data/dke'
    save_keywords(keywords, filenames, path_to_demonstrative_file, path_to_results_dir)


