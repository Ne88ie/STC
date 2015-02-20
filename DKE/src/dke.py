from __future__ import division, print_function
import numpy as np
import cPickle as pickle
from topic_modeling import topic_model_on_zlda
range = xrange

__author__ = 'annie'


class Text:
    def __init__(self, text_number, text, dke, lambda_=0.75):
        self.text = set(text)
        self.text_number = text_number
        self.dke = dke
        self.keywords = []
        self.lambda_ = lambda_
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
            rewards += self.dke.theta[self.text_number, topic] * r_s_z_buffer[topic] ** self.lambda_
        self.r_s_z_buffer.append(r_s_z_buffer)
        return rewards

    def keywords_extract(self):
        while len(self.keywords) < self.dke.number_of_keywords:
            words = list(self.text - set(self.keywords))
            word = max(words, key=self.reward_function)
            self.keywords.append(word)
            index = words.index(word)
            self.r_s_z += self.r_s_z_buffer[index]
            self.r_s_z_buffer = []
        return self.keywords

class DKE:
    def __init__(self, docs, vocab, zlabels=None, num_topics=5, number_of_keywords=10):
        self.docs = docs
        self.vocab = vocab
        self.num_topics = num_topics
        self.number_of_keywords = number_of_keywords
        self.phi, self.theta = topic_model_on_zlda(docs, vocab, num_topics, number_of_keywords, zlabels)
        self.keywords = []

    def keywords_ind_extract(self):
        for i, doc in enumerate(self.docs):
            keywords = Text(i, doc, self).keywords_extract()
            self.keywords.append(keywords)
        return self.keywords

    def keywords_extract(self):
        self.keywords_ind_extract()
        return [[self.vocab[word] for word in doc] for doc in self.keywords]


if __name__ == "__main__":
    with open('../data/docs', 'rb') as f:
        docs = pickle.load(f)
    with open('../data/vocabulary', 'rb') as f:
        vocab = pickle.load(f)
        vocab = {v: k for k, v in vocab.items()}

    dke = DKE(docs, vocab)
    keywords = dke.keywords_extract()
    print('\nkeywords\n', keywords)