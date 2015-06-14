# coding=utf-8
from __future__ import print_function
import os
from zlabelLDA import zlabelLDA
import numpy as np
import cPickle as pickle
from sklearn import decomposition
from feature_extract import TOKEN_PATTERN, iseq_del_meaningless_words, iseg_normalize, get_tokenizer,\
    MAX_DF, USE_IDF, get_vocabulary_analyzer_transform, COMPRESSED_CATALOGS, count_transform_files, snowball_stemme

from __init__ import TRAINING_DIR, STOP_WORDS, NUM_TOPICS, BASE_DIR, TEST_DIR, TEMP_DIR


class TTopic_Model_ON_ZLABELS:
    def __init__(self):
        self.words_topics = None
        self.docs_topics = None

    def fit_topic_model(self, docs, len_vocab, do_recalculation_of_conditional_probability=True,
                        num_topics=NUM_TOPICS, zlabels=None, eta=0.95):
        """
        See http://pages.cs.wisc.edu/~andrzeje/research/zl_lda.html
        :param docs:
        :param vocab:
        :param num_topics:
        :param zlabels: each entry is ignored unless it is a List.
        :param eta: confidence in the our labels. If eta = 0 --> don't use z-labels,
               if eta = 1 --> "hard" z-labels.
        :param file_out:
        :return: Phi - P(w|z), Theta - P(z|d)
        """
        alpha = .1 * np.ones((1, num_topics))
        beta = .1 * np.ones((num_topics, len_vocab))
        numsamp = 100
        randseed = 194582

        if not zlabels:
            zlabels = [[0]*len(text) for text in docs]

        '''Theta - P(z|d), Phi - P(w|z)'''
        phi, theta, sample = zlabelLDA(docs, zlabels, eta, alpha, beta, numsamp, randseed)
        if do_recalculation_of_conditional_probability:
            phi = phi * phi.sum(axis=1)[:,np.newaxis]/(phi.sum(axis=0)[np.newaxis, :])
        words_topics = phi.T
        self.words_topics = words_topics
        self.docs_topics = theta
        self.sample = sample

    def get_top_topic_words(self, inverse_vocab, num_top_words=10):
        top_topics_words = []

        for topic in self.words_topics.T:
            word_idx = np.argsort(topic)[::-1][0:num_top_words]
            top_topics_words.append([inverse_vocab[i] for i in word_idx])
        return top_topics_words

if __name__ == '__main__':
    """----------------- получем dtm и vocab из training files ------------------------"""

    # TRAINING_FILES = [os.path.join(COMPRESSED_CATALOGS, file_name)
    #                   for file_name in os.listdir(COMPRESSED_CATALOGS) if file_name.endswith('.txt')]
    ''' кажется лучше не сливать все файлы каталога в один '''
    TRAINING_FILES = []
    for catalog in os.listdir(TRAINING_DIR):
        catalog_path = os.path.join(TRAINING_DIR, catalog)
        if os.path.isdir(catalog_path):
            for file_name in os.listdir(catalog_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(catalog_path, file_name)
                    TRAINING_FILES.append(file_path)


    settings_for_tokenizer = {
        'token_pattern': TOKEN_PATTERN,
        'remover': iseq_del_meaningless_words,   #pymorphy2_del_meaningless_words,
        'normalizer': iseg_normalize,  #iseg_normalize,   #pymorphy2_normalize,
        'stemmer': None  # snowball_stemme
    }
    tokenizer = get_tokenizer(**settings_for_tokenizer)

    setings_for_vectorizer = {
        'tokenizer': tokenizer,
        'preprocessor': None,
        'stop_words': STOP_WORDS,
        'treshhold': MAX_DF,  # 0.7
        'min_df': 2,  # 2
        'use_idf': USE_IDF,  # True
        'vocabulary': None
    }

    vocab, analyzer, _t = get_vocabulary_analyzer_transform(TRAINING_FILES, setings_for_vectorizer)
    """-------------------------------------- получаем документы ----------------------------"""
    docs = []
    for doc_path in TRAINING_FILES:
        docs.append([vocab[word] for word in analyzer(doc_path) if word in vocab])

    """------- получаем topic model из z-labels -------------"""
    topic_model = TTopic_Model_ON_ZLABELS()
    topic_model.fit_topic_model(docs, len(vocab), NUM_TOPICS)
    words_topics = topic_model.words_topics
    docs_topics = topic_model.docs_topics
    ''' взять 10 топ слов для каждой темы '''
    inverse_vocab = {v: k for k, v in vocab.iteritems()}
    top_topic_words = topic_model.get_top_topic_words(inverse_vocab)
    """------------------------------------------------------------------------------------"""

    """---------------------------- печатем topic model -----------------------------------"""
    print('words_topics', words_topics.shape, words_topics)
    ''' проверка что сумма каждого слова по темам == 1 '''
    print('words_topics == 1:', np.sum(words_topics, axis=1))

    print('docs_topics', docs_topics.shape, docs_topics)
    for i, words in enumerate(top_topic_words):
        print('topic', i, ':', u', '.join(words))
    """------------------------------------------------------------------------------------"""


