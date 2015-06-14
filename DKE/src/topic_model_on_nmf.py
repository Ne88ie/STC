# coding=utf-8
from __future__ import print_function
import os
import numpy as np
import cPickle as pickle
from sklearn import decomposition
from feature_extract import COMPRESSED_CATALOGS, TOKEN_PATTERN, iseq_del_meaningless_words, iseg_normalize, \
    get_tokenizer, MAX_DF, USE_IDF, get_vocabulary_analyzer_transform, count_transform_files, snowball_stemme

from __init__ import BASE_DIR, TEST_DIR, TRAINING_DIR, TEMP_DIR, STOP_WORDS, NUM_TOPICS



class TTopic_Model_ON_NMF:
    def __init__(self):
        self.words_topics = None
        self.docs_topics = None

    def fit_topic_model(self, dtm, do_recalculation_of_conditional_probability, num_topics=NUM_TOPICS):
        """
        :param dtm: matrix (dok-word-freq) - только count_matrix
        """
        '''non-negative matrix factorization'''
        clf = decomposition.NMF(n_components=num_topics, random_state=1)
        doctopic = clf.fit_transform(dtm)
        '''words associated with topics'''
        topics_words = clf.components_
        ''' считаем p(z|w) - words_topics из данных p(w|z) - topics_words на основе условной вероятности '''
        topics_words += 0.0001
        if do_recalculation_of_conditional_probability:
            topics_words = topics_words * topics_words.sum(axis=1)[:,np.newaxis]/(topics_words.sum(axis=0)[np.newaxis, :])
        ''' нормировка '''
        topics_words = topics_words / np.sum(topics_words, axis=0, keepdims=True)
        self.words_topics = topics_words.T
        self.docs_topics = doctopic / np.sum(doctopic, axis=1, keepdims=True)

    def get_top_topic_words(self, inverse_vocab, num_top_words=10):
        """ вернуть 10 топ слов для каждой темы """
        top_topics_words = []

        for topic in self.words_topics.T:
            word_idx = np.argsort(topic)[::-1][0:num_top_words]
            top_topics_words.append([inverse_vocab[i] for i in word_idx])

        return top_topics_words


if __name__ == '__main__':
    """----------------- читаем  данные полученые из feature_extract -----------------"""
    # with open(os.path.join(TEMP_DIR, 'count_matrix_1445.mtx'), 'rb') as f:
    #     dtm = pickle.loads(f.read())
    # with open(os.path.join(TEMP_DIR, 'vocabulary_1445.dict'), 'rb') as f:
    #     vocab = pickle.load(f)
    """--------------------------------------------------------------------------------"""

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

    vocab, _a, _t = get_vocabulary_analyzer_transform(TRAINING_FILES, setings_for_vectorizer)
    setings_for_vectorizer['vocabulary'] = vocab
    dtm = count_transform_files(TRAINING_FILES, setings_for_vectorizer)
    """------------------------------------------------------------------------------------"""

    """------- получаем topic model из non-negative matrix factorization (NMF) -------------"""
    topic_model = TTopic_Model_ON_NMF()
    topic_model.fit_topic_model(dtm)
    words_topics = topic_model.words_topics
    docs_topics = topic_model.docs_topics
    ''' взять 10 топ слов для каждой темы '''
    inverse_vocab = {v: k for k, v in vocab.iteritems()}
    top_topic_words = topic_model.get_top_topic_words(inverse_vocab)
    """------------------------------------------------------------------------------------"""

    """---------------------------- печатем topic model -----------------------------------"""
    print('words_topics', words_topics.shape, words_topics)
    ''' проверка что сумма каждого слова по темам == 1 '''
    # print('words_topics == 1', np.sum(words_topics, axis=1))

    print('docs_topics', docs_topics.shape, docs_topics)
    for i, words in enumerate(top_topic_words):
        print('topic', i, ':', u', '.join(words))
    """------------------------------------------------------------------------------------"""



