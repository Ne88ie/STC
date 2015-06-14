# coding=utf-8
from __future__ import division, print_function
import os
from shutil import rmtree
import numpy as np
from DKE.src.preprocessing_data import copy_files_for_testing, copy_plane_files_for_base, copy_plane_files_for_testing
from DKE.src.test_files_to_catalogs import plane_test_files, map_base_name_to_number
from DKE.src.topic_model_on_nmf import TTopic_Model_ON_NMF
from evaluation import PythonROUGE
from feature_extract import get_vocabulary_analyzer_transform, TOKEN_PATTERN, iseq_del_meaningless_words, \
    iseg_normalize, get_tokenizer, MAX_DF, USE_IDF, count_transform_files
from __init__ import open_read, STOP_WORDS, TRAINING_DIR, open_write, BASE_DIR, TEST_DIR

__author__ = 'annie'

LAMBDA = 0.75

class DKE:
    def __init__(self, words_topics, docs_topics, vocab, analyzer, lambda_=LAMBDA):
        self.words_topics = words_topics # 1450 x 5
        self.docs_topics = docs_topics # 50 x 5
        self.num_topics = self.docs_topics.shape[1]
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.iteritems()}
        self.analyzer = analyzer  # vectorizer.build_analyzer()
        self.lambda_ = lambda_

        self.temp_prev_reward_value = 0
        self.temp_distribution_of_topics = None
        self.temp_text = None

    def _get_document_words(self, document_path):
        text = self.analyzer(document_path)
        text = set([self.vocab[i] for i in text if i in self.vocab])
        return text

    def _b_z(self, topic_ind):
        b_z = 0
        for word in self.temp_text:
            b_z += self.words_topics[word, topic_ind]
        b_z /= float(len(self.temp_text))
        return b_z

    def _reward_function(self, word):
        result = 0
        for topic_ind, topic_f in enumerate(self.temp_distribution_of_topics):
            result += self._b_z(topic_ind) * pow(self.words_topics[word, topic_ind] + self.temp_prev_reward_value, self.lambda_)
        return result

    def _get_next_keyword(self):
        words = list(self.temp_text)
        rewards = map(self._reward_function, words)
        ind_argmax = np.argmax(rewards)
        self.temp_prev_reward_value = rewards[ind_argmax]
        return words[ind_argmax]

    def _extract_keywords_from_one_document(self, document, number_of_keywords):
        document_keywords = []
        self.temp_text = self._get_document_words(document)
        while self.temp_text and len(document_keywords) < number_of_keywords:
            next_keyword = self._get_next_keyword()
            document_keywords.append(self.inverse_vocab[next_keyword])
            self.temp_text -= {next_keyword}
        return document_keywords

    def extract_keywords(self, list_of_documents, number_of_keywords=10):
        result = []
        for i, document in enumerate(list_of_documents):
            self.temp_distribution_of_topics = self.docs_topics[i]
            keywords = self._extract_keywords_from_one_document(document, number_of_keywords)
            result.append(keywords)
        return result


if __name__ == '__main__':
    """----------------- перекладываем файлы из каталогов в одну папку ------------------------"""
    TEMP_BASE_DIR = BASE_DIR + '.plane'
    # copy_plane_files_for_base(BASE_DIR, TEMP_BASE_DIR)

    TEMP_BASE_FILES = []
    for file_name in sorted(os.listdir(TEMP_BASE_DIR)):
        file_path = os.path.join(TEMP_BASE_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            TEMP_BASE_FILES.append(file_path)

    """----------------- получем dtm и vocab из testing files ------------------------"""
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
    vocab, analyzer, _ = get_vocabulary_analyzer_transform(TEMP_BASE_FILES, setings_for_vectorizer)
    # setings_for_vectorizer['vocabulary'] = vocab
    # dtm = count_transform_files(TEMP_BASE_FILES, setings_for_vectorizer)
    #
    # """------- получаем topic model из non-negative matrix factorization (NMF) -------------"""
    # topic_model = TTopic_Model_ON_NMF()
    # topic_model.fit_topic_model(dtm)
    # words_topics = topic_model.topics_words.T  # (650, 6)
    # docs_topics = topic_model.docs_topics  # (110, 6)
    #
    # """---------------------------------- запускае dke -------------------------------------"""
    # dke = DKE(words_topics, docs_topics, vocab, analyzer)
    # keywords = dke.extract_keywords(TEMP_BASE_FILES)
    #
    # """---------------------------------- записываем dke -------------------------------------"""
    # def write_test_keywords(testing_dir):
    #     if os.path.exists(testing_dir):
    #         rmtree(testing_dir)
    #     os.mkdir(testing_dir)
    #
    #     for i, path_file in enumerate(TEMP_BASE_FILES):
    #         file_name = os.path.basename(path_file).decode('utf-8')
    #         if file_name in plane_test_files:
    #             file_path = os.path.join(testing_dir, map_base_name_to_number.get(file_name))
    #             with open_write(file_path) as f:
    #                 f.write(u'\n'.join(keywords[i]))
    #
    DKE_KEYWORDS_DIR = os.path.join(TEST_DIR, 'dke')
    # write_test_keywords(DKE_KEYWORDS_DIR)

    """---------------------------------- оценка результата -------------------------------------"""
    def my_preprocessor(text):
        text = tokenizer(text)
        return u' '.join(text)

    settings = {
        'txtTemp': '../data2/tempSettings.txt',  # tempSetingsTxt
        'pathTemp': '../data2/tempDir',  # pathTempDir
        'ROUGE_output_path': '../data2/ROUGE_result.txt',  # ROUGE_result
        'pathToRefs': ['../data2/test_files/' + i for i in ['d', 'k', 'm']],
        'ngramOrder': 1,
        'skipBigram': 1,
        'reverseSkipBigram': 'u',
        'useRank': False,
        'preprocessor': my_preprocessor
    }

    pr = PythonROUGE(DKE_KEYWORDS_DIR, **settings)
    commonR, commonP, commonF = pr.run()



