# coding=utf-8
from __future__ import print_function
from collections import namedtuple
import os
from pprint import pprint
from shutil import rmtree
from topic_model_on_zlabelspy import TTopic_Model_ON_ZLABELS
from dke import DKE, LAMBDA
from test_files_to_catalogs import plane_test_files, map_base_name_to_number
from feature_extract import TOKEN_PATTERN, iseq_del_meaningless_words, iseg_normalize, get_tokenizer, MAX_DF, \
    USE_IDF, MIN_DF, get_vocabulary_analyzer_transform, count_transform_files, pymorphy2_del_meaningless_words, \
    pymorphy2_normalize, snowball_stemme
from topic_model_on_nmf import TTopic_Model_ON_NMF
from evaluation import PythonROUGE
from preprocessing_data import merge_all_dirs, Catalog, copy_files_for_training

from __init__ import BASE_DIR, TEST_DIR, TRAINING_DIR, STOP_WORDS, open_write, NUM_TOPICS, WORD_LABELS, open_read

__author__ = 'annie'

TPRF = namedtuple('TPRF', ['precision', 'recall', 'fmera'])


def main_test(res_prefix, remover=iseq_del_meaningless_words, normalizer=iseg_normalize, stemmer=None,
              treshhold=MAX_DF, min_df=MIN_DF, use_idf=USE_IDF, flag_topic_model='zlbels_w_labels',
              num_topics=NUM_TOPICS, lambda_=LAMBDA, useRank=False,
              do_recalculation_of_conditional_probability=True):

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
        'remover': remover,  # iseq_del_meaningless_words, #pymorphy2_del_meaningless_words,
        'normalizer': normalizer,  #iseg_normalize,   #pymorphy2_normalize,
        'stemmer': stemmer  # None, #snowball_stemme
    }
    tokenizer = get_tokenizer(**settings_for_tokenizer)

    setings_for_vectorizer = {
        'tokenizer': tokenizer,
        'preprocessor': None,
        'stop_words': STOP_WORDS,
        'treshhold': treshhold,  #MAX_DF,  # 0.7
        'min_df': min_df,  #MIN_DF,  # 2
        'use_idf': use_idf,  #USE_IDF,  # True
        'vocabulary': None
    }
    vocab, analyzer, _ = get_vocabulary_analyzer_transform(TEMP_BASE_FILES, setings_for_vectorizer)
    setings_for_vectorizer['vocabulary'] = vocab

    def topic_model(flag_topic_mode):
        if flag_topic_mode == 'nmf':
            ''' получаем topic model из non-negative matrix factorization (NMF) '''
            dtm = count_transform_files(TEMP_BASE_FILES, setings_for_vectorizer)
            topic_model = TTopic_Model_ON_NMF()
            topic_model.fit_topic_model(dtm, do_recalculation_of_conditional_probability, num_topics)
            words_topics = topic_model.words_topics  # (650, 6)
            docs_topics = topic_model.docs_topics  # (110, 6)
        elif flag_topic_mode.startswith('zlbels_'):  #simple':
            docs = []
            for doc_path in TEMP_BASE_FILES:
                docs.append([vocab[word] for word in analyzer(doc_path) if word in vocab])

            zlabels = False

            if flag_topic_mode.endswith('w_labels'):
                word_labels = WORD_LABELS
                inverse_vocab = {v: k for k, v in vocab.iteritems()}
                zlabels = []
                for text in docs:
                    zlabels_for_text = []
                    for ind_word in text:
                        word = inverse_vocab[ind_word]
                        if word in word_labels:
                            zlabels_for_text.append(word_labels[word])
                        else:
                            zlabels_for_text.append(0)
                    zlabels.append(zlabels_for_text)

            topic_model = TTopic_Model_ON_ZLABELS()
            topic_model.fit_topic_model(docs, len(vocab), do_recalculation_of_conditional_probability, num_topics, zlabels)
            words_topics = topic_model.words_topics  # (650, 6)
            docs_topics = topic_model.docs_topics  # (110, 6)
        return words_topics, docs_topics

    """---------------------------------- запускае dke -------------------------------------"""
    words_topics, docs_topics = topic_model(flag_topic_model)
    dke = DKE(words_topics, docs_topics, vocab, analyzer, lambda_)
    keywords = dke.extract_keywords(TEMP_BASE_FILES)

    """---------------------------------- записываем dke -------------------------------------"""
    def write_test_keywords(testing_dir):
        if os.path.exists(testing_dir):
            rmtree(testing_dir)
        os.mkdir(testing_dir)

        for i, path_file in enumerate(TEMP_BASE_FILES):
            file_name = os.path.basename(path_file).decode('utf-8')
            if file_name in plane_test_files:
                file_path = os.path.join(testing_dir, map_base_name_to_number.get(file_name))
                with open_write(file_path) as f:
                    f.write(u'\n'.join(keywords[i]))

    DKE_KEYWORDS_DIR = os.path.join(TEST_DIR, 'dke'+res_prefix)
    write_test_keywords(DKE_KEYWORDS_DIR)

    """---------------------------------- оценка результата -------------------------------------"""
    def my_preprocessor(text):
        text = tokenizer(text)
        return u' '.join(text)

    settings = {
        'txtTemp': '../data2/tempSettings.txt',  # tempSetingsTxt
        'pathTemp': '../data2/tempDir',  # pathTempDir
        'ROUGE_output_path': '../data2/ROUGE_result'+res_prefix+'.txt',  # ROUGE_result
        'pathToRefs': ['../data2/test_files/' + i for i in ['d', 'k', 'm']],
        'ngramOrder': 1,
        'skipBigram': 1,
        'reverseSkipBigram': 'u',
        'useRank': useRank,  #False, # Detailed keywords is ignored. The default is False.
        'preprocessor': my_preprocessor
    }

    pr = PythonROUGE(DKE_KEYWORDS_DIR, **settings)
    commonR, commonP, commonF = pr.run()
    return commonR, commonP, commonF


def run_test_stand(tests, tester_options):
    results = []

    for i in range(len(tests)):
        name_test = tests[i]
        print('test', i, ':', name_test)

        tester_option_1 = {}
        tester_option_2 = {}
        for option_name, dict_ in tester_options.iteritems():
            options = dict_.get(name_test)
            if options is None:
                tester_option_1[option_name] = dict_['default']
                tester_option_2[option_name] = dict_['default']
            else:
                tester_option_1[option_name] = options[0]
                tester_option_2[option_name] = options[1]

        temp_res = []
        for options in (tester_option_1, tester_option_2):
            commonR, commonP, commonF = main_test(**options)
            temp_res.append((options['res_prefix'], commonR, commonP, commonF))

        results.append({'name_test': name_test, 'res': temp_res})
    print('results =', results)

if __name__ == '__main__':

    """----------------- Запускаем тестовый стенд ------------------"""
    tests = [
        # 'MIN_DF=2 vs MIN_DF=1',  #0
        # 'topic_model: zlbels_simple vs zlbels_w_labels',  #1
        # 'topic_model: nmf vs zlabels_simple',  #2
        # 'iseg vs pymorphy2',  #3
        # 'treshhold: 0.7 vs 0.5', #4
        # 'treshhold: 0.55 vs 0.6', #4
        # 'treshhold: 0.8 vs 0.9', #5
        # 'lambda: 0.5 vs 0.75', #6
        # 'lambda: 0.65 vs 0.7', #6
        # 'lambda: 0.8 vs 0.85', #7
        # 'lambda: 0.9 vs 0.95', #8
        # 'lambda: 0.88 vs 1', #9
        # 'num_topics: 5 vs 6',  #10
        # 'num_topics: 7 vs 8',  #11
        # 'num_topics: 9 vs 10',  #12
        # 'no doStem vs doStem', #13
        # 'useTFIDF vs no useTFIDF', #14



    ]

    tester_options = {
        'res_prefix': {  # это будет в названии файла и dke выходных данных
            'MIN_DF=2 vs MIN_DF=1': [
                '_MIN_DF=2',
                '_MIN_DF=1'],

            'topic_model: zlbels_simple vs zlbels_w_labels': [
                '_zlbels_simple',
                '_zlbels_w_labels'],

            'topic_model: nmf vs zlabels_simple': [
                '_nmf',
                '_zlbels_simple'],

            'iseg vs pymorphy2': [
                '_iseg',
                '_pymorphy2'],

            'num_topics: 5 vs 6': [
                '_num_topics_5',
                '_num_topics_6'],

            'num_topics: 7 vs 8': [
                '_num_topics_7',
                '_num_topics_8'],

            'num_topics: 9 vs 10': [
                '_num_topics_9',
                '_num_topics_10'],

            'useTFIDF vs no useTFIDF': [
                '_useTFIDF',
                '_no_useTFIDF'],

            'no doStem vs doStem': [
                '_doStem',
                '_no_doStem'],

            'treshhold: 0.7 vs 0.5': [
                '_treshhold_0.7',
                '_treshhold_0.5'],

            'treshhold: 0.8 vs 0.9': [
                '_treshhold_0.8',
                '_treshhold_0.9'],

            'treshhold: 0.55 vs 0.6': [
                '_treshhold_0.55',
                '_treshhold_0.6'],

            'lambda: 0.5 vs 0.75': [
                '_lambda_0.5',
                '_lambda_0.75'],

            'lambda: 0.65 vs 0.7': [
                '_lambda_0.65',
                '_lambda_0.7'],

            'lambda: 0.8 vs 0.85': [
                '_lambda_0.80',
                '_lambda_0.85'],

            'lambda: 0.9 vs 0.95': [
                '_lambda_0.9',
                '_lambda_0.95'],

            'lambda: 0.88 vs 1': [
                '_lambda_0.88',
                '_lambda_0.1'],


        },
        'remover':    {
            'default': iseq_del_meaningless_words,
            'iseg vs pymorphy2': [iseq_del_meaningless_words, pymorphy2_del_meaningless_words],
        },

        'normalizer': {
            'default': iseg_normalize,
            'iseg vs pymorphy2': [iseg_normalize, pymorphy2_normalize],
        },

        'stemmer': {
            'default': None,
            'no doStem vs doStem': [None, snowball_stemme],
        },

        'treshhold': {  # MAX_DF = 0.7
            'default': MAX_DF,
            'treshhold: 0.7 vs 0.5': [MAX_DF, 0.5],
            'treshhold: 0.55 vs 0.6': [0.55, 0.6],
            'treshhold: 0.8 vs 0.9': [0.8, 0.9],
        },

        'min_df': {  # MIN_DF = 1
            'default': MIN_DF,
            'MIN_DF=2 vs MIN_DF=1': [2, MIN_DF],
        },

        'use_idf': {  # USE_IDF = True
            'default': USE_IDF,
            'useTFIDF vs no useTFIDF': [USE_IDF, False],
        },

        'flag_topic_model': {  # nmf, zlbels_simple, zlbels_w_labels
            'default': 'zlbels_w_labels',
            'topic_model: zlbels_simple vs zlbels_w_labels': ['zlbels_simple', 'zlbels_w_labels'],
            'topic_model: nmf vs zlabels_simple': ['nmf', 'zlbels_simple'],
        },

        'num_topics': {  # NUM_TOPICS = 6, число тем вообще 5
            'default': NUM_TOPICS,
            'num_topics: 5 vs 6': [5, NUM_TOPICS],
            'num_topics: 7 vs 8': [7, 8],
            'num_topics: 9 vs 10': [9, 10],
        },

        'lambda_': {  # LAMBDA=0.75
            'default': LAMBDA,
            'lambda: 0.5 vs 0.75': [0.5, 0.75],
            'lambda: 0.65 vs 0.7': [0.65, 0.7],
            'lambda: 0.8 vs 0.85': [0.8, 0.85],
            'lambda: 0.9 vs 0.95': [0.9, 0.95],
            'lambda: 0.88 vs 1': [0.88, 1.0],
        }
        # 'do_recalculation_of_conditional_probability':  # параметр только для меня, делать
        #     # do_recalculation vs no_recalculation
        #     True, # 0.16
        #     False, #  0.09
        #
        # 'useRank':  # Detailed keywords is ignored. The default is False.
        #     # useRank vs no useRank
        #     True, # 0.16
        #     False, # 0.16
    }
    'тестируем'
    # run_test_stand(tests, tester_options)



    ############~############~############~############~############~############~############~
    # TEXTRANK_KEYWORDS_DIR = os.path.join(TEST_DIR, 'textrank')
    TEXTRANK_KEYWORDS_DIR = os.path.join(TEST_DIR, 'ng_wn_3')
    'сравнение c textrank'
    def get_ROUGE_matrics_for_textrank(res_prefix='ng_wn_3', remover=iseq_del_meaningless_words, normalizer=iseg_normalize, stemmer=None,
              treshhold=MAX_DF, min_df=MIN_DF, use_idf=USE_IDF, flag_topic_model='zlbels_w_labels',
              num_topics=NUM_TOPICS, lambda_=LAMBDA, useRank=False,
              do_recalculation_of_conditional_probability=True):
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
            'remover': remover,  # iseq_del_meaningless_words, #pymorphy2_del_meaningless_words,
            'normalizer': normalizer,  #iseg_normalize,   #pymorphy2_normalize,
            'stemmer': stemmer  # None, #snowball_stemme
        }
        tokenizer = get_tokenizer(**settings_for_tokenizer)

        setings_for_vectorizer = {
            'tokenizer': tokenizer,
            'preprocessor': None,
            'stop_words': STOP_WORDS,
            'treshhold': treshhold,  #MAX_DF,  # 0.7
            'min_df': min_df,  #MIN_DF,  # 2
            'use_idf': use_idf,  #USE_IDF,  # True
            'vocabulary': None
        }
        vocab, analyzer, _ = get_vocabulary_analyzer_transform(TEMP_BASE_FILES, setings_for_vectorizer)

        def my_preprocessor(text):
            text = tokenizer(text)
            return u' '.join(text)

        new_TEXTRANK_KEYWORDS_DIR = TEXTRANK_KEYWORDS_DIR + 'tokinoze'
        def tokinoze_text():
            # new_TEXTRANK_KEYWORDS_DIR = TEXTRANK_KEYWORDS_DIR + 'tokinoze'

            if os.path.exists(new_TEXTRANK_KEYWORDS_DIR):
                rmtree(new_TEXTRANK_KEYWORDS_DIR)
            os.mkdir(new_TEXTRANK_KEYWORDS_DIR)

            for file_name in os.listdir(TEXTRANK_KEYWORDS_DIR):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(TEXTRANK_KEYWORDS_DIR, file_name)
                    new_file_path = os.path.join(new_TEXTRANK_KEYWORDS_DIR, file_name)
                    with open_read(file_path) as f:
                        new_text = my_preprocessor(f.read())
                    with open_write(new_file_path) as f:
                        f.write(new_text)

            return new_TEXTRANK_KEYWORDS_DIR

        settings = {
            'txtTemp': '../data2/tempSettings.txt',  # tempSetingsTxt
            'pathTemp': '../data2/tempDir',  # pathTempDir
            'ROUGE_output_path': '../data2/ROUGE_result'+res_prefix+'.txt',  # ROUGE_result
            'pathToRefs': ['../data2/test_files/' + i for i in ['d', 'k', 'm']],
            'ngramOrder': 1,
            'skipBigram': 1,
            'reverseSkipBigram': 'u',
            'useRank': useRank,  #False, # Detailed keywords is ignored. The default is False.
            'preprocessor': my_preprocessor
        }

        # tokinoze_text()
        pr = PythonROUGE(new_TEXTRANK_KEYWORDS_DIR, **settings)
        commonR, commonP, commonF = pr.run()

        return commonR, commonP, commonF

    get_ROUGE_matrics_for_textrank()

