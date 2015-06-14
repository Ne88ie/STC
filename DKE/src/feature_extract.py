# coding=utf-8
from __future__ import print_function
import os
import cPickle as pickle
import re
from shutil import rmtree
import pymorphy2
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from snowballstemmer import stemmer
from __init__ import open_write, open_read
from __init__ import BASE_DIR, TEST_DIR, TRAINING_DIR, TEMP_DIR, STOP_WORDS

__author__ = 'annie'

COMPRESSED_CATALOGS = TRAINING_DIR + '.compressed'
TOKEN_PATTERN = u'(?u)[A-zА-я]{2,}-[A-zА-я]{1,}|[A-zА-я]{3,}'
MAX_DF = 0.7
MIN_DF = 1
USE_IDF = True
__morph = pymorphy2.MorphAnalyzer()
__stemmer = stemmer('russian')
__mystem = Mystem() # iseg

def pymorphy2_del_meaningless_words(list_of_words):
    """
    Removes meaningless words from the dictionary.
    See https://pymorphy2.readthedocs.org/en/latest/user/grammemes.html?highlight=числительное#grammeme-docs
    :param word:
    """
    result = []
    stop_tags = {'NUMR', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}
    for word in list_of_words:
        p = __morph.parse(word)[0]
        if p.tag.POS not in stop_tags:
            result.append(word)
    return result

def iseq_del_meaningless_words(list_of_words):
    """
    Расшифровки см. https://tech.yandex.ru/mystem/doc/grammemes-values-docpage.
    """
    def is_valid_part_of_speech(word):
        invalid_parts = ('ADVPRO', 'ANUM', 'CONJ', 'INTJ', 'NUM', 'PART', 'PR', 'SPRO')
        analyze = __mystem.analyze(word)
        if analyze:
            analysis = analyze[0]['analysis']
            if analysis:
                gr = analysis[0]['gr']
                for part in invalid_parts:
                    if gr.startswith(part):
                        return False
        return True

    result = []
    for word in list_of_words:
        if is_valid_part_of_speech(word):
            result.append(word)
    return result

def pymorphy2_normalize(list_of_words):
    """Substitutes word to its normal form."""
    result = []
    for word in list_of_words:
        result.append(__morph.normal_forms(word)[0])
    return result

def iseg_normalize(list_of_words):
    result = []
    for word in list_of_words:
        analyze = __mystem.analyze(word)
        if analyze:
            analysis = analyze[0]['analysis']
            if analysis:
                word = analysis[0]['lex']
        result.append(word)
    return result


def snowball_stemme(list_of_words):
    """Substitutes each word to its stemm. Here work bad."""
    return __stemmer.stemWords(list_of_words)

# def iseg_stemme(list_of_words):
#     result = []
#     for word in list_of_words:
#         lemmas = __mystem.lemmatize(word)
#         if lemmas:
#             word = lemmas[0]
#         result.append(word)
#     return result


def get_tokenizer(token_pattern, remover=None, normalizer=None, stemmer=None):
    """

    :param token_pattern: string
    :param remover:
    :param normalizer:
    :param stemmer:
    :return:
    """
    token_pattern = re.compile(token_pattern)

    def wrapper(doc):
        list_of_words = token_pattern.findall(doc)
        for func in (remover, normalizer, stemmer):
            if func:
                list_of_words = func(list_of_words)
        return list_of_words
    return wrapper


def build_count_vectorizer(tokenizer=None, preprocessor=None, stop_words=None, treshhold=MAX_DF, min_df=MIN_DF, vocabulary=None, **args):
    return CountVectorizer(
        input='filename',
        encoding='utf-8',
        lowercase=True,
        preprocessor=preprocessor,  # или None. Если есть, то lowercase не учитывается
        tokenizer=tokenizer,  # сначала обрабатывается текст preprocessor потом идёт tokenizer
        token_pattern=TOKEN_PATTERN,  # findall(token_pattern, doc)
        stop_words=stop_words,
        analyzer='word',
        max_df=treshhold,  # ignore terms that have a term frequency strictly higher than the given threshold (corpus specific stop words)
        min_df=min_df,  # ignore terms that have a term frequency strictly lower than the given threshold If float, the parameter represents a proportion of documents, integer absolute counts
        vocabulary=vocabulary  # Mapping or iterable
    )

def build_tf_idf_vectorizer(tokenizer=None, preprocessor=None, stop_words=None, treshhold=MAX_DF, min_df=MIN_DF, use_idf=USE_IDF, **args):
    return TfidfVectorizer(
        input='filename',
        encoding='utf-8',
        lowercase=True,
        preprocessor=preprocessor,  # или None. Если есть, то lowercase не учитывается
        tokenizer=tokenizer,  # сначала обрабатывается текст preprocessor потом идёт tokenizer
        token_pattern=TOKEN_PATTERN,  # findall(token_pattern, doc)
        stop_words=stop_words,
        analyzer='word',
        max_df=treshhold,  # ignore terms that have a term frequency strictly higher than the given threshold (corpus specific stop words)
        min_df=min_df,  # ignore terms that have a term frequency strictly lower than the given threshold If float, the parameter represents a proportion of documents, integer absolute counts
        norm='l2',  # cosine
        use_idf=use_idf,
        smooth_idf=True,
        sublinear_tf=False # или True -> logarithmic. Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        # vocabulary= Mapping or iterable
    )


def get_vocabulary_analyzer_transform(training_files, setings):
    tf_idf_vectorizer = build_tf_idf_vectorizer(**setings)
    transform = tf_idf_vectorizer.fit_transform(training_files)
    return tf_idf_vectorizer.vocabulary_, tf_idf_vectorizer.build_analyzer(), transform
    # return tf_idf_vectorizer.get_feature_names(), transform


def count_transform_files(test_files, setings):
    count_vectorizer = build_count_vectorizer(**setings)
    return count_vectorizer.fit_transform(test_files)

def tf_idf_transform_test_files(test_files, setings):
    setings['use_idf'] = True
    tf_idf_vectorizer = build_tf_idf_vectorizer(**setings)
    return tf_idf_vectorizer.fit_transform(test_files)

def tf_transform_test_files(test_files, setings):
    setings['use_idf'] = False
    tf_vectorizer = build_tf_idf_vectorizer(**setings)
    return tf_vectorizer.fit_transform(test_files)


def write_training_catalog_to_file(catalog_dir, catalog_file):
    with open_write(catalog_file) as f_write:
        all_text = []
        for file_name in os.listdir(catalog_dir):
            file_path = os.path.join(catalog_dir, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                with open_read(file_path) as f_read:
                    all_text.append(f_read.read().strip(u'\n'))
        f_write.write(u'\n'.join(all_text))


def compress_catalogs_to_one_file(training_files_dir, result_dir):
    if os.path.exists(result_dir):
        rmtree(result_dir)
    os.mkdir(result_dir)

    for catalog_name in os.listdir(training_files_dir):
        catalog_dir_path = os.path.join(training_files_dir, catalog_name)
        if os.path.isdir(catalog_dir_path):
            catalog_file_path = os.path.join(result_dir, catalog_name + '.txt')
            write_training_catalog_to_file(catalog_dir_path, catalog_file_path)


if __name__ == '__main__':

    """------------------- Делаем словарь и count_matrix -------------------"""
    '''сливаем файлы каждого каталога в один файл'''
    # compress_catalogs_to_one_file(TRAINING_DIR, COMPRESSED_CATALOGS)

    TRAINING_FILES = [os.path.join(COMPRESSED_CATALOGS, file_name)
                      for file_name in os.listdir(COMPRESSED_CATALOGS) if file_name.endswith('.txt')]
    TEST_FILES = [os.path.join(TEST_DIR, file_name)
                  for file_name in os.listdir(TEST_DIR) if file_name.endswith('.txt')]

    settings_for_tokenizer = {
        'token_pattern': TOKEN_PATTERN,
        'remover': iseq_del_meaningless_words,   #pymorphy2_del_meaningless_words,
        'normalizer': iseg_normalize,   #pymorphy2_normalize,
        'stemmer': None  # snowball_stemme
    }
    tokenizer = get_tokenizer(**settings_for_tokenizer)

    setings_for_vectorizer = {
        'tokenizer': tokenizer,
        'preprocessor': None,
        'stop_words': STOP_WORDS,
        'treshhold': MAX_DF,  # 0.7
        'min_df': 1,  # 2
        'use_idf': USE_IDF,  # True
        'vocabulary': None
    }

    vocabulary, _a, _t = get_vocabulary_analyzer_transform(TRAINING_FILES, setings_for_vectorizer)
    # rmtree(COMPRESSED_CATALOGS)
    setings_for_vectorizer['vocabulary'] = vocabulary
    count_matrix = count_transform_files(TEST_FILES, setings_for_vectorizer)
    tf_idf_matrix = tf_idf_transform_test_files(TEST_FILES, setings_for_vectorizer)
    tf_matrix = tf_transform_test_files(TEST_FILES, setings_for_vectorizer)

    """------------------------------ Вывод и сохранение -----------------------------"""
    '''печатаем словарь'''
    print('vocabulary', len(vocabulary))
    for k, v in sorted(vocabulary.iteritems()):
        print(u"u'{0}': {1},".format(k, v))

    ''' печатаем матрицу
    getrow(0).indices - все индексы слов для первого доуемнты,
    getrow - все слова документа,
    getcol - встречаемость слова,
    tolil - все значение в виде: (r, c) v'''
    print('count_matrix', count_matrix.tolil())
    print('tf_idf_matrix', tf_idf_matrix.tolil())
    print('tf_matrix', tf_matrix.tolil())

    '''сохраняем в файлы'''
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)

    with open(os.path.join(TEMP_DIR, 'vocabulary_.dict'), 'wb') as f:
        pickle.dump(vocabulary, f)

    with open(os.path.join(TEMP_DIR, 'count_matrix_.mtx'), 'wb') as f:
        f.write(pickle.dumps(count_matrix))

    with open(os.path.join(TEMP_DIR, 'tf_idf_matrix_.mtx'), 'wb') as f:
        f.write(pickle.dumps(tf_idf_matrix))

    with open(os.path.join(TEMP_DIR, 'tf_matrix_.mtx'), 'wb') as f:
        f.write(pickle.dumps(tf_matrix))

    """----------------------------------------------------------------------------"""

    """----------------читаем из файла матрицу-------------------"""
    # with open(os.path.join(TEMP_DIR, 'count_matrix.mtx'), 'rb') as f:
    #     count_matrix = pickle.loads(f.read())
    # print(count_matrix.tolil())

    """----------------------------------------------------------------------------"""
