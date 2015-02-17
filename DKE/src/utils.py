# coding=utf-8
import re
import cPickle as pickle
import codecs
import pymorphy2
from snowballstemmer import stemmer

__author__ = 'moiseeva'

__morph = pymorphy2.MorphAnalyzer()
__stemmer = stemmer('russian')
__pattern = re.compile(u'(?u)[A-zА-я]{2,}')

def str_dict(dict_):
    """
    Right conversion dict to string. Without nesting level of values.
    :param dict_: some dict
    :return: str
    """
    ans = [u'{0}: {1}'.format(k, v) for k, v in sorted(dict_.items())]
    return u'\n'.join(ans)


def print_dict(dict_):
    print(str_dict(dict_), u'\n\n{0} ключей'.format(len(dict_)))


def save_dict(dict_, path):
    with codecs.open(path, encoding='utf-8', mode='w') as f:
        f.write(str_dict(dict_))
    with open(path[:-4], 'wb') as f:
        pickle.dump(dict_, f)


def get_union_words(list_files):
    words = set()
    for file in list_files:
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            words.update(f.read().split())
    return sorted(words)


def stemmer(text):
    """
    Substitutes each word to its stemm. Here work bad.
    :param text:
    :return:
    """
    text = __stemmer.stemWords(text.split())
    return u' '.join(text)

def normalized(matchobj):
    return __morph.parse(matchobj.group(0))[0].normal_form

def lemmer(text):
    """
    Substitutes each word to its normal form. 1231 -> 820 tokens.
    :param text: str
    :return: str
    """
    return __pattern.sub(normalized, text)

def get_prepared_texts(files):
    """
    Returns lemmated texts from list of pathes to texts.
    :param files: [path, path, ...]
    :return: list[str, str, ...]
    """
    list_stemmed = []
    for file in files:
        with open_read(file) as f:
            list_stemmed.append(lemmer(f.read()))
    return list_stemmed


def fix_vocabulary(vocabulary):
    """
    Removes meaningless words from the dictionary. 820 -> 785 tokens.
    See https://pymorphy2.readthedocs.org/en/latest/user/grammemes.html?highlight=числительное#grammeme-docs
    :param vocabulary:
    """
    stop_tags = {'NUMR', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'} # 820->711
    # stop_tags = {'NUMR'} # 820 -> 785
    i = 0
    for word in vocabulary.keys():
        p = __morph.parse(word)[0]
        if p.tag.POS in stop_tags:
            del vocabulary[word]
        else:
            vocabulary[word] = i
            i += 1

def get_stop_lemms(path_to_stop_words):
    """
    Union equal norm forms from file containing stop words.
    :param path_to_stop_words:
    :return: list of lemms
    """
    with open_read(path_to_stop_words) as f:
        return sorted(set(f.read().split()))

def dump_words_from_file(file):
    """
    Read words from file, split them to list, save list by dump.
    :param file: contains words
    """
    with open_read(file) as f_from:
        with open(file[:-4], 'wb') as f_to:
            pickle.dump(f_from.read().split(), f_to)

open_write = lambda file: codecs.open(file, encoding='utf-8', mode='w')
open_read = lambda file: codecs.open(file, encoding='utf-8', mode='r')

