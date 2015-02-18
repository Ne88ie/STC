# coding=utf-8
from __future__ import print_function

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
from utils import str_dict, print_dict, save_dict, open_write, open_read, fix_vocabulary, lemmer
import cPickle as pickle
import traceback
from snowballstemmer import stemmer

__author__ = 'moiseeva'

treshhold = 0.5
with open('../data/stop_lemms', 'rb') as f:
    """
    811 -> 687 tokens
    """
    stop_lemms = pickle.load(f)



vectorizer = CountVectorizer(input=u'filename',
                             encoding=u'utf-8',
                             lowercase=True,
                             preprocessor=lemmer, # None
                             tokenizer=None,
                             token_pattern=u'(?u)[A-zА-я\-]{2,}',
                             stop_words=stop_lemms,
                             analyzer=u'word',
                             max_df=treshhold,
                             min_df=0.0,
                             binary=False,                 # True
                             )

def get_stop_words(vectorizer, collection, treshhold=0.5, save_to=None):
    """
    Return list of stop words.
    >> stop_words = get_stop_words(vectorizer, filenames, treshhold, 'stop_words')
    :param vectorizer: object of class CountVectorizer
    :param collection: raw_documents
    :param treshhold: ignore terms that have a document frequency strictly lower than the given threshold.
    :param save_to: path to save serialized stop words
    :return: list of stop words
    """
    temp_token_pattern, vectorizer.token_pattern = vectorizer.token_pattern, u'(?u)[A-zА-я\-]+'
    temp_max_df, vectorizer.max_df = vectorizer.max_df, 1.0
    temp_min_df, vectorizer.min_df = vectorizer.min_df, treshhold
    temp_stop_words, vectorizer.stop_words = vectorizer.stop_words, None
    temp_vocabulary, vectorizer.vocabulary_ = getattr(vectorizer, 'vocabulary_', None), None
    vectorizer.fit(collection)
    stop_words = vectorizer.vocabulary_.keys()
    vectorizer.token_pattern = temp_token_pattern
    vectorizer.min_df = temp_min_df
    vectorizer.max_df = temp_max_df
    vectorizer.stop_words = temp_stop_words
    vectorizer.vocabulary_ = temp_vocabulary
    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump(stop_words, f)
        with open_write(save_to+'.txt') as f:
            f.write(u'\n'.join(sorted(stop_words)))
    return stop_words



path_to_dir = 'C:/Users/moiseeva/SElabs/data/utf_new_RGD/txt/validFiles'
path_to_test = 'C:/Users/moiseeva/SElabs/data/utf_new_RGD/txt/NOTvalidFiles/005.txt'
filenames = [os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir)]


vectorizer.fit(filenames)
fix_vocabulary(vectorizer.vocabulary_)
# save_dict(vectorizer.vocabulary_, '../data/vocabulary.txt')


transform = vectorizer.transform([path_to_test])
print(transform[0])

if __name__ == '__main__':
    pass
    # а-ля интерактивный режим. Баг: зацикливается на exit()
    # while True:
    #     try:
    #         eval(str(input()))
    #     except:
    #         traceback.print_exc(file=sys.stdout)
    #         print()

