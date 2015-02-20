# coding=utf-8
from __future__ import print_function
import os
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from utils import save_dict, open_write, del_meaningless_words, lemmer

__author__ = 'annie'


def get_stop_words(vectorizer, collection, treshhold=0.5, save_to=None):
    """
    Return list of stop words, those words that have document-frequency above treshhold.
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


def get_docs(vectorizer, filenames):
    analyzer = vectorizer.build_analyzer()
    docs = []
    for file in filenames:
        text = analyzer(file)
        text = [vectorizer.vocabulary_[i] for i in text if i in vectorizer.vocabulary_]
        docs.append(text)
    return docs

def get_docs_vocab(filenames, treshhold, stop_lemms, path_to_save_vocab=None, path_to_save_docs=None):
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

    vectorizer.fit(filenames)
    del_meaningless_words(vectorizer.vocabulary_)
    if path_to_save_vocab:
        save_dict(vectorizer.vocabulary_, path_to_save_vocab)

    docs = get_docs(vectorizer, filenames)
    if path_to_save_docs:
        with open(path_to_save_docs, 'wb') as f:
            pickle.dump(docs, f)
    return docs, vectorizer.vocabulary_

if __name__ == '__main__':
    treshhold = 0.5
    with open('../data/stop_lemms', 'rb') as f:
        stop_lemms = pickle.load(f)
    path_to_dir = '/Users/annie/SELabs/practice/txt'
    filenames = sorted(os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir) if file[-4:] == '.txt')
    path_to_save_vocab = '../data/vocabulary.txt'
    path_to_save_docs = '../data/docs'

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



    vectorizer.fit(filenames)
    del_meaningless_words(vectorizer.vocabulary_)
    save_dict(vectorizer.vocabulary_, '../data/vocabulary.txt')

    transform = vectorizer.transform(filenames)
    with open('../data/transforms', 'wb') as f:
        pickle.dump(transform, f)

    docs = get_docs(vectorizer, filenames)
    with open('../data/docs', 'wb') as f:
        pickle.dump(docs, f)
