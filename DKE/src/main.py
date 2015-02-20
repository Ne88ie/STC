import os
import cPickle as pickle
from feature_extraction import get_docs_vocab
from dke import DKE, save_keywords

__author__ = 'annie'

def main():
    treshhold = 0.5
    with open('../data/stop_lemms', 'rb') as f:
        stop_lemms = pickle.load(f)
    path_to_dir = '/Users/annie/SELabs/data/utf_new_RGD/txt/validFiles'
    filenames = sorted(os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir))
    docs, vocab = get_docs_vocab(filenames, treshhold, stop_lemms)
    vocab = {v: k for k, v in vocab.items()}

    num_topics = 5
    number_of_keywords = 10
    zlabels = None
    eta = 0.95
    dke = DKE(docs, vocab, num_topics, number_of_keywords, zlabels, eta)
    keywords = dke.keywords_extract()

    path_to_demonstrative_file = '../data/keywords1.txt'
    path_to_results_dir = '../data/dke'

    save_keywords(keywords, filenames, path_to_demonstrative_file, path_to_results_dir)

if __name__ == '__main__':
    main()