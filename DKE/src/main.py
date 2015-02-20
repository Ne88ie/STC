import os
import cPickle as pickle
from utils import lemmer
from feature_extraction import get_docs_vocab
from dke import DKE, save_keywords
from evaluation import PythonROUGE

__author__ = 'annie'

def extract_keyword():
    treshhold = 0.5
    with open('../data/stop_lemms', 'rb') as f:
        stop_lemms = pickle.load(f)
    path_to_dir = '/Users/annie/SELabs/practice/txt'
    filenames = sorted(os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir) if file[-4:] == '.txt')
    docs, vocab = get_docs_vocab(filenames, treshhold, stop_lemms)
    vocab = {v: k for k, v in vocab.items()}

    num_topics = 5
    number_of_keywords = 10
    zlabels = None
    eta = 0.95
    lambda_ = 0.75
    dke = DKE(docs, vocab, num_topics, number_of_keywords, zlabels, eta, lambda_)
    keywords = dke.keywords_extract()

    path_to_demonstrative_file = '../data/keywords.txt'
    path_to_results_dir = '../data/dke'

    save_keywords(keywords, filenames, path_to_demonstrative_file, path_to_results_dir)


def evaluate():
    pathToGuess    = '../data/dke'
    tempSetingsTxt = '../data/tempSettings.txt'
    pathTempDir    = '../data/tempDir'
    ROUGE_result   = '../data/ROUGE_result.txt'
    pathesToRefs   = ['/Users/annie/PycharmProjects/ROUGE/data/txt/' + i for i in ['d', 'k', 'm']]
    ngramOrder = 2
    skipBigram = 2
    reverseSkipBigram = 'U'
    preprocessor = lemmer

    pr = PythonROUGE(pathToGuess, pathesToRefs, tempSetingsTxt, pathTempDir, ROUGE_result, ngramOrder, skipBigram, reverseSkipBigram, preprocessor)
    pr.run()

if __name__ == '__main__':
    # extract_keyword()
    evaluate()