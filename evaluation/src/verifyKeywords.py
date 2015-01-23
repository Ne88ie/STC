"""
Created on Jan 19 2015
author: annie

Create two tables for each file marked-up text.
Table 1
    * selected manually keywords;
    * counted whether the phrase in the evaluation;
    * got the collocation at all in the list of all ngram identified by the TextRank, and its weight if hit;
    * a list of all similar ngram included in the list and their weight.

Table 2
    * the list of selected TextRank keywords that do not appear in the manual markup;
    * their weight.
"""

import os
import re
from snowballstemmer import stemmer


class verifier:
    """

    """
    __stemmer = stemmer('russian')
    __split_on_weight = re.compile('^(\d+\.\d+)\s(\d+\.\d+\s){3}(.*)')

    def __init__(self, pathToGuess='', pathToRefs=[], pathToResult='', mode='w', mixing_references='union', use_rank=True, min_weight=0.1):
        """
        :param pathToGuess:
        :param pathToRefs:
        :param pathToResult:
        :param mode: 'a' for add or 'w' for rewrite. The default is 'w'
        :param mixing_references: intersection, union, n - the minimum number of texts that should be a coincidence.
                                  The default is 'union'
        :param use_rank: distinguish whether general keywords (1 in marked file) and detailed keywords (0 in marked file)
                    Detailed keywords is ignored. The default is True
        :param min_weight: accepted for evaluation keywords textrank that weights are equal to or above this threshold
        :return:
        """
        self.pathToGuess = pathToGuess
        self.pathToRefs = pathToRefs
        self.pathToResult = pathToResult
        self.mode = mode
        self.mix = mixing_references
        self.use_rank = use_rank
        self.min_weight = min_weight

    def __trim_word(self, words):
        """

        :param words:
        :return:
        """
        words = list(map(str.strip, words.lower().replace('-', ' ').split()))
        words = self.__stemmer.stemWords(words)
        words = ' '.join(sorted(words))
        return words

    def __get_gues_keywords(self, pathToFile):
        """
        Format of input is file containing lines such as: x.x x.x x.x x.x keyword
        :param pathToFile:
        :return: dict {keyword: weight, ...}
        """
        guess_keywords_weights = {}
        guess_keywords_orig = {}
        with open(pathToFile, 'r', encoding='utf-8') as file:
            for line in file:
                weight_keyword = self.__split_on_weight.findall(line)
                if weight_keyword:
                    orig_word = weight_keyword[0][-1].replace('.', '')
                    keyword = self.__trim_word(orig_word)
                    guess_keywords_weights[keyword] = float(weight_keyword[0][0])
                    guess_keywords_orig[keyword] = orig_word
        return guess_keywords_weights, guess_keywords_orig

    def __get_ref_keywords(self, pathToFile):
        ref_keywords = {}
        with open(pathToFile, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    lineRank = line.split(',', maxsplit=1)
                    line = lineRank[0]
                    if self.use_rank:
                        rank = lineRank[1]
                        if int(rank) == 0:
                            continue
                    line = map(str.strip, line.replace('.', '').split())
                    keywords = ' '.join([word for word in line if word[0] != '*'])
                    ref_keywords[self.__trim_word(keywords)] = keywords
        return ref_keywords

    def __get_similar_keywords(self, word, guess_keywords_orig):
        similar_keywords = set()
        str_word = word
        word = set(word.split())
        for guess_keyword, orig_keyword in guess_keywords_orig.items():
            if guess_keyword != str_word:
                guess_keyword = set(map(str.strip, guess_keyword.split()))
                intersection = guess_keyword.intersection(word)
                if intersection and abs(len(word) - len(intersection)) <= 1 and len(guess_keyword) - len(intersection) <= 1:
                    similar_keywords.add(orig_keyword)
        return sorted(similar_keywords)

    def __mixing_references_on_n(self, list_refs):
        all = {x: [0, ''] for x in set.union(*map(set, list_refs))}
        for ref in list_refs:
            for word, origin in ref.items():
                all[word][0] += 1
                all[word][1] = origin
        return {word: origin for word, (count, origin) in all.items() if count >= self.mix}

    def verify(self):
        print('Start')
        with open(self.pathToResult, self.mode) as res:
            for guess in os.listdir(self.pathToGuess):
                if guess[-4:] in ('.txt', '.csv'):
                    nameFileGuess = os.path.normpath(os.path.join(self.pathToGuess, guess))
                    guess_keywords_weights, guess_keywords_orig = self.__get_gues_keywords(nameFileGuess)
                    refs_keywords = []
                    for dirToRefs in self.pathToRefs:
                        nameFileRef = os.path.normpath(os.path.join(dirToRefs, 'keywords_' + os.path.splitext(guess)[0].split('_')[-1] + '.csv'))
                        if os.path.isfile(nameFileRef):
                            refs_keywords.append(self.__get_ref_keywords(nameFileRef))
                    if self.mix == 'union':
                        refs = {}
                        for d in refs_keywords:
                            refs.update(d)
                        refs_keywords = refs
                    elif self.mix == 'intersection':
                        refs_keywords = {x: refs_keywords[0][x] for x in set.intersection(*map(set, refs_keywords))}
                    elif str(self.mix).isdigit():
                        self.mix = int(self.mix)
                        refs_keywords = self.__mixing_references_on_n(refs_keywords)
                    else:
                        raise ValueError("Неверный формат значения mixing_references! "
                                         "Должно быть либо 'union', либо 'intersection' либо число.")

                    res.write('Файл: ' + guess + '\n')
                    # Table 1
                    res.write('Таблица 1\nРучные ключевые слова\tСтемма\tУчтено?\tЕсть среди нграмм?\tВес\tПохожие ключевые слова у textrank\n')
                    rows = []
                    for word, original_words in refs_keywords.items():
                        temp_row = []
                        if word in guess_keywords_weights:
                            temp_row.append(original_words)
                            temp_row.append(word)
                            temp_row.append('да' if guess_keywords_weights.get(word) >= self.min_weight else 'нет')
                            temp_row.append('да')
                            temp_row.append(str(guess_keywords_weights[word]))
                            del guess_keywords_weights[word]
                        else:
                            temp_row = [original_words, word, 'нет', 'нет', '0']
                        similar_keywords = ', '.join(self.__get_similar_keywords(word, guess_keywords_orig))
                        similar_keywords = similar_keywords[:-1] if similar_keywords and similar_keywords[-1] == ',' else similar_keywords
                        temp_row.append(similar_keywords)
                        rows.append(temp_row)
                    rows = sorted(rows) # sorted on first colomn ascendingly
                    rows = sorted(rows, key=lambda row: [row[4]], reverse=True) # sorted on weight descendingly
                    rows = list(map('\t'.join, rows))
                    res.write('\n'.join(rows) + '\n')

                    # Table 2
                    res.write('\nТаблица 2\nНеучтенные ключевые слова textrank\tСтемма\tВес\n')
                    rows = []
                    for word, weight in guess_keywords_weights.items():
                        rows.append([guess_keywords_orig[word], word, str(weight)])
                    rows = sorted(rows) # sorted on first colomn ascendingly
                    rows = sorted(rows, key=lambda row: [row[2]], reverse=True) # sorted on weight descendingly
                    rows = list(map('\t'.join, rows))
                    res.write('\n'.join(rows) + '\n\n\n')
                    print(guess + ' done')
        print('Finish')


if __name__ == "__main__":
    pathToGuess = 'C:/Users/moiseeva/PycharmProjects/evaluation/data/txt/textrank/ng_wn_3'
    pathToRefs = ['C:/Users/moiseeva/PycharmProjects/evaluation/data/txt/denisov',
                  'C:/Users/moiseeva/PycharmProjects/evaluation/data/txt/kiseleva',
                  'C:/Users/moiseeva/PycharmProjects/evaluation/data/txt/moiseeva']
    pathToResult = '../data/new verify/reults_for_verify.csv'
    use_rank = False
    # use_rank = True
    # mixing_references = 'union'
    # mixing_references = 'intersection'
    mixing_references = 2
    verifier(pathToGuess, pathToRefs, pathToResult, mixing_references=mixing_references, use_rank=use_rank).verify()