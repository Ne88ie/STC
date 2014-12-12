"""
Created on Nov 11 2014
author: annie

Wrapper function to use ROUGE from Python easily.
Before use is recommended use checkValidData from checkValidData.py for reference summaries.
IMPORTANT: All the reference summaries must be in the same directory!
           And ROUGE does not work with Russian characters!
So we create a temporary directory and store the translited version for each file there.

Inputs:
    pathToGuess   - folders in which are candidates for summarization.
    pathToRefs    - list of folders for reference summarizations.
    pathTempDir   - path to temporary directory where we store the translited version for each file.
    temSetingsTxt - temporary settings file for ROUGE.
    ROUGE_result  - output file for ROUGE.
    ngramOrder    - (optional) the order of the N-grams used to compute ROUGE. The default is 2 (bigrams).
    skipBigram    - (optional) max-gap-length. The default is 2.
    reverseSkipBigram - (optional) compute ROUGE-SUx or ROUGE-Sx or both. The default is 'U' - both.
    doStem        - (optional) use stemmer for Russian language. The default is False.
    useRank       - (optional) distinguish whether general keywords (1 in marked file) and detailed keywords (0 in marked file).
                    Detailed keywords is ignored. The default is False.

Output:
    The ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-SU4 stored in the file (ROUGE_result). And out Average ROUGE results.

Example usage:
    1) out Average ROUGE results in stdout
    >>> from PythonROUGE import PythonROUGE
    >>> pr = PythonROUGE(pathToGuess, pathesToRefs, tempSetingsTxt, pathTempDir, ROUGE_result)
    >>> pr.run()

    2) out Average ROUGE results in stdout and in "../../data/Average ROUGE results.txt"
    >>> from PythonROUGE import PythonROUGE
    >>> pr = PythonROUGE(pathToGuess, pathesToRefs, tempSetingsTxt, pathTempDir, ROUGE_result)
    >>> pr.run(outToFile="../../data/Average ROUGE results.txt")

    More see at the end of this file.

Depending:
    (1) Python3

    (2) transliterate, snowballstammer, PyStemmer from PyPI
        >> pip install transliterate
        >> pip install snowballstammer
        >> pip install PyStemmer

    (3) Perl interpreter (for windows work Strawberry Perl http://strawberryperl.com/).

    (4) XML::DOM from http://www.cpan.org.
        >> perl -MCPAN -e 'install XML::DOM'

    (5) You need to have DB_File installed. If the Perl script complains
        about database version incompatibility, you can create a new
        WordNet-2.0.exc.db by running the buildExceptionDB.pl script in
        the "data/WordNet-2.0-Exceptions" subdirectory.
        >> cd data/WordNet-2.0-Exceptions/
        >> perl buildExeptionDB.pl . exc ../WordNet-2.0.exc.db
"""

import os
import shutil
from transliterate import translit
from snowballstemmer import stemmer
from sys import stdout

class PythonROUGE:

    __stemmer = stemmer('russian')

    def __init__(self, pathToGuess='', pathToRefs='', txtTemp='', pathTemp='', ROUGE_output_path='',
                 ngramOrder=2, skipBigram=2, reverseSkipBigram='U', doStem=True, useRank=False):
        self.pathToGuess = pathToGuess
        self.pathToRefs = pathToRefs
        self.txtTemp = txtTemp
        self.pathTemp = pathTemp
        self.ngramOrder = ngramOrder
        self.skipBigram = skipBigram
        self.reverseSkipBigram = reverseSkipBigram
        self.doStem = doStem
        self.useRank = useRank
        self.ROUGE_output_path = ROUGE_output_path

    def __getNameWithLabels(self):
        resultName = self.ROUGE_output_path
        if self.doStem:
            resultName = os.path.splitext(resultName)[0] + '_doStem' + os.path.splitext(resultName)[1]
        if self.useRank:
            resultName = os.path.splitext(resultName)[0] + '_useRank' + os.path.splitext(resultName)[1]
        return resultName

    def __getGuessRefsSettings(self):
        """
        Create temporary directory and settings file for ROUGE.
        """
        guess_summary_list = []
        ref_summary_list = []
        os.mkdir(self.pathTemp)
        with open(self.txtTemp, 'w+') as temp:
            i = 0
            for guess in os.listdir(self.pathToGuess):
                if guess[-4:] in ('.txt', '.csv'):
                    nameFileGuess = os.path.normpath(os.path.join(self.pathToGuess, guess))
                    newNameFileGuess = os.path.normpath(os.path.join(self.pathTemp, os.path.basename(self.pathToGuess) + '_' + guess))
                    self.__saveTranslite(nameFileGuess, newNameFileGuess, self.doStem, False)
                    temp.write('%s' % newNameFileGuess)
                    guess_summary_list.append(newNameFileGuess)
                    ref_summary_list.append([])
                    for dirToRefs in self.pathToRefs:
                        dirName = os.path.basename(dirToRefs)
                        nameFileRef = os.path.normpath(os.path.join(dirToRefs, 'keywords_' + os.path.splitext(guess)[0].split('_')[-1] + '.csv'))
                        if os.path.isfile(nameFileRef):
                            newNameFileRef = os.path.normpath(os.path.join(self.pathTemp, dirName + '_' + os.path.splitext(guess)[0] + '.csv'))
                            self.__saveTranslite(nameFileRef, newNameFileRef, self.doStem, self.useRank)
                            temp.write(' %s' % newNameFileRef)
                            ref_summary_list[i].append(newNameFileRef)
                    i += 1
                    temp.write('\n')
            if i == 0:
                print("The folder {0} contains no reference generalizations".format(self.pathToGuess))
                exit()
        return guess_summary_list, ref_summary_list

    def __runROUGE(self):
        """
        Wrapper function to use ROUGE.
        """
        ROUGE_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.normpath('RELEASE-1.5.5/ROUGE-1.5.5.pl'))
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.normpath('RELEASE-1.5.5/data'))
        '''
        options for ROUGE:
            -f A|B (scoring formula)
                A - (by default) use model average to compute the overall ROUGE scores when there are multiple references. "-f A" option is better
                    when use ROUGE in summarization evaluations.
                B - use the best matching score among the referenes as the final score. "-f B" option is better when use ROUGE in machine translation.
            -m (steming),
            -n (count ROUGE-N),
            -s (dell stopwords),
            -2 x -u (this is ROUGE-SUx, may be without -u ROUGE-S4, but it's worse),
                -u (use unigramms)
                -U (same as -u but also compute regular skip-bigram)
            -x (do not calculate ROUGE-L)
            -w weight (weighting factor for WLCS - 'L^weight' typically this is set to 1.2 or other number greater than 1.
        '''
        unigramInSkipBigram = '' if self.reverseSkipBigram == 's' else '-' + self.reverseSkipBigram
        options = '-a -2 {0} {1} -n {2}'.format(str(self.skipBigram), unigramInSkipBigram, str(self.ngramOrder))

        # >> RELEASE-1.5.5/ROUGE-1.5.5.pl -e RELEASE-1.5.5/data -a -2 4 -u -n 2 -z SPL ../../data/tempSettings.txt > ../../data/ROUGE_result.txt
        exec_command = os.path.normpath(ROUGE_path) + \
                       ' -e ' + os.path.normpath(data_path) + ' ' +\
                       options + ' -z SPL ' + os.path.normpath(self.txtTemp) +\
                       ' > ' + os.path.normpath(self.__getNameWithLabels())
        os.system(exec_command)

    def __purge(self):
        """
        Delete all temporary files.
        """
        if os.path.exists(self.pathTemp): shutil.rmtree(self.pathTemp)
        if os.path.exists(self.txtTemp): os.remove(self.txtTemp)

    def __saveTranslite(self, pathFromFile, pathToFile, doStem=False, useRank=False):
        """
        Save translited version of file.
        """
        with open(pathFromFile, 'r', encoding='utf-8') as fromFile:
            with open(pathToFile, 'w+') as toFile:
                for line in fromFile:
                    if line.strip():
                        lineRank = line.split(',', maxsplit=1)
                        line = lineRank[0]
                        if useRank:
                            rank = lineRank[1]
                            if int(rank) == 0:
                                continue
                        listWords = [word for word in line.replace('-', ' ').split() if word[0] != '*']
                        if doStem:
                            listWords = self.__stemmer.stemWords(listWords)
                        line = ' '.join(listWords) + '\n'
                        toFile.write(translit(line, 'ru', reversed=True))

    @staticmethod
    def getAverageMetrics(ROUGE_output_path, outToFile='', stdOut=True):
        """
        Computation average Recall, average Precision, average F-measure on all metrics.
        If outToFile given then be output to the specified file.
        If stdOut=False then average ROUGE metrics not shown.
        """
        commonR, commonP, commonF = [], [], []

        with open(ROUGE_output_path) as file:
            for lineNum, line in enumerate(file):
                if lineNum % 4 == 1:
                    commonR.append(float(line.split()[3]))
                if lineNum % 4 == 2:
                    commonP.append(float(line.split()[3]))
                if lineNum % 4 == 3:
                    commonF.append(float(line.split()[3]))

        commonR = sum(commonR)/max(len(commonR), 0.000001)
        commonP = sum(commonP)/max(len(commonP), 0.000001)
        commonF = sum(commonF)/max(len(commonF), 0.000001)
        if stdOut:
            print('Average ROUGE metrics for {0}:'.format(ROUGE_output_path))
            print('Average Recall    = {0:.2f}\n'
                  'Average Precision = {1:.2f}\n'
                  'Average F-measure = {2:.2f}\n'.format(commonR, commonP, commonF))
        if outToFile:
            with open(outToFile, 'w') as out:
                print('Average ROUGE metrics for {0}:'.format(ROUGE_output_path), file=out)
                print('Average Recall    = {0:.2f}\n'
                      'Average Precision = {1:.2f}\n'
                      'Average F-measure = {2:.2f}\n'.format(commonR, commonP, commonF), file=out)
        return commonR, commonP, commonF

    def run(self, outToFile='', stdOut=True):
        """
        Return average Recall, average Precision, average F-measure on all metrics.
        If out=None then Average Metrics not shown.
        """
        try:
            self.__getGuessRefsSettings()
            self.__runROUGE()
        finally:
            self.__purge()
        return self.getAverageMetrics(self.__getNameWithLabels(), outToFile, stdOut)


if __name__ == '__main__':
    """
    Examples usage:
    """
    def example_1():
        """
        Run ROUGE for 3 cases: without stemming, with stemming, with stemming and with use rank of keywords.
        It outputs the average Recall, Precision, F-measure over all metrics for each case.
        See ../../data/Example_1_ results.txt
        """
        print('############# Example 1 #############')
        pathToGuess    = '../../data/txt/textrank/ng_wn_3'
        tempSetingsTxt = '../../data/tempSettings.txt'
        pathTempDir    = '../../data/tempDir'
        ROUGE_result   = '../../data/ROUGE_result.txt'
        pathesToRefs   = ['../../data/txt/reference1', '../../data/txt/reference1', '../../data/txt/reference3']

        parametrs = ({'doStem': False, 'useRank': False, 'doc': 'without stemming'},
                     {'doStem': True, 'useRank': False, 'doc': 'with stemming'},
                     {'doStem': True, 'useRank': True, 'doc': 'with stemming and with use rank of keywords'})

        pr = PythonROUGE(pathToGuess, pathesToRefs, tempSetingsTxt, pathTempDir, ROUGE_result)

        for parametr in parametrs:
            pr.doStem, pr.useRank = parametr['doStem'], parametr['useRank']
            print('By all reference summarizations {0}:'.format(parametr['doc']))
            pr.run()

    def example_2():
        """
        It outputs the similarity with textrank for each executor.
        See ../../data/Example_2_ results.txt
        """
        print('############# Example 2 #############')
        pathToGuess     = '../../data/txt/textrank/ng_wn_3'
        tempSetingsTxt  = '../../data/tempSettings.txt'
        pathTempDir     = '../../data/tempDir'
        doStem, useRank = True, False # with stemming

        pathesToRefs = (['../../data/txt/denisov'],
                        ['../../data/txt/kiseleva'],
                        ['../../data/txt/moiseeva'])
        ROUGE_results = ('../../data/ROUGE_result_reference1.txt',
                         '../../data/ROUGE_result_reference2.txt',
                         '../../data/ROUGE_result_reference3.txt')

        pr = PythonROUGE(pathToGuess, '', tempSetingsTxt, pathTempDir, '', doStem=doStem, useRank=useRank)

        for iter, result in enumerate(ROUGE_results):
            pr.pathToRefs = pathesToRefs[iter]
            pr.ROUGE_output_path = result
            commonF = pr.run(stdOut=False)[2]
            print("{0}'s similarity with textrank (with stemming): {1:.2f}\n".format(os.path.basename(pathesToRefs[iter][0]), commonF))

    example_1()
    example_2()
