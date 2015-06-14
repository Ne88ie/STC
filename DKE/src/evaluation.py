from __future__ import division, print_function
import sys
import os
import shutil
from __init__ import open_write, open_read
from transliterate import translit

__author__ = 'annie'


class PythonROUGE:
    """
    pathToGuess   - folders in which are candidates for summarization.
    pathToRefs    - list of folders for reference summarizations.
    pathTempDir   - path to temporary directory where we store the translited version for each file.
    temSetingsTxt - temporary settings file for ROUGE.
    ROUGE_result  - output file for ROUGE.
    ngramOrder    - (optional) the order of the N-grams used to compute ROUGE. The default is 2
                    (bigrams).
    skipBigram    - (optional) max-gap-length. The default is 2.
    reverseSkipBigram - (optional) compute ROUGE-SUx ('u') or ROUGE-Sx ('s') or both ('U'). The default
                        is 'U'.
    doStem        - (optional) use stemmer for Russian language. The default is False.
    useRank       - (optional) distinguish whether general keywords (1 in marked file) and detailed
                    keywords (0 in marked file).
                    Detailed keywords is ignored. The default is False.
    """
    def __init__(self, pathToGuess='', pathToRefs='', txtTemp='', pathTemp='', ROUGE_output_path='',
                 ngramOrder=2, skipBigram=2, reverseSkipBigram='U', preprocessor=None, useRank=False):
        self.pathToGuess = pathToGuess
        self.pathToRefs = pathToRefs
        self.txtTemp = txtTemp
        self.pathTemp = pathTemp
        self.ngramOrder = ngramOrder
        self.skipBigram = skipBigram
        self.reverseSkipBigram = reverseSkipBigram
        self.preprocessor = preprocessor
        self.useRank = useRank
        self.ROUGE_output_path = ROUGE_output_path

    def __getNameWithLabels(self):
        resultName = self.ROUGE_output_path
        if self.preprocessor:
            resultName = os.path.splitext(resultName)[0] + '_prepr' + os.path.splitext(resultName)[1]
        if self.useRank:
            resultName = os.path.splitext(resultName)[0] + '_useRank' + os.path.splitext(resultName)[1]
        return resultName

    def __getGuessRefsSettings(self):
        """
        Create temporary directory and settings file for ROUGE.
        """
        os.mkdir(self.pathTemp)
        with open(self.txtTemp, 'w+') as temp:
            i = 0
            for guess in os.listdir(self.pathToGuess):
                if guess[-4:] in ('.txt', '.csv'):
                    nameFileGuess = os.path.normpath(os.path.join(self.pathToGuess, guess))
                    newNameFileGuess = os.path.normpath(
                        os.path.join(
                            self.pathTemp,
                            os.path.basename(self.pathToGuess) + '_' + guess
                        )
                    )
                    self.__save_translite_gues(nameFileGuess, newNameFileGuess)
                    temp.write('%s' % newNameFileGuess)
                    for dirToRefs in self.pathToRefs:
                        dirName = os.path.basename(dirToRefs)
                        nameFileRef = os.path.normpath(
                            os.path.join(
                                dirToRefs,
                                'keywords_' + os.path.splitext(guess)[0].split('_')[-1] + '.csv'
                            )
                        )
                        if os.path.isfile(nameFileRef):
                            newNameFileRef = os.path.normpath(
                                os.path.join(
                                    self.pathTemp, dirName + '_' + os.path.splitext(guess)[0] + '.csv'
                                )
                            )
                            self.__save_translite_refs(nameFileRef, newNameFileRef)
                            temp.write(' %s' % newNameFileRef)
                    i += 1
                    temp.write('\n')
            if i == 0:
                print("The folder {0} contains no reference annotation".format(self.pathToGuess))
                exit()

    def __runROUGE(self):
        """
        Wrapper function to use ROUGE.
        """
        ROUGE_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.normpath('RELEASE-1.5.5/ROUGE-1.5.5.pl')
        )
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.normpath('RELEASE-1.5.5/data')
        )
        '''
        options for ROUGE:
            -f A|B (scoring formula)
                A - (by default) use model average to compute the overall ROUGE scores when there are
                    multiple references. "-f A" option is better
                    when use ROUGE in summarization evaluations.
                B - use the best matching score among the referenes as the final score. "-f B" option
                    is better when use ROUGE in machine translation.
            -m (steming),
            -n (count ROUGE-N),
            -s (dell stopwords),
            -2 x -u (this is ROUGE-SUx, may be without -u ROUGE-S4, but it's worse),
                -u (use unigramms)
                -U (same as -u but also compute regular skip-bigram)
            -x (do not calculate ROUGE-L)
            -w weight (weighting factor for WLCS - 'L^weight' typically this is set to 1.2 or other
               number greater than 1.
        '''
        unigramInSkipBigram = '' if self.reverseSkipBigram == 's' else '-' + self.reverseSkipBigram
        options = '-a -2 {0} {1} -n {2}'.format(
            str(self.skipBigram),
            unigramInSkipBigram,
            str(self.ngramOrder)
        )

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

    def __save_translite_gues(self, pathFromFile, pathToFile):
        with open_read(pathFromFile) as fromFile:
            with open_write(pathToFile) as toFile:
                toFile.write(translit(fromFile.read(), 'ru', reversed=True))

    def __save_translite_refs(self, pathFromFile, pathToFile):
        """
        Save translited version of file.
        """
        with open_read(pathFromFile) as fromFile:
            with open_write(pathToFile) as toFile:
                for line in fromFile:
                    if line.strip():
                        lineRank = line.split(u',', 1)
                        if self.useRank and len(lineRank) == 2:
                            rank = lineRank[1]
                            if rank.isdigit() and int(rank) == 0:
                                continue
                        line = lineRank[0]
                        line = ' '.join([word for word in line.split() if not word.startswith(u'*')])
                        line += '\n'
                        if self.preprocessor:
                            line = self.preprocessor(line)
                        toFile.write(translit(line, 'ru', reversed=True))

    @staticmethod
    def getAverageMetrics(ROUGE_output_path):
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
                elif lineNum % 4 == 2:
                    commonP.append(float(line.split()[3]))
                elif lineNum % 4 == 3:
                    commonF.append(float(line.split()[3]))

        commonR = sum(commonR)/(len(commonR) or 0.000001)
        commonP = sum(commonP)/(len(commonP) or 0.000001)
        commonF = sum(commonF)/(len(commonF) or 0.000001)

        return commonR, commonP, commonF

    def run(self, out=sys.stdout):
        """
        Return average Recall, average Precision, average F-measure on all metrics.
        If out=None then Average Metrics not shown.
        """
        try:
            self.__getGuessRefsSettings()
            self.__runROUGE()
        finally:
            self.__purge()
        ROUGE_output_path = self.__getNameWithLabels()
        commonR, commonP, commonF = self.getAverageMetrics(ROUGE_output_path)

        print('Average ROUGE metrics for {0}:'.format(ROUGE_output_path), file=out)
        print('Average Recall    = {0:.4f}\n'
              'Average Precision = {1:.4f}\n'
              'Average F-measure = {2:.4f}\n'.format(commonR, commonP, commonF), file=out)
        return commonR, commonP, commonF

