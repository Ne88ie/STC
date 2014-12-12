__author__ = 'annie'


import os
import configparser
from PythonROUGE import PythonROUGE

config = configparser.ConfigParser()

pathToIni = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python_rouge.ini')

config.read(pathToIni)

pathToGuess = config['PythonROUGE']['pathToGuess']

pathesToRefs = config['PythonROUGE']['pathesToRefs'].split()

tempSetingsTxt = config['PythonROUGE']['tempSetingsTxt']

pathTempDir = config['PythonROUGE']['pathTempDir']

ROUGE_result = config['PythonROUGE']['ROUGE_result']

ngramOrder = int(config['PythonROUGE']['ngramOrder'])

skipBigram = int(config['PythonROUGE']['skipBigram'])

reverseSkipBigram = config['PythonROUGE']['reverseSkipBigram']

doStem = config['PythonROUGE']['doStem'] == '1'

useRank = config['PythonROUGE']['useRank'] == '1'

outAverageMetrics = config['PythonROUGE']['outAverageMetrics']

rouge = PythonROUGE(pathToGuess, pathesToRefs, tempSetingsTxt, pathTempDir, ROUGE_result, ngramOrder, skipBigram, reverseSkipBigram, doStem, useRank)

rouge.run(outAverageMetrics)


