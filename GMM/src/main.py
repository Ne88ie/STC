# coding=utf-8
from __future__ import print_function, division
import os
import time
from IOData import getGmm, getFeatures, saveGmm
from calculations import getNewMeans, getNewMeans3D, criterionNeymanPearson
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'annie'

def main():
    """
    GMM - Gaussian Mixture Model. Реализация и использование алгоритма MAP-адаптации GMM для создания голосовых моделей
    дикторов, а также построения на его основе простейшей системы распознавания дикторов по голосу.
    """
    pathToData = '/Users/annie/SELabs/Kudashev/Lab1/data'
    pathToModelsDir = os.path.join(pathToData, 'models')
    pathToTestsDir = os.path.join(pathToData, 'tests')
    pathToUbm = os.path.join(pathToData, 'ubm.gmm')
    pathToGmmsDir = os.path.join(pathToData, 'gmms')
    pathToProtocolsDir = os.path.join(pathToData, 'protocols')
    pathToAnswersDir = os.path.join(pathToData, pathToProtocolsDir, 'answers')
    r = 20

    def handle(path=pathToModelsDir):
        """
        Выполняем первую часть задания: получаем апостериорные оценки средних значений гауссойд распределения GMM,
        взяв в качестве априорного распределения парметров значения UBM.
        """
        print('Start handle')
        tBegAll = time.time()
        try:
            if not os.path.exists(pathToGmmsDir):
                os.mkdir(pathToGmmsDir)
            ubm = getGmm(pathToUbm, preprocessing=True)
            for i, model in enumerate(sorted(os.listdir(path))):
                if os.path.splitext(model)[-1] == '.features_bin':
                    tBeg = time.time()
                    print('Model', i+1)
                    features = getFeatures(os.path.join(path, model))
                    newMeans = getNewMeans3D(ubm, features, r)
                    pathToNewUbm = os.path.join(pathToGmmsDir, os.path.splitext(model)[0]) + '.gmm'
                    saveGmm(pathToNewUbm, pathToUbm, newMeans)
                    deltaTime = time.time() - tBeg
                    print('\tHandled for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
        finally:
            deltaTimeAll = time.time() - tBegAll
            print('All models were handled for {0:.0f}m {1:.0f}s\n'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    handle()

    def compareProtocols(path=pathToProtocolsDir):
        """
        Выполняем второую часть задания: для targets- и imposters-протоколов подсчитываем среднее значение log
        правдоподобия по критерию Неймана-Пирсона.
        """
        print('Start compareProtocols')
        tBegAll = time.time()
        try:
            if not os.path.exists(pathToAnswersDir):
                os.mkdir(pathToAnswersDir)
            ubm = getGmm(pathToUbm, preprocessing=True)
            for protocol in sorted(os.listdir(path)):
                if os.path.splitext(protocol)[-1] == '.txt':
                    i = 1
                    with open(os.path.join(path, protocol)) as p:
                        with open(os.path.join(pathToAnswersDir, os.path.splitext(protocol)[0] + '_answers.txt'), 'w') as answers:
                            for gmms in p:
                                if gmms:
                                    tBeg = time.time()
                                    print('{0} {1}'.format(os.path.splitext(protocol)[0][:-1].capitalize(), i))
                                    modelSample, testSample = gmms.split()
                                    pathToSpeakerGmm = os.path.join(pathToGmmsDir, os.path.splitext(modelSample)[0] + '.gmm')
                                    pathToTestFeatures = os.path.join(pathToTestsDir, os.path.splitext(testSample)[0] + '.features_bin')
                                    speakerGmm = getGmm(pathToSpeakerGmm, preprocessing=True)
                                    testFeatures = getFeatures(pathToTestFeatures)
                                    logLikelihood = criterionNeymanPearson(ubm, speakerGmm, testFeatures)
                                    answers.write(str(logLikelihood) + '\n')
                                    deltaTime = time.time() - tBeg
                                    print('\tHandled for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
                                    i += 1
        finally:
            deltaTimeAll = time.time() - tBegAll
            print('All protocols were compared for {0:.0f}m {1:.0f}s\n'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    compareProtocols()

    def drawEER():
        """
        Выполняем третью часть задания: посчитать EER (Equal Error Rate).
        """
        with open(os.path.join(pathToAnswersDir, 'targets_answers.txt')) as targAnsFiles:
            with open(os.path.join(pathToAnswersDir, 'impostors_answers.txt')) as impAnsFile:
                targAnswers = np.array([np.float64(i) for i in targAnsFiles.readlines() if i])
                impAnswers = np.array([np.float64(i) for i in impAnsFile.readlines() if i])

        minBin = min(min(targAnswers), min(impAnswers))
        maxBin = max(max(targAnswers), max(impAnswers))

        gaps = 1000
        bins = np.linspace(minBin, maxBin, gaps)

        tarRate, tarBins  = np.histogram(targAnswers, bins=bins)
        impRate, impBins  = np.histogram(impAnswers, bins=bins)

        tarRate = tarRate/targAnswers.size
        impRate = impRate/impAnswers.size

        tarRate = np.add.accumulate(tarRate)
        impRate = np.add.accumulate(impRate[::-1])[::-1]
        arg = np.argmin(np.abs(tarRate - impRate))
        eer = (tarRate[arg] + impRate[arg])/2

        plt.plot(tarBins[1:], tarRate, 'b-', impBins[1:], impRate, 'r-', impBins[arg], eer, 'og')
        plt.annotate('EER = {0:.2%}'.format(eer), xy=(impBins[arg], eer), xytext=(impBins[arg] + 0.02, eer + 0.03),
                     arrowprops=dict(facecolor='green', width=0.01, headwidth=7, shrink=0.1),)
        plt.legend(['target  FR', 'impostor FA'], loc='center left')
        plt.xlabel('logLikelihood')
        plt.ylabel('pdf')
        plt.title('FR-FA')
        plt.show()

    drawEER()

if __name__ == '__main__':
    main()
