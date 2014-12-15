# coding=utf-8
from __future__ import print_function, division
import os
import time
from IOData import getGmm, getFeatures, saveGmm
from calculations import getNewMeans, getNewMeans3D, criterionNeymanPearson

__author__ = 'annie'

def main():
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
            print('\nHandle all models for {0:.0f}m {1:.0f}s'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    # handle()

    def compareProtocols(path=pathToProtocolsDir):
        """
        Выполняем второую часть задания: для targets- и imposters-протоколов подсчитываем среднее значение log
        правдоподобия по критерию Неймана-Пирсона.
        """
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
            print('\nHandle all protocols for {0:.0f}m {1:.0f}s'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    compareProtocols()


if __name__ == '__main__':
    main()
