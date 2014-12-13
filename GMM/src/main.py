from __future__ import print_function, division
import os
import time
from readData import getUbm, getFeatures, saveUbm
from countMeans import getNewMeans, saveLogGMM

__author__ = 'annie'

def main():
    pathToData = '/Users/annie/SELabs/Kudashev/Lab1/data'
    pathToModels = os.path.join(pathToData, 'models')
    pathToTests = os.path.join(pathToData, 'tests')
    pathToUbm = os.path.join(pathToData, 'ubm.gmm')
    pathToUbmsDir = os.path.join(pathToData, 'ubms')
    pathToLogDir = os.path.join(pathToData, 'tempLogGMM')

    def handle(path):
        tBegAll = time.time()
        ubm = getUbm(pathToUbm, preprocessing=True)
        if not os.path.exists(pathToUbmsDir):
            os.mkdir(pathToUbmsDir)
        if not os.path.exists(pathToLogDir):
            os.mkdir(pathToLogDir)
        for i, model in enumerate(os.listdir(path)):
            if os.path.splitext(model)[-1] == '.features_bin':
                tBeg = time.time()
                print('Model', i+1)
                features = getFeatures(os.path.join(path, model))
                newMeans = getNewMeans(ubm, features, 20)
                pathToNewUbm = os.path.join(pathToUbmsDir, os.path.splitext(model)[0]) + '.gmm'
                saveUbm(pathToNewUbm, pathToUbm, newMeans)
                newUbm = getUbm(pathToNewUbm, preprocessing=True)
                pathToLog = os.path.join(pathToLogDir, os.path.splitext(model)[0] + '.log_gmm')
                saveLogGMM(newUbm, features, pathToLog)
                deltaTime = time.time() - tBeg
                print('\tHandled for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
        deltaTimeAll = time.time() - tBegAll
        print('\nHandle all models for {0:.0f}m {1:.0f}s'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    handle(pathToModels)
    handle(pathToTests)

if __name__ == '__main__':
    main()