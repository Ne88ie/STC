from __future__ import print_function, division
import os
import time
from readData import getUbm, getFeatures, saveUbm
from countMeans import getNewMeans

__author__ = 'annie'

def main():
    pathToData = '/Users/annie/SELabs/Kudashev/Lab1/data'
    pathToModels = os.path.join(pathToData, 'models')
    pathToUbm = os.path.join(pathToData, 'ubm.gmm')
    pathToUbms = os.path.join(pathToData, 'ubms')

    tBegAll = time.time()
    ubm = getUbm(pathToUbm)
    if not os.path.exists(pathToUbms):
        os.mkdir(pathToUbms)
    for i, model in enumerate(os.listdir(pathToModels)):
        if os.path.splitext(model)[-1] == '.features_bin':
            tBeg = time.time()
            print('Model', i+1)
            features = getFeatures(os.path.join(pathToModels, model))
            newMeans = getNewMeans(ubm, features, 16)
            pathToNewUbm = os.path.join(pathToUbms, os.path.splitext(model)[0]) + '.gmm'
            saveUbm(pathToNewUbm, pathToUbm, newMeans)
            deltaTime = time.time() - tBeg
            print('\tHandled for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
        break
    deltaTimeAll = time.time() - tBegAll
    print('\nHandle all models for {0:.0f}m {1:.0f}s'.format(deltaTimeAll / 60, deltaTimeAll % 60))


if __name__ == '__main__':
    main()