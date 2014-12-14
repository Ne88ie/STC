from __future__ import print_function, division
import os
import time
from IOData import getUbm, getFeatures, saveUbm
from calculations import getNewMeans, getNewMeans3D, criterionNeymanPearson

__author__ = 'annie'

def main():
    pathToData = '/Users/annie/SELabs/Kudashev/Lab1/data'
    pathToModelsDir = os.path.join(pathToData, 'models')
    pathToTestsDir = os.path.join(pathToData, 'tests')
    pathToUbm = os.path.join(pathToData, 'ubm.gmm')
    pathToUbmsDir = os.path.join(pathToData, 'ubms')
    pathToProtocolsDir = os.path.join(pathToData, 'protocols')
    r = 20

    def handle(path=pathToModelsDir):
        tBegAll = time.time()
        ubm = getUbm(pathToUbm, preprocessing=True)
        if not os.path.exists(pathToUbmsDir):
            os.mkdir(pathToUbmsDir)
        for i, model in enumerate(os.listdir(path)):
            if os.path.splitext(model)[-1] == '.features_bin':
                tBeg = time.time()
                print('Model', i+1)
                features = getFeatures(os.path.join(path, model))
                newMeans = getNewMeans3D(ubm, features, r)
                pathToNewUbm = os.path.join(pathToUbmsDir, os.path.splitext(model)[0]) + '.gmm'
                saveUbm(pathToNewUbm, pathToUbm, newMeans)
                deltaTime = time.time() - tBeg
                print('\tHandled for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
        deltaTimeAll = time.time() - tBegAll
        print('\nHandle all models for {0:.0f}m {1:.0f}s'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    # handle()

    def compareProtocols(path=pathToProtocolsDir):
        tBegAll = time.time()
        for protocol in sorted(os.listdir(path), reverse=True):
            if os.path.splitext(protocol)[-1] == '.txt':
                i = 1
                with open(os.path.join(path, protocol)) as p:
                    with open(os.path.join(path, 'answers', os.path.splitext(protocol)[0] + '_answers.txt'), 'w') as answers:
                        for gmms in p:
                            if gmms:
                                tBeg = time.time()
                                print('{0} {1}'.format(os.path.splitext(protocol)[0][:-1].capitalize(), i))
                                modelSample, testSample = gmms.split()
                                pathToModelUbm = os.path.join(pathToUbmsDir, os.path.splitext(modelSample)[0] + '.gmm')
                                pathToTestFeatures = os.path.join(pathToTestsDir, os.path.splitext(testSample)[0] + '.features_bin')
                                modelUbm = getUbm(pathToModelUbm, preprocessing=True)
                                testFeatures = getFeatures(pathToTestFeatures)
                                answers.write(str(criterionNeymanPearson(modelUbm, testFeatures, r)) + '\n')
                                deltaTime = time.time() - tBeg
                                print('\tHandled for {0:.0f}m {1:.0f}s'.format(deltaTime / 60, deltaTime % 60))
                                i += 1
        deltaTimeAll = time.time() - tBegAll
        print('\nHandle all protocols for {0:.0f}m {1:.0f}s'.format(deltaTimeAll / 60, deltaTimeAll % 60))

    compareProtocols()


if __name__ == '__main__':
    main()
    pass