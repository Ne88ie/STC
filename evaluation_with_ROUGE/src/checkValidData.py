__author__ = 'moiseeva'

import os

def checkValidData(pathToDirData):
    """
    Checks the files in the folder to conform specified format.
    """
    for file in os.listdir(pathToDirData):
        pathToFile = os.path.normpath(os.path.join(pathToDirData, file))
        with open(pathToFile, encoding='utf-8') as text:
            if not text.read().strip():
                print('Exception:\n\tFile "{0}" is empty.'.format(pathToFile))
                continue
            for line in text:
                if line.strip():
                    substr = line.split(',')
                    if len(substr) != 2:
                        print('Exception:\n\tFile "{0}" has an invalid format line (problem with commas): {1}'.format(pathToFile, line))
                    elif not substr[1].strip().isdecimal():
                        print('Exception:\n\tFile "{0}" has an invalid format line (problem with keyword rank): {1}'.format(pathToFile, line))
                    substr = substr[0].split()
                    for word in substr:
                        if word[1:].find('*') != -1:
                            print('Exception:\n\tFile "{0}" has an invalid format line (problem with asterisk): {1}'.format(pathToFile, line))


if __name__ == '__main__':
    pathToDirData1 = r'C:\Users\moiseeva\PycharmProjects\evaluation\data\txt\kiseleva'
    pathToDirData2 = r'C:\Users\moiseeva\PycharmProjects\evaluation\data\txt\moiseeva'
    p = r'C:\Users\moiseeva\SElabs\textrank\evaluation\data\txt\kiseleva'
    pathToDirData3 = r'C:\Users\moiseeva\PycharmProjects\evaluation\data\txt\denisov'
    for path in (pathToDirData1, pathToDirData2, pathToDirData3, p):
        checkValidData(path)