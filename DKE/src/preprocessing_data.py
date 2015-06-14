# coding=utf-8
from __future__ import print_function

from collections import defaultdict, OrderedDict, namedtuple
import os
from hashlib import md5
from shutil import copyfile, rmtree
from __init__ import open_write, open_read
from test_files_to_catalogs import test_files, plane_test_files, map_base_name_to_number

__author__ = 'annie'


def strip_prefix(text):
    prefix = u'Sentence='
    if text.startswith(prefix):
        return text[len(prefix):].strip()


def get_one_text(file_asking, file_answer):
    with open_read(file_asking) as asking,\
         open_read(file_answer) as answer:
        return strip_prefix(asking.read()) + u' ' + strip_prefix(answer.read())


def merge_two_speakers_into_one_file_in_dir(speakers_dir, result_speakers_dir):
    """
    слить двух дикторов в один файл
    :param speakers_dir: папка с парными файлами
    """
    files = defaultdict(list)

    asking = '_l.txt'
    answer = '_r.txt'
    for file_name in os.listdir(speakers_dir):
        if file_name.endswith(asking) or file_name.endswith(answer):
            files[file_name[:-len(asking)]].append(file_name)
    files = {k: sorted(v, key=lambda s: s[-len(asking):]) for k, v in files.iteritems() if len(v) == 2}

    if os.path.exists(result_speakers_dir):
        rmtree(result_speakers_dir)
    os.mkdir(result_speakers_dir)

    for name, two_files in files.iteritems():
        file_asking = os.path.join(speakers_dir, two_files[0])
        file_answer = os.path.join(speakers_dir, two_files[1])
        one_text = get_one_text(file_asking, file_answer)
        if one_text.strip() != u'':
            with open_write(os.path.join(result_speakers_dir, name + '.txt')) as f:
                f.write(one_text)


def merge_all_dirs(base_dir, result_dir):
    if os.path.exists(result_dir):
        rmtree(result_dir)
    os.mkdir(result_dir)

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            result_dir_path = os.path.join(result_dir, dir_name)
            merge_two_speakers_into_one_file_in_dir(dir_path, result_dir_path)


def get_hash(file_path):
    with open_read(file_path) as f:
        text = f.read()[:50].rstrip().encode('utf-8')
        return md5(text).hexdigest()

TFile = namedtuple('TFile', ['hash', 'name'])

def build_map_file_to_catalog(dir_all_ctalogs):
    '''
    строим отбражение для хеш от файла в кактлог - откуда этот файл
    :param dir_all_ctalogs: путь до базы с папками-каталогами
    :return: словарь
    '''
    map_catalog_to_files = defaultdict(list)
    for dir_name in os.listdir(dir_all_ctalogs):
        dir_path = os.path.join(dir_all_ctalogs, dir_name)
        if os.path.isdir(dir_path):
            files = [TFile(get_hash(os.path.join(dir_path, file_name)), file_name)
                     for file_name in os.listdir(dir_path)
                     if file_name.endswith('.txt')]
            map_catalog_to_files[dir_name].extend(files)

    map_file_to_catalog = {}
    for catalog, files in map_catalog_to_files.iteritems():
        for file_ in files:
            map_file_to_catalog[file_.hash] = (catalog, file_.name)

    return map_file_to_catalog


class Catalog:
    def __init__(self, dir_all_ctalogs=None):
        if not dir_all_ctalogs:
            dir_all_ctalogs = '../data2/Base_RGD'
        self.map_file_to_catalog = build_map_file_to_catalog(dir_all_ctalogs)

    def find_catalog(self, file_path):
        return self.map_file_to_catalog.get(get_hash(file_path))

    def find_catalog_for_all_files(self, dir_path):
        map_files_to_catalogs = OrderedDict()
        for file_name in sorted(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file_name)
            map_files_to_catalogs[file_name] = self.find_catalog(file_path)

        return map_files_to_catalogs


def copy_files_for_training(base_dir, training_dir):
    if os.path.exists(training_dir):
        rmtree(training_dir)
    os.mkdir(training_dir)

    for catalog_name in os.listdir(base_dir):
        catalog_path = os.path.join(base_dir, catalog_name)
        if os.path.isdir(catalog_path):
            training_catalog_path = os.path.join(training_dir, catalog_name)
            if not os.path.exists(training_catalog_path):
                os.mkdir(training_catalog_path)
            for file_name in os.listdir(catalog_path):
                if file_name.decode('utf-8') not in test_files:
                    file_path = os.path.join(catalog_path, file_name)
                    copyfile(file_path, os.path.join(training_catalog_path, file_name))


def copy_files_for_testing(base_dir, testing_dir):
    if os.path.exists(testing_dir):
        rmtree(testing_dir)
    os.mkdir(testing_dir)

    for catalog_name in os.listdir(base_dir):
        catalog_path = os.path.join(base_dir, catalog_name)
        if os.path.isdir(catalog_path):
            for file_name in os.listdir(catalog_path):
                if file_name.decode('utf-8') in test_files:
                    file_path = os.path.join(catalog_path, file_name)
                    file_new_path = os.path.join(testing_dir, catalog_name + '_' + file_name)
                    copyfile(file_path, file_new_path)


def copy_plane_files_for_base(base_dir, testing_dir):
    if os.path.exists(testing_dir):
        rmtree(testing_dir)
    os.mkdir(testing_dir)

    for catalog_name in os.listdir(base_dir):
        catalog_path = os.path.join(base_dir, catalog_name)
        if os.path.isdir(catalog_path):
            for file_name in os.listdir(catalog_path):
                file_path = os.path.join(catalog_path, file_name)
                file_new_path = os.path.join(testing_dir, catalog_name + '_' + file_name)
                copyfile(file_path, file_new_path)


def copy_plane_files_for_testing(base_dir, testing_dir):
    if os.path.exists(testing_dir):
        rmtree(testing_dir)
    os.mkdir(testing_dir)

    for file_name in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file_name)
        if file_name.decode('utf-8') in plane_test_files:
            file_new_path = os.path.join(testing_dir, map_base_name_to_number.get(file_name.decode('utf-8')))
            copyfile(file_path, file_new_path)


if __name__ == '__main__':
    """-----------------Подготовка данных из базы------------------"""
    '''сливаем файлы спикеров'''
    # merge_all_dirs('/Users/annie/SELabs/data/utf_Base_RGD', BASE_DIR)

    '''ищем из каких каталогов файлы (см. DKE/src2/test_files_to_catalogs.py)'''
    # catalog = Catalog(BASE_DIR)
    # files_to_catalogs = catalog.find_catalog_for_all_files(FILES_FOR_EVALUATIONS)
    # for k, v in files_to_catalogs.iteritems():
    #     print("u'{0}': (u'{1}', u'{2}'),".format(k, *v))

    '''отбираем файлы для обучения'''
    # copy_files_for_training(BASE_DIR, TRAINING_DIR)
    """-------------------------------------------------------------"""