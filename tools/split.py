# -*- coding: utf-8 -*-

import os
import sys
import urllib2
import shutil

sys.path.append('./')

from core import util


folds = [
    ('http://www.eecs.qmul.ac.uk/~sss31/TASLP/train_fold_1.txt', 'http://www.eecs.qmul.ac.uk/~sss31/TASLP/test_fold_1.txt'),
    ('http://www.eecs.qmul.ac.uk/~sss31/TASLP/train_fold_2.txt', 'http://www.eecs.qmul.ac.uk/~sss31/TASLP/test_fold_2.txt'),
    ('http://www.eecs.qmul.ac.uk/~sss31/TASLP/train_fold_3.txt', 'http://www.eecs.qmul.ac.uk/~sss31/TASLP/test_fold_3.txt'),
    ('http://www.eecs.qmul.ac.uk/~sss31/TASLP/train_fold_4.txt', 'http://www.eecs.qmul.ac.uk/~sss31/TASLP/test_fold_4.txt')
]


def contains(list, file):
    for name in list:
        if name in file:
            return True
    return False


def copyall(input, output, files):
    if not os.path.exists(output):
        os.makedirs(output)

    for sample in util.wav_walk(input):
        if 'MAPS_MUS' not in sample:
            continue

        if contains(files, sample + '.wav'):
            print os.path.split(sample)
            filename = os.path.split(sample)[1]
            shutil.copyfile(sample + '.wav', os.path.join(output, filename + '.wav'))
            shutil.copyfile(sample + '.txt', os.path.join(output, filename + '.txt'))


def split(input, output):
    i = 0
    for fold in folds:
        i += 1
        train = urllib2.urlopen(fold[0]).read()
        copyall(input, os.path.join(output, 'fold' + str(i), 'train'), train.split('\n'))

        test = urllib2.urlopen(fold[1]).read()
        copyall(input, os.path.join(output, 'fold' + str(i), 'test'), test.split('\n'))


if __name__ == '__main__':
    split(sys.argv[1], sys.argv[2])
