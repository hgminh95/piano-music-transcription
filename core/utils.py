# -*- coding: utf-8 -*-

import os
import fnmatch
import logging


_logger = logging.getLogger(__name__)


def wav_walk(path, recursive=True):
    for root, dirs, files in os.walk(path):
        for wav_name in fnmatch.filter(files, '*.wav'):
            wav_full = os.path.join(root, wav_name)

            yield wav_full[:-4]


def read_txt(path):
    index = 0
    res = []
    with open(path) as f:
        for line in f:
            if index != 0:
                data = line.rstrip().split('\t')
                res.append((float(data[0]), float(data[1]), int(data[2])))

            index += 1

    return res
