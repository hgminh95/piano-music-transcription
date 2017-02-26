# -*- coding: utf-8 -*-

import os
import fnmatch
import logging

import numpy as np
import mir_eval

_logger = logging.getLogger(__name__)


def incif(a, b):
    if a < b:
        return a + 1
    else:
        return a


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


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


def defaultgetkey(x):
    return x[0][0]


def leftjoin(ans, expect, window=0.05, getkey=defaultgetkey):
    ans_len = len(ans)
    expect_len = len(expect)

    i, j = 0, 0
    notes = []
    while i < ans_len and j < expect_len:
        if abs(getkey(ans[i]) - expect[j][0]) <= window:
            notes.append(expect[j][2])
            j = incif(j, expect_len)
        elif getkey(ans[i]) > expect[j][0]:
            j = incif(j, expect_len)
        else:
            yield ans[i], notes
            i = incif(i, ans_len)
            notes = []


def transform(res):
    intervals = np.array(map(lambda x: (x[0], x[1]), res), dtype=float)
    pitches = np.array(map(lambda x: mir_eval.util.midi_to_hz(x[2]), res), dtype=float)

    return intervals, pitches
