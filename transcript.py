# -*- coding: utf-8 -*-

import os
import argparse
import logging
import random

import numpy as np
import mir_eval

from core import MetaExtractor, MetaModel
import core.util as util

import models
import features

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s\t %(filename)s\t %(message)s")

_logger = logging.getLogger(__name__)

extractor = None
model = None
args = None


def construct():
    total_samples = 0
    n_features = 0

    with open(args.output, 'wb') as f:
        for sample in util.wav_walk(args.input):
            if 'MAPS_MUS' not in sample:
                continue

            wav_path = sample + '.wav'
            txt_path = sample + '.txt'

            _logger.info("Extract feature from {0}".format(wav_path))

            for meta, feature, notes in extractor.extract(wav_path, truth=txt_path):
                extractor.dump(f, (meta, feature, notes))

                total_samples += 1
                n_features = len(feature)

    _logger.info("Total samples: {0}".format(total_samples))
    _logger.info("Features per sample: {0}".format(n_features))


def infinite_samples(input, loop=True):
    while True:
        with open(input, 'rb') as f:
            while True:
                try:
                    meta, feature, classes = extractor.load(f)
                    yield feature, classes
                except Exception:
                    break

        if not loop:
            break


def data_generator(input, loop=True):
    cnt = 0
    n_features = 0
    X = []
    Y = []
    for sample in infinite_samples(input, loop=loop):
        if cnt % 40000 == 0:
            if cnt > 0:
                seed = random.randint(0, 1000)
                X = np.array(X).reshape(-1, n_features)
                Y = np.array(Y, dtype=int).reshape(-1, 88)
                np.random.seed(seed)
                np.random.shuffle(X)
                np.random.seed(seed)
                np.random.shuffle(Y)

                yield X, Y
            X = []
            Y = []

        feature, classes = sample
        n_features = len(feature)
        X.append(feature)
        Y.append(classes)

        cnt += 1

    if len(X) > 0:
        seed = random.randint(0, 1000)
        X = np.array(X).reshape(-1, n_features)
        Y = np.array(Y, dtype=int).reshape(-1, 88)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(Y)

        yield X, Y


def train():
    # Calculate mean and std in entire collection
    mean = 0
    std = 0
    for X, Y in data_generator(args.input, loop=False):
        # Only take the first batch, will be fixed later
        mean = X.mean(axis=0).tolist()
        std = X.std(axis=0).tolist()
        break

    model.parameters['mean'] = mean
    model.parameters['std'] = std

    for X, Y in data_generator(args.input, loop=False):
        model.fit(X, Y)

    if args.test:
        max_matched = 0
        best_threshold = 0.1
        for threshold in np.arange(0.1, 1.0, 0.1):
            model.parameters['threshold'] = threshold

            matched = 0
            for X, y in data_generator(args.test, loop=False):
                y_pred = model.predict(X)

                matched += (np.count_nonzero(y_pred) + np.count_nonzero(y) - np.count_nonzero(y_pred - y)) / 2

            if matched > max_matched:
                best_threshold = threshold
                max_matched = matched

        model.parameters['threshold'] = best_threshold

        _logger.info("Best threshold: {} - matched: {}".format(best_threshold, max_matched))

    model.save(args.output)


def eval():
    matched = 0
    total_ref = 0
    total_est = 0

    model.load(args.modelfile)

    if os.path.isdir(args.test):
        for sample in util.wav_walk(args.test):
            wav_path = sample + '.wav'
            txt_path = sample + '.txt'

            expect = util.read_txt(txt_path)
            ref_intervals, ref_pitches = util.transform(expect)

            ans = []
            for (time, frame), feature in extractor.extract(wav_path):
                if len(feature) != 1265:
                    continue
                y = model.predict(feature.reshape(1, -1))

                for i in xrange(88):
                    if y[0][i] == 1:
                        ans.append((time, time + 0.2, i + 9))

            if len(ans) > 3 * len(expect):
                _logger.warn("Precision is too low (< 33%) ans: {}, expect: {}".format(len(ans), len(expect)))
                continue

            est_intervals, est_pitches = util.transform(ans)

            if len(ans) > 0:
                matched += len(mir_eval.transcription.match_notes(
                    ref_intervals, ref_pitches,
                    est_intervals, est_pitches,
                    onset_tolerance=0.05,
                    offset_ratio=None))

            total_ref += len(ref_intervals)
            total_est += len(est_intervals)
    else:
        for X, y in data_generator(args.test, loop=False):
            y_pred = model.predict(X)

            total_est += np.count_nonzero(y_pred)
            total_ref += np.count_nonzero(y)
            matched += (np.count_nonzero(y_pred) + np.count_nonzero(y) - np.count_nonzero(y_pred - y)) / 2

    _logger.info("Result")
    util.score(matched, total_est, total_ref)


def transcribe():
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Piano Transcription')
    parser.add_argument(
        'action',
        choices=['construct', 'train', 'eval', 'transcribe'],
        help='Action perform',
        default='train')
    parser.add_argument(
        '-e', '--extract',
        help='Feature extraction method',
        default='mul.cqt')
    parser.add_argument(
        '-m', '--model',
        help='Model used',
        default='nn')
    parser.add_argument(
        '-mf', '--modelfile',
        help='Model file to load')
    parser.add_argument(
        '-i', '--input',
        help='Input')
    parser.add_argument(
        '-o', '--output',
        help='Output')
    parser.add_argument(
        '-t', '--test',
        help='Test data')

    args = parser.parse_args()

    extractor = MetaExtractor.method_to_extractor[args.extract]()
    model = MetaModel.method_to_model[args.model]()

    _logger.info("Extractor: {}".format(extractor._description))
    _logger.info("Model: {}".format(model._description))

    if args.action == 'construct':
        construct()
    elif args.action == 'train':
        train()
    elif args.action == 'eval':
        eval()
    elif args.action == 'transcribe':
        transcribe()
