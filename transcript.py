# -*- coding: utf-8 -*-

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


def infinite_samples(loop=True):
    while True:
        with open(args.input, 'rb') as f:
            while True:
                try:
                    meta, feature, classes = extractor.load(f)
                    yield feature, classes
                except Exception:
                    break

        if not loop:
            break


def data_generator(loop=True):
    cnt = 0
    n_features = 0
    X = []
    Y = []
    for sample in infinite_samples(loop=loop):
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
    for X, Y in data_generator(loop=False):
        model.parameters['mean'] = X.mean(axis=0).tolist()
        model.parameters['std'] = X.std(axis=0).tolist()
        model.fit(X, Y)
        break

    model.save(args.output)


def eval():
    matched = 0
    total_ref = 0
    total_est = 0

    for sample in util.wav_walk(args.input):
        wav_path = sample + '.wav'
        txt_path = sample + '.txt'

        expect = util.read_txt(txt_path)
        ref_intervals, ref_pitches = util.transform(expect)

        model.load(args.modelfile)
        ans = []
        for (time, frame), feature in extractor.extract(wav_path):
            y = model.predict(feature.reshape(1, -1))

            for i in xrange(88):
                if y[0][i] == 1:
                    ans.append((time, time + 0.2, i + 9))

        if len(ans) > 3 * len(expect):
            _logger.warn("Precision is too low (< 33%)")
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

    precision = 1.0 * matched / total_est
    recall = 1.0 * matched / total_ref
    f1_measure = 2 * (precision * recall) / (precision + recall)

    _logger.info("Total ref/Total est: {}/{}".format(total_ref, total_est))
    _logger.info("Matched: {}".format(matched))
    _logger.info("Precision: {0}".format(precision))
    _logger.info("Recall: {0}".format(recall))
    _logger.info("F1 measure: {0}".format(f1_measure))


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
