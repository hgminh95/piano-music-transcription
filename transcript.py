# -*- coding: utf-8 -*-

import argparse
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import mir_eval

from sklearn.metrics import matthews_corrcoef

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


def train():
    X = []
    Y = []

    _logger.info('Loading train data from file')
    n_samples = 0
    n_features = 0
    with open(args.input, 'rb') as f:
        while True:
            try:
                meta, feature, classes = extractor.load(f)
            except Exception:
                break

            X.append(feature)
            Y.append(classes)

            n_features = len(feature)
            n_samples += 1

    _logger.info('Load {0} samples'.format(n_samples))

    X = np.array(X).reshape(-1, n_features)
    Y = np.array(Y, dtype=int).reshape(-1, 88)

    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0)

    _logger.info('Split into train and test set')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)

    model.fit(X_train, y_train)

    threshold = np.arange(0.1, 0.9, 0.1)

    out = model.predict(X_test)

    acc = []
    accuracies = []
    best_threshold = np.zeros(out.shape[1])
    for i in range(out.shape[1]):
        y_prob = np.array(out[:, i])
        for j in threshold:
            y_pred = [1 if prob >= j else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(y_test[:, i], y_pred))
        acc = np.array(acc)
        index = np.where(acc == acc.max())
        accuracies.append(acc.max())
        best_threshold[i] = threshold[index[0][0]]
        acc = []

    print best_threshold
    print accuracies

    y_pred = np.array(
        [[1 if out[i, j] >= best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

    print "Precision: {0}".format(precision_score(y_test, y_pred, average='macro'))
    print "Recall: {0}".format(recall_score(y_test, y_pred, average='macro'))
    print "F1 score: {0}".format(f1_score(y_test, y_pred, average='macro'))

    print np.unique(y_test, return_counts=True)
    print np.unique(y_pred, return_counts=True)
    print np.unique(y_test - y_pred, return_counts=True)

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
        data = extractor.extract(wav_path)
        ans = []
        for (time, frame), feature in data:
            y = model.predict(feature.reshape(1, -1))
            y[y > 0.4] = 1
            y[y <= 0.4] = 0

            for i in xrange(88):
                if y[0][i] == 1:
                    ans.append((time, time + 0.2, i + 9))
        print len(data)
        est_intervals, est_pitches = util.transform(ans)

        matched += len(mir_eval.transcription.match_notes(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,
            offset_ratio=None))

        total_ref += len(ref_intervals)
        total_est += len(est_intervals)

    _logger.info((matched, total_ref, total_est))
    _logger.info("Precision: {0}".format(1.0 * matched / total_est))
    _logger.info("Recall: {0}".format(1.0 * matched / total_ref))


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

    print extractor
    print model

    if args.action == 'construct':
        construct()
    elif args.action == 'train':
        train()
    elif args.action == 'eval':
        eval()
    elif args.action == 'transcribe':
        transcribe()
