# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np

from core import MetaExtractor, MetaModel
import core.util as util

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

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

    with open(args.output, 'w') as f:
        for sample in util.wav_walk(args.input):
            wav_path = sample + '.wav'
            txt_path = sample + '.txt'

            _logger.info("Extract feature from {0}".format(wav_path))

            data = extractor.extract(wav_path)
            truth = util.read_txt(txt_path)

            for (meta, feature), notes in util.leftjoin(data, truth):
                f.write("{0} {1}\n".format(meta[0], int(meta[1])))
                f.write(' '.join(map(str, feature)))
                f.write('\n')
                if len(notes) == 0:
                    f.write('-1')
                else:
                    f.write(' '.join(map(str, notes)))
                f.write('\n')

                total_samples += 1
                n_features = len(feature)

    _logger.info("Total samples: {0}".format(total_samples))
    _logger.info("Features per sample: {0}".format(n_features))


def train():
    piano_range = xrange(9, 96)
    X = []
    Y = []

    _logger.info('Loading train data from file')
    n_samples = 0
    n_features = 0
    with open(args.input, 'r') as f:
        while True:
            line = f.readline()
            if line is None or line == '':
                break
            line = f.readline()
            feature = map(float, line.rstrip().split(' '))
            n_features = len(feature)
            line = f.readline()
            notes = map(int, line.rstrip().split(' '))

            # Convert to sklearn format
            X.append(feature)

            classes = [0] * 88
            for note in notes:
                if note in piano_range:
                    classes[note - 9] = 1

            Y.append(classes)

            n_samples += 1

    _logger.info('Load {0} samples'.format(n_samples))

    X = np.array(X).reshape(-1, n_features)
    Y = np.array(Y, dtype=int).reshape(-1, 88)

    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0)

    _logger.info('Split into train and test set')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_pred = y_pred.astype(int)

    print "Precision: {0}".format(precision_score(y_test, y_pred, average='macro'))
    print "Recall: {0}".format(recall_score(y_test, y_pred, average='macro'))
    print "F1 score: {0}".format(f1_score(y_test, y_pred, average='macro'))

    print np.unique(y_test, return_counts=True)
    print np.unique(y_pred, return_counts=True)
    print np.unique(y_test - y_pred, return_counts=True)

    model.save(args.output)


def eval():
    pass


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
