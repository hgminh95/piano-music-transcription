# -*- coding: utf-8 -*-

import numpy as np
import mir_eval
import logging

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation

from sklearn.metrics import f1_score, precision_score, recall_score

from core import transcriber
from core.utils import read_txt
import evaluation.algo as algo
from dummy import SVMTranscriber

_logger = logging.getLogger(__name__)


class NeuralNetworkTranscriber(transcriber.Transcriber):

    _name = "nn"
    _description = "Neural Network Transcriber"

    @classmethod
    def eval(cls, filename, model):
        # Load ref from ground truth file
        _logger.info('Load ground truth value from file {0}'.format(filename + '.txt'))
        expect = read_txt(filename + '.txt')

        ref_intervals = map(lambda x: (x[0], x[1]), expect)
        ref_intervals = np.array(ref_intervals, dtype=float)

        ref_pitches = map(lambda x: mir_eval.util.midi_to_hz(x[2]), expect)
        ref_pitches = np.array(ref_pitches, dtype=float)

        # Load model
        _logger.info('Load model from file {0}'.format(model))
        classifier = load_model(model)

        # Use classifier
        _logger.info('Extract features')
        onset, D, onset_strength = SVMTranscriber.extract_features(filename + '.wav')

        _logger.info('Predicting...')
        ans = []
        for time, frame in onset:
            frame = int(frame)
            X = D[:, frame - 2:frame + 3].ravel()
            X = np.append(X, onset_strength[frame - 2:frame + 3])
            X = X.reshape(1, -1)
            y = classifier.predict(X)
            y[y > 0.5] = 1
            y[y <= 0.5] = 0
            y = y.astype(int)
            for i in xrange(88):
                if y[0][i] == 1:
                    ans.append((time, time + 0.2, i + 9))

        _logger.debug('Onset length = {0}'.format(len(onset)))
        _logger.debug('Est length = {0}'.format(len(ans)))

        est_intervals = map(lambda x: (x[0], x[1]), ans)
        est_intervals = np.array(est_intervals, dtype=float)

        est_pitches = map(lambda x: mir_eval.util.midi_to_hz(x[2]), ans)
        est_pitches = np.array(est_pitches, dtype=float)

        mir_eval.transcription.validate(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches)

        print len(mir_eval.transcription.match_notes(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,
            offset_ratio=None))
        print mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,
            offset_ratio=None)

    @classmethod
    def train(cls, filename, output):
        X_train, X_test, y_train, y_test = SVMTranscriber.obtain_data(filename)

        n_input = X_train.shape[1]
        n_output = y_train.shape[1]
        print n_input, n_output
        model = Sequential([
            Dense(100, input_dim=n_input),
            Activation('relu'),
            Dense(n_output),
            Activation('sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

        model.fit(X_train, y_train, nb_epoch=5)

        y_predict = model.predict(X_test)
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict <= 0.5] = 0
        y_predict = y_predict.astype(int)

        print "Precision: {0}".format(precision_score(y_test, y_predict, average='macro'))
        print "Recall: {0}".format(recall_score(y_test, y_predict, average='macro'))
        print "F1 score: {0}".format(f1_score(y_test, y_predict, average='macro'))

        print np.count_nonzero(y_test)
        print np.unique(y_test, return_counts=True)
        print np.unique(y_predict, return_counts=True)
        print np.unique(y_test - y_predict, return_counts=True)

        model.save(output)
