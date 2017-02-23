# -*- coding: utf-8 -*-

import librosa
import numpy as np
import mir_eval
import logging

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from core import transcriber
from core.utils import read_txt
import evaluation.algo as algo

_logger = logging.getLogger(__name__)


class SVMTranscriber(transcriber.Transcriber):

    _name = "librosa_onset"
    _description = "Dummy Transcriber"

    @classmethod
    def transcribe(cls, filename):
        expect = read_txt(filename + '.txt')

        y, sr = librosa.load(filename + '.wav')

        o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=o_env, sr=sr,
            pre_max=0.03 * sr // 512, post_max=0.00 * sr // 512 + 1,
            pre_avg=0.10 * sr // 512, post_avg=0.10 * sr // 512 + 1,
            delta=0.01, wait=0.02 * sr // 512)

        _logger.debug('Onset length = {0}'.format(len(onset_frames)))
        ans = librosa.frames_to_time(onset_frames, sr=sr)
        expect = algo.unique(map(lambda x: x[0], expect))

        print mir_eval.onset.f_measure(np.array(expect), np.array(ans))

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
        classifier = joblib.load(model)

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
            y = classifier.predict(X).astype(int)
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
    def construct(cls, filename, output):
        expect = read_txt(filename + '.txt')

        ans, D, onset_strength = SVMTranscriber.extract_features(filename + '.wav')

        for time, frame, notes in algo.leftjoin(ans, expect):
            frame = int(frame)
            if frame < 3:
                continue
            output.write("{0} {1}".format(time, frame))
            output.write('\n')
            features = D[:, frame - 2:frame + 3].ravel()
            features = np.append(features, onset_strength[frame - 2:frame + 3])
            output.write(' '.join(map(str, features)))
            output.write('\n')
            if len(notes) == 0:
                output.write('-1')
            else:
                output.write(' '.join(map(str, notes)))
            output.write('\n')

    @classmethod
    def extract_features(cls, filename):
        y, sr = librosa.load(filename)
        D = librosa.core.cqt(y, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36, real=False)
        D = np.abs(D)

        o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=o_env, sr=sr,
            pre_max=0.03 * sr // 512, post_max=0.00 * sr // 512 + 1,
            pre_avg=0.10 * sr // 512, post_avg=0.10 * sr // 512 + 1,
            delta=0.01, wait=0.02 * sr // 512)
        onset_frames = onset_frames.reshape(-1, 1)

        ans = librosa.frames_to_time(onset_frames, sr=sr)
        ans = ans.reshape(-1, 1)
        ans = np.hstack((ans, onset_frames))

        return ans, D, o_env

    @classmethod
    def obtain_data(cls, filename):
        piano_range = xrange(9, 96)
        X = []
        Y = []

        _logger.info('Loading train data from file')
        n_samples = 0
        n_features = 0
        with open(filename) as f:
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

        return X_train, X_test, y_train, y_test

    @classmethod
    def train(cls, filename, output):
        X_train, X_test, y_train, y_test = SVMTranscriber.obtain_data(filename)

        _logger.info('Training...')
        classifier = OneVsRestClassifier(LinearSVC())
        print y_train.shape
        classifier.fit(X_train, y_train)

        y_predict = classifier.predict(X_test).astype(int)

        print y_predict.shape
        print np.count_nonzero(y_test)
        print np.unique(y_test, return_counts=True)
        print np.unique(y_predict, return_counts=True)
        print np.unique(y_test - y_predict, return_counts=True)

        _logger.info("Save to file {0}".format(output))
        joblib.dump(classifier, output)
