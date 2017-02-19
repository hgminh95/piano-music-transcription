# -*- coding: utf-8 -*-

import librosa
import numpy as np
import mir_eval
import logging

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

from core import transcriber
from core.utils import read_txt
import evaluation.algo as algo

_logger = logging.getLogger(__name__)


class LibrosaOnsetDectector(transcriber.Transcriber):

    _name = "librosa_onset"
    _description = "Dummy Transcriber"

    @classmethod
    def transcribe(cls, filename):
        expect = read_txt(filename + '.txt')

        y, sr = librosa.load(filename + '.wav')

        o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

        ans = librosa.frames_to_time(onset_frames, sr=sr)
        expect = algo.unique(map(lambda x: x[0], expect))

        print mir_eval.onset.f_measure(np.array(expect), np.array(ans))

    @classmethod
    def construct(cls, filename, output):
        expect = read_txt(filename + '.txt')

        y, sr = librosa.load(filename + '.wav')
        # D = librosa.core.cqt(y, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36, real=False)
        D = librosa.core.stft(y)
        D = np.abs(D)
        print D.shape

        o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        onset_frames = onset_frames.reshape(-1, 1)

        ans = librosa.frames_to_time(onset_frames, sr=sr)
        ans = ans.reshape(-1, 1)
        ans = np.hstack((ans, onset_frames))

        for time, frame, notes in algo.leftjoin(ans, expect):
            frame = int(frame)
            output.write("{0} {1}".format(time, frame))
            output.write('\n')
            output.write(' '.join(map(str, D[:, frame])))
            output.write('\n')
            if len(notes) == 0:
                output.write('-1')
            else:
                output.write(' '.join(map(str, notes)))
            output.write('\n')

    @classmethod
    def train(cls, filename, output):
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

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(Y)

        X = X - X.mean(axis=0, keepdims=True)
        X = X / X.std(axis=0)

        _logger.info('Split into train and test set')
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)

        _logger.info('Training...')
        classifier = OneVsRestClassifier(LinearSVC())
        classifier.fit(X_train, y_train)

        y_predict = classifier.predict(X_test).astype(int)

        print np.count_nonzero(y_test)
        print np.unique(y_test, return_counts=True)
        print np.unique(y_predict, return_counts=True)
        print np.unique(y_test - y_predict, return_counts=True)

        _logger.info("Save to file {0}".format(output))
        joblib.dump(classifier, output)
