# -*- coding: utf-8 -*-

import librosa
import numpy as np
import logging

import core

_logger = logging.getLogger(__name__)


def getfeatures(D, o_env, frame, left=0, right=0):
    return np.append(
        D[:, frame - left:frame + right + 1].ravel(),
        o_env[frame - left: frame + right + 1])


class SignalExtractor(core.Extractor):

    def __init__(self):
        self.D = None
        self.o_env = None

    def transform(self, y, sr):
        return y

    def features_at(self, frame):
        return getfeatures(self.D, self.o_env, frame)

    def extract(self, filename):
        y, sr = librosa.load(filename)
        self.D = self.transform(y, sr)

        self.o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=self.o_env, sr=sr,
            pre_max=0.03 * sr // 512, post_max=0.00 * sr // 512 + 1,
            pre_avg=0.10 * sr // 512, post_avg=0.10 * sr // 512 + 1,
            delta=0.01, wait=0.02 * sr // 512)
        onset_frames = onset_frames.reshape(-1, 1)

        onset_time = librosa.frames_to_time(onset_frames, sr=sr)
        onset_time = onset_time.reshape(-1, 1)
        onset = np.hstack((onset_time, onset_frames))

        features = map(lambda x: (
            (x[0], x[1]),
            self.features_at(int(x[1]))
        ), onset)

        return features


class STFT(SignalExtractor):

    _name = "stft"

    def transform(self, y, sr):
        return np.abs(librosa.core.stft(y, sr=sr))


class CQT(SignalExtractor):

    _name = "cqt"

    def transform(self, y, sr):
        return np.abs(
            librosa.core.cqt(y, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36, real=False))


class MultiSTFT(STFT):

    _name = "mul.stft"

    def features_at(self, frame):
        return getfeatures(self.D, self.o_env, frame, left=2, right=2)


class MultiCQT(CQT):

    _name = "mul.cqt"

    def features_at(self, frame):
        return getfeatures(self.D, self.o_env, frame, left=2, right=2)
