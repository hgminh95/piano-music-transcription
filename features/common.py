# -*- coding: utf-8 -*-

import librosa
import numpy as np
import logging
import struct

import core
from core import util

_logger = logging.getLogger(__name__)


def getfeatures(D, o_env, frame, left=0, right=0):
    return np.append(
        np.lib.pad(
            D[:, frame - left:frame + right + 1],
            ((0, 0), (max(0, left - frame), max(0, - D.shape[1] + frame + right + 1))),
            'constant',
            constant_values=(0, 0)
        ).ravel(),
        np.lib.pad(
            o_env[frame - left: frame + right + 1],
            (max(0, left - frame), max(0, - o_env.shape[0] + frame + right + 1)),
            'constant',
            constant_values=(0, 0))
    )


class SignalExtractor(core.Extractor):

    def __init__(self):
        self.D = None
        self.o_env = None

    def transform(self, y, sr):
        return y

    def features_at(self, frame):
        return getfeatures(self.D, self.o_env, frame)

    def extract(self, filename, truth=None):
        data = self._extract(filename)

        if truth:
            truth = util.read_txt(truth)

            for (meta, feature), notes in util.leftjoin(data, truth):
                yield meta, feature, notes
        else:
            for meta, feature in data:
                yield meta, feature

    def _extract(self, filename):
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

        features = map(lambda x: self.features_at(int(x[1])), onset)
        features = np.array(features)

        mean = features.mean(axis=0)
        std = features.std(axis=0)

        features = (features - mean) / std

        return zip(onset, features)

    def dump(self, out, data):
        meta, features, notes = data

        out.write(struct.pack('>fi', meta[0], meta[1]))
        out.write(struct.pack('>i', len(features)))
        out.write(struct.pack('>' + 'f' * len(features), *(features.tolist())))
        out.write(struct.pack('>i', len(notes)))
        out.write(struct.pack('>' + 'i' * len(notes), *notes))

    def load(self, inp):
        meta_bytes = inp.read(8)
        if meta_bytes == b'':
            raise Exception("EOF")

        meta = struct.unpack_from('>fi', meta_bytes)
        features_len = struct.unpack_from('>i', inp.read(4))[0]
        features = struct.unpack_from('>' + 'f' * features_len, inp.read(4 * features_len))
        notes_len = struct.unpack_from('>i', inp.read(4))[0]
        notes = struct.unpack_from('>' + 'i' * notes_len, inp.read(4 * notes_len))

        piano_range = xrange(9, 96)
        classes = [0] * 88
        for note in notes:
            if note in piano_range:
                classes[note - 9] = 1

        return meta, features, classes


class STFT(SignalExtractor):

    _name = "stft"
    _description = "Short Time Fourier Transform"

    def transform(self, y, sr):
        return np.abs(librosa.core.stft(y))


class CQT(SignalExtractor):

    _name = "cqt"
    _description = "Constant Q transform"

    def transform(self, y, sr):
        return np.abs(
            librosa.core.cqt(y, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36))


class MultiSTFT(STFT):

    _name = "mul.stft"
    _description = "Short Time Fourier Transform with Multiple Frames"

    def features_at(self, frame):
        return getfeatures(self.D, self.o_env, frame, left=2, right=2)


class MultiCQT(CQT):

    _name = "mul.cqt"
    _description = "Constant Q Transform with Multiple Frames"

    def features_at(self, frame):
        return getfeatures(self.D, self.o_env, frame, left=2, right=2)
