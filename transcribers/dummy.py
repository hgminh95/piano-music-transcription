# -*- coding: utf-8 -*-

import librosa
import numpy as np
import matplotlib.pyplot as plt
import mir_eval

from core import transcriber
from core.utils import read_txt
import evaluation.algo as algo


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
        D = librosa.core.cqt(y, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36, real=False)
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
