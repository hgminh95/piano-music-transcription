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

        # D = np.abs(librosa.stft(y)) ** 2
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
        #                          x_axis='time', y_axis='log')
        # plt.title('Power spectrogram')
        # plt.subplot(2, 1, 2)
        # plt.plot(o_env, label='Onset strength')
        # plt.vlines(onset_frames, 0, o_env.max(), color='r', alpha=0.9,
        #            linestyle='--', label='Onset')

        # plt.xticks([])
        # plt.axis('tight')
        # plt.legend(frameon=True, framealpha=0.75)

        # plt.show()

    @classmethod
    def construct(cls, filename):
        expect = read_txt(filename + '.txt')

        y, sr = librosa.load(filename + '.wav')

        o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

        ans = librosa.frames_to_time(onset_frames, sr=sr)
        ans = map(lambda x: (x, 0, 0), ans)

        for sample in algo.leftjoin(ans, expect):
            print sample
