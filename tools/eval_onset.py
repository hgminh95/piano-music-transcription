# -*- coding: utf-8 -*-

import sys
import mir_eval
import librosa
import numpy as np

sys.path.append('./')

from core import util


if __name__ == '__main__':
    for sample in util.wav_walk(sys.argv[1]):
        if 'MAPS_MUS' not in sample:
            continue

        txt_path = sample + '.txt'
        wav_path = sample + '.wav'

        # Truth
        expect = util.read_txt(txt_path)
        ref_intervals, _ = util.transform(expect)
        ref = np.unique(map(lambda x: x[0], ref_intervals))
        print ref.shape

        # Sample
        y, sr = librosa.load(wav_path)

        o_env = librosa.onset.onset_strength(y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=o_env, sr=sr,
            pre_max=0.02 * sr // 512,                # 30ms
            post_max=0.00 * sr // 512 + 1,           # 0ms
            pre_avg=0.05 * sr // 512,                # 100ms
            post_avg=0.00 * sr // 512 + 1,           # 100ms
            delta=0.0005,
            wait=0.005 * sr // 512)                  # 30ms
        onset_frames = onset_frames.reshape(-1, 1)

        onset_time = librosa.frames_to_time(onset_frames, sr=sr)
        onset_time = onset_time.reshape(-1)

        print onset_time.shape
        print sample
        print mir_eval.onset.evaluate(ref, onset_time)
        print "________"
