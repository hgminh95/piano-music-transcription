# -*- coding: utf-8 -*-

import os
import fnmatch
import logging


_logger = logging.getLogger(__name__)


def wav_walk(path, recursive=True, wavonly=False):
    for root, dirs, files in os.walk(path):
        for wav_name in fnmatch.filter(files, '*.wav'):
            mid_name = wav_name[:-4] + '.mid'

            mid_full = os.path.join(root, mid_name)
            wav_full = os.path.join(root, wav_name)

            if wavonly:
                yield wav_full
            else:
                if os.path.exists(mid_full):
                    yield (wav_full, mid_full)
