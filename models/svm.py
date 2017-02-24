# -*- coding: utf-8 -*-

import librosa
import numpy as np
import mir_eval
import logging

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import core

_logger = logging.getLogger(__name__)


class SVM(core.Model):

    _name = "svm"
