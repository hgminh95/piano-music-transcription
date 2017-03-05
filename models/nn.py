# -*- coding: utf-8 -*-

import numpy as np
import logging

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import core

_logger = logging.getLogger(__name__)


class NeuralNetwork(core.Model):

    _name = "nn"

    def fit(self, X, y):
        if self.model is None:
            self._build_model(X.shape[1], y.shape[1])

        self.model.fit(X, y, nb_epoch=5)

    def fit_generator(self, generator):
        X, y = generator().next()
        self._build_model(X.shape[1], y.shape[1])

        self.model.fit_generator(generator(), samples_per_epoch=50000, nb_epoch=20)

    def save(self, output):
        self.model.save(output)

        self._save_metadata(output)

    def predict(self, X):
        y = self.model.predict(X)

        y[y > 0.3] = 1
        y[y <= 0.3] = 0

        return y.astype(int)

    def load(self, filename):
        self.model = load_model(filename)

        self._load_metadata(filename)

    def _build_model(self, n_input, n_output):
        print "Build model with n_input = {}, n_output = {}".format(n_input, n_output)
        self.model = Sequential([
            Dense(200, input_dim=n_input),
            # Activation('relu'),
            # Dense(200),
            Activation('relu'),
            Dense(n_output),
            Activation('sigmoid')
        ])
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])
