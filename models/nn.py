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
        n_input = X.shape[1]
        n_output = y.shape[1]

        self.model = Sequential([
            Dense(200, input_dim=n_input),
            Activation('relu'),
            Dense(200),
            Activation('relu'),
            Dense(n_output),
            Activation('sigmoid')
        ])
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

        self.model.fit(X, y, nb_epoch=5)

    def save(self, output):
        self.model.save(output)

    def predict(self, X):
        y = self.model.predict(X)
        return y
        # y[y > 0.5] = 1
        # y[y <= 0.5] = 0

        # return y.astype(int)

    def load(self, filename):
        self.model = load_model(filename)
