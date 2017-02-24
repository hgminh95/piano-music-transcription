# -*- coding: utf-8 -*-

import numpy as np
import logging

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation

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
            Dense(n_output),
            Activation('sigmoid')
        ])
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

        self.model.fit(X, y, nb_epoch=5)

    def save(self, output):
        self.model.save(output)

    def predict(self, inp):
        return self.model.predict(inp)
