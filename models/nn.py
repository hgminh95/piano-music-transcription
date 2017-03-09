# -*- coding: utf-8 -*-

import numpy as np
import logging

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import core

_logger = logging.getLogger(__name__)


class NeuralNetwork(core.Model):

    _name = "nn"
    _description = "Neural Network"

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.parameters = {
            'layer': 1,
            'unit': 200,
            'dropout': 0.3,
            'activation': 'relu',
            'optimizer': 'sgd',
            'mean': None,
            'std': None,
            'threshold': 0.5,
        }

    def fit(self, X, y):
        if self.model is None:
            self._build_model(X.shape[1], y.shape[1])

        if self.parameters['mean'] is not None and self.parameters['std'] is not None:
            X = (X - self.parameters['mean']) / self.parameters['std']

        self.model.fit(X, y, nb_epoch=10)

    def fit_generator(self, generator):
        X, y = generator().next()
        self._build_model(X.shape[1], y.shape[1])

        self.model.fit_generator(generator(), samples_per_epoch=50000, nb_epoch=20)

    def save(self, output):
        self.model.save(output)

        self._save_metadata(output)

    def predict(self, X):
        if self.parameters['mean'] is not None and self.parameters['std'] is not None:
            X = (X - self.parameters['mean']) / self.parameters['std']

        y = self.model.predict(X)

        y[y > self.parameters['threshold']] = 1
        y[y <= self.parameters['threshold']] = 0

        return y.astype(int)

    def load(self, filename):
        self.model = load_model(filename)

        self._load_metadata(filename)

    def _build_model(self, n_input, n_output):
        _logger.debug("Build model with n_input = {}, n_output = {}".format(n_input, n_output))
        self.model = Sequential([
            Dense(125, input_dim=n_input),
            Activation('relu'),
            # Dropout(0.3),
            Dense(125),
            Activation('relu'),
            # Dropout(0.3),
            # Dense(125),
            # Activation('relu'),
            # Dropout(0.3),
            Dense(n_output),
            Activation('sigmoid')
        ])
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=sgd,
            metrics=['accuracy', 'precision', 'recall', 'fmeasure'])
