# -*- coding: utf-8 -*-

import numpy as np
import logging

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Dropout

import core

_logger = logging.getLogger(__name__)


class NeuralNetwork(core.Model):

    _name = "nn"
    _description = "Neural Network"

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.parameters = {
            'layer': 2,
            'unit': 150,
            'dropout': 0.0,
            'activation': 'relu',
            'optimizer': 'sgd',
            'mean': None,
            'std': None,
            'threshold': 0.5,
            'epoch': 5,
        }

    def fit(self, X, y):
        if self.model is None:
            self._build_model(X.shape[1], y.shape[1])

        if self.parameters['mean'] is not None and self.parameters['std'] is not None:
            X = (X - self.parameters['mean']) / self.parameters['std']

        self.model.fit(X, y, nb_epoch=self.parameters['epoch'])

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
        self.model = Sequential()

        self.model.add(
            Dense(
                self.parameters['unit'],
                input_dim=n_input, activation=self.parameters['activation']))

        for i in xrange(self.parameters['layer'] - 1):
            self.model.add(
                Dense(
                    self.parameters['unit'],
                    activation=self.parameters['activation']))
            if self.parameters['dropout'] > 0.0001:
                self.model.add(
                    Dropout(self.parameters['dropout']))

        self.model.add(
            Dense(n_output, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=self.parameters['optimizer'],
            metrics=['precision', 'recall', 'fmeasure'])
