# -*- coding: utf-8 -*-

import numpy as np
import logging

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD

import nn


class ConvolutionNeuralNetwork(nn.NeuralNetwork):

    _name = 'cnn'
    _description = 'Convolution Neural Network'

    def __init__(self):
        super(ConvolutionNeuralNetwork, self).__init__()

        self.parameters['optimizer'] = 'adagrad'
        self.parameters['epoch'] = 3
        self.parameters['layer'] = 2
        self.parameters['unit'] = 200

    def fit(self, X, y):
        X = self._transform(X)
        X = X.reshape(X.shape + (1, ))

        if self.model is None:
            self._build_model((X.shape[1], X.shape[2], 1), y.shape[1])

        self.model.fit(X, y, epochs=self.parameters['epoch'])

    def predict(self, X):
        X = self._transform(X)
        X = X.reshape(X.shape + (1, ))

        y = self.model.predict(X)

        y[y > self.parameters['threshold']] = 1
        y[y <= self.parameters['threshold']] = 0

        return y.astype(int)

    def _build_model(self, input_shape, n_output):
        nb_filter = 25
        kernel_size = (2, 2)

        self.model = Sequential()

        self.model.add(
            Conv2D(
                nb_filter, kernel_size,
                data_format='channels_last', input_shape=input_shape,
                activation=self.parameters['activation']))

        self.model.add(
            Conv2D(
                nb_filter, kernel_size,
                activation=self.parameters['activation']))

        self.model.add(MaxPooling2D())
        self.model.add(Flatten())

        for i in xrange(self.parameters['layer']):
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
            metrics=[])

    def _transform(self, X):
        for feature_per_frame in [253, 11111]:
            if X.shape[1] % feature_per_frame != 0:
                continue

            frame = X.shape[1] / feature_per_frame
            if frame not in range(1, 9, 2):
                continue

            newX = []
            for x in X:
                feature = []
                for i in xrange(frame):
                    feature.append(
                        np.concatenate([
                            [x[-(i + 1)]],
                            x[i * feature_per_frame:(i + 1) * feature_per_frame]]))

                newX.append(feature)

            newX = np.array(newX)
            return newX
