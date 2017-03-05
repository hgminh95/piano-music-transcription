# -*- coding: utf-8 -*-

import os
import json


class MetaExtractor(type):

    method_to_extractor = {}

    def __init__(self, name, bases, attrs):
        if self._name is not None:
            self.method_to_extractor[self._name] = self

    def __new__(meta, name, bases, attrs):
        return type.__new__(meta, name, bases, attrs)


class Extractor(object):

    __metaclass__ = MetaExtractor
    _name = None
    _description = None

    def extract(cls, filename):
        pass


class MetaModel(type):

    method_to_model = {}

    def __init__(self, name, bases, attrs):
        if self._name is not None:
            self.method_to_model[self._name] = self

    def __new__(meta, name, bases, attrs):
        return type.__new__(meta, name, bases, attrs)


class Model(object):

    __metaclass__ = MetaModel
    _name = None
    _description = None

    def __init__(self):
        self.model = None
        self.parameters = {}

    def fit(self, X, y):
        pass

    def save(self, filename):
        pass

    def _save_metadata(self, filename):
        filename, ext = os.path.splitext(filename)

        with open(filename + '.meta', 'w') as f:
            json.dump(self.parameters, f, indent=4)

    def predict(self, inp):
        return self.model.predict(inp)

    def load(self, filename):
        pass

    def _load_metadata(self, filename):
        filename, ext = os.path.splitext(filename)

        with open(filename + '.meta', 'r') as f:
            self.parameters = json.load(f)
