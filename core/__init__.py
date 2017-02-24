# -*- coding: utf-8 -*-


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

    def fit(self, X, y):
        pass

    def save(self, filename):
        self.model.save(filename)

    def predict(self, inp):
        return self.model.predict(inp)
