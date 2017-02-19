# -*- coding: utf-8 -*-


class MetaTranscriber(type):

    method_to_transcriber = {}

    def __init__(self, name, bases, attrs):
        if self._name is not None:
            self.method_to_transcriber[self._name] = self

    def __new__(meta, name, bases, attrs):
        return type.__new__(meta, name, bases, attrs)


class Transcriber(object):

    __metaclass__ = MetaTranscriber
    _name = None
    _description = None

    @classmethod
    def construct(cls, filename, output):
        pass

    @classmethod
    def transcribe(cls, filename):
        pass

    @classmethod
    def eval(cls, filename, model):
        pass

    @classmethod
    def train(cls, filename):
        pass
