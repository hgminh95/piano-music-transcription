# -*- coding: utf-8 -*-

from collections import defaultdict


class MetaTranscriber(type):

    method_to_transcriber = defaultdict(list)

    def __init__(self, name, bases, attrs):
        if self._name is not None:
            self.method_to_transcriber[self._name].append(self)

    def __new__(meta, name, bases, attrs):
        return type.__new__(meta, name, bases, attrs)


class Transcriber(object):

    __metaclass__ = MetaTranscriber
    _name = None
    _description = None

    @classmethod
    def transcribe(cls, audio):
        pass
