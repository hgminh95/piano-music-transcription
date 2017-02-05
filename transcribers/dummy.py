# -*- coding: utf-8 -*-

from core import transcriber


class DummyTranscriber(transcriber.Transcriber):

    _name = "dummy"
    _description = "Dummy Transcriber"


    @classmethod
    def transcribe(cls, audio):
        return None
