# -*- coding: utf-8 -*-

import argparse
import logging

from core.transcriber import MetaTranscriber
import transcribers

logging.basicConfig(level=logging.DEBUG,
                    format="%(lineno)d in %(filename)s: %(message)s")

_logger = logging.getLogger(__name__)


def do_transcribe(args):
    for name, transcriber in MetaTranscriber.method_to_transcriber.iteritems():
        if name == args.method:
            _logger.info(name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Piano Transcription')
    subparsers = parser.add_subparsers(help='commands')

    # Parser for transcribe command
    transcribe_parser = subparsers.add_parser(
        'transcribe',
        help='Transcribe audio music to midi file')
    transcribe_parser.add_argument(
        '-m', '--method',
        help='method used to transcribe music',
        default='dummy')
    transcribe_parser.add_argument(
        '-i', '--input',
        help='input file or folder')
    transcribe_parser.add_argument(
        '-o', '--output',
        help='output file or folder')
    transcribe_parser.set_defaults(sub='transcribe')

    args = parser.parse_args()

    if args.sub == 'transcribe':
        do_transcribe(args)
    else:
        pass
