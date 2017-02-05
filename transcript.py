# -*- coding: utf-8 -*-

import argparse
import logging

from core.transcriber import MetaTranscriber
from core import utils
import transcribers

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s\t %(filename)s\t %(message)s")

_logger = logging.getLogger(__name__)


def do_transcribe(args):
    # Load input file
    total_files = 0
    for elem in utils.wav_walk(args.input):
        total_files += 1
        _logger.debug('Found wav file: {0}'.format(elem))
    _logger.info('Found {0} sample(s)'.format(total_files))

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
        help='input file or folder',
        default='.')
    transcribe_parser.add_argument(
        '-o', '--output',
        help='output file or folder')
    transcribe_parser.set_defaults(sub='transcribe')

    args = parser.parse_args()

    if args.sub == 'transcribe':
        do_transcribe(args)
    else:
        pass
