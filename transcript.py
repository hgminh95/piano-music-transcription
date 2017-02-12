# -*- coding: utf-8 -*-

import argparse
import logging

from core.transcriber import MetaTranscriber
from core import utils
import transcribers

logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s\t %(filename)s\t %(message)s")

_logger = logging.getLogger(__name__)


def transcriber_do(action, args):
    transcriber = MetaTranscriber.method_to_transcriber[args.method]

    if action == 'train':
        transcriber.train(args.input)
        return

    # Load input file
    files = []
    for elem in utils.wav_walk(args.input):
        files.append(elem)
        _logger.debug('Found wav file: {0}'.format(elem))
    _logger.info('Found {0} sample(s)'.format(len(files)))

    output = open(args.output, 'w')

    for file in files:
        if action == 'transcribe':
            transcriber.transcribe(file)
        elif action == 'construct':
            transcriber.construct(file, output)

    output.close()


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
        default='librosa_onset')
    transcribe_parser.add_argument(
        '-i', '--input',
        help='input file or folder',
        default='.')
    transcribe_parser.add_argument(
        '-o', '--output',
        help='output file or folder')
    transcribe_parser.set_defaults(sub='transcribe')

    construct_parser = subparsers.add_parser(
        'construct',
        help='Construct data for training')
    construct_parser.add_argument(
        '-m', '--method',
        help='method used to construct data',
        default='librosa_onset')
    construct_parser.add_argument(
        '-i', '--input',
        help='input file or folder',
        default='.')
    construct_parser.add_argument(
        '-o', '--output',
        help='output file or folder',
        default='a.out')
    construct_parser.set_defaults(sub='construct')

    train_parser = subparsers.add_parser(
        'train',
        help='Train model')
    train_parser.add_argument(
        '-m', '--method',
        help='method used to train',
        default='librosa_onset')
    train_parser.add_argument(
        '-i', '--input',
        help='input file',
        default='.')
    train_parser.set_defaults(sub='train')

    args = parser.parse_args()

    if args.sub == 'transcribe':
        transcriber_do('transcribe', args)
    elif args.sub == 'construct':
        transcriber_do('construct', args)
    elif args.sub == 'train':
        transcriber_do('train', args)
        pass
