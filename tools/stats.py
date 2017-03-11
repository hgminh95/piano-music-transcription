# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

import core.util as util


def bar(value, block):
    nbar = value / block + 1

    return '.' * nbar


def key_to_note(key):
    return key


def print_distribution(dist):
    values = list(dist.values())
    max_value = max(values)
    block = max_value / 60
    print max_value, block

    for key, value in dist.items():
        print key_to_note(key), bar(value, block), value


def stats(input_dir):
    total_files = 0
    total_events = 0

    notes = {}
    for sample in util.wav_walk(input_dir):
        if 'MAPS_MUS' not in sample:
            continue

        txt_path = sample + '.txt'
        truth = util.read_txt(txt_path)

        for start, end, note in truth:
            if note not in notes:
                notes[note] = 1
            else:
                notes[note] += 1
            total_events += 1

        total_files += 1

    print "====================="
    print "General"
    print "====================="
    print ""
    print "Total files:          {}".format(total_files)
    print "Total events:         {}".format(total_events)
    print "Avg events per files: {}".format(1.0 * total_events / total_files)
    print ""

    keys = list(notes.keys())
    note_range = xrange(min(keys), max(keys) + 1)
    missing_note = filter(lambda x: x not in notes, note_range)
    print "====================="
    print "Notes"
    print "====================="
    print ""
    print "Range: {}".format(note_range)
    print "Missing notes: {}".format(missing_note)
    print "Distribution"
    print_distribution(notes)


if __name__ == '__main__':
    stats(sys.argv[1])
