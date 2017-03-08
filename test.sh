#!/bin/sh

source activate tensorflow

python transcript.py construct -i . -o test.data
python transcript.py train -i test.data -o test.pkl
python transcript.py eval -mf test.pkl -t .
python transcript.py eval -mf test.pkl -t dislavier.mul.cqt.data
