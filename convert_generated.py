#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import os


def main(args):
    with open(args.output, 'w', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['DOC_ID', 'Report Generated'])
        with gzip.open(args.gen, 'rt', encoding='utf-8') as f:
            for line in f:
                entry = line.rstrip().split(' ')
                did = entry[0].split('__')[0]
                text = rewrite(' '.join(entry[2:]))
                writer.writerow([did, text])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def rewrite(text):
    text = text.replace(" ' ", "'")
    text = text.replace(" n't", "n't")
    text = text.replace(' - ', '-')
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    return text


if __name__ == '__main__':
    args = parse_args()
    main(args)
