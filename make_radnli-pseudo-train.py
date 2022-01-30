#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import json
import os
import re
from collections import OrderedDict
from nltk.tokenize import sent_tokenize


def main(args):
    reports = {}
    with gzip.open(args.sections, 'rt', encoding='utf-8') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            findings = row[2]
            findings = findings.replace('\n', ' ')
            findings = re.sub('\\s+', ' ', findings)
            reports[row[0]] = findings
    with open('radnli_pseudo-train.jsonl', 'w', encoding='utf-8') as out:
        with open(os.path.join('resources', 'radnli_pseudo-train_indexes.jsonl'), encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                sent1 = entry['sentence1'].split(',')
                idxs = sent1[1].split(':')
                entry['sentence1'] = reports[sent1[0]][int(idxs[0]):int(idxs[1])]
                sent2 = entry['sentence2'].split(',')
                idxs = sent2[1].split(':')
                entry['sentence2'] = reports[sent2[0]][int(idxs[0]):int(idxs[1])]
                out.write(json.dumps(entry) + '\n')
    print('Wrote: radnli_pseudo-train.jsonl')

def parse_args():
    parser = argparse.ArgumentParser(description='Make RadNLI pseudo training data from MIMIC-CXR')
    parser.add_argument('sections', type=str, help='A path to the MIMIC-CXR sectioned file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
