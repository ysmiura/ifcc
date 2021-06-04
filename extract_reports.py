#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import os
import re


def main(args):
    space_pattern = re.compile('\\s+')

    doc_ids = {}
    with gzip.open(args.split, 'rt', encoding='utf-8') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if row[3] == 'test':
                sid = row[1]
                doc_ids[sid] = True
    print('{0} test reports'.format(len(doc_ids)))

    with gzip.open(args.sections, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        sections = {}
        reader = csv.reader(f)
        for row in reader:
            report = {}
            for i, sec in enumerate(header):
                report[sec] = row[i]
            sections[row[0]] = report

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    with open(os.path.join(args.output, 'reports.csv'), 'w', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['DOC_ID', 'Report Impression'])
        for sid in doc_ids:
            text = sections['s' + sid]['impression']
            if len(text) == 0:
                text = sections['s' + sid]['findings']
            text = text.replace('\n', ' ')
            text = space_pattern.sub(' ', text)
            if len(text) > 0:
                writer.writerow([sid, text])


def parse_args():
    parser = argparse.ArgumentParser(description='Extract radiology named entities')
    parser.add_argument('sections', type=str, help='A path to the MIMIC-CXR sectioned file')
    parser.add_argument('split', type=str, help='A path to the MIMIC-CXR split file')
    parser.add_argument('output', type=str, help='An output path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
