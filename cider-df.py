#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import os
import pickle
from clinicgen.data.areport import AReportData
from clinicgen.data.mimiccxr import MIMICCXRData
from clinicgen.eval import GenEval
from clinicgen.text.textfilter import get_textfilter
from clinicgen.text.tokenfilter import get_tokenfilter
from clinicgen.text.tokenizer import get_tokenizer


def main(args):
    tokenizer = get_tokenizer(args.tokenizer)
    textfilter = get_textfilter(args.textfilter)
    tokenfilter = get_tokenfilter(args.tokenfilter)

    texts = []
    if args.corpus == 'mimic-cxr':
        path = os.path.join(args.data, 'mimic-cxr-resized', '2.0.0', MIMICCXRData.SPLITS_PATH)
        train_ids = {}
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            header = f.readline()
            reader = csv.reader(f)
            for row in reader:
                if row[3] == 'train':
                    train_ids['s' + row[1]] = True
        path = os.path.join(args.data, 'mimic-cxr-resized', '2.0.0', MIMICCXRData.SECTIONED_PATH)
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            header = f.readline()
            reader = csv.reader(f)
            for row in reader:
                if row[0] in train_ids:
                    if args.section == 'impression':
                        text = row[1]
                    else:
                        text = row[2]
                    if len(text) > 0:
                        texts.append(text)
    elif args.corpus == 'a':
        dump_dir = os.path.join(args.cache, 'a', 'train')
        dataset = AReportData(args.data, section=args.section, anatomy=args.anatomy, exclude_ids=args.exclude_ids,
                              meta=args.meta, split='train', cache_image=True, cache_text=True, dump_dir=dump_dir,
                              multi_image=2)
        for target in dataset.targets:
            target = gzip.decompress(target).decode('utf-8')
            target = dataset.extract_section(target)
            if len(target) > 0:
                texts.append(target)
    else:
        raise ValueError('Unknown corpus {0}'.format(args.corpus))
    print('{0} texts'.format(len(texts)))

    ftexts = []
    for text in texts:
        toks = tokenizer.tokenize(textfilter.filter(text))
        toks = tokenfilter.filter(toks)
        ftext = ' '.join(toks)
        ftexts.append(ftext)

    df = GenEval.compute_cider_df(texts)
    with gzip.open(args.output, 'w') as f:
        pickle.dump(df, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute CIDEr DFs')
    parser.add_argument('data', type=str, help='A path to clinical data')
    parser.add_argument('output', type=str, help='An output path')
    parser.add_argument('--anatomy', type=str, default=None, help='An anatomy')
    parser.add_argument('--cache', type=str, default=None, help='A cache path')
    parser.add_argument('--corpus', type=str, default='mimic-cxr', help='Corpus name')
    parser.add_argument('--exclude-ids', type=str, default=None, help='Exclude IDs')
    parser.add_argument('--meta', type=str, default=None, help='A meta data path')
    parser.add_argument('--section', type=str, default='findings', help='Target section')
    parser.add_argument('--textfilter', type=str, default='lower', help='Text filter')
    parser.add_argument('--tokenfilter', type=str, default='none', help='Token filter')
    parser.add_argument('--tokenizer', type=str, default='nltk', choices=['nltk', 'none', 'stanford', 'whitespace'], help='Tokenizer name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
