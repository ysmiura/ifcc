#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import json
import os
import stanza
from stanza import Pipeline
from clinicgen.data.mimiccxr import MIMICCXRData
from clinicgen.data.openi import OpenIData
from clinicgen.text.textfilter import get_textfilter


def main(args):
    # Download
    if args.stanza_download:
        stanza.download('en', processors='tokenize,lemma,pos,ner')
        stanza.download('en', package='radiology')
    # Text processors
    textfilter = get_textfilter(args.textfilter)
    nlp = Pipeline(lang='en', package='radiology', processors={'lemma': 'default', 'pos': 'default',
                                                               'tokenize': 'default', 'ner': 'radiology'})
    # Extract texts
    texts = []
    if args.corpus == 'mimic-cxr':
        path = os.path.join(args.data, 'files', 'mimic-cxr-resized', '2.0.0', MIMICCXRData.SECTIONED_PATH)
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            header = f.readline()
            reader = csv.reader(f)
            for row in reader:
                if args.section == 'impression':
                    text = row[1]
                else:
                    text = row[2]
                if len(text) > 0:
                    texts.append((row[0][1:], text))
    elif args.corpus == 'open-i':
        dataset = OpenIData(args.data, section=args.section, meta=args.splits, split='train', multi_image=2)
        for tid, target in zip(dataset.doc_ids, dataset.targets):
            target = gzip.decompress(target).decode('utf-8')
            target = dataset.extract_section(target)
            if len(target) > 0:
                texts.append((tid, target))
    else:
        raise ValueError('Unknown corpus {0}'.format(args.corpus))
    print('{0} texts'.format(len(texts)))
    # Extract NEs
    count = 0
    with gzip.open(args.output, 'wt', encoding='utf-8') as out:
        for tid, text in texts:
            ftext = textfilter.filter(text)
            doc = nlp(ftext)
            i = 0
            for sentence in doc.sentences:
                token_starts, token_ends = {}, {}
                j = 0
                text_tokens = []
                for token in sentence.tokens:
                    token_starts[token.start_char] = j
                    token_ends[token.end_char] = j
                    text_tokens.append(token.text)
                    j += 1
                lemmas, poses = [], []
                for word in sentence.words:
                    lemmas.append(word.lemma)
                    poses.append(word.pos)
                ne_tuples = []
                for entity in sentence.ents:
                    ne_tuples.append({'text': entity.text, 'type': entity.type,
                                      'start': token_starts[entity.start_char],
                                      'end': token_ends[entity.end_char] + 1})
                ins = {'id': '{0}__{1}'.format(tid, i), 'nes': ne_tuples, 'text': sentence.text,
                       'tokens': text_tokens, 'lemmas': lemmas, 'poses': poses}
                out.write('{0}\n'.format(json.dumps(ins)))
                i += 1
                count += 1
                if count % 10000 == 0:
                    print('Processed {0}'.format(count))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract radiology named entities')
    parser.add_argument('data', type=str, help='A path to a clinical dataset')
    parser.add_argument('output', type=str, help='An output path')
    parser.add_argument('--cache', type=str, default=None, help='A cache path')
    parser.add_argument('--corpus', type=str, default='mimic-cxr', help='Corpus name')
    parser.add_argument('--section', type=str, default='findings', help='Target section')
    parser.add_argument('--splits', type=str, default=None, help='A path to a file defining splits')
    parser.add_argument('--stanza-download', default=False, action='store_true', help='Download Stanza clinical model')
    parser.add_argument('--textfilter', type=str, default='lower', help='Text filter')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
