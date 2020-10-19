#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import os
import torch.utils.data as data


class MedNLIData(data.Dataset):
    def __init__(self):
        self.ids = []
        self.samples = []

    def __getitem__(self, index):
        sentence1, sentence2, gold_label = self.samples[index]
        iid = self.ids[index]
        return iid, sentence1, sentence2, gold_label

    def __len__(self):
        return len(self.samples)

    def load(self, root, split=None):
        if split == 'validation':
            split = 'dev'
        path = os.path.join(root, 'mednli_bionlp19_shared_task_ground_truth.csv')
        if os.path.exists(path):
            form = 'jsonl-csv'
        else:
            path = os.path.join(root, 'mli_{0}_v1.jsonl'.format(split))
            if os.path.exists(path):
                form = 'jsonl'
            else:
                form = 'tsv'

        if form == 'tsv':
            if split == 'dev' or split == 'test':
                path = os.path.join(root, '{0}.tsv'.format(split))
                if not os.path.exists(path):
                    path = os.path.join(root, '{0}_fact240.tsv'.format(split))
            elif split == 'train':
                path = os.path.join(root, 'train.tsv')
                if not os.path.exists(path):
                    items = []
                    for item in os.listdir(root):
                        if not item.startswith('.') and item.endswith('.tsv'):
                            items.append(item)
                    assert len(items) == 1
                    path = os.path.join(root, items[0])
            else:
                raise ValueError('Unknown split {0}'.format(split))
            with open(path, encoding='utf-8') as f:
                for line in f:
                    entry = line.rstrip().split('\t')
                    self.ids.append(entry[0])
                    self.samples.append((entry[1], entry[2], entry[-1]))
        elif form == 'jsonl':
            path = os.path.join(root, 'mli_{0}_v1.jsonl'.format(split))
            with open(path, encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.ids.append(entry['pairID'])
                    self.samples.append((entry['sentence1'], entry['sentence2'], entry['gold_label']))
        elif form == 'jsonl-csv':
            path = os.path.join(root, 'mednli_bionlp19_shared_task_ground_truth.csv')
            labels = {}
            with open(path, encoding='utf-8') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    labels[row[0]] = row[1]
            path = os.path.join(root, 'mednli_bionlp19_shared_task.jsonl')
            with open(path, encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    pid = entry['pairID']
                    self.ids.append(pid)
                    self.samples.append((entry['sentence1'], entry['sentence2'], labels[pid]))
        else:
            raise ValueError('Unknown format {0}'.format(form))
