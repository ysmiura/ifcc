#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import torch
import torch.nn as nn
import utils
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer
from constants import *
from models.bert_labeler import bert_labeler


def label(checkpoint_path, texts):
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_imp = tokenize(texts, tokenizer)

    with torch.no_grad():
        batches, ls, buffer = [], [], []
        for data in encoded_imp:
            buffer.append(data)
            if len(buffer) >= BATCH_SIZE:
                batch, bl = make_batch(buffer)
                batches.append(batch)
                ls.append(bl)
                buffer = []
        if len(buffer) > 0:
            batch, bl = make_batch(buffer)
            batches.append(batch)
            ls.append(bl)

        for batch, bl in zip(batches, ls):
            batch = batch.to(device)
            src_len = bl
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)

    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    return y_pred


def main(args):
    if args.uncertain:
        avg, pos = 'micro', [1, 3]
    else:
        avg, pos = 'binary', 1

    ids = {}
    path = os.path.join(args.ref, 'reports.csv')
    with open(path, encoding='utf-8') as f:
        f.readline()
        reader = csv.reader(f)
        idx = 1
        for row in reader:
            ids[idx] = row[0]
            idx += 1
    refs = {}
    path = os.path.join(args.ref, 'labeled_reports.csv')
    with open(path, encoding='utf-8') as f:
        f.readline()
        reader = csv.reader(f)
        idx = 1
        for row in reader:
            labels = {}
            for lidx, l in enumerate(row[1:]):
                labels[CONDITIONS[lidx]] = l
            refs[ids[idx]] = labels
            idx += 1
    print('{0} references'.format(len(refs)))

    gids, texts = [], []
    with open(args.gen, 'rt', encoding='utf-8') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            gids.append(row[0])
            texts.append(row[1])
    print('{0} generated'.format(len(texts)))

    with open('chexbert.pth', 'rb') as f:
        rs = label(f, texts)

    result = {}
    for lidx, labels in enumerate(rs):
        for idx, l in enumerate(labels):
            did = gids[idx]
            if did not in result:
                result[did] = {}
            if CONDITIONS[lidx] not in result[did]:
                result[did][CONDITIONS[lidx]] = l

    with open(args.out, 'w', encoding='utf-8') as out:
        out.write('DOC_ID,{0}\n'.format(','.join(CONDITIONS)))
        for did in gids:
            l = []
            for c in CONDITIONS:
                v = result[did][c]
                l.append(str(v))
            out.write('{0},{1}\n'.format(did, ','.join(l) ))

    trues, preds = {}, {}
    for did, gen_labels in result.items():
        for gen_label, v in gen_labels.items():
            if gen_label not in preds:
                preds[gen_label] = []
            if v == 3 and not args.uncertain:
                v = 1
            elif v == 2:
                v = 0
            preds[gen_label].append(v)
        for true_label, v in refs[did].items():
            if true_label not in trues:
                trues[true_label] = []
            if v == '-1.0':
                v = 1 if not args.uncertain else 3
            elif v == '1.0':
                v = 1
            elif v == '0.0' or v == '':
                v = 0
            trues[true_label].append(v)

    prs, rcs, fb1s = [], [], []
    for c in CONDITIONS:
        if c in trues and c in preds:
            acc = accuracy_score(trues[c], preds[c])
            pr, rc, fb1, _ = precision_recall_fscore_support(trues[c], preds[c], labels=pos, average=avg)
            prs.append(pr)
            rcs.append(rc)
            fb1s.append(fb1)
            print('{0} {1} {2} {3} {4}'.format(c, acc, pr, rc, fb1))

    trues5, preds5 = [], []
    for c in ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']:
        trues5 += trues[c]
        preds5 += preds[c]
    pr, rc, fb1, _ = precision_recall_fscore_support(trues5, preds5, labels=pos, average=avg)
    print('5-micro {0} {1} {2}'.format(pr, rc, fb1))
    print('5-acc {0}'.format(accuracy_score(trues5, preds5)))


def make_batch(buffer):
    max_len = 0
    for data in buffer:
        if data.shape[0] > max_len:
            max_len = data.shape[0]
    batch = torch.zeros((len(buffer), max_len), dtype=torch.long)
    bl = []
    for i, data in enumerate(buffer):
        batch[i][:data.shape[0]] = data
        bl.append(data.shape[0])
    return batch, bl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str, help='A path to reference reports')
    parser.add_argument('gen', type=str, help='A path to generated reports')
    parser.add_argument('out', type=str, help='A path to output CheXbert outputs')
    parser.add_argument('--uncertain', default=False, action='store_true', help='Treat uncertain as an independent class')
    return parser.parse_args()


def tokenize(impressions, tokenizer):
    new_impressions = []
    for impression in impressions:
        tokenized_imp = tokenizer.tokenize(impression)
        if tokenized_imp: #not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512: #length exceeds maximum size
                print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(torch.LongTensor(res))
        else: #an empty report
            new_impressions.append(torch.LongTensor([tokenizer.cls_token_id, tokenizer.sep_token_id]))
    return new_impressions


if __name__ == '__main__':
    args = parse_args()
    main(args)
