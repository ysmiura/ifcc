#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import os
import random
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch import softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from clinicgen.data.chexpert import CheXpertData
from clinicgen.data.utils import Data
from clinicgen.models.image import ImageClassification
from clinicgen.utils import data_cuda


def eval_model(pbar_vals, outs, epoch, data_n, model, optimizer, scheduler, val_loader, test_loader, bests,
               device=None):
    for split, data_loader in [('val', val_loader), ('test', test_loader)]:
        if data_loader is not None:
            scores = eval_split(model, data_loader, device=device)
            pbar_vals['{0}_score'.format(split)] = scores[0]
            outs[split].write('{0}-{1} {2} {3}\n'.format(epoch, data_n, scores[0], scores[1]))
            outs[split].flush()
            if split == 'val':
                updates = update_bests(bests, scores)
                for update in updates:
                    save_model(os.path.join(args.out, 'model_{0}.dict.gz'.format(update)), epoch, model, optimizer,
                               scheduler, bests)


def eval_split(model, data_loader, device=None):
    with torch.no_grad():
        model.eval()
        y_true5, y_true14, y_score5, y_score14 = [], [], [], []
        for _, inp, targ, _, _, _ in data_loader:
            inp, _ = data_cuda(inp, targ, device=device, non_blocking=False)
            out = model(inp)
            probs = softmax(out.permute(0, 2, 1)[:, :, 1:3], dim=-1)
            probs = probs.detach().cpu().numpy()
            targ = targ.numpy()
            for i in range(probs.shape[0]):
                for j in range(probs.shape[1]):
                    true_val = 1 if targ[i][j] == 1 else 0
                    score_val = probs[i][j][1]
                    y_true14.append(true_val)
                    y_score14.append(score_val)
                    if j == 2 or j == 5 or j == 6 or j == 8 or j == 10:
                        y_true5.append(true_val)
                        y_score5.append(score_val)
    y_true5 = np.array(y_true5)
    y_score5 = np.array(y_score5)
    y_true14 = np.array(y_true14)
    y_score14 = np.array(y_score14)
    rocauc5 = roc_auc_score(y_true5, y_score5, average='macro')
    rocauc14 = roc_auc_score(y_true14, y_score14, average='macro')
    model.train()
    return rocauc5, rocauc14


def main(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    else:
        print('ERROR: {0} already exists'.format(args.out))
        exit(1)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model configurations
    model = ImageClassification(args.model, CheXpertData.NUM_LABELS, CheXpertData.NUM_CLASSES, args.multi_image,
                                dropout=args.dropout, pretrained=args.pretrained)
    if args.cuda:
        device = 'gpu'
        model = model.cuda(0)
    else:
        device = 'cpu'

    # Data
    t = time.time()
    datasets = Data.get_datasets(args.data, args.corpus, None, None, None, None, None, None, None,
                                 multi_image=args.multi_image, img_mode=args.img_trans, img_augment=args.img_augment,
                                 cache_data=args.cache_data, anatomy=args.anatomy, meta=args.splits,
                                 ignore_blank=args.ignore_blank, exclude_ids=args.exclude_ids, filter_reports=False)
    nw = 0 if args.cache_data else args.num_workers
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=nw,
                              pin_memory=False)
    batch_size_test = args.batch_size if args.batch_size_test is None else args.batch_size_test
    val_loader = DataLoader(datasets['validation'], batch_size=batch_size_test, shuffle=False, num_workers=nw,
                            pin_memory=False)
    if 'test' in datasets:
        test_loader = DataLoader(datasets['test'], batch_size=batch_size_test, shuffle=False, num_workers=nw,
                                 pin_memory=False)
        test_size = len(test_loader.dataset.samples)
    else:
        test_loader, test_size = None, 0
    print('Data: train={0}, validation={1}, test={2} (load time {3:.2f}s)'.format(len(train_loader.dataset.samples),
                                                                                  len(val_loader.dataset.samples),
                                                                                  test_size, time.time() - t))

    # Train and test processes
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, args.lr_step, args.lr_gamma)
    pbar_vals = {'loss': None, 'val_score': None, 'test_score': None}
    outs, bests = {}, {'auc5': 0.0, 'auc14': 0.0}
    try:
        outs['val'] = open(os.path.join(args.out, 'val.txt'), 'w', encoding='utf-8')
        outs['test'] = open(os.path.join(args.out, 'test.txt'), 'w', encoding='utf-8')
        for epoch in range(args.epochs):
            loss_log = []

            with tqdm(total=len(train_loader.dataset.samples)) as pbar:
                pbar.set_description('Epoch {0}/{1}'.format(epoch + 1, args.epochs))
                data_n, eval_interval, tqdm_interval = 0, 0, 0

                for _, inp, targ, _ in train_loader:
                    # Train
                    loss_val = model.train_step(inp, targ, optimizer, clip_grad=args.clip_grad, device=device)
                    loss_log.append(loss_val)
                    # Validation / Test
                    data_n += inp.shape[0]
                    eval_interval += inp.shape[0]
                    if args.eval_interval is not None and eval_interval >= args.eval_interval:
                        eval_model(pbar_vals, outs, epoch, data_n, model, optimizer, scheduler, val_loader, test_loader,
                                   bests, device)
                        eval_interval -= args.eval_interval
                    # Progress updates
                    tqdm_interval += inp.shape[0]
                    if args.tqdm_interval is None or tqdm_interval >= args.tqdm_interval:
                        pbar_vals['loss'] = np.mean(loss_log)
                        pbar.set_postfix(**pbar_vals)
                        pbar.update(tqdm_interval)
                        tqdm_interval -= args.tqdm_interval if args.tqdm_interval is not None else 0
            # Epoch end processes
            scheduler.step()
            eval_model(pbar_vals, outs, epoch, None, model, optimizer, scheduler, val_loader, test_loader, bests,
                       device)
    finally:
        for _, out in outs.items():
            out.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification')
    parser.add_argument('data', type=str, help='A path to clinical data')
    parser.add_argument('model', type=str, help='A model name')
    parser.add_argument('out', type=str, help='An output path')
    parser.add_argument('--anatomy', type=str, default=None, help='A specific anatomy to target')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--batch-size-test', type=int, default=None, help='Batch size (test)')
    parser.add_argument('--cache-data', type=str, default=None, help='Cache images and texts to memory and disk')
    parser.add_argument('--clip-grad', type=float, default=None, help='Clip gradients')
    parser.add_argument('--corpus', type=str, default='chexpert', choices=['a', 'chexpert', 'mimic-cxr', 'open-i'], help='Corpus name')
    parser.add_argument('--cuda', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=12, help='Epoch num')
    parser.add_argument('--eval-interval', type=int, default=None, help='Evaluation interval')
    parser.add_argument('--exclude-ids', type=str, default=None, help='IDs to exclude from the data')
    parser.add_argument('--ignore-blank', default=False, action='store_true', help='Ignore blank labels')
    parser.add_argument('--img-no-augment', dest='img_augment', default=True, action='store_false', help='Do not augment images')
    parser.add_argument('--img-trans', type=str, default='pad', help='Image transformation mode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='A learning rate scheduler gamma')
    parser.add_argument('--lr-step', type=int, default=16, help='A learning rate scheduler step')
    parser.add_argument('--multi-image', type=int, default=2, help='Multi image number')
    parser.add_argument('--scratch', dest='pretrained', default=True, action='store_false', help='Train a model from scratch')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--splits', type=str, default=None, help='A path to a file defining splits')
    parser.add_argument('--tqdm-interval', type=int, default=None, help='tqdm interval')
    return parser.parse_args()


def save_model(path, epoch, model, optimizer, scheduler, bests):
    with gzip.open(path, 'wb') as out:
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'bests': bests}
        torch.save(state, out)


def update_bests(bests, scores):
    updates = []
    if scores[0] > bests['auc5']:
        bests['auc5'] = scores[0]
        updates.append('auc5')
    if scores[1] > bests['auc14']:
        bests['auc14'] = scores[1]
        updates.append('auc14')
    return updates


if __name__ == '__main__':
    args = parse_args()
    main(args)
