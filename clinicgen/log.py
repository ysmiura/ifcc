#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import inspect
import json
import math
import os
import shutil
import sys
import numpy as np
import torch
from clinicgen.eval import GenEval
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


class EpochLog:
    @classmethod
    def log_datasets(cls, logger, pbar_vals, epoch, data_n, epoch_loss, val_loader, test_loader, progress=False,
                     save=True):
        if epoch_loss is not None:
            logger.log_train(epoch, data_n, epoch_loss)
        results = {}
        logger.evaluator.setup()
        if logger.device == 'gpu':
            logger.evaluator.cuda()
        for split, data_loader in [('val', val_loader), ('test', test_loader)]:
            prog_name = split if progress else None
            results[split] = logger.evaluator.generate_and_eval(data_loader, prog_name)
            metric_idxs = logger.pbar_indexes()
            scores = ','.join(['%.2f' % results[split][GenEval.EVAL_SCORE][i] for i in metric_idxs])
            pbar_vals['{0}_scores'.format(split)] = scores
        logger.evaluator.cleanup()
        logger.log(epoch, data_n, results, save)
        return pbar_vals


class FileLogger:
    MODE_ALL = 'all'
    MODE_BEST = 'best'

    def __init__(self, root, state, model, evaluator, optimizers, schedulers, batch_schedulers, scheduler_tfr,
                 mode='best', device='gpu', test=None):
        if root is None:
            root = state
        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(state):
            os.makedirs(state)
        self.scores = {}
        if test is not None:
            self.splits = ['val', 'test']
            fm = 'w'
        else:
            self.splits = ['train', 'val', 'test']
            fm = 'a'
        for split in self.splits:
            fn = '{0}-{1}.txt'.format(split, test) if test is not None else '{0}.txt'.format(split)
            path = os.path.join(state, fn)
            self.scores[split] = open(path, fm, encoding='utf-8')
        self.summary_writer = SummaryWriter(os.path.join(state, 'log'))
        self.bests = {}
        for i in evaluator.metrics().keys():
            self.bests[i] = 0.0
        self.root = root
        self.state = state
        self.model = model
        self.evaluator = evaluator
        self.metrics = evaluator.metrics()
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.batch_schedulers = batch_schedulers
        self.scheduler_tfr = scheduler_tfr
        self.mode = mode
        self.device = device
        self.test = test

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def load_model(cls, path, model, optimizers, schedulers, batch_schedulers, scheduler_tfr, bests, device='gpu'):
        with gzip.open(path, 'rb') as f:
            d = torch.device('cpu') if device == 'cpu' else torch.device('cuda')
            state = torch.load(f, map_location=d)
        model.load_state_dict(state['model'])
        if optimizers is not None:
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(state[name + '_optimizer'])
        if schedulers is not None:
            for name, scheduler in schedulers.items():
                scheduler.load_state_dict(state[name + '_scheduler'])
        if batch_schedulers is not None:
            for name, batch_scheduler in batch_schedulers.items():
                batch_scheduler.load_state_dict(state[name + '_batch_scheduler'])
        if scheduler_tfr is not None:
            scheduler_tfr.load_state_dict(state['scheduler_tfr'])
        if bests is not None:
            for k, v in state['bests'].items():
                bests[k] = v
        return state['epoch']

    @classmethod
    def load_parameters(cls, path):
        path = os.path.join(path, 'parameters.json')
        with open(path, encoding='utf-8') as f:
            params = json.load(f)
        return params

    @classmethod
    def save_model(cls, path, epoch, model, optimizers, schedulers, batch_schedulers, scheduler_tfr, bests,
                   device='gpu'):
        if device == 'gpu' and torch.cuda.device_count() > 1:
            model.to_single()
        with gzip.open(path, 'wb') as out:
            state = {'epoch': epoch, 'model': model.state_dict(), 'bests': bests}
            for name, optimizer in optimizers.items():
                state[name + '_optimizer'] = optimizer.state_dict()
            for name, scheduler in schedulers.items():
                state[name + '_scheduler'] = scheduler.state_dict()
            if batch_schedulers is not None:
                for name, batch_scheduler in batch_schedulers.items():
                    state[name + '_batch_scheduler'] = batch_scheduler.state_dict()
            if scheduler_tfr is not None:
                state['scheduler_tfr'] = scheduler_tfr.state_dict()
            torch.save(state, out)
        if device == 'gpu' and torch.cuda.device_count() > 1:
            model.to_parallel()

    def close(self):
        for f in self.scores.values():
            f.close()
        self.summary_writer.close()

    def epoch_loss(self):
        eloss = {}
        for i in range(len(self.model.loss_names())):
            eloss[i] = []
        return eloss

    def epoch_loss_update(self, eloss, losses):
        if not isinstance(losses, tuple) and not isinstance(losses, list):
            losses = (losses,)
        for i, loss in enumerate(losses):
            if not math.isnan(loss):
                eloss[i].append(loss)
            else:
                sys.stderr.write('WARNING: nan in loss(es)\n')
        return eloss

    def load_baseline(self, path, optimizers=False):
        if optimizers:
            return self.load_model(path, self.model, self.optimizers, self.schedulers, self.batch_schedulers,
                                   self.scheduler_tfr, None, device=self.device)
        else:
            self.load_model(path, self.model, None, None, None, None, None, device=self.device)
            return -1

    def log(self, epoch, data_n, results, save=True):
        updated = self.updates(results['val'][GenEval.EVAL_SCORE]) if 'val' in results else []
        for split in ['val', 'test']:
            if split in results:
                result = results[split]
                scores = result[GenEval.EVAL_SCORE]
                self.log_scores(epoch, data_n, split, scores)
                self.log_samples(epoch, data_n, split, updated, result[GenEval.EVAL_ID], result[GenEval.EVAL_REPORT],
                                 result[GenEval.EVAL_SCORE_DETAILED])

        if 'val' in results:
            for i, (val_score, test_score) in enumerate(zip(results['val'][GenEval.EVAL_SCORE],
                                                            results['test'][GenEval.EVAL_SCORE])):
                self.summary_writer.add_scalars(self.metrics[i], {'val': val_score, 'test': test_score}, epoch + 1)
        else:
            for i, test_score in enumerate(results['test'][GenEval.EVAL_SCORE]):
                self.summary_writer.add_scalars(self.metrics[i], {'test': test_score}, epoch + 1)
        self.summary_writer.flush()

        if save and self.test is None:
            if self.mode == self.MODE_BEST:
                names = map(lambda s: '{0}_best'.format(s), updated)
            else:
                names = ['{0}-{1}'.format(epoch, data_n)]
            prev_path = None
            for name in names:
                path = os.path.join(self.root, 'model_{0}.dict.gz'.format(name))
                if prev_path is not None:
                    shutil.copy(prev_path, path)
                else:
                    self.save_model(path, epoch, self.model, self.optimizers, self.schedulers, self.batch_schedulers,
                                    self.scheduler_tfr, self.bests, device=self.device)
                prev_path = path

    def log_scores(self, epoch, data_n, split, scores):
        lid = '{0}-{1}'.format(epoch, data_n)
        if not isinstance(scores, list):
            scores = [scores]
        self.scores[split].write('{0} {1}\n'.format(lid, ' '.join(map(lambda v: str(v), scores))))
        self.scores[split].flush()

    def log_samples(self, epoch, data_n, split, updated, report_ids, samples, scores_detailed):
        sample_scores = []
        for i in range(len(report_ids)):
            sample_score = [scores_detailed[j][i] for j in range(len(scores_detailed))]
            sample_scores.append(','.join(map(lambda s: str(s), sample_score)))

        if self.mode == self.MODE_BEST:
            names = list(map(lambda s: '{0}_best'.format(s), updated))
            names.append('current')
        else:
            names = ['{0}-{1}'.format(epoch, data_n)]
        for name in names:
            sp_name = '{0}-{1}'.format(split, self.test) if self.test is not None else split
            path = os.path.join(self.root, '{0}_{1}_samples.txt.gz'.format(sp_name, name))
            with gzip.open(path, 'wt', encoding='utf8') as out:
                for rid, sample_score, sample in zip(report_ids, sample_scores, samples):
                    out.write('{0} {1} {2}\n'.format(rid, sample_score, sample))

    def log_train(self, epoch, data_n, epoch_loss):
        scores = []
        for i in range(len(epoch_loss)):
            scores.append(np.mean(epoch_loss[i]))
        self.log_scores(epoch, data_n, 'train', scores)
        d = {}
        for score, name in zip(scores, self.model.loss_names()):
            d[name] = score
        self.summary_writer.add_scalars('loss', d, epoch + 1)
        self.summary_writer.flush()

    def pbar_indexes(self):
        metric_idxs = {}
        for idx, name in self.metrics.items():
            metric_idxs[name] = idx

        pbar_idxs = []
        for name in ['BLEU1', 'CIDEr', 'NLISentBERTScore', 'NLISentBERTScoreT', 'NLISentBERTScoreM',
                     'NLISentBERTScoreP', 'BLEU4', 'NLISentAll', 'NLIEntail', 'SPICE', 'ROUGE', 'BERTScoreF']:
            if name in metric_idxs:
                pbar_idxs.append(metric_idxs[name])
        return pbar_idxs[:3]

    def resume(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.state, 'model_current.dict.gz')
        if os.path.exists(model_path):
            epoch = self.load_model(model_path, self.model, self.optimizers, self.schedulers, self.batch_schedulers,
                                    self.scheduler_tfr, self.bests, device=self.device)
            return epoch + 1
        else:
            return 0

    def save_current_model(self, epoch):
        current = os.path.join(self.state, 'model_current.dict.gz')
        self.save_model(current, epoch, self.model, self.optimizers, self.schedulers, self.batch_schedulers,
                        self.scheduler_tfr, self.bests, device=self.device)

    def save_parameters(self, args):
        path = os.path.join(self.state, 'parameters.json')
        dic = OrderedDict()
        for k, v in inspect.getmembers(args):
            if not k.startswith('_'):
                dic[k] = v
        with open(path, 'w', encoding='utf-8') as out:
            json.dump(dic, out)

    def updates(self, scores):
        updated = []
        for i, score in enumerate(scores):
            if score > self.bests[i]:
                self.bests[i] = score
                updated.append(self.metrics[i])
        return updated
