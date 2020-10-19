#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import json
import os
import site
import sys
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, OrderedDict
from bert_score.utils import bert_cos_score_idf, cache_scibert, get_idf_dict, get_model, lang2model, model2layers
from cachetools import LRUCache
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import DataParallel
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from clinicgen.models.bertnli import BERTNLI
from clinicgen.text.sentsplit import get_sentsplitter
from clinicgen.text.tokenizer import get_tokenizer


class _NLIScorer:
    AVERAGE_FROM = 'from'
    AVERAGE_TO = 'to'
    COMPARE_ALL = 'all'
    COMPARE_BERT_SCORE = 'bert-score'
    COMPARE_BERT_SCORE_FIX_THRESH = 'bert-score-thresh'
    COMPARE_TFIDF = 'tfidf'
    COMPARE_DOC = 'doc'
    LABEL_ALL = 'all'
    LABEL_CONTRADICT = 'contradiction'
    LABEL_ENTAIL = 'entailment'
    LABEL_NEUTRAL = 'neutral'
    MERGE_FROM = 'from'
    MERGE_TO = 'to'
    METRIC_FB1 = 'f'
    METRIC_FB1_HARD = 'fh'
    METRIC_FB1_PENALTY = 'fp'
    METRIC_PRECISION = 'p'
    METRIC_RECALL = 'r'
    PENALTY_SIGMA = 6.0
    SIM_MIN = 'min'
    SIM_MUL = 'mul'
    SIM_NONE = 'none'
    THRESH_FIX = 'fix'
    THRESH_NONE = 'none'

    def __init__(self, neutral_score=(1.0 / 3), batch=16, nthreads=2, pin_memory=False, bert_score=None,
                 sentsplitter='none', cache=None, verbose=False):
        self.model = None
        self.neutral_score = neutral_score
        self.batch = batch
        self.nthreads = nthreads
        self.pin_memory = pin_memory

        self.sentsplitter = get_sentsplitter(sentsplitter, linebreak=False)
        self.tokenizer = get_tokenizer('nltk')
        if bert_score is not None:
            self.bert_score_model = BERTScorer(model_type=bert_score, batch_size=batch, nthreads=nthreads,
                                               lang='en', rescale_with_baseline=True)
        else:
            self.bert_score_model = None
        self.cache = LRUCache(cache) if cache is not None else None
        self.verbose = verbose
        self.gpu = False

    def cuda(self):
        self.model = self.model.cuda()
        self.gpu = True
        if self.bert_score_model is not None:
            self.bert_score_model = self.bert_score_model.cuda()
        return self

    def predict(self, premises, hypotheses):
        raise NotImplementedError

    def sentence_scores(self, texts1, texts2, compare='bert-score', label='entailment', tfidf_vectorizer=None, prf='f'):
        if self.verbose:
            t = time.time()
        if compare == self.COMPARE_ALL:
            result = self.sentence_scores_all(texts1, texts2, label, prf=prf)
        elif compare == self.COMPARE_BERT_SCORE:
            result = self.sentence_scores_bert_score(texts1, texts2, label, prf=prf)
        elif compare == self.COMPARE_BERT_SCORE_FIX_THRESH:
            result = self.sentence_scores_bert_score(texts1, texts2, label, thresh=self.THRESH_FIX, prf=prf)
        elif compare == self.COMPARE_TFIDF:
            result = self.sentence_scores_tfidf(texts1, texts2, label, tfidf_vectorizer)
        else:
            raise ValueError('Unknown comparison method {0}'.format(compare))
        if self.verbose:
            print('NLI {0} pairs with gpu={1}: {2}s'.format(len(texts1), self.gpu, time.time() - t))
        return result

    def sentence_scores_all(self, texts1, texts2, label='entailment', prf='f'):
        pids, pairs, prems, hypos = [], OrderedDict(), [], []
        probs_pairs = {}
        for pid in range(len(texts1)):
            probs_pairs[pid] = []
            text1, text2 = texts1[pid], texts2[pid]
            sents1 = self.sentsplitter.split(text1)
            sents2 = self.sentsplitter.split(text2)
            pairs[pid] = (sents1, sents2)

            if prf != self.METRIC_PRECISION:
                for sent1 in sents1:
                    for sent2 in sents2:
                        # ref sent: prem=gen & hypo=ref
                        prems.append(sent2)
                        hypos.append(sent1)
                        pids.append(pid)
            if prf != self.METRIC_RECALL:
                for sent2 in sents2:
                    for sent1 in sents1:
                        # gen sent: prem=ref & hypo=gen
                        prems.append(sent1)
                        hypos.append(sent2)
                        pids.append(pid)
        probs_all, _ = self.predict(prems, hypos)
        for i, pid in enumerate(pids):
            probs_pairs[pid].append(probs_all[i])

        prs, rcs, fb1s, stats = [], [], [], []
        for pid, (sents1, sents2) in pairs.items():
            # NLI probabilities: (s2_1, s1_1), (s2_2, s1_1), ..., (s1_1, s2_1), (s1_2, s2_1), ...
            probs = np.array(probs_pairs[pid])
            idx = 0
            sent_probs, sent_idxs = [{}, {}], [{}, {}]
            rc_pr = [[], []]
            if prf == 'f' or prf == 'h':
                triple = [0, 1], (sents1, sents2), (sents2, sents1)
            elif prf == 'p':
                triple = [1], (sents2,), (sents1,)
            elif prf == 'r':
                triple = [0], (sents1,), (sents2,)
            else:
                raise ValueError('Unknown metric {0}'.format(prf))
            for k, rp_hypos, rp_prems in zip(triple[0], triple[1], triple[2]):
                # k=0: (s2_1, s1_1), (s2_2, s1_1), ...
                # k=1: (s1_1, s2_1), (s1_2, s2_1), ...
                for i in range(len(rp_hypos)):
                    scores = []
                    sent_probs[k][i] = {}
                    for j in range(len(rp_prems)):
                        if prf == 'h':
                            # Hard scoring
                            best_prob, best_label = 0.0, self.LABEL_ENTAIL
                            for lb, pr in probs[idx].items():
                                if pr > best_prob:
                                    best_prob, best_label = pr, lb
                            tlabel = label if label != self.LABEL_ALL else self.LABEL_ENTAIL
                            prob = 1.0 if tlabel == best_label else 0.0
                            if label == self.LABEL_CONTRADICT:
                                prob = 1.0 - prob
                            scores.append(prob)
                        else:
                            # Soft scoring
                            if label == self.LABEL_ALL:
                                prob = probs[idx]
                                scores.append(prob[self.LABEL_ENTAIL])
                            elif label == self.LABEL_CONTRADICT:
                                prob = 1.0 - probs[idx][label]
                                scores.append(prob)
                            else:
                                prob = probs[idx][label]
                                scores.append(prob)
                        sent_probs[k][i][j] = (prob, rp_hypos[i], rp_prems[j])
                        idx += 1
                    if len(scores) > 0:
                        ridx = np.argmin(scores) if label == self.LABEL_CONTRADICT else np.argmax(scores)
                        sent_idxs[k][i] = ridx
                        score = scores[ridx]
                    else:
                        score = 0.0
                    rc_pr[k].append(score)
            mean_precision = np.mean(rc_pr[1]) if len(rc_pr[1]) > 0 else 0.0
            mean_recall = np.mean(rc_pr[0]) if len(rc_pr[0]) > 0 else 0.0
            if mean_precision + mean_recall > 0.0:
                fb1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            else:
                fb1 = 0.0
            prs.append(mean_precision)
            rcs.append(mean_recall)
            fb1s.append(fb1)
            stats.append({'scores': sent_probs, 'indexes': sent_idxs})
        if prf == self.METRIC_PRECISION:
            fb1s = prs
        elif prf == self.METRIC_RECALL:
            fb1s = rcs
        return prs, rcs, fb1s, stats

    def sentence_scores_bert_score(self, texts1, texts2, label='entailment', thresh='none', prf='f'):
        # Calculate BertScores
        tids, tsents1, tsents2, bsents1, bsents2 = [], {}, {}, [], []
        bsents1t, bsents2t, thresh_nums = [], [], [0, 0]
        for tid, (text1, text2) in enumerate(zip(texts1, texts2)):
            tids.append(tid)
            sents1 = self.sentsplitter.split(text1)
            tsents1[tid] = sents1
            sents2 = self.sentsplitter.split(text2)
            tsents2[tid] = sents2
            for i, sent1 in enumerate(sents1):
                for j, sent2 in enumerate(sents2):
                    bsents1.append(sent1)
                    bsents2.append(sent2)
            if thresh != self.THRESH_NONE:
                for k, sents in enumerate([sents1, sents2]):
                    for i in range(len(sents)):
                        sent1 = sents[i]
                        for j in range(i + 1, len(sents)):
                            sent2 = sents[j]
                            bsents1t.append(sent1)
                            bsents2t.append(sent2)
                            thresh_nums[k] += 1
        bsents1 += bsents1t
        bsents2 += bsents2t
        # Store BERTScores to dictionaries of (#text, #sentence)
        _, _, bf = self.bert_score_model.score(bsents1, bsents2)
        bf = bf.numpy()
        idx = 0
        scores1, scores2 = {}, {}
        for tid in tids:
            scores1[tid], scores2[tid] = {}, {}
            for i in range(len(tsents1[tid])):
                scores1[tid][i] = []
                for j in range(len(tsents2[tid])):
                    if j not in scores2[tid]:
                        scores2[tid][j] = []
                    score = bf[idx]
                    scores1[tid][i].append(score)
                    scores2[tid][j].append(score)
                    idx += 1
        if thresh == self.THRESH_FIX:
            thresh1, thresh2 = 0.5, 0.5
        else:
            thresh1, thresh2 = -100.0, -100.0
        # Obtain high BertScore premise-hypothesis pairs
        tids_all, prems_all, hypos_all = [], [], []
        prems, hypos, pbfs, fidxs = {}, {}, {}, {}
        valid_sents, cache_sents = {}, {}
        for k in [0, 1]:
            prems[k], hypos[k], pbfs[k], fidxs[k] = {}, {}, {}, {}
            valid_sents[k], cache_sents[k] = {}, {}
        for tid in tids:
            for k, quad in enumerate([(scores1, tsents1, tsents2, thresh1), (scores2, tsents2, tsents1, thresh2)]):
                # ref sent: prem=gen & hypo=ref, gen sent: prem=ref & hypo=gen
                scores, sents_hypos, sents_prems, thresh = quad
                prems[k][tid], hypos[k][tid], pbfs[k][tid], fidxs[k][tid] = {}, {}, {}, {}
                valid_sents[k][tid], cache_sents[k][tid] = {}, {}
                for i in range(len(scores[tid])):
                    if len(scores[tid][i]) > 0:
                        fidx = np.argmax(scores[tid][i])
                        fscore = scores[tid][i][fidx]
                        # Align with the best scoring sentence
                        prems[k][tid][i] = sents_prems[tid][fidx]
                        hypos[k][tid][i] = sents_hypos[tid][i]
                        pbfs[k][tid][i] = fscore
                        fidxs[k][tid][i] = fidx
                        # Set IDs, premises, and hypothesis for NLI
                        prem = sents_prems[tid][fidx]
                        hypo = sents_hypos[tid][i]
                        if self.cache is not None and (prem, hypo) in self.cache:
                            cache_sents[k][tid][i] = self.cache[(prem, hypo)]
                            valid_sents[k][tid][i] = None
                        else:
                            if scores[tid][i][fidx] >= thresh and ((k == 0 and prf != 'p') or (k == 1 and prf != 'r')):
                                tids_all.append(tid)
                                prems_all.append(prem)
                                hypos_all.append(hypo)
                                valid_sents[k][tid][i] = (prem, hypo)
                            else:
                                valid_sents[k][tid][i] = None

        # NLI probabilities
        probs_all, _ = self.predict(prems_all, hypos_all)
        probs_tids = OrderedDict()
        for i, tid in enumerate(tids_all):
            if tid not in probs_tids:
                probs_tids[tid] = []
            probs_tids[tid].append(probs_all[i])
        prs, rcs, fb1s, stats = [], [], [], []
        for tid in tids:
            rc_pr = {0: [], 1: []}
            sent_probs = [{}, {}]
            idx = 0

            for k, num_sent in zip([0, 1], (len(tsents1[tid]), len(tsents2[tid]))):
                for i in range(num_sent):
                    if i in cache_sents[k][tid]:
                        prob = cache_sents[k][tid][i]
                    elif i in valid_sents[k][tid] and valid_sents[k][tid][i] is not None:
                        prem_hypo = valid_sents[k][tid][i]
                        prob = probs_tids[tid][idx]
                        if self.cache is not None:
                            self.cache[prem_hypo] = prob
                        idx += 1
                    else:
                        prob = None
                    if prf == self.METRIC_FB1_HARD:
                        # Hard scoring
                        if prob is not None:
                            best_prob, best_label = 0.0, self.LABEL_ENTAIL
                            for lb, pr in prob.items():
                                if pr > best_prob:
                                    best_prob, best_label = pr, lb
                            if label == self.LABEL_ALL:
                                prob = {}
                                for lb in [self.LABEL_ENTAIL, self.LABEL_NEUTRAL, self.LABEL_CONTRADICT]:
                                    if lb == best_label:
                                        prob[lb] = 1.0
                                    else:
                                        prob[lb] = 0.0
                            else:
                                prob = 1.0 if label == best_label else 0.0
                                if label == self.LABEL_CONTRADICT:
                                    prob = 1.0 - prob
                            rc_pr[k].append(prob)
                        else:
                            if label == self.LABEL_ALL:
                                prob = {self.LABEL_ENTAIL: -1.0, self.LABEL_NEUTRAL: -1.0, self.LABEL_CONTRADICT: -1.0}
                            else:
                                prob = -1.0
                    else:
                        # Soft scoring
                        if label == self.LABEL_ALL:
                            if prob is not None:
                                score = prob[self.LABEL_ENTAIL]
                                rc_pr[k].append(score)
                            else:
                                prob = {self.LABEL_ENTAIL: -1.0, self.LABEL_NEUTRAL: -1.0, self.LABEL_CONTRADICT: -1.0}
                        else:
                            if prob is not None:
                                prob_l = prob[label]
                                if label == self.LABEL_CONTRADICT:
                                    prob_l = 1.0 - prob_l
                                score = prob_l
                                neutral_prob = prob[self.LABEL_NEUTRAL]
                                if label == self.LABEL_ENTAIL:
                                    if self.neutral_score > 0.0 and neutral_prob > prob_l and neutral_prob > prob[self.LABEL_CONTRADICT]:
                                        prob_l = -1.0
                                        rc_pr[k].append(self.neutral_score)
                                    else:
                                        rc_pr[k].append(score)
                                else:
                                    rc_pr[k].append(score)
                                prob = prob_l
                            else:
                                prob = -1.0
                    if i in prems[k][tid] and i in hypos[k][tid]:
                        sent_probs[k][i] = (prob, pbfs[k][tid][i], prems[k][tid][i], hypos[k][tid][i], fidxs[k][tid][i])
            mean_precision = np.mean(rc_pr[1]) if len(rc_pr[1]) > 0 else 0.0
            mean_recall = np.mean(rc_pr[0]) if len(rc_pr[0]) > 0 else 0.0
            if mean_precision + mean_recall > 0.0:
                fb1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            else:
                fb1 = 0.0
            # Gaussian penalty
            if prf == self.METRIC_FB1_PENALTY:
                l1, l2 = 0, 0
                for sent in tsents1[tid]:
                    toks = self.tokenizer.tokenize(sent)
                    l1 += len(toks)
                for sent in tsents2[tid]:
                    toks = self.tokenizer.tokenize(sent)
                    l2 += len(toks)
                penalty = np.e ** (-((l1 - l2) ** 2) / (2 * self.PENALTY_SIGMA ** 2))
                fb1 *= penalty
            prs.append(mean_precision)
            rcs.append(mean_recall)
            fb1s.append(fb1)
            stats.append({'scores': sent_probs, 'threshes': (thresh1, thresh2)})
        if prf == self.METRIC_PRECISION:
            fb1s = prs
        elif prf == self.METRIC_RECALL:
            fb1s = rcs
        return prs, rcs, fb1s, stats

    def sentence_scores_tfidf(self, texts1, texts2, label='entailment', vectorizer=None):
        prs, rcs, fb1s, stats = [], [], [], []
        # Calculate TFIDF similarities
        probs_tids = OrderedDict()
        tids_all, prems_all, hypos_all = [], [], []
        prems, hypos, pbfs, fidxs, sent_lengths = {}, {}, {}, {}, {}
        for tid, (text1, text2) in enumerate(zip(texts1, texts2)):
            probs_tids[tid] = []
            prems[tid], hypos[tid], pbfs[tid], fidxs[tid] = [], [], [], []
            sents1 = self.sentsplitter.split(text1)
            sents2 = self.sentsplitter.split(text2)
            if len(sents1) > 0 and len(sents2) > 0:
                vecs1 = vectorizer.transform(sents1)
                vecs2 = vectorizer.transform(sents2)
                scores = cosine_similarity(vecs1, vecs2)
                scores1 = np.argmax(scores, axis=1)
                scores2 = np.argmax(scores, axis=0)
                sent_lengths[tid] = (len(scores1), len(scores2))
                for i, j in enumerate(scores1):
                    tids_all.append(tid)
                    prems_all.append(sents2[i])
                    hypos_all.append(sents1[j])
                    prems[tid].append(sents2[i])
                    hypos[tid].append(sents1[j])
                    pbfs[tid].append(scores[i, j])
                    fidxs[tid].append(j)
                for j, i in enumerate(scores2):
                    tids_all.append(tid)
                    prems_all.append(sents1[j])
                    hypos_all.append(sents2[i])
                    prems[tid].append(sents1[j])
                    hypos[tid].append(sents2[i])
                    pbfs[tid].append(scores[i, j])
                    fidxs[tid].append(i)
        # NLI probabilities
        probs_all, _ = self.predict(prems_all, hypos_all)
        for i, tid in enumerate(tids_all):
            probs_tids[tid].append(probs_all[i])

        for tid, probs in probs_tids.items():
            sent_len1, sent_len2 = sent_lengths[tid]
            idx = 0
            sent_probs = [{}, {}]
            rc_pr = [[], []]
            if len(probs) > 0:
                for k, sent_len in zip([0, 1], [sent_len1, sent_len2]):
                    for i in range(sent_len):
                        if label == self.LABEL_ALL:
                            prob = probs[idx]
                            rc_pr[k].append(prob[self.LABEL_ENTAIL])
                        else:
                            prob = probs[idx][label]
                            neutral_prob = probs[idx][self.LABEL_NEUTRAL]
                            if neutral_prob > prob and neutral_prob > probs[idx][self.LABEL_CONTRADICT]:
                                prob = -1.0
                                rc_pr[k].append(0.5)
                            else:
                                rc_pr[k].append(prob)
                        sent_probs[k][i] = (prob, pbfs[tid][idx], prems[tid][idx], hypos[tid][idx], fidxs[tid][idx])
                        idx += 1
            mean_precision = sum(rc_pr[1]) / len(rc_pr[1]) if len(rc_pr[1]) > 0 else 0.0
            mean_recall = sum(rc_pr[0]) / len(rc_pr[0]) if len(rc_pr[0]) > 0 else 0.0
            if mean_precision + mean_recall > 0.0:
                fb1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            else:
                fb1 = 0.0
            prs.append(mean_precision)
            rcs.append(mean_recall)
            fb1s.append(fb1)
            stats.append({'scores': sent_probs})
        return prs, rcs, fb1s, stats

    def stop(self):
        pass


class BERTScorer:
    PENALTY_SIGMA = 6.0

    def __init__(self, refs=None, model_type=None, num_layers=None, verbose=False, idf=False, batch_size=16, nthreads=2,
                 all_layers=False, lang=None, rescale_with_baseline=False, penalty=False):
        assert lang is not None or model_type is not None, 'Either lang or model_type should be specified'
        if rescale_with_baseline:
            assert lang is not None, 'Need to specify Language when rescaling with baseline'

        if model_type is None:
            lang = lang.lower()
            model_type = lang2model[lang]
        if num_layers is None:
            num_layers = model2layers[model_type]

        if model_type.startswith('scibert'):
            tokenizer = AutoTokenizer.from_pretrained(cache_scibert(model_type))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = get_model(model_type, num_layers, all_layers)

        if not idf:
            idf_dict = defaultdict(lambda: 1.)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[tokenizer.sep_token_id] = 0
            idf_dict[tokenizer.cls_token_id] = 0
        elif isinstance(idf, dict):
            if verbose:
                print('using predefined IDF dict...')
            idf_dict = idf
        else:
            if verbose:
                print('preparing IDF dict...')
            start = time.perf_counter()
            idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)
            if verbose:
                print('done in {:.2f} seconds'.format(time.perf_counter() - start))

        self.batch_size = batch_size
        self.verbose = verbose
        self.all_layers = all_layers
        self.penalty = penalty

        self.tokenizer = tokenizer
        self.model = model
        self.idf_dict = idf_dict
        self.device = 'cpu'

        self.baselines = None
        if rescale_with_baseline:
            baseline_path = None
            for sitepackage in site.getsitepackages():
                if baseline_path is None:
                    candidate_path = os.path.join(sitepackage, 'bert_score',
                                                  f'rescale_baseline/{lang}/{model_type}.tsv')
                    if os.path.exists(candidate_path):
                        baseline_path = candidate_path
            if baseline_path is not None and os.path.isfile(baseline_path):
                if not all_layers:
                    baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[num_layers].to_numpy())[1:].float()
                else:
                    baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()
                self.baselines = baselines
            else:
                print(f'Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}', file=sys.stderr)

    def cuda(self):
        self.device = 'cuda:0'
        self.model.cuda()
        return self

    def score(self, cands, refs):
        assert len(cands) == len(refs)

        if self.verbose:
            print('calculating scores...')
        start = time.perf_counter()
        all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, self.idf_dict, verbose=self.verbose,
                                       device=self.device, batch_size=self.batch_size, all_layers=self.all_layers).cpu()
        if self.baselines is not None:
            all_preds = (all_preds - self.baselines) / (1 - self.baselines)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2] # P, R, F
        if self.penalty:
            for idx, (cand, ref) in enumerate(zip(cands, refs)):
                toks1 = self.tokenizer.tokenize(cand)
                toks2 = self.tokenizer.tokenize(ref)
                penalty = np.e ** (-((len(toks1) - len(toks2)) ** 2) / (2 * self.PENALTY_SIGMA ** 2))
                out[-1][idx] *= penalty

        if self.verbose:
            time_diff = time.perf_counter() - start
            print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

        return out


class SimpleNLI(_NLIScorer):
    CONFIG_FILENAME = 'config.json'
    DEFAULT_BERT_TYPE = 'bert'
    DEFAULT_NAME = 'bert-base-uncased'
    DEFAULT_STATES = 'model_mednli_13k'
    RADNLI_STATES = 'model_medrad_19k'
    STATES_FILENAME = 'model.dict.gz'

    def __init__(self, model, neutral_score=(1.0 / 3), batch=16, nthreads=2, pin_memory=False, bert_score=None,
                 sentsplitter='none', cache=None, verbose=False):
        super(SimpleNLI, self).__init__(neutral_score, batch, nthreads, pin_memory, bert_score, sentsplitter, cache,
                                        verbose)
        self.model = DataParallel(model)
        self.model.eval()

    @classmethod
    def load_model(cls, states=None):
        resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources')
        if states is None:
            name = cls.DEFAULT_NAME
            states = os.path.join(resource_dir, cls.DEFAULT_STATES, cls.STATES_FILENAME)
            bert_type = cls.DEFAULT_BERT_TYPE
        else:
            config_path = os.path.join(states, cls.CONFIG_FILENAME)
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)
            bert_type = config['bert_type']
            if bert_type == 'bert':
                name = 'bert-base-uncased'
            elif bert_type == 'distilbert':
                name = 'distilbert-base-uncased'
            else:
                raise ValueError('Unknown BERT type {0}'.format(bert_type))
            states = os.path.join(states, cls.STATES_FILENAME)
        bertnli = BERTNLI(name, bert_type=bert_type, length=384, force_lowercase=True, device='cpu')
        with gzip.open(states, 'rb') as f:
            states_dict = torch.load(f, map_location=torch.device('cpu'))
        bertnli.load_state_dict(states_dict)
        return bertnli

    def predict(self, sent1s, sent2s):
        batches, buf1, buf2 = [], [], []
        for sent1, sent2 in zip(sent1s, sent2s):
            buf1.append(sent1)
            buf2.append(sent2)
            if len(buf1) >= self.batch:
                batches.append((buf1, buf2))
                buf1, buf2 = [], []
        if len(buf1) > 0:
            batches.append((buf1, buf2))

        probs, preds = [], []
        with torch.no_grad():
            for b1, b2 in batches:
                out = self.model(b1, b2)
                out = softmax(out, dim=-1).detach().cpu()
                _, idxs = out.max(dim=-1)
                for i, idx in enumerate(idxs):
                    idx = int(idx)
                    probs.append({'entailment': float(out[i][BERTNLI.LABEL_ENTAILMENT]),
                                  'neutral': float(out[i][BERTNLI.LABEL_NEUTRAL]),
                                  'contradiction': float(out[i][BERTNLI.LABEL_CONTRADICTION])})
                    if idx == BERTNLI.LABEL_ENTAILMENT:
                        preds.append('entailment')
                    elif idx == BERTNLI.LABEL_NEUTRAL:
                        preds.append('neutral')
                    elif idx == BERTNLI.LABEL_CONTRADICTION:
                        preds.append('contradiction')
                    else:
                        raise ValueError('Unknown label index {0}'.format(idx))
        return probs, preds

