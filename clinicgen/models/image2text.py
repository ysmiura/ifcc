#! /bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import torch
import torch.nn
from torch import exp, tanh
from torch.distributions.categorical import Categorical
from torch.nn import Linear
from torch.nn.functional import log_softmax, softmax
from torch.nn.utils import clip_grad_norm_
from clinicgen.data.image2text import PretrainedEmbeddings
from clinicgen.eval import GenEval
from clinicgen.optmizer import Optimizers
from clinicgen.utils import data_cuda


class _Image2Text(torch.nn.Module):
    EOS_SCORE = 1000.0
    MULTI_MERGE_ATT = 'att'
    MULTI_MERGE_MAX = 'max'
    NLI_CACHE = 200000
    OP_ADD = 'add'
    OP_MUL = 'mul'

    def __init__(self, max_word, multi_image=1, multi_merge='att', teacher_forcing=None, image_finetune_epoch=None,
                 rl_opts=None, word_idxs=None, verbose=False):
        super(_Image2Text, self).__init__()
        self.max_word = max_word
        self.multi_image = multi_image
        self.multi_merge = multi_merge
        self.teacher_forcing = teacher_forcing
        self.image_finetune_epoch = image_finetune_epoch
        self.verbose = verbose
        # Self-critical RL process
        self.rl_train = False
        if rl_opts is not None:
            self.rl_opts = rl_opts
            self.evaluator = GenEval(None, word_idxs, 1, bleu=rl_opts.bleu, rouge=rl_opts.rouge, cider=rl_opts.cider,
                                     cider_df=rl_opts.cider_df, spice=rl_opts.spice, bert_score=rl_opts.bert_score,
                                     bert_score_penalty=rl_opts.bert_score_penalty, nli=rl_opts.nli,
                                     nli_compare=rl_opts.nli_compare, nli_label=rl_opts.nli_label,
                                     nli_neutral_score=rl_opts.nli_neutral_score, nli_prf=rl_opts.nli_prf,
                                     nli_batch=rl_opts.nli_batch, nli_cache=self.NLI_CACHE,
                                     entity_match=rl_opts.entity_match, entity_mode=rl_opts.entity_mode,
                                     nthreads=rl_opts.nthreads, pin_memory=rl_opts.pin_memory, sentsplitter='nltk',
                                     verbose=verbose)
            self.evaluator.setup()
            self.eval_idxs = []
            for idx, name in self.evaluator.metrics().items():
                if name in self.rl_opts.metrics:
                    self.eval_idxs.append(idx)
        else:
            self.rl_opts = None
            self.evaluator, self.eval_idxs = None, None

    @staticmethod
    def _beam_buffer(batch_size, beam_size, initial=False):
        beam_buffer = np.zeros((batch_size, beam_size), dtype='object')
        if initial:
            for m in range(batch_size):
                for k in range(beam_size):
                    beam_buffer[m, k] = []
        return beam_buffer

    @classmethod
    def dummy_stops(cls, words):
        return torch.tensor(np.zeros((words.shape[0], 1), dtype='float'))

    @classmethod
    def optimize(cls, name, epoch, image_finetune_epoch):
        if name == Optimizers.IMAGE:
            if epoch is not None and image_finetune_epoch is not None and epoch < image_finetune_epoch:
                return False
            else:
                return True
        else:
            return True

    @classmethod
    def truncate_targ(cls, targ):
        mask = ((targ > 0).sum(dim=(1, 2))).type(torch.long)
        max_len = int(mask.max().cpu())
        return targ[:, :, :max_len]

    def _decode_words_beam(self, words, states, beam_size, allow_stop=True, recover_words=None, diversity_rate=0.0):
        if beam_size == 1:
            return self._decode_words_greedy(words, states, allow_stop=allow_stop, recover_words=recover_words)
        elif diversity_rate > 0.0:
            return self._decode_words_diverse(words, states, beam_size, recover_words, diversity_rate)

        logs = {'word': [], 'score': [], 'score_current': [], 'beam_buffer': []} if recover_words is not None else None
        n_states = len(states)
        states = [states]
        scores = words.new_zeros((words.shape[0], 1), dtype=torch.float)
        stop_states = np.ones(words.shape[0], dtype='uint8')
        zero_index = words.new_zeros((1,), dtype=torch.long)[0]
        beam_buffer = self._beam_buffer(words.shape[0], beam_size, True)

        # Decodes words
        for j in range(self.max_word):
            if stop_states.sum() <= 0 and allow_stop:
                # Skip the word generation since all generation are stopped
                for m in range(words.shape[0]):
                    for k in range(beam_size):
                        beam_buffer[m, k].append(PretrainedEmbeddings.INDEX_PAD)
            else:
                # Generate j-th words for the current batch
                idx_buffer, next_word_buffer, next_score_buffer, score_buffer = [], [], [], []
                state_buffer = [[] for _ in range(n_states)]
                for k in range(words.shape[1]):
                    # For each previous word in beam size B
                    w = words[:, k]
                    s = states[k]
                    # proc_word should be overridden in subclasses
                    p, s = self.proc_word(w, s)
                    # p is a pre-softmax word distribution
                    p = -log_softmax(p, dim=-1).detach()
                    ps, idxs = p.topk(beam_size, largest=False, dim=-1)
                    for l in range(idxs.shape[-1]):
                        # For each current word in beam size B
                        idx_buffer.append(np.full(words.shape[0], k, dtype='int64'))
                        for i in range(n_states):
                            state_buffer[i].append(s[i])
                        next_word_buffer.append(idxs[:, l])
                        next_score_buffer.append(ps[:, l] + scores[:, k])
                        if recover_words is not None:
                            score_buffer.append(ps[:, l])
                # Buffers in the size of (NxB**2)
                idx_buffer = np.stack(idx_buffer, axis=1)
                state_buffer = list(map(lambda sbuf: torch.stack(sbuf, dim=1), state_buffer))
                next_word_buffer = torch.stack(next_word_buffer, dim=1)
                next_score_buffer = torch.stack(next_score_buffer, dim=1)

                if recover_words is not None:
                    rwords = []
                    for idxs in next_word_buffer.cpu().numpy():
                        rwords.append(recover_words.array(idxs))
                    logs['word'].append(rwords)
                    logs['score'].append(exp(-next_score_buffer.cpu()).numpy())
                    score_buffer = torch.stack(score_buffer, dim=1)
                    logs['score_current'].append(exp(-score_buffer.cpu()).numpy())
                # Take top-B of B**2 candidates
                ps, idxs = next_score_buffer.topk(beam_size, largest=False, dim=1)
                idxs_np = idxs.cpu().numpy()
                new_words, scores = [], []
                new_states = [[] for _ in range(n_states)]
                new_beam_buffer = self._beam_buffer(words.shape[0], beam_size)
                rc_beam_buffer = self._beam_buffer(words.shape[0], beam_size) if recover_words is not None else None
                for m in range(next_word_buffer.shape[0]):
                    nwb = next_word_buffer[m, idxs[m]]
                    if stop_states[m] != 0:
                        # Choose top-B for each instance in the batch
                        ei = torch.where(nwb == PretrainedEmbeddings.INDEX_EOS)
                        stop = False
                        if len(ei[0]) > 0:
                            eos_idx = ei[0]
                            if torch.equal(eos_idx[0], zero_index):
                                # If the most probable word is EOS, stop the generation of that batch.
                                stop_states[m] = 0
                                stop = True
                            else:
                                # If top-B includes EOS, force EOS to be not selected in the next iteration.
                                ps[m][eos_idx] = ps.new_full((len(eos_idx),), self.EOS_SCORE)
                        nwb_cpu = nwb.cpu()
                        if not stop:
                            # Update the beam buffer.
                            for current_idx in range(len(nwb)):
                                prev_idx = idx_buffer[m, idxs_np[m]][current_idx]
                                w = int(nwb_cpu[current_idx])
                                new_beam_buffer[m, current_idx] = beam_buffer[m, prev_idx] + [w]
                                if recover_words is not None:
                                    rc_beam_buffer[m, current_idx] = recover_words.array(beam_buffer[m, prev_idx] +
                                                                                         [w])
                        else:
                            # Force the decoder to output EOS.
                            for current_idx in range(len(nwb)):
                                prev_idx = idx_buffer[m, idxs_np[m]][current_idx]
                                w = PretrainedEmbeddings.INDEX_EOS
                                new_beam_buffer[m, current_idx] = beam_buffer[m, prev_idx] + [w]
                                if recover_words is not None:
                                    rc_beam_buffer[m, current_idx] = recover_words.array(beam_buffer[m, prev_idx] +
                                                                                         [w])
                    else:
                        # Force the decoder to output PAD.
                        for current_idx in range(len(nwb)):
                            w = PretrainedEmbeddings.INDEX_PAD
                            new_beam_buffer[m, current_idx] = beam_buffer[m, current_idx] + [w]
                            if recover_words is not None:
                                rc_beam_buffer[m, current_idx] = recover_words.array(beam_buffer[m, current_idx] +
                                                                                     [w])
                    new_words.append(nwb)
                    for i, state in enumerate(state_buffer):
                        if state.shape[0] == 1:
                            new_states[i].append(state[0, idxs[m]])
                        else:
                            new_states[i].append(state[m, idxs[m]])
                    scores.append(ps[m])
                # Update top-B words and scores (NxB)
                words = torch.stack(new_words, dim=0)
                new_states = [torch.stack(s, dim=1) for s in new_states]
                states = []
                for k in range(words.shape[1]):
                    s = []
                    for i in range(n_states):
                        s.append(new_states[i][k])
                    states.append(tuple(s))
                scores = torch.stack(scores, dim=0)
                beam_buffer = new_beam_buffer
                if recover_words is not None:
                    logs['beam_buffer'].append(rc_beam_buffer)

        # Convert the beam search result to a tensor of (NxBxW)
        beam_words = np.zeros((beam_buffer.shape[0], beam_buffer.shape[1], self.max_word), dtype='long')
        for m in range(beam_buffer.shape[0]):
            for k in range(beam_buffer.shape[1]):
                beam_words[m, k] = np.array(beam_buffer[m, k], dtype='long')
        return torch.tensor(beam_words), logs

    def _decode_words_diverse(self, words, states, beam_size, recover_words=None, diversity_rate=0.0):
        logs = {'word': [], 'score': [], 'score_current': [], 'beam_buffer': []} if recover_words is not None else None
        n_states = len(states)
        states = [states]
        scores = words.new_zeros((words.shape[0], 1), dtype=torch.float)
        zero_index = words.new_zeros((1,), dtype=torch.long)[0]
        beam_buffer = self._beam_buffer(words.shape[0], beam_size, True)
        beam_results = {}
        for m in range(words.shape[0]):
            beam_results[m] = []

        # Decodes words
        for j in range(self.max_word):
            # Generate j-th words for the current batch
            idx_buffer, next_word_buffer, next_score_buffer, score_buffer = [], [], [], []
            state_buffer = [[] for _ in range(n_states)]
            for k in range(words.shape[1]):
                # For each previous word in beam size B
                w = words[:, k]
                s = states[k]
                # proc_word should be overridden in subclasses
                p, s = self.proc_word(w, s)
                # p is a pre-softmax word distribution
                p = -log_softmax(p, dim=-1).detach()
                ps, idxs = p.topk(beam_size, largest=False, dim=-1)
                for l in range(idxs.shape[-1]):
                    # For each current word in beam size B
                    idx_buffer.append(np.full(words.shape[0], k, dtype='int64'))
                    for i in range(n_states):
                        state_buffer[i].append(s[i])
                    next_word_buffer.append(idxs[:, l])
                    # diversity penalty
                    penalty = float(l + 1) * diversity_rate if j > 0 else 0.0
                    next_score_buffer.append(ps[:, l] + penalty + scores[:, k])
                    if recover_words is not None:
                        score_buffer.append(ps[:, l])
            # Buffers in the size of (NxB**2)
            idx_buffer = np.stack(idx_buffer, axis=1)
            state_buffer = list(map(lambda sbuf: torch.stack(sbuf, dim=1), state_buffer))
            next_word_buffer = torch.stack(next_word_buffer, dim=1)
            next_score_buffer = torch.stack(next_score_buffer, dim=1)

            if recover_words is not None:
                rwords = []
                for idxs in next_word_buffer.cpu().numpy():
                    rwords.append(recover_words.array(idxs))
                logs['word'].append(rwords)
                logs['score'].append(next_score_buffer.cpu().numpy())
                score_buffer = torch.stack(score_buffer, dim=1)
                logs['score_current'].append(score_buffer.cpu().numpy())
            # Take top-B of B**2 candidates
            ps, idxs = next_score_buffer.topk(beam_size, largest=False, dim=1)
            idxs_np = idxs.cpu().numpy()
            new_words, scores = [], []
            new_states = [[] for _ in range(n_states)]
            new_beam_buffer = self._beam_buffer(words.shape[0], beam_size)
            rc_beam_buffer = self._beam_buffer(words.shape[0], beam_size) if recover_words is not None else None
            for m in range(next_word_buffer.shape[0]):
                nwb = next_word_buffer[m, idxs[m]]
                # Choose top-B for each instance in the batch
                ei = torch.where(nwb == PretrainedEmbeddings.INDEX_EOS)
                if len(ei[0]) > 0:
                    eos_idx = ei[0]
                    if torch.equal(eos_idx[0], zero_index):
                        # If the most probable word is EOS, add it to the beam result.
                        prev_idx = idx_buffer[m, idxs_np[m]][eos_idx[0]]
                        beam_results[m].append(beam_buffer[m, prev_idx] + [int(nwb[eos_idx[0]])])
                    # If top-B includes EOS, force EOS to be not selected in the next iteration.
                    ps[m][eos_idx] = ps.new_full((len(eos_idx),), self.EOS_SCORE)
                nwb_cpu = nwb.cpu()
                # Update the beam buffer.
                for current_idx in range(len(nwb)):
                    prev_idx = idx_buffer[m, idxs_np[m]][current_idx]
                    w = int(nwb_cpu[current_idx])
                    new_beam_buffer[m, current_idx] = beam_buffer[m, prev_idx] + [w]
                    if recover_words is not None:
                        rc_beam_buffer[m, current_idx] = recover_words.array(beam_buffer[m, prev_idx] + [w])
                new_words.append(nwb)
                for i, state in enumerate(state_buffer):
                    if state.shape[0] == 1:
                        new_states[i].append(state[0, idxs[m]])
                    else:
                        new_states[i].append(state[m, idxs[m]])
                scores.append(ps[m])
            # Update top-B words and scores (NxB)
            words = torch.stack(new_words, dim=0)
            new_states = [torch.stack(s, dim=1) for s in new_states]
            states = []
            for k in range(words.shape[1]):
                s = []
                for i in range(n_states):
                    s.append(new_states[i][k])
                states.append(tuple(s))
            scores = torch.stack(scores, dim=0)
            beam_buffer = new_beam_buffer
            if recover_words is not None:
                logs['beam_buffer'].append(rc_beam_buffer)

        # Convert the beam search result to a tensor of (NxBxW)
        beam_words = np.zeros((beam_buffer.shape[0], beam_buffer.shape[1], self.max_word), dtype='long')
        for m in range(beam_buffer.shape[0]):
            for k in range(beam_buffer.shape[1]):
                rs_len = len(beam_results[m])
                if rs_len - 1 >= k:
                    buf = beam_results[m][k]
                    for _ in range(self.max_word - len(buf)):
                        buf.append(PretrainedEmbeddings.INDEX_PAD)
                    beam_words[m, k] = np.array(buf, dtype='long')
                else:
                    beam_words[m, k] = np.array(beam_buffer[m, k - rs_len], dtype='long')
        return torch.tensor(beam_words), logs

    def _decode_words_greedy(self, words, states, allow_stop=True, recover_words=None):
        logs = {'word': []} if recover_words is not None else None
        words = words[:, 0]
        stop_states = np.ones(words.shape[0], dtype='uint8')
        greedy_buffer = torch.full((words.shape[0], 1, self.max_word), PretrainedEmbeddings.INDEX_PAD, dtype=torch.long)

        # Decodes words
        for j in range(self.max_word):
            if stop_states.sum() <= 0 and allow_stop:
                # Skip the word generation since all generation are stopped
                greedy_buffer[:, 0, j] = torch.full((words.shape[0],), PretrainedEmbeddings.INDEX_PAD, dtype=torch.long)
            else:
                # proc_word should be overridden in subclasses
                p, states = self.proc_word(words, states)
                # p is a pre-softmax word distribution
                p = -log_softmax(p, dim=-1).detach()
                # Pick greedy words
                words = p.argmin(dim=-1)
                words_cpu = words.cpu()
                words_np = words_cpu.numpy()
                stop_states *= (words_np != PretrainedEmbeddings.INDEX_EOS).astype('uint8')
                greedy_buffer[:, 0, j] = words_cpu
                if recover_words is not None:
                    logs['word'].append(recover_words.array(words_np))
        return greedy_buffer, logs

    def _init_multi_image(self, image_dim, visual_num, rnn_dim):
        if self.multi_image > 1 and self.multi_merge == self.MULTI_MERGE_ATT:
            self.att_vg = Linear(image_dim, image_dim)
            self.att_vg_a = Linear(image_dim, 1)

    def _nll_step(self, encoded_data, targ, device, non_blocking, ids=None):
        """
        Process an NLL training step
        :param encoded_data: Encoded input images and meta data
        :param targ: Output texts
        :param device: A CUDA device
        :param non_blocking: A non-blocking option for CUDA data
        :param ids: Instance IDs
        :return: A tuple of (total loss, losses and rewards)
        """
        raise NotImplementedError()

    def _rl_step(self, encoded_data, targ, device, non_blocking, ids=None):
        """
        Process an RL training step
        :param encoded_data: Encoded input images and meta data
        :param targ: Output texts
        :param device: A CUDA device
        :param non_blocking: A non-blocking option for CUDA data
        :param ids: Instance IDs
        :return: A tuple of (total loss, losses and rewards)
        """
        raise NotImplementedError()

    def _sample_words(self, word, states, nucleus_p=None, allow_stop=True):
        if nucleus_p is not None:
            return self._sample_words_nucleus(word, states, pthresh=nucleus_p, allow_stop=allow_stop)
        words, lprobs = [], []
        stop_states = None
        for j in range(self.max_word):
            if stop_states is not None and stop_states.sum() <= 0 and allow_stop:
                words.append(word.new_full((word.shape[0],), PretrainedEmbeddings.INDEX_PAD, dtype=torch.long))
                lprobs.append(word.new_zeros((word.shape[0],), dtype=torch.float))
            else:
                p, states = self.proc_word(word, states)
                lp = log_softmax(p, dim=-1)
                cat = Categorical(logits=lp)
                word = cat.sample()
                word_np = word.cpu().numpy()
                if stop_states is None:
                    stop_states = (word_np != PretrainedEmbeddings.INDEX_EOS).astype('uint8')
                else:
                    stop_states *= (word_np != PretrainedEmbeddings.INDEX_EOS).astype('uint8')
                words.append(word)
                idx = np.array([np.arange(word.shape[0]), word.detach().cpu().numpy()])
                lprobs.append(lp[idx])
        words = torch.stack(words, dim=1)
        lprobs = torch.stack(lprobs, dim=1)
        return words, lprobs

    def _sample_words_nucleus(self, word, states, pthresh=0.95, allow_stop=True):
        words, lprobs = [], []
        stop_states = None
        for j in range(self.max_word):
            if stop_states is not None and stop_states.sum() <= 0 and allow_stop:
                words.append(word.new_full((word.shape[0],), PretrainedEmbeddings.INDEX_PAD, dtype=torch.long))
                lprobs.append(word.new_zeros((word.shape[0],), dtype=torch.float))
            else:
                p, states = self.proc_word(word, states)
                vals, idxs = p.sort(descending=True)
                cvals = softmax(vals, dim=-1).cumsum(dim=-1)
                rcidxs = cvals > pthresh
                rcidxs[:, 1:] = rcidxs[:, :-1].clone()
                rcidxs[:, 0] = False
                for k in range(rcidxs.shape[0]):
                    ridxs = idxs[k][rcidxs[k]]
                    p[k][ridxs] = -float('inf')
                cat = Categorical(logits=p)
                word = cat.sample()
                word_np = word.cpu().numpy()
                if stop_states is None:
                    stop_states = (word_np != PretrainedEmbeddings.INDEX_EOS).astype('uint8')
                else:
                    stop_states *= (word_np != PretrainedEmbeddings.INDEX_EOS).astype('uint8')
                words.append(word)
                idx = np.array([np.arange(word.shape[0]), word.detach().cpu().numpy()])
                lprobs.append(log_softmax(p[idx], dim=-1))
        words = torch.stack(words, dim=1)
        lprobs = torch.stack(lprobs, dim=1)
        return words, lprobs

    def cuda(self, device=None):
        super(_Image2Text, self).cuda()
        if self.evaluator is not None:
            self.evaluator = self.evaluator.cuda()
        return self

    def decode_beam(self, encoded_data, beam_size, allow_stop=True, recover_words=None, diversity_rate=0.0):
        """
        Decode texts with beam search.
        :param encoded_data: Encoded data (Input images (NxCxHxW) and additional meta data)
        :param beam_size: Beam size B
        :param allow_stop: Suspend decoding when all sentences or words are in stop states
        :param recover_words: Log decoded words and scores
        :param diversity_rate: Diversity rate
        :return: Decoded texts in (NxBxW)
        """
        raise NotImplementedError()

    def deflatten_image(self, x):
        if self.multi_image > 1:
            if len(x.shape) == 2:
                x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, x.shape[1])
            elif len(x.shape) == 3:
                x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, x.shape[1], x.shape[2])
            else:
                raise ValueError('Unknown flattened image shape {0}'.format(x.shape))
        return x

    def encode(self, inp, meta):
        raise NotImplementedError()

    def flatten_image(self, x):
        if self.multi_image > 1:
            return x.flatten(start_dim=0, end_dim=1)
        else:
            return x

    def image_features_with_mask(self, x, model):
        mask = (x.detach().sum(dim=(1, 2, 3)) != 0).type(torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.multi_image > 1:
            nz = mask.squeeze().nonzero().squeeze()
            x_nz = x[nz]
            if len(nz.shape) == 0:
                x_nz = x_nz.unsqueeze(dim=0)
            x_nz = model(x_nz)
            x = x.new_zeros(x.shape[0], x_nz.shape[1], x_nz.shape[2], x_nz.shape[3])
            if len(nz.shape) == 0:
                x_nz = x_nz.squeeze(dim=0)
            x[nz] += x_nz
        else:
            x = model(x)
        return x, mask

    def loss_names(self):
        names = ['total', 'word']
        if self.rl_opts.epoch is not None:
            names.append('rl')
            names += self.rl_opts.metrics
        return names

    def loss_progress(self, loss_logs):
        if self.rl_opts.epoch is not None:
            return '%.2f,%.2f' % (np.mean(loss_logs[1]), np.mean(loss_logs[2]))
        else:
            return '%.2f' % np.mean(loss_logs[0])

    def meta_cuda(self, meta, device='gpu', non_blocking=False):
        return meta

    def multi_vg(self, vg):
        if self.multi_image > 1:
            if self.multi_merge == self.MULTI_MERGE_ATT:
                alpha = self.att_vg_a(tanh(self.att_vg(vg))).squeeze(dim=2)
                alpha = softmax(alpha, dim=-1)
                vg = (vg * alpha.unsqueeze(dim=2)).sum(dim=1)
            elif self.multi_merge == self.MULTI_MERGE_MAX:
                vg, _ = torch.max(vg, dim=1)
            else:
                raise ValueError('Unknown multi merge {0}'.format(self.multi_merge))
        return vg

    def proc_word(self, w, states):
        raise NotImplementedError()

    def sample(self, encoded_data, nucleus_p=None):
        """
        :param encoded_data: Encoded data (Input images (NxCxHxW) and additional meta data)
        :param nucleus_p: Nucleus sampling threshold
        :return: Decoded texts in (NxBxW)
        """

    def self_critical_reward(self, gens_sample, gens_greedy, references, masks_sample, log_probs, ids=None):
        """
        Calculate a self critical reward between sample texts and greedy-decoded texts
        :param gens_sample: Sampled texts (N,)
        :param gens_greedy: Greedy-decoded texts (N,)
        :param references: Reference texts (N,)
        :param masks_sample: Word masks for sampled texts (N, W) or (N, S, W)
        :param log_probs: Word log probabilities in shape of (N, W) or (N, S, W)
        :param ids: Instance IDs
        :return:
        """
        hids, hypos, refs, rids = [], {}, {}, []
        num_words = masks_sample.sum(axis=(1, 2)) if len(masks_sample.shape) == 3 else masks_sample.sum(axis=1)
        # Evaluation pairs between references and samples
        for i in range(len(gens_sample)):
            sample_id = 's{0}'.format(i)
            hids.append(sample_id)
            hypos[sample_id] = [gens_sample[i]]
            refs[sample_id] = [references[i]]
            rids.append(ids[i])
        # Evaluation pairs between references and greedy-decodes
        for i in range(len(gens_greedy)):
            greedy_id = 'g{0}'.format(i)
            hids.append(greedy_id)
            hypos[greedy_id] = [gens_greedy[i]]
            refs[greedy_id] = [references[i]]
            rids.append(ids[i])
        _, scores_detailed = self.evaluator.eval(hids, refs, hypos, self.rl_opts.tfidf, ref_ids=rids)
        # Rewards and losses
        log_probs *= log_probs.new_tensor(masks_sample)
        # Calculate mean log word probabilities considering pads
        log_probs = log_probs.sum(axis=(1, 2)) if len(masks_sample.shape) == 3 else log_probs.sum(axis=1)
        log_probs /= log_probs.new_tensor(num_words)
        rewards, losses = [], []
        if self.rl_opts.op == self.OP_ADD:
            for i, idx in enumerate(self.eval_idxs):
                scores = np.array(scores_detailed[idx])
                # Calculate a mean reward difference between samples and greedy-decodes
                reward = scores[:len(gens_sample)] - scores[len(gens_sample):]
                rewards.append(reward.mean())
                loss = -log_probs.new_tensor(reward) * log_probs
                # Multiply a corresponding RL weight (Note that index 0 is an NLL weight)
                loss = self.rl_opts.weights[i + 1] * loss.mean()
                losses.append(loss)
            loss_acc = sum(losses)
        elif self.rl_opts.op == self.OP_MUL:
            mscores = []
            for i, idx in enumerate(self.eval_idxs):
                scores = np.array(scores_detailed[idx])
                scores += self.rl_opts.weights[i + 1]
                mscores.append(scores)
            # Calculate a mean reward difference between samples and greedy-decodes
            scores = np.prod(mscores, axis=0)
            reward = scores[:len(gens_sample)] - scores[len(gens_sample):]
            rewards.append(reward.mean())
            loss = -log_probs.new_tensor(reward) * log_probs
            loss_acc = loss.mean()
            v = (1.0 - self.rl_opts.weights[0]) / (1.0 + sum(self.rl_opts.weights[1:]))
            loss_acc *= v
        else:
            raise ValueError('Unknown RL operator {0}'.format(self.rl_opts.op))
        return rewards, loss_acc

    def train_step(self, inp, targ, optimizers, ids=None, schedulers=None, meta=None, clip_grad=None, device='gpu',
                   non_blocking=False, epoch=None):
        """
        Process a single training step
        :param inp: Input images
        :param targ: Output texts
        :param optimizers: Optimizers
        :param ids: Instance IDs
        :param schedulers: Schedulers for batch update
        :param meta: Input meta data
        :param clip_grad: A clip gradient parameter
        :param device: A CUDA device
        :param non_blocking: A non-blocking option for CUDA data
        :param epoch: The current number of epoch
        :return: Losses and rewards
        """
        # Define update steps (_nll_step for NLL and _rl_step for RL)
        if self.rl_opts.epoch is not None and epoch >= self.rl_opts.epoch:
            # NLL+RL optimization (Individual RL weights are processed in self_critical_reward)
            steps = [(self._nll_step, self.rl_opts.weights[0]), (self._rl_step, 1.0)]
            self.rl_train = True
        else:
            # NLL optimization
            steps = [(self._nll_step, 1.0)]
            self.rl_train = False

        total_loss, vals = None, []
        inp = data_cuda(inp, device=device, non_blocking=non_blocking)
        meta = self.meta_cuda(meta, device=device, non_blocking=non_blocking)
        # Initialize optimizers
        for name, optimizer in optimizers.items():
            if self.optimize(name, epoch, self.image_finetune_epoch):
                optimizer.zero_grad()
        # Encode data (shared among multiple decoding processes)
        encoded_data = self.encode(inp, meta)
        # Decode data (NLL and/or RL)
        for step, weight in steps:
            loss, loss_reward_vals = step(encoded_data, targ, device, non_blocking, ids=ids)
            vals.append(float(loss.detach()))
            vals += loss_reward_vals
            loss *= weight
            total_loss = loss if total_loss is None else total_loss + loss
        # Optimize
        if not torch.isnan(total_loss):
            total_loss.backward()
        if clip_grad is not None:
            clip_grad_norm_(self.parameters(), clip_grad)
        if schedulers is not None:
            for _, scheduler in schedulers.items():
                scheduler.batch_step()
        for name, optimizer in optimizers.items():
            if self.optimize(name, epoch, self.image_finetune_epoch):
                optimizer.step()
        # Return losses and rewards
        vals = [float(total_loss.detach())] + vals
        return vals


class StepTFR(object):
    MIN_TEACHER_FORCING_RATE = 0.75

    def __init__(self, base_tfr, step_size, gamma=0.05, last_epoch=-1):
        self.base_tfr = base_tfr
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_tfr(self):
        if self.step_size is not None:
            tfr = max(self.base_tfr - self.gamma * (self.last_epoch // self.step_size), self.MIN_TEACHER_FORCING_RATE)
            return tfr
        else:
            return self.base_tfr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
