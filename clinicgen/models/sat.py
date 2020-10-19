#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
from torch import sigmoid, tanh
from torch.distributions.categorical import Categorical
from torch.nn import Dropout, Embedding, Linear, LSTMCell
from torch.nn.functional import cross_entropy, relu, softmax
from clinicgen.models.image import ImageClassification
from clinicgen.models.image2text import _Image2Text, PretrainedEmbeddings
from clinicgen.utils import data_cuda


class ShowAttendAndTell(_Image2Text):
    VISUAL_NUM = 196

    def __init__(self, embeddings, max_word=32, multi_image=1, multi_merge='att', context_dim=512, lstm_dim=1000,
                 lambda_a=1.0, teacher_forcing=None, image_model=None, image_pretrained=None, finetune_image=False,
                 image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                 verbose=False):
        super(ShowAttendAndTell, self).__init__(max_word, multi_image, multi_merge, teacher_forcing,
                                                image_finetune_epoch, rl_opts, word_idxs, verbose)
        self.feat_dim = context_dim
        self.lstm_dim = lstm_dim
        self.lambda_a = lambda_a

        self.dropout = Dropout(0.5)
        # Image processes
        if image_model is None:
            image_model = 'vgg'
        self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
                                                                         image_pretrained, device)
        self._init_multi_image(image_dim, self.VISUAL_NUM, lstm_dim)
        self.image_proj = Linear(image_dim, context_dim)
        # Word processes
        self.init_h = Linear(context_dim, lstm_dim)
        self.init_c = Linear(context_dim, lstm_dim)
        self.att_v = Linear(image_dim, image_dim)
        self.att_h = Linear(lstm_dim, image_dim)
        self.att_a = Linear(image_dim, 1)
        self.gate = Linear(lstm_dim, image_dim)
        input_dim = image_dim + embeddings.shape[1]
        self.lstm_word = LSTMCell(input_dim, lstm_dim)
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False,
                                                    padding_idx=PretrainedEmbeddings.INDEX_PAD)
        self.embed_num = self.embeddings.num_embeddings
        # Deep output
        self.lh = Linear(lstm_dim, embeddings.shape[1])
        self.lz = Linear(image_dim, embeddings.shape[1])
        self.lo = Linear(embeddings.shape[1], embeddings.shape[0], bias=False)

    def _init_multi_image(self, image_dim, visual_num, rnn_dim):
        super(ShowAttendAndTell, self)._init_multi_image(image_dim, visual_num, rnn_dim)
        if self.multi_image > 1 and self.multi_merge == self.MULTI_MERGE_ATT:
            self.att_z_v = Linear(image_dim, image_dim)
            self.att_z_h = Linear(rnn_dim, image_dim)
            self.att_z_a = Linear(image_dim, 1)

    def _nll_step(self, encoded_data, targ, device, non_blocking, ids=None):
        vl, vg = encoded_data
        y = data_cuda(targ, device=device, non_blocking=non_blocking)
        words, alphas = self.decode_teacher_forcing(y, vl, vg)
        loss, loss_alpha, loss_word = self.loss_nll(y, words, alphas)
        if loss_alpha is None:
            return loss, [float(loss_word.detach().cpu()), 0.0]
        else:
            return loss, [float(loss_word.detach().cpu()), float(loss_alpha.detach().cpu())]

    def _rl_step(self, encoded_data, targ, device, non_blocking, ids=None):
        self.eval()
        with torch.no_grad():
            _, words_greedy, _ = self.decode_beam(encoded_data, beam_size=1)
            gens_greedy, _ = self.evaluator.recover_words(words_greedy.squeeze(dim=1).squeeze(dim=1))
        self.train()
        words, log_probs = self.sample(encoded_data)
        gens_sample, masks_sample = self.evaluator.recover_words(words)
        gens_ref, _ = self.evaluator.recover_words(targ.squeeze(dim=1))
        rewards, loss_acc = self.self_critical_reward(gens_sample, gens_greedy, gens_ref, masks_sample, log_probs,
                                                      ids=ids)
        return loss_acc, rewards

    def decode_beam(self, encoded_data, beam_size, allow_stop=True, recover_words=None, diversity_rate=0.0):
        vl, vg = encoded_data
        # Initialize word process states
        ws = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        h = self.init_h(vg)
        c = self.init_c(vg)
        states = (vl, h, c, None)
        # Decode words
        beam_words, logs = self._decode_words_beam(ws, states, beam_size, allow_stop, recover_words, diversity_rate)
        dummy_stops = self.dummy_stops(beam_words)
        return dummy_stops, beam_words.unsqueeze(dim=1), logs

    def decode_teacher_forcing(self, y, vl, vg):
        # Masks
        not_masked = y.new_ones(1, dtype=torch.bool)[0]
        mask = ((y > 0).sum(dim=(0, 1)) > 0)
        # Initialize word states
        w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        vg = self.dropout(vg)
        h = self.init_h(vg)
        c = self.init_c(vg)
        states = (vl, h, c, None)
        # Process words
        words, alphas = [], []
        for j in range(self.max_word):
            if torch.equal(mask[j], not_masked):
                p, states = self.proc_word(w, states)
                words.append(p)
                _, _, _, alpha = states
                if alpha is not None:
                    alphas.append(alpha)
                if self.teacher_forcing is None or self.teacher_forcing.get_tfr() >= random.random():
                    w = y[:, 0, j]
                else:
                    p = softmax(p, dim=-1)
                    cat = Categorical(probs=p)
                    w = cat.sample()
            else:
                p = vg.new_ones(vg.shape[0], self.embed_num) / self.embed_num
                words.append(p)
                if self.teacher_forcing is None or self.teacher_forcing.get_tfr() >= random.random():
                    w = y[:, 0, j]
                else:
                    w = vg.new_zeros(vg.shape[0])
        words = torch.stack(words, dim=1)
        if len(alphas) > 0:
            alphas = torch.stack(alphas, dim=1)
        return words, alphas

    def encode(self, x, meta):
        return self.encode_image(x)

    def encode_image(self, x):
        # VGG-19 features
        x = self.flatten_image(x)
        x, mask = self.image_features_with_mask(x, self.image_feats)
        # Nx512x14x14 -> Nx196x512
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.permute(0, 2, 1)
        vl = self.deflatten_image(x * mask)
        # Projection to word embedding dim (Nx196x512)
        vg = relu(self.image_proj(x)) * mask
        # Mean visual feature (Nx512)
        vg = vg.mean(dim=1)
        vg = self.deflatten_image(vg)
        vg = self.multi_vg(vg)
        return vl, vg

    def forward(self, x, y, meta):
        # Encode image
        vl, vg = self.encode(x, meta)
        # Decode text
        return self.decode_teacher_forcing(y, vl, vg)

    def loss_nll(self, y, words, alphas):
        # Word loss
        words = words.unsqueeze(dim=1).permute(0, 3, 1, 2)
        loss_word = cross_entropy(words, y, ignore_index=PretrainedEmbeddings.INDEX_PAD, reduction='mean')
        if self.rl_train:
            loss = loss_word
            loss_alpha = None
        else:
            # Attention loss
            loss_alpha = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            # Total loss
            loss = loss_word + self.lambda_a * loss_alpha
        return loss, loss_alpha, loss_word

    def loss_names(self):
        names = ['total', 'word_att', 'word', 'att']
        if self.rl_opts.epoch is not None:
            names.append('rl')
            names += self.rl_opts.metrics
        return names

    def loss_progress(self, loss_logs):
        if self.rl_opts.epoch is not None:
            return '%.2f,%.2f,%.2f' % (np.mean(loss_logs[2]), np.mean(loss_logs[3]), np.mean(loss_logs[4]))
        else:
            return '%.2f,%.2f' % (np.mean(loss_logs[2]), np.mean(loss_logs[3]))

    def multi_z(self, z, h):
        if self.multi_image > 1:
            if self.multi_merge == self.MULTI_MERGE_ATT:
                alpha = self.att_z_a(tanh(self.att_z_z(z) + self.att_z_h(h).unsqueeze(dim=1))).squeeze(dim=2)
                alpha = softmax(alpha, dim=-1)
                z = (z * alpha.unsqueeze(dim=2)).sum(dim=1)
            elif self.multi_merge == self.MULTI_MERGE_MAX:
                z, _ = torch.max(z, dim=1)
            else:
                raise ValueError('Unknown multi merge {0}'.format(self.multi_merge))
        return z

    def proc_word(self, word, states):
        vl, h, c, _ = states
        if self.multi_image > 1:
            h_a = self.att_h(h).unsqueeze(dim=1).unsqueeze(dim=1)
            h_a = h_a.repeat(1, vl.shape[1], vl.shape[2], 1)
        else:
            h_a = self.att_h(h).unsqueeze(dim=1)
            h_a = h_a.repeat(1, vl.shape[1], 1)
        alpha = self.att_a(tanh(self.att_v(vl) + h_a)).squeeze(dim=-1)
        alpha = softmax(alpha, dim=-1)
        z = (vl * alpha.unsqueeze(dim=-1)).sum(dim=-2)
        z = self.multi_z(z, h)
        beta = sigmoid(self.gate(h))
        z = beta * z
        e = self.embeddings(word).squeeze(1)
        xw = torch.cat([e, z], dim=1)
        h, c = self.lstm_word(xw, (h, c))
        # deep output
        h2 = self.dropout(h)
        o = tanh(e + self.lh(h2) + self.lz(z))
        o = self.dropout(o)
        p = self.lo(o)
        if not self.rl_train:
            # Alpha will not be used in RL
            alpha = torch.zeros((word.shape[0],), dtype=torch.float32)
        else:
            # Alpha should be back trackable only in NLL training
            if not self.training:
                alpha = alpha.detach()
        return p, (vl, h, c, alpha)

    def sample(self, encoded_data, nucleus_p=None):
        vl, vg = encoded_data
        w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        vg = self.dropout(vg)
        h = self.init_h(vg)
        c = self.init_c(vg)
        states = (vl, h, c, None)
        return self._sample_words(w, states, nucleus_p)
