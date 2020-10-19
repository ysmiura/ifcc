#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import torch
from torch import sigmoid, tanh
from torch.distributions.categorical import Categorical
from torch.nn import Dropout, Embedding, Linear, LSTMCell
from torch.nn.functional import cross_entropy, relu, softmax
from clinicgen.models.image import ImageClassification
from clinicgen.models.image2text import _Image2Text, PretrainedEmbeddings
from clinicgen.utils import data_cuda


class KnowingWhenToLook(_Image2Text):
    VISUAL_NUM = 49

    def __init__(self, embeddings, feat_dim=512, max_word=32, multi_image=1, multi_merge='att', teacher_forcing=False,
                 image_model=None, image_pretrained=None, finetune_image=False, image_finetune_epoch=None, rl_opts=None,
                 word_idxs=None, device='gpu', verbose=False):
        super(KnowingWhenToLook, self).__init__(max_word, multi_image, multi_merge, teacher_forcing,
                                                image_finetune_epoch, rl_opts, word_idxs, verbose)
        self.feat_dim = feat_dim

        self.dropout = Dropout(0.5)
        # Image processes
        if image_model is None:
            image_model = 'resnet'
        self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
                                                                         image_pretrained, device)
        self._init_multi_image(image_dim, self.VISUAL_NUM, feat_dim)
        self.image_proj_l = Linear(image_dim, feat_dim)
        self.image_proj_g = Linear(image_dim, feat_dim)
        # Visual sentinel
        input_dim = feat_dim + embeddings.shape[1]
        self.vs_att_h = Linear(self.VISUAL_NUM, 1, bias=False)
        self.vs_att_v = Linear(feat_dim, self.VISUAL_NUM, bias=False)
        self.vs_att_g = Linear(feat_dim, self.VISUAL_NUM, bias=False)
        self.vs_att_s = Linear(feat_dim, self.VISUAL_NUM, bias=False)
        self.vs_dense1 = Linear(input_dim, feat_dim, bias=False)
        self.vs_dense2 = Linear(feat_dim, feat_dim, bias=False)
        # Word processes
        self.lstm_word = LSTMCell(input_dim, feat_dim)
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False,
                                                    padding_idx=PretrainedEmbeddings.INDEX_PAD)
        self.embed_num = self.embeddings.num_embeddings
        self.word_dense = Linear(feat_dim, embeddings.shape[0], bias=False)

    def _init_multi_image(self, image_dim, visual_num, rnn_dim):
        super(KnowingWhenToLook, self)._init_multi_image(image_dim, visual_num, rnn_dim)
        if self.multi_image > 1 and self.multi_merge == self.MULTI_MERGE_ATT:
            sentinel_num = visual_num + 1
            self.att_z_z = Linear(sentinel_num, sentinel_num)
            self.att_z_h = Linear(rnn_dim, sentinel_num)
            self.att_z_a = Linear(sentinel_num, 1)

    def _nll_step(self, encoded_data, targ, device, non_blocking, ids=None):
        vl, vg = encoded_data
        y = data_cuda(targ, device=device, non_blocking=non_blocking)
        words = self.decode_teacher_forcing(y, vl, vg)
        return self.loss_nll(y, words), []

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
        w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        h, c = vg.new_zeros((vg.shape[0], self.feat_dim)), vg.new_zeros((vg.shape[0], self.feat_dim))
        states = (vg, vl, h, c)
        # Decode words
        beam_words, logs = self._decode_words_beam(w, states, beam_size, allow_stop, recover_words, diversity_rate)
        dummy_stops = self.dummy_stops(beam_words)
        return dummy_stops, beam_words.unsqueeze(dim=1), logs

    def decode_teacher_forcing(self, y, vl, vg):
        # Masks
        not_masked = y.new_ones(1, dtype=torch.bool)[0]
        mask = ((y > 0).sum(dim=(0, 1)) > 0)

        h, m = vg.new_zeros((vg.shape[0], self.feat_dim)), vg.new_zeros((vg.shape[0], self.feat_dim))
        w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        states = (vg, vl, h, m)

        words = []
        for j in range(self.max_word):
            if torch.equal(mask[j], not_masked):
                p, states = self.proc_word(w, states)
                words.append(p)
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
        return torch.stack(words, dim=1)

    def encode(self, x, meta):
        return self.encode_image(x)

    def encode_image(self, x):
        # Resnet-152 features
        x = self.flatten_image(x)
        x, mask = self.image_features_with_mask(x, self.image_feats)
        # Nx2048x7x7 -> Nx49x512
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.permute(0, 2, 1)
        vl = self.dropout(x)
        vl = relu(self.image_proj_l(vl))
        vl *= mask
        vl = self.deflatten_image(vl)
        # Mean visual feature (Nx512)
        vg = (x * mask).mean(dim=1)
        vg = self.deflatten_image(vg)
        vg = self.multi_vg(vg)
        vg = self.dropout(vg)
        vg = relu(self.image_proj_g(vg))
        return vl, vg

    def forward(self, x, y, meta):
        # Encode image
        vl, vg = self.encode(x, meta)
        # Decode text
        return self.decode_teacher_forcing(y, vl, vg)

    def loss_nll(self, y, words):
        # Word loss
        words = words.unsqueeze(dim=1).permute(0, 3, 1, 2)
        loss_word = cross_entropy(words, y, ignore_index=PretrainedEmbeddings.INDEX_PAD, reduction='mean')
        return loss_word

    def multi_cb(self, zs, cv, b, h):
        if self.multi_image > 1:
            if self.multi_merge == self.MULTI_MERGE_ATT:
                h_a = self.att_z_h(h).unsqueeze(dim=1).repeat(1, zs.shape[1], 1)
                alpha = self.att_z_a(tanh(self.att_z_z(zs) + h_a)).squeeze(dim=2)
                alpha = softmax(alpha, dim=-1)
                alpha = alpha.unsqueeze(dim=2)
                cv = (cv * alpha).sum(dim=1)
                b = (b * alpha).sum(dim=1)
            elif self.multi_merge == self.MULTI_MERGE_MAX:
                cv, _ = torch.max(cv, dim=1)
                b, _ = torch.min(b, dim=1)
            else:
                raise ValueError('Unknown multi merge {0}'.format(self.multi_merge))
        return cv, b

    def proc_word(self, w, states):
        vg, vl, hw, mw = states
        z, hw, mw, s = self.visual_sentinel(w, vg, vl, hw, mw)
        a, b, c = self.visual_sentinel_attention(vl, hw, z, s)
        # Word generation probability (NxW), note that softmax is not applied here.
        p = self.word_dense(self.dropout(c + hw))
        return p, (vg, vl, hw, mw)

    def sample(self, encoded_data, nucleus_p=None):
        vl, vg = encoded_data
        w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        h, c = vg.new_zeros((vg.shape[0], self.feat_dim)), vg.new_zeros((vg.shape[0], self.feat_dim))
        states = (vg, vl, h, c)
        return self._sample_words(w, states, nucleus_p)

    def visual_sentinel(self, w, vg, vl, h, m):
        # Visual features (Nx49)
        z1 = self.vs_att_v(vl)
        if self.multi_image > 1:
            z2 = self.vs_att_g(h).unsqueeze(dim=1).unsqueeze(dim=1)
            z2 = z2.repeat(1, vl.shape[1], vl.shape[2], 1)
        else:
            z2 = self.vs_att_g(h).unsqueeze(dim=1)
            z2 = z2.repeat(1, vl.shape[1], 1)
        z = self.vs_att_h(tanh(z1 + z2)).squeeze(dim=-1)
        # Visual sentinel (Nx512)
        e = self.embeddings(w).squeeze(1)
        xw = torch.cat((vg, e), dim=1)
        g = sigmoid(self.vs_dense1(xw) + self.vs_dense2(h))
        h, m = self.lstm_word(xw, (h, m))
        s = g * tanh(m)
        return z, h, m, s

    def visual_sentinel_attention(self, vl, h, z, s):
        # Attention to visual sentinel & visual features (Nx50)
        a1 = softmax(z, dim=-1)
        zs = tanh(self.vs_att_s(s) + self.vs_att_g(h))
        if self.multi_image > 1:
            zs = zs.unsqueeze(dim=1).repeat(1, vl.shape[1], 1)
        zs = torch.cat((z, self.vs_att_h(zs)), dim=-1)
        a2 = softmax(zs, dim=-1)
        # Mixture of visual sentinel and visual features (Nx512)
        cv = torch.sum(a1.unsqueeze(-1) * vl, dim=-2)
        b = a2[:, :, -1].unsqueeze(dim=-1) if self.multi_image > 1 else a2[:, -1].unsqueeze(dim=-1)
        cv, b = self.multi_cb(zs, cv, b, h)
        c = b * s + (cv.new_ones((cv.shape[0], 1)) - b) * cv
        return a2, b, c
