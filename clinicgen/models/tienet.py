#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import gzip
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


class TieNet(_Image2Text):
    DOC_IMAGE_SEPARATOR = '__'
    VISUAL_NUM = 196
    DISEASE_NUM = 13

    def __init__(self, embeddings, max_word=32, multi_image=1, multi_merge='att', labels=None, aete_s=2000, aete_r=5,
                 lstm_dim=256, lambda_a=0.85, teacher_forcing=None, image_model=None, image_pretrained=None,
                 finetune_image=False, image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                 verbose=False):
        super(TieNet, self).__init__(max_word, multi_image, multi_merge, teacher_forcing, image_finetune_epoch,
                                     rl_opts, word_idxs, verbose)
        # Label statistics
        self.chexpert_labels, self.lp, self.ln, self.lq = self._load_labels(labels)
        # Various NN parameters
        self.feat_dim = lstm_dim
        self.lstm_dim = lstm_dim
        self.lambda_a = lambda_a
        self.dropout = Dropout(0.5)
        # Image processes
        if image_model is None:
            image_model = 'resnet50'
        self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
                                                                         image_pretrained, device)
        self._init_multi_image(image_dim, self.VISUAL_NUM, lstm_dim)
        self.image_proj = Linear(image_dim, lstm_dim)
        # Word processes
        self.init_h = Linear(lstm_dim, lstm_dim)
        self.init_c = Linear(lstm_dim, lstm_dim)
        self.att_v = Linear(image_dim, image_dim)
        self.att_h = Linear(lstm_dim, image_dim)
        self.att_a = Linear(image_dim, 1)
        self.gate = Linear(lstm_dim, image_dim)
        input_dim = image_dim + embeddings.shape[1]
        self.lstm_word = LSTMCell(input_dim, lstm_dim)
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False,
                                                    padding_idx=PretrainedEmbeddings.INDEX_PAD)
        self.embed_num = self.embeddings.num_embeddings
        self.word_dense = Linear(lstm_dim, embeddings.shape[0], bias=False)
        # AETE processes
        self.aete1 = Linear(lstm_dim, aete_s)
        self.aete2 = Linear(aete_s, aete_r)
        # Joint
        self.joint = Linear(lstm_dim + image_dim, self.DISEASE_NUM * 2)

    @classmethod
    def _load_labels(cls, path):
        # labels
        chexpert_labels = {}
        if path is not None:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                header = f.readline()
                reader = csv.reader(f)
                for row in reader:
                    ls = np.zeros((13,), dtype='long')
                    for i, l in enumerate(row[2:]):
                        if i != 8:
                            j = i if i < 8 else i - 1
                            if l == '1.0' or l == '-1.0':
                                ls[j] = 1
                            else:
                                ls[j] = 0
                    chexpert_labels[row[1]] = ls
        # weights
        p, n = 0, 0
        q = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
        if path is not None:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                header = f.readline()
                reader = csv.reader(f)
                for row in reader:
                    obs = False
                    for i, v in enumerate(row[2:]):
                        if i != 8:
                            if v == '1.0' or v == '-1.0':
                                j = i if i < 8 else i - 1
                                q[j] += 1
                                obs = True
                    if obs:
                        p += 1
                    else:
                        n += 1
        return chexpert_labels, p, n, q

    def _init_multi_image(self, image_dim, visual_num, rnn_dim):
        super(TieNet, self)._init_multi_image(image_dim, visual_num, rnn_dim)
        if self.multi_image > 1 and self.multi_merge == self.MULTI_MERGE_ATT:
            self.att_z_v = Linear(image_dim, image_dim)
            self.att_z_h = Linear(rnn_dim, image_dim)
            self.att_z_a = Linear(image_dim, 1)

    def _nll_step(self, encoded_data, targ, device, non_blocking, ids=None):
        vl, vg = encoded_data
        yl = np.zeros((targ.shape[0], self.DISEASE_NUM), dtype='long')
        for i, iid in enumerate(ids):
            did = iid.split(self.DOC_IMAGE_SEPARATOR)[0]
            yl[i] = self.chexpert_labels[did]
        yl = torch.tensor(yl)
        y = data_cuda(targ, device=device, non_blocking=non_blocking)
        yl = data_cuda(yl, device=device, non_blocking=non_blocking)
        words, dis = self.decode_teacher_forcing(y, vl, vg)
        loss, loss_dis, loss_word = self.loss_nll(y, yl, words, dis)
        if loss_dis is None:
            return loss, [float(loss_word.detach().cpu()), 0.0]
        else:
            return loss, [float(loss_word.detach().cpu()), float(loss_dis.detach().cpu())]

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
        words, hs, alphas = [], [], []
        for j in range(self.max_word):
            if torch.equal(mask[j], not_masked):
                p, states = self.proc_word(w, states)
                words.append(p)
                _, h, _, alpha = states
                hs.append(h)
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
        # Attention Encoded Text Embedding
        # Concat hidden states NxTxd
        hs = torch.stack(hs, dim=1)
        # Projection to NxTxr rows
        g = softmax(self.aete2(tanh(self.aete1(hs))), dim=2)
        # Weighted sum over T to Nxrxd
        m = g.permute(0, 2, 1).matmul(hs)
        # AETE embedding Nxd
        x1 = m.max(dim=1)[0]
        # Saliency Weighted Global Average Pooling
        alphas = torch.stack(alphas, dim=1)
        if self.multi_image > 1:
            # Spatial attention maps NxMxT
            aws = (alphas * g.max(dim=2)[0].unsqueeze(dim=-1).unsqueeze(dim=-1)).sum(dim=1)
            # SWGAP Nx1024
            x2 = (aws.unsqueeze(dim=-1) * vl).sum(dim=2)
            x2 = x2.max(dim=1)[0]
        else:
            # Spatial attention maps NxT
            aws = (alphas * g.max(dim=2)[0].unsqueeze(dim=-1)).sum(dim=1)
            # SWGAP Nx1024
            x2 = (aws.unsqueeze(dim=-1) * vl).sum(dim=1)
        # Joint Nx14
        dis = self.joint(torch.cat([x1, x2], dim=1))
        dis = dis.view((dis.shape[0], self.DISEASE_NUM, 2))
        return words, dis

    def encode(self, x, meta):
        return self.encode_image(x)

    def encode_image(self, x):
        # ResNet-50 features
        x = self.flatten_image(x)
        x, mask = self.image_features_with_mask(x, self.image_feats)
        # Nx2048x14x14 -> Nx196x2048
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.permute(0, 2, 1)
        vl = self.deflatten_image(x * mask)
        # Projection to word embedding dim (Nx196x2048)
        vg = relu(self.image_proj(x)) * mask
        # Mean visual feature (Nx2048)
        vg = vg.mean(dim=1)
        vg = self.deflatten_image(vg)
        vg = self.multi_vg(vg)
        return vl, vg

    def forward(self, x, y, meta):
        # Encode image
        vl, vg = self.encode(x, meta)
        # Decode text
        return self.decode_teacher_forcing(y, vl, vg)

    def loss_names(self):
        names = ['total', 'word_att', 'word', 'att']
        if self.rl_opts.epoch is not None:
            names.append('rl')
            names += self.rl_opts.metrics
        return names

    def loss_nll(self, y, yl, words, dis):
        # Word loss
        words = words.unsqueeze(dim=1).permute(0, 3, 1, 2)
        loss_word = cross_entropy(words, y, ignore_index=PretrainedEmbeddings.INDEX_PAD, reduction='mean')
        if self.rl_train:
            loss = loss_word
            loss_dis = None
        else:
            # Disease classification loss
            loss_dis = None
            for i in range(yl.shape[1]):
                wp = self.ln / (self.lp + self.ln)
                wn = self.lp / (self.lp + self.ln)
                wl = (self.lp + self.ln - self.lq[i]) / (self.lp + self.ln)
                weight = dis.new([wn * wl, wp * wl])
                loss_label = cross_entropy(dis[:, i], yl[:, i], weight=weight)
                if loss_dis is None:
                    loss_dis = loss_label
                else:
                    loss_dis += loss_label
            loss_dis /= yl.shape[1]
            # Total loss
            loss = (1.0 - self.lambda_a) * loss_word + self.lambda_a * loss_dis
        return loss, loss_dis, loss_word

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
        p = self.word_dense(h)
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
