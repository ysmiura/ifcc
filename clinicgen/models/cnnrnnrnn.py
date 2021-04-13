#! /bin/env python
# -*- encoding: utf-8 -*-

import random
import numpy as np
import torch
from torch import sigmoid, tanh
from torch.distributions.categorical import Categorical
from torch.nn import Dropout, Embedding, Linear, LSTM, LSTMCell
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, log_softmax, relu, softmax
from clinicgen.models.image import ImageClassification
from clinicgen.models.image2text import _Image2Text, PretrainedEmbeddings
from clinicgen.utils import data_cuda


class CNNRNNRNN(_Image2Text):
    VISUAL_NUM = 49

    def __init__(self, embeddings, max_sent=4, max_word=32, multi_image=1, multi_merge='att', feat_dim=256,
                 lambda_sent=5.0, lambda_word=1.0, topic_as_state=False, simple_proj=False, teacher_forcing=None,
                 parallel_sent=False, view_position=False, image_model=None, image_pretrained=None,
                 finetune_image=False, image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                 verbose=False):
        super(CNNRNNRNN, self).__init__(max_word, multi_image, multi_merge, teacher_forcing, image_finetune_epoch,
                                        rl_opts, word_idxs, verbose)
        self.max_sent = max_sent
        self.feat_dim = feat_dim
        self.lambda_sent = lambda_sent
        self.lambda_word = lambda_word
        self.topic_as_state = topic_as_state
        self.simple_proj = simple_proj
        self.dropout = Dropout(0.5)

        # Image processes
        if image_model is None:
            image_model = 'densenet'
        self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
                                                                         image_pretrained, device)
        self.image_proj = Linear(image_dim, feat_dim)
        if not simple_proj:
            self.image_proj2 = Linear(image_dim, feat_dim)
        self.view_postion = view_position
        image_dim = feat_dim + 3 if view_position else feat_dim
        self._init_multi_image(feat_dim, self.VISUAL_NUM, feat_dim)
        # Sentence processes
        self.parallel_sent = parallel_sent
        if parallel_sent:
            self.lstm_sent = LSTM(image_dim, feat_dim, batch_first=True)
        else:
            self.lstm_sent = LSTMCell(image_dim, feat_dim)
        self.topic_dense = Linear(feat_dim, feat_dim)
        self.stop_dense = Linear(feat_dim, 1)
        # Visual sentinel
        if not topic_as_state:
            input_dim = image_dim + feat_dim + embeddings.shape[1]
        else:
            input_dim = image_dim + embeddings.shape[1]
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
        self.word_dense = Linear(feat_dim, embeddings.shape[0], bias=False)

    def _decode_loop_word(self, vl, vg, t, beam_size, allow_stop, recover_words=None, diversity_rate=0.0):
        if self.topic_as_state:
            hw, mw = t, t
        else:
            hw, mw = vg.new_zeros(vg.shape[0], self.feat_dim), vg.new_zeros(vg.shape[0], self.feat_dim)
        ws = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        states = (t, vg, vl, hw, mw)
        sent_words, logs = self._decode_words_beam(ws, states, beam_size, allow_stop, recover_words, diversity_rate)
        return sent_words, logs

    def _init_multi_image(self, image_dim, visual_num, rnn_dim):
        super(CNNRNNRNN, self)._init_multi_image(image_dim, visual_num, rnn_dim)
        if self.multi_image > 1 and self.multi_merge == self.MULTI_MERGE_ATT:
            sentinel_num = visual_num + 1
            self.att_z_z = Linear(sentinel_num, sentinel_num)
            self.att_z_h = Linear(rnn_dim, sentinel_num)
            self.att_z_a = Linear(sentinel_num, 1)

    def _loop_word(self, vl, vg, t, yw, mask, not_masked):
        if self.topic_as_state:
            hw, mw = t, t
        else:
            hw, mw = vg.new_zeros(vg.shape[0], self.feat_dim), vg.new_zeros(vg.shape[0], self.feat_dim)
        w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        states = (t, vg, vl, hw, mw)

        sent_words = []
        for j in range(self.max_word):
            if torch.equal(mask[j], not_masked):
                p, states = self.proc_word(w, states)
                sent_words.append(p)
                if self.teacher_forcing is None or self.teacher_forcing.get_tfr() >= random.random():
                    w = yw[j]
                else:
                    p = softmax(p, dim=-1)
                    cat = Categorical(probs=p)
                    w = cat.sample()
            else:
                p = vg.new_ones(vg.shape[0], self.embeddings.num_embeddings) / self.embeddings.num_embeddings
                sent_words.append(p)
                if self.teacher_forcing is None or self.teacher_forcing.get_tfr() >= random.random():
                    w = yw[j]
                else:
                    w = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_PAD)
        return torch.stack(sent_words, dim=1)

    def _loop_word_sample(self, vl, vg, t):
        if self.topic_as_state:
            hw, mw = t, t
        else:
            hw, mw = vg.new_zeros(vg.shape[0], self.feat_dim), vg.new_zeros(vg.shape[0], self.feat_dim)
        word = vg.new_full((vg.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        states = (t, vg, vl, hw, mw)

        sent_words, sent_lprobs = [], []
        for j in range(self.max_word):
            p, states = self.proc_word(word, states)
            lp = log_softmax(p, dim=-1)
            cat = Categorical(logits=lp)
            word = cat.sample()
            sent_words.append(word)
            idx = np.array([np.arange(word.shape[0]), word.detach().cpu().numpy()])
            sent_lprobs.append(lp[idx])
        sent_words = torch.stack(sent_words, dim=1)
        sent_lprobs = torch.stack(sent_lprobs, dim=1)
        return sent_words, sent_lprobs

    def _nll_step(self, encoded_data, targ, device, non_blocking, ids=None):
        vl, vg = encoded_data
        y = data_cuda(targ, device=device, non_blocking=non_blocking)
        stops, words = self.decode_teacher_forcing(y, vl, vg)
        loss, loss_sent, loss_word = self.loss_nll(y, stops, words)
        return loss, [float(loss_word.detach().cpu()), float(loss_sent.detach().cpu())]

    def _rl_step(self, encoded_data, targ, device, non_blocking, ids=None):
        self.eval()
        with torch.no_grad():
            stops_greedy, words_greedy, _ = self.decode_beam(encoded_data, beam_size=1)
            gens_greedy, _ = self.evaluator.recover_words(stops_greedy, words_greedy.squeeze(dim=2))
        self.train()
        stops, words, log_probs = self.sample(encoded_data)
        gens_sample, masks_sample = self.evaluator.recover_words(stops, words)
        targ_stops = ((targ > 0).sum(dim=-1, keepdim=True) == 0.0).type(torch.float)
        gens_ref, _ = self.evaluator.recover_words(targ_stops, targ, normalized=True)
        rewards, loss_acc = self.self_critical_reward(gens_sample, gens_greedy, gens_ref, masks_sample, log_probs,
                                                      ids=ids)
        return loss_acc, rewards

    def decode_beam(self, encoded_data, beam_size, allow_stop=True, recover_words=None, diversity_rate=0.0):
        logs = {'word': [], 'score': [], 'score_current': [], 'beam_buffer': []} if recover_words is not None else None

        # Encode image
        vl, vg = encoded_data
        batch_size = vg.shape[0]

        if self.parallel_sent:
            # Process max_sent sentences
            hs, ms = vg.new_zeros((1, batch_size, self.feat_dim)), vg.new_zeros((1, batch_size, self.feat_dim))
            t, stops, _, _ = self.proc_sentence(vg, hs, ms)
            stops = stops.detach().cpu()
            # Process each word
            # Repeat visual features for max_sent times (N -> NxS)
            vl = vl.repeat(self.max_sent, 1, 1, 1) if self.multi_image > 1 else vl.repeat(self.max_sent, 1, 1)
            vg = vg.repeat(self.max_sent, 1)
            # Combine the batch dimension and the sentence dimension
            t = t.view((batch_size * self.max_sent, self.feat_dim))
            # Decodes words
            words, logs_sent = self._decode_loop_word(vl, vg, t, beam_size, allow_stop, recover_words, diversity_rate)
            # Convert decoded words to NxBxSxW
            words = words.view((batch_size, self.max_sent, beam_size, self.max_word)).permute((0, 2, 1, 3))
            if recover_words is not None:
                logs['word'].append(logs_sent['word'])
                logs['score'].append(logs_sent['score'])
                logs['beam_buffer'].append(logs_sent['beam_buffer'])
        else:
            # Process each sentence
            batch_stop = vg.new_zeros(1, dtype=torch.uint8)[0]
            stops, words = [], []
            hs, ms = vg.new_zeros((batch_size, self.feat_dim)), vg.new_zeros((batch_size, self.feat_dim))
            for i in range(self.max_sent):
                t, u, hs, ms = self.proc_sentence(vg, hs, ms)
                stops.append(u.detach().cpu())
                stop_state = (sigmoid(u) < 0.5).sum(dtype=torch.uint8)
                logs_sent = {'word': [], 'score': []} if recover_words is None else None

                if torch.equal(stop_state, batch_stop) and allow_stop:
                    words.append(torch.zeros((batch_size, beam_size, self.max_word), dtype=torch.long))
                else:
                    # Process each word
                    sent_words, logs_sent = self._decode_loop_word(vl, vg, t, beam_size, allow_stop, recover_words,
                                                                   diversity_rate)
                    words.append(sent_words)

                if recover_words is not None:
                    logs['word'].append(logs_sent['word'])
                    logs['score'].append(logs_sent['score'])
                    logs['score_current'].append(logs_sent['score_current'])
                    logs['beam_buffer'].append(logs_sent['beam_buffer'])
            stops = torch.stack(stops, dim=1)
            words = torch.stack(words, dim=1)
        return stops, words, logs

    def decode_teacher_forcing(self, y, vl, vg):
        not_masked = y.new_ones(1, dtype=torch.bool)[0]
        batch_size = vg.shape[0]

        if self.parallel_sent:
            # Make masks
            batch_mask_words = ((y > 0).sum(dim=(0, 1)) > 0)
            # Process max_sent sentences
            hs, ms = vg.new_zeros((1, batch_size, self.feat_dim)), vg.new_zeros((1, batch_size, self.feat_dim))
            t, stops, _, _ = self.proc_sentence(vg, hs, ms)
            vl = vl.repeat(self.max_sent, 1, 1, 1) if self.multi_image > 1 else vl.repeat(self.max_sent, 1, 1)
            vg = vg.repeat(self.max_sent, 1)
            t = t.view((batch_size * self.max_sent, self.feat_dim))
            y2 = y.view((batch_size * self.max_sent, self.max_word)).permute(1, 0)
            words = self._loop_word(vl, vg, t, y2, batch_mask_words, not_masked)
            words = words.view((batch_size, self.max_sent, self.max_word, self.embeddings.num_embeddings))
        else:
            # Make masks
            batch_mask_words = ((y > 0).sum(dim=0) > 0)

            # Process each sentence
            stops, words = [], []
            hs, ms = vg.new_zeros((batch_size, self.feat_dim)), vg.new_zeros((batch_size, self.feat_dim))
            for i in range(self.max_sent):
                t, u, hs, ms = self.proc_sentence(vg, hs, ms)
                stops.append(u)
                # Process each word
                y2 = y.permute(1, 2, 0)
                sent_words = self._loop_word(vl, vg, t, y2[i], batch_mask_words[i], not_masked)
                words.append(sent_words)
            stops = torch.stack(stops, dim=1)
            words = torch.stack(words, dim=1)
        return stops, words

    def encode(self, x, meta):
        vp = meta[0]
        if not self.view_postion:
            vp = None
        return self.encode_image(x, vp)

    def encode_image(self, x, vp=None):
        # DenseNet-121 features
        x = self.flatten_image(x)
        x, mask = self.image_features_with_mask(x, self.image_feats)
        # Nx1024x7x7 -> Nx49x1024
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.permute(0, 2, 1)
        if self.simple_proj:
            # Projection to word embedding dim (Nx49x256)
            vl = self.image_proj(x) * mask
            # Mean visual feature (Nx256)
            vg = vl.mean(dim=1)
        else:
            # Projection to word embedding dim (Nx49x256)
            vl = relu(self.image_proj(x)) * mask
            # Mean visual feature (Nx256)
            vg = self.image_proj2(x)
            vg *= mask
            vg = vg.mean(dim=1)
            vg = relu(vg)
        vl = self.deflatten_image(vl)
        vg = self.deflatten_image(vg)
        vg = self.multi_vg(vg)

        # Dropouts
        vg = self.dropout(vg)
        # Image meta-data
        if vp is not None:
            if self.multi_image > 1:
                vp = vp.sum(dim=1).clamp(0, 1)
            vg = torch.cat([vg, vp], dim=-1)
        return vl, vg

    def forward(self, x, y, meta):
        # Encode image
        vl, vg = self.encode(x, meta)
        # Decode text
        return self.decode_teacher_forcing(y, vl, vg)

    def loss_nll(self, y, stops, words):
        # Make masks
        mask_words = (y > 0).type(torch.float32)
        y_stops = (mask_words.sum(dim=-1, keepdim=True) == 0).type(torch.float32)
        # Compute losses
        loss_sent = binary_cross_entropy_with_logits(stops, y_stops, reduction='mean')
        loss_word = cross_entropy(words.permute(0, 3, 1, 2), y, ignore_index=PretrainedEmbeddings.INDEX_PAD,
                                  reduction='mean')
        loss = loss_sent * self.lambda_sent + loss_word * self.lambda_word
        return loss, loss_sent, loss_word

    def loss_names(self):
        names = ['total', 'word_sent', 'word', 'sent']
        if self.rl_opts.epoch is not None:
            names.append('rl')
            names += self.rl_opts.metrics
        return names

    def loss_progress(self, loss_logs):
        if self.rl_opts.epoch is not None:
            return '%.2f,%.2f,%.2f' % (np.mean(loss_logs[2]), np.mean(loss_logs[3]), np.mean(loss_logs[4]))
        else:
            return '%.2f,%.2f' % (np.mean(loss_logs[2]), np.mean(loss_logs[3]))

    def meta_cuda(self, meta, device='gpu', non_blocking=False):
        vp = meta[0]
        if self.view_postion:
            vp = data_cuda(vp, device=device, non_blocking=non_blocking)
        return (vp,)

    def multi_cb(self, zs, cv, b, hw):
        if self.multi_image > 1:
            if self.multi_merge == self.MULTI_MERGE_ATT:
                h_a = self.att_z_h(hw).unsqueeze(dim=1).repeat(1, zs.shape[1], 1)
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

    def proc_sentence(self, vg, hs, ms):
        # Topic vector (Nx256) & stop signal vector (Nx1)
        # For parallel_sent, Topic vector (NxSx256) & stop signal vector (NxSx1)
        # sigmoid is applied later to the stop signal vector for computational stability
        if self.parallel_sent:
            vg = vg.unsqueeze(dim=1)
            vg = vg.repeat((1, self.max_sent, 1))
            os, (hs, ms) = self.lstm_sent(vg, (hs, ms))
        else:
            hs, ms = self.lstm_sent(vg, (hs, ms))
            os = hs
        t = relu(self.topic_dense(os))
        u = self.stop_dense(os)
        return t, u, hs, ms

    def proc_word(self, w, states):
        t, vg, vl, hw, mw = states
        z, hw, mw, s = self.visual_sentinel(w, t, vg, vl, hw, mw)
        a, b, c = self.visual_sentinel_attention(vl, hw, z, s)
        # Word generation probability (NxW), note that softmax is not applied here.
        p = self.word_dense(self.dropout(c + hw))
        return p, (t, vg, vl, hw, mw)

    def sample(self, encoded_data, nucleus_p=None):
        if nucleus_p is not None:
            raise ValueError('Nucleus sampling not supported in CNNRNNRNN')

        vl, vg = encoded_data
        batch_size = vg.shape[0]

        if self.parallel_sent:
            # Process max_sent sentences
            hs, ms = vg.new_zeros((1, batch_size, self.feat_dim)), vg.new_zeros((1, batch_size, self.feat_dim))
            t, stops, _, _ = self.proc_sentence(vg, hs, ms)
            vl = vl.repeat(self.max_sent, 1, 1, 1) if self.multi_image > 1 else vl.repeat(self.max_sent, 1, 1)
            vg = vg.repeat(self.max_sent, 1)
            t = t.view((batch_size * self.max_sent, self.feat_dim))
            words, lprobs = self._loop_word_sample(vl, vg, t)
            words = words.view((batch_size, self.max_sent, self.max_word))
            lprobs = lprobs.view((batch_size, self.max_sent, self.max_word))
        else:
            # Process each sentence
            stops, words, lprobs = [], [], []
            hs, ms = vg.new_zeros((batch_size, self.feat_dim)), vg.new_zeros((batch_size, self.feat_dim))
            for i in range(self.max_sent):
                t, u, hs, ms = self.proc_sentence(vg, hs, ms)
                stops.append(u)
                # Process each word
                sent_words, sent_lprobs = self._loop_word_sample(vl, vg, t)
                words.append(sent_words)
                lprobs.append(sent_lprobs)
            stops = torch.stack(stops, dim=1)
            words = torch.stack(words, dim=1)
            lprobs = torch.stack(lprobs, dim=1)
        return stops, words, lprobs

    def visual_sentinel(self, w, t, vg, vl, hw, mw):
        # Visual features (Nx49)
        z1 = self.vs_att_v(vl)
        if self.multi_image > 1:
            z2 = self.vs_att_g(hw).unsqueeze(dim=1).unsqueeze(dim=1)
            z2 = z2.repeat(1, vl.shape[1], vl.shape[2], 1)
        else:
            z2 = self.vs_att_g(hw).unsqueeze(dim=1)
            z2 = z2.repeat(1, vl.shape[1], 1)
        z = self.vs_att_h(tanh(z1 + z2)).squeeze(dim=-1)
        # Visual sentinel (Nx256)
        e = self.embeddings(w).squeeze(1)
        xw = torch.cat((vg, e), dim=1) if self.topic_as_state else torch.cat((vg, t, e), dim=1)
        g = sigmoid(self.vs_dense1(xw) + self.vs_dense2(hw))
        hw, mw = self.lstm_word(xw, (hw, mw))
        s = g * tanh(mw)
        return z, hw, mw, s

    def visual_sentinel_attention(self, vl, hw, z, s):
        # Attention to visual sentinel & visual features (Nx50)
        a1 = softmax(z, dim=-1)
        zs = tanh(self.vs_att_s(s) + self.vs_att_g(hw))
        if self.multi_image > 1:
            zs = zs.unsqueeze(dim=1).repeat(1, vl.shape[1], 1)
        zs = torch.cat((z, self.vs_att_h(zs)), dim=-1)
        a2 = softmax(zs, dim=-1)
        # Mixture of visual sentinel and visual features (Nx256)
        cv = torch.sum(a1.unsqueeze(-1) * vl, dim=-2)
        b = a2[:, :, -1].unsqueeze(dim=-1) if self.multi_image > 1 else a2[:, -1].unsqueeze(dim=-1)
        cv, b = self.multi_cb(zs, cv, b, hw)
        c = b * s + (cv.new_ones((cv.shape[0], 1)) - b) * cv
        return a2, b, c
