#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import torch
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module
from torch.distributions.categorical import Categorical
from torch.nn.functional import cross_entropy, log_softmax, relu, softmax
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from clinicgen.models.image import ImageClassification
from clinicgen.models.image2text import _Image2Text, PretrainedEmbeddings
from clinicgen.utils import data_cuda


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class PositionalEncoding2D(PositionalEncoding):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding2D, self).__init__(d_model, max_len=max_len)

    def forward(self, x):
        pe_w = self.pe[:x.size(0), :].repeat(self.pe.shape[0], 1, 1)
        pe_h = self.pe[:x.size(0), :].repeat_interleave(self.pe.shape[0], dim=0)
        x = x + pe_w + pe_h
        return x


class _TransformerCaptioner(_Image2Text):
    VISUAL_NUM = 49

    def __init__(self, embeddings, feat_dim=512, max_word=32, multi_image=1, image_pe=True, layer_norm=False,
                 teacher_forcing=False, image_model=None, image_pretrained=None, finetune_image=False,
                 image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu', verbose=False):
        super(_TransformerCaptioner, self).__init__(max_word, multi_image, None, teacher_forcing, image_finetune_epoch,
                                                    rl_opts, word_idxs, verbose)
        self.feat_dim = feat_dim

        self.dropout = Dropout(0.1)
        self.layer_norm = LayerNorm(feat_dim) if layer_norm else None
        # Image processes
        if image_model is None:
            image_model = 'densenet'
        self.image_feats, image_dim = ImageClassification.image_features(image_model, not finetune_image, True,
                                                                         image_pretrained, device)
        self.image_proj_l = Linear(image_dim, feat_dim)
        image_len = int(math.sqrt(self.VISUAL_NUM))
        self.image_weight = math.sqrt(image_len)
        if image_pe:
            self.image_pe = PositionalEncoding2D(feat_dim, image_len)
        # Word processes
        self.embeddings = Embedding.from_pretrained(embeddings, freeze=False,
                                                    padding_idx=PretrainedEmbeddings.INDEX_PAD)
        self.embed_num = self.embeddings.num_embeddings
        self.word_weight = math.sqrt(max_word)
        self.word_pe = PositionalEncoding(feat_dim, max_len=max_word + 1)
        self.word_dense = Linear(feat_dim, embeddings.shape[0], bias=False)

    def _nll_step(self, encoded_data, targ, device, non_blocking, ids=None):
        v = encoded_data
        targ = self.truncate_targ(targ)
        y = data_cuda(targ, device=device, non_blocking=non_blocking)
        words = self.decode_teacher_forcing(y, v)
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
        v = encoded_data
        # Initialize word process states
        w = v.new_full((v.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
        states = (torch.tensor([0]), None, v)
        # Decode words
        beam_words, logs = self._decode_words_beam(w, states, beam_size, allow_stop, recover_words, diversity_rate)
        dummy_stops = self.dummy_stops(beam_words)
        return dummy_stops, beam_words.unsqueeze(dim=1), logs

    def decode_teacher_forcing(self, y, v):
        # Masks
        w = v.new_full((v.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)

        if self.teacher_forcing is None:
            wt = torch.cat([w, y.squeeze(dim=1)], dim=1)[:, :-1]
            tgt_mask = wt.new_full((wt.shape[1], wt.shape[1]), float('-inf'), dtype=torch.float).triu(1)
            d = self.proc_word_sequence(wt, v, mask=tgt_mask)
            p = self.word_dense(d)
            words = p.permute(1, 0, 2)
        else:
            words = []
            states = (torch.tensor([0]), None, v)
            for j in range(y.shape[-1]):
                p, states = self.proc_word(w, states)
                words.append(p)
                if self.teacher_forcing.get_tfr() >= random.random():
                    w = y[:, 0, j]
                else:
                    p = softmax(p, dim=-1)
                    cat = Categorical(probs=p)
                    w = cat.sample()
            words = torch.stack(words, dim=1)
        return words

    def deflatten_image(self, x):
        if self.multi_image > 1:
            x = x.view(int(x.shape[0] / self.multi_image), self.multi_image, x.shape[1], x.shape[2])
        return x

    def encode(self, x, meta):
        return self.encode_image(x)

    def encode_image(self, x):
        # CNN features
        x = self.flatten_image(x)
        v, _ = self.image_features_with_mask(x, self.image_feats)
        # Merge multiple images
        v = self.deflatten_image(v)
        return v

    def forward(self, x, y, meta):
        # Encode image
        v = self.encode(x, meta)
        # Decode text
        return self.decode_teacher_forcing(y, v)

    def image_features_with_mask(self, x, model):
        raise NotImplementedError()

    def loss_nll(self, y, words):
        # Word loss
        words = words.unsqueeze(dim=1).permute(0, 3, 1, 2)
        loss_word = cross_entropy(words, y, ignore_index=PretrainedEmbeddings.INDEX_PAD, reduction='mean')
        return loss_word

    def proc_word_sequence(self, wt, v, mask=None):
        raise NotImplementedError()

    def proc_word(self, w, states):
        c, wt, v = states
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=-1)
        if wt is not None:
            wt = torch.cat([wt, w], dim=1)
        else:
            wt = w
        d = self.proc_word_sequence(wt, v)
        d = d[c[0]]
        # Word generation probability (NxW), note that softmax is not applied here.
        p = self.word_dense(d)
        c[0] += 1
        return p, (c, wt, v)

    def sample(self, encoded_data, nucleus_p=None):
        self.eval()
        v = encoded_data
        with torch.no_grad():
            # Initialize word process states
            w = v.new_full((v.shape[0], 1), PretrainedEmbeddings.INDEX_START, dtype=torch.long)
            states = (torch.tensor([0]), None, v)
            words, lprobs = self._sample_words(w, states, nucleus_p)
        y = words.unsqueeze(dim=1)
        y = self.truncate_targ(y)
        out = self.decode_teacher_forcing(y, v)
        out = log_softmax(out, dim=-1)
        words = words[:, :y.shape[2]]
        lprobs2 = []
        for b in range(out.shape[0]):
            idx = np.array([np.arange(out.shape[1]), words[b]])
            lprobs2.append(out[b][idx])
        lprobs2 = torch.stack(lprobs2, dim=0)
        self.train()
        return words, lprobs2


class TransformerSimpleCaptioner(_TransformerCaptioner):
    def __init__(self, embeddings, feat_dim=512, max_word=32, multi_image=1, image_pe=True, layer_norm=False,
                 num_layers=6, teacher_forcing=False, image_model=None, image_pretrained=None, finetune_image=False,
                 image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu', verbose=False):
        super(TransformerSimpleCaptioner, self).__init__(embeddings, feat_dim, max_word, multi_image, image_pe,
                                                         layer_norm, teacher_forcing, image_model, image_pretrained,
                                                         finetune_image, image_finetune_epoch, rl_opts, word_idxs,
                                                         device, verbose)
        # Transformer Decoder
        decoder_layer = TransformerMaxDecoderLayer(feat_dim, nhead=8)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def image_features_with_mask(self, x, model):
        mask = (x.detach().sum(dim=(1, 2, 3)) != 0).type(torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.multi_image > 1:
            nz = mask.squeeze().nonzero().squeeze()
            if len(nz.shape) == 0:
                nz = x.new([0]).type(torch.long)
            x_nz = x[nz]
        else:
            x_nz, nz = x, None
        # Image features
        x_nz = model(x_nz)
        x_nz = x_nz.flatten(start_dim=-2, end_dim=-1)
        x_nz = x_nz.permute(0, 2, 1)
        x_nz = relu(self.image_proj_l(x_nz))
        x_nz = self.dropout(x_nz)
        if self.layer_norm is not None:
            x_nz = self.layer_norm(x_nz)
        # Positional encodings + Transformer encoder
        if hasattr(self, 'image_pe'):
            x_nz = x_nz.permute(1, 0, 2)
            x_nz *= self.image_weight
            x_nz = self.image_pe(x_nz)
            x_nz = x_nz.permute(1, 0, 2)
        if self.multi_image > 1:
            x = x.new_zeros(x.shape[0], x_nz.shape[1], x_nz.shape[2])
            x[nz] += x_nz
        else:
            x = x_nz
        return x, mask

    def proc_word_sequence(self, wt, v, mask=None):
        e = self.embeddings(wt)
        # Positional encoding + Transformer decoder
        e = e.permute(1, 0, 2)
        e *= self.word_weight
        e = self.word_pe(e)
        if self.multi_image > 1:
            v2 = v.permute(1, 2, 0, 3)
        else:
            v2 = v.permute(1, 0, 2)
        return self.decoder(e, v2, tgt_mask=mask)


class TransformerCaptioner(TransformerSimpleCaptioner):
    def __init__(self, embeddings, feat_dim=512, max_word=32, multi_image=1, image_pe=True, layer_norm=False,
                 num_enc_layers=6, num_dec_layers=6, teacher_forcing=False, image_model=None, image_pretrained=None,
                 finetune_image=False, image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                 verbose=False):
        super(TransformerCaptioner, self).__init__(embeddings, feat_dim, max_word, multi_image, image_pe, layer_norm,
                                                   num_dec_layers, teacher_forcing, image_model, image_pretrained,
                                                   finetune_image, image_finetune_epoch, rl_opts, word_idxs, device,
                                                   verbose)
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(feat_dim, nhead=8)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

    def decode_beam(self, encoded_data, beam_size, allow_stop=True, recover_words=None, diversity_rate=0.0):
        return super(TransformerCaptioner, self).decode_beam(encoded_data, beam_size, allow_stop, recover_words,
                                                                 diversity_rate)

    def image_features_with_mask(self, x, model):
        mask = (x.detach().sum(dim=(1, 2, 3)) != 0).type(torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.multi_image > 1:
            nz = mask.squeeze().nonzero().squeeze()
            if len(nz.shape) == 0:
                nz = x.new([0]).type(torch.long)
            x_nz = x[nz]
        else:
            x_nz, nz = x, None
        # Image features
        x_nz = model(x_nz)
        x_nz = x_nz.flatten(start_dim=-2, end_dim=-1)
        x_nz = x_nz.permute(0, 2, 1)
        x_nz = relu(self.image_proj_l(x_nz))
        x_nz = self.dropout(x_nz)
        if self.layer_norm is not None:
            x_nz = self.layer_norm(x_nz)
        # Positional encodings + Transformer encoder
        x_nz = x_nz.permute(1, 0, 2)
        if hasattr(self, 'image_pe'):
            x_nz *= self.image_weight
            x_nz = self.image_pe(x_nz)
        x_nz = self.encoder(x_nz)
        if self.multi_image > 1:
            x_nz = x_nz.permute(1, 0, 2)
            x = x.new_zeros(x.shape[0], x_nz.shape[1], x_nz.shape[2])
            x[nz] += x_nz
        else:
            x = x_nz
            x = x.permute(1, 0, 2)
        return x, mask


class TransformerMaxDecoderLayer(TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if len(memory.shape) == 4:
            tgt_stack = []
            for i in range(memory.shape[0]):
                tgt2 = self.multihead_attn(tgt, memory[i], memory[i], attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)[0]
                tgt_stack.append(tgt2)
            tgt_stack = torch.stack(tgt_stack, dim=0)
            tgt2, _ = torch.max(tgt_stack, dim=0)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
