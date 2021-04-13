#! /bin/env python
# -*- encoding: utf-8 -*-

from clinicgen.models.cnnrnnrnn import CNNRNNRNN
from clinicgen.models.kwl import KnowingWhenToLook
from clinicgen.models.m2transformer import M2Transformer
from clinicgen.models.sat import ShowAttendAndTell
from clinicgen.models.tienet import TieNet
from clinicgen.models.transformer import TransformerCaptioner, TransformerSimpleCaptioner
from clinicgen.nli import SimpleNLI


class Models:
    @classmethod
    def get_model(cls, name, embeddings, hidden_size, max_word, max_sent=None, multi_image=1, multi_merge='att',
                  teacher_forcing=None, image_model=None, image_pretrained=None, finetune_image=True,
                  view_position=False, image_finetune_epoch=None, rl_opts=None, word_idxs=None, device='gpu',
                  parallel_sent=True, cnnrnnrnn_topic_state=False, cnnrnnrnn_simple_proj=False, sat_lstm_dim=1000,
                  trans_image_pe=True, trans_layers=6, trans_enc_layers=None, trans_layer_norm=False, m2_memory=40,
                  tienet_labels=None, verbose=False):
        if trans_enc_layers is None:
            trans_enc_layers = trans_layers
        if name == 'cnnrnnrnn':
            model = CNNRNNRNN(embeddings, feat_dim=hidden_size, max_sent=max_sent, max_word=max_word,
                              multi_image=multi_image, multi_merge=multi_merge, topic_as_state=cnnrnnrnn_topic_state,
                              simple_proj=cnnrnnrnn_simple_proj, teacher_forcing=teacher_forcing,
                              parallel_sent=parallel_sent, view_position=view_position, image_model=image_model,
                              image_pretrained=image_pretrained, finetune_image=finetune_image,
                              image_finetune_epoch=image_finetune_epoch, rl_opts=rl_opts, word_idxs=word_idxs,
                              device=device, verbose=verbose)
        elif name == 'kwl':
            model = KnowingWhenToLook(embeddings, feat_dim=hidden_size, max_word=max_word, multi_image=multi_image,
                                      multi_merge=multi_merge, teacher_forcing=teacher_forcing, image_model=image_model,
                                      image_pretrained=image_pretrained, finetune_image=finetune_image,
                                      image_finetune_epoch=image_finetune_epoch, rl_opts=rl_opts, word_idxs=word_idxs,
                                      device=device, verbose=verbose)
        elif name == 'm2trans':
            model = M2Transformer(embeddings, feat_dim=hidden_size, max_word=max_word, multi_image=multi_image,
                                  layer_norm=trans_layer_norm, num_enc_layers=trans_enc_layers,
                                  num_dec_layers=trans_layers, num_memory=m2_memory, teacher_forcing=teacher_forcing,
                                  image_model=image_model, image_pretrained=image_pretrained,
                                  finetune_image=finetune_image, image_finetune_epoch=image_finetune_epoch,
                                  rl_opts=rl_opts, word_idxs=word_idxs, device=device, verbose=verbose)
        elif name == 'sat':
            model = ShowAttendAndTell(embeddings, context_dim=hidden_size, lstm_dim=sat_lstm_dim,  max_word=max_word,
                                      multi_image=multi_image, multi_merge=multi_merge, teacher_forcing=teacher_forcing,
                                      image_model=image_model, image_pretrained=image_pretrained,
                                      finetune_image=finetune_image, image_finetune_epoch=image_finetune_epoch,
                                      rl_opts=rl_opts, word_idxs=word_idxs, device=device, verbose=verbose)
        elif name == 'tienet':
            model = TieNet(embeddings, lstm_dim=hidden_size, max_word=max_word, multi_image=multi_image,
                           multi_merge=multi_merge, labels=tienet_labels, teacher_forcing=teacher_forcing,
                           image_model=image_model, image_pretrained=image_pretrained,
                           finetune_image=finetune_image, image_finetune_epoch=image_finetune_epoch,
                           rl_opts=rl_opts, word_idxs=word_idxs, device=device, verbose=verbose)
        elif name == 'trans':
            model = TransformerCaptioner(embeddings, feat_dim=hidden_size, max_word=max_word, multi_image=multi_image,
                                         image_pe=trans_image_pe, layer_norm=trans_layer_norm,
                                         num_enc_layers=trans_enc_layers, num_dec_layers=trans_layers,
                                         teacher_forcing=teacher_forcing, image_model=image_model,
                                         image_pretrained=image_pretrained, finetune_image=finetune_image,
                                         image_finetune_epoch=image_finetune_epoch, rl_opts=rl_opts,
                                         word_idxs=word_idxs, device=device, verbose=verbose)
        elif name == 'trans-s':
            model = TransformerSimpleCaptioner(embeddings, feat_dim=hidden_size, max_word=max_word,
                                               multi_image=multi_image, image_pe=trans_image_pe,
                                               layer_norm=trans_layer_norm, num_layers=trans_layers,
                                               teacher_forcing=teacher_forcing, image_model=image_model,
                                               image_pretrained=image_pretrained, finetune_image=finetune_image,
                                               image_finetune_epoch=image_finetune_epoch, rl_opts=rl_opts,
                                               word_idxs=word_idxs, device=device, verbose=verbose)
        else:
            raise ValueError('Unknown model {0}'.format(name))
        return model

    @classmethod
    def hierarchical(cls, name):
        if name == 'cnnrnnrnn':
            return True
        else:
            return False


class RLOptions:
    def __init__(self, epoch=None, metrics=None, weights=None, cider_df=None, tfidf=None, bert_score=None,
                 bert_score_penalty=False, op='add', nli='mednli', nli_label='entailment', nli_neutral_score=(1.0 / 3),
                 nli_prf='f', nli_batch=16, entity_match=None, entity_mode='exact', nthreads=2, pin_memory=False):
        self.epoch = epoch
        self.metrics = metrics.split(',') if metrics is not None else None
        self.weights = list(map(lambda v: float(v), weights.split(','))) if weights is not None else None
        self.cider_df = cider_df
        self.tfidf = tfidf
        self.bert_score_penalty = bert_score_penalty
        self.nli = nli
        self.nli_label = nli_label
        self.nli_neutral_score = nli_neutral_score
        self.nli_prf = nli_prf
        self.nli_batch = nli_batch
        self.op = op
        self.nthreads = nthreads
        self.pin_memory = pin_memory

        # Self-critical RL metrics
        self.rl_train = False
        self.bleu, self.rouge, self.cider, self.spice, self.bert_score = False, False, False, False, None
        self.nli_compare = []
        self.entity_match, self.entity_mode = None, None
        if metrics is not None:
            for metric in self.metrics:
                if metric.startswith('BLEU'):
                    self.bleu = True
                elif metric == 'ROUGE':
                    self.rouge = True
                elif metric == 'CIDEr':
                    self.cider = True
                elif metric == 'SPICE':
                    self.spice = True
                elif metric.startswith('BERTScore'):
                    self.bert_score = bert_score
                elif metric.startswith('NLI'):
                    if metric == 'NLISentBERTScore':
                        self.nli_compare.append(SimpleNLI.COMPARE_BERT_SCORE)
                    elif metric == 'NLISentBERTScoreT':
                        self.nli_compare.append(SimpleNLI.COMPARE_BERT_SCORE_FIX_THRESH)
                    elif metric == 'NLISentTFIDF':
                        self.nli_compare.append(SimpleNLI.COMPARE_TFIDF)
                    elif metric == 'NLISentAll':
                        self.nli_compare.append(SimpleNLI.COMPARE_ALL)
                    else:
                        self.nli_compare.append(SimpleNLI.COMPARE_DOC)
                elif metric == 'EntityMatchExact':
                    self.entity_match = entity_match
                    if self.entity_mode is None:
                        buf = entity_mode.split('-')
                        buf[0] = 'exact'
                        self.entity_mode = '-'.join(buf)
                elif metric == 'EntityMatchNLI':
                    self.entity_match = entity_match
                    self.entity_mode = entity_mode
                else:
                    raise ValueError('Unknown RL metric {0}'.format(metric))
