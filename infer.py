#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import os
import pickle
import random
import sys
import time
import traceback
import nltk
import numpy as np
import stanza
import torch.utils.data
from torch.utils.data import DataLoader
from clinicgen.data.image2text import PretrainedEmbeddings
from clinicgen.data.utils import Data
from clinicgen.eval import GenEval
from clinicgen.log import EpochLog, FileLogger
from clinicgen.models.utils import Models


def main(args):
    # Download
    if args.nltk_download:
        nltk.download('punkt')
    if args.stanza_download:
        stanza.download('en', processors='tokenize,lemma,pos,ner')
        stanza.download('en', package='radiology')
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Embeddings
    embeddings, word_idxs = PretrainedEmbeddings.load_embeddings(args.embeddings)
    # Data
    t = time.time()
    hierarchical = Models.hierarchical(args.model)
    max_sent = args.max_sent if hierarchical else None
    datasets = Data.get_datasets(args.data, args.corpus, word_idxs, args.sentsplitter, args.tokenizer, args.textfilter,
                                 args.tokenfilter, max_sent, args.max_word, multi_image=args.multi_image,
                                 img_mode=args.img_trans, img_augment=args.img_augment, single_test=args.single_test,
                                 cache_data=args.cache_data, section=args.section, anatomy=args.anatomy,
                                 meta=args.splits, exclude_ids=args.exclude_ids, a_labels=args.a_labels,
                                 test_only=True)
    nw = 0 if args.cache_data else args.num_workers
    val_loader = DataLoader(datasets['validation'], batch_size=args.batch_size, shuffle=False, num_workers=nw,
                            pin_memory=args.pin_memory)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=nw,
                             pin_memory=args.pin_memory)
    print('Data: validation={0}, test={1} (load time {2:.2f}s)'.format(len(val_loader.dataset),
                                                                       len(test_loader.dataset), time.time() - t))
    if args.cider_df is not None:
        with gzip.open(args.cider_df) as f:
            cider_df = pickle.load(f)
        print('CIDEr df: {0}'.format(len(cider_df)))
    else:
        cider_df = None
    # Model configurations
    device = 'gpu' if args.cuda else 'cpu'

    model = Models.get_model(args.model, embeddings, args.hidden_size, args.max_word, args.max_sent,
                             multi_image=args.multi_image, multi_merge=args.multi_merge, image_model=args.img_model,
                             image_pretrained=args.img_pretrained, view_position=args.view_position,
                             parallel_sent=args.parallel_sent, word_idxs=word_idxs, device=device,
                             cnnrnnrnn_topic_state=args.cnnrnnrnn_topic_state,
                             cnnrnnrnn_simple_proj=args.cnnrnnrnn_simple_proj, sat_lstm_dim=args.sat_lstm_dim,
                             trans_image_pe=args.img_pe, trans_layers=args.trans_layers,
                             trans_enc_layers=args.trans_enc_layers, trans_layer_norm=args.trans_layer_norm,
                             m2_memory=args.m2_memory, tienet_labels=args.tienet_labels, verbose=args.verbose)
    if device == 'gpu':
        model = model.cuda()

    # Train and test processes
    evaluator = GenEval(model, word_idxs, beam_size=args.beam_size, cider_df=cider_df, spice=args.spice,
                        bert_score=args.bert_score, bert_score_penalty=args.bert_score_penalty, nli=args.nli,
                        nli_compare=args.nli_comp, nli_label=args.nli_label, nli_neutral_score=args.nli_neutral_score,
                        nli_prf=args.nli_prf, nli_batch=args.nli_batch, entity_match=args.entity_match,
                        entity_mode=args.entity_mode, nthreads=args.nthreads, pin_memory=args.pin_memory,
                        verbose=args.verbose)
    with FileLogger(args.out, args.out, model, evaluator, None, None, None, None, mode='all', device=device) as logger:
        try:
            pbar_vals = {'losses': None, 'val_scores': None, 'test_scores': None}
            bep = logger.load_baseline(args.baseline_model)
            print('Starting from baseline {0} ({1})'.format(os.path.basename(args.baseline_model), bep + 1))
            # Run evaluation on the baseline model
            EpochLog.log_datasets(logger, pbar_vals, 0, 0, None, val_loader, test_loader, save=False, progress=True)
            logger.save_parameters(args)
        except BaseException:
            print('Unexpected exception: {0}'.format(traceback.format_exc()))


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from a model for report generation')
    parser.add_argument('data', type=str, help='A path to clinical data')
    parser.add_argument('baseline_model', type=str, help='A baseline model to start from')
    parser.add_argument('embeddings', type=str, help='A path to word embeddings')
    parser.add_argument('out', type=str, help='An output path')
    parser.add_argument('--a-labels', type=str, default=None, help='AReport validity labels')
    parser.add_argument('--anatomy', type=str, default=None, help='A specific anatomy to target')
    parser.add_argument('--batch-size', type=int, default=24, help='Batch size')
    parser.add_argument('--beam-size', type=int, default=4, help='Beam size')
    parser.add_argument('--bert-score', type=str, default=None, help='BERTScore model type')
    parser.add_argument('--bert-score-penalty', default=False, action='store_true', help='Add a Gaussian penalty to BERTScore')
    parser.add_argument('--cache-data', type=str, default=None, help='Cache images and texts to memory and disk')
    parser.add_argument('--cider-df', type=str, default=None, help='A path to CIDEr DF')
    parser.add_argument('--cnnrnnrnn-mlp-proj', dest='cnnrnnrnn_simple_proj', default=True, action='store_false', help='An MLP visual feature projection for CNNRNNRNN')
    parser.add_argument('--cnnrnnrnn-topic-state', default=False, action='store_true', help='Use topic as an initial word LSTM state')
    parser.add_argument('--corpus', type=str, default='a', choices=['a', 'flickr30k', 'mimic-cxr', 'open-i'], help='Corpus name')
    parser.add_argument('--cuda', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--entity-match', type=str, default=None, help='A path to reference entities')
    parser.add_argument('--entity-mode',type=str, default='nli-f', help='Entity match mode')
    parser.add_argument('--exclude-ids', type=str, default=None, help='IDs to exclude from the data')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden unit size')
    parser.add_argument('--img-no-augment', dest='img_augment', default=True, action='store_false', help='Do not augment images')
    parser.add_argument('--img-pe', default=False, action='store_true', help='Add positional encodings for images')
    parser.add_argument('--img-trans', type=str, default='pad', choices=['center', 'pad'], help='Image transformation mode')
    parser.add_argument('--img-model', type=str, default=None, help='An image model')
    parser.add_argument('--img-pretrained', type=str, default=None, help='Pre-trained image model')
    parser.add_argument('--iter-sent', dest='parallel_sent', default=True, action='store_false', help='Iteratively process sentences')
    parser.add_argument('--m2-memory', type=int, default=40, help='M2 Transformer memory size')
    parser.add_argument('--max-sent', type=int, default=1, help='Max sentence num')
    parser.add_argument('--max-word', type=int, default=128, help='Max word num')
    parser.add_argument('--model', type=str, default='m2trans', choices=['cnnrnnrnn', 'kwl', 'm2trans', 'sat', 'tienet', 'trans', 'trans-s'])
    parser.add_argument('--multi-image', type=int, default=3, help='Multi image number')
    parser.add_argument('--multi-merge', type=str, default='max', choices=['att', 'max'], help='A merge method for multi images')
    parser.add_argument('--nli', type=str, default=None, choices=['mednli', 'mednli-rad'], help='NLI model type')
    parser.add_argument('--nli-batch', type=int, default=24, help='NLI batch size')
    parser.add_argument('--nli-comp', type=str, default='bert-score', help='NLI comparison method')
    parser.add_argument('--nli-label', type=str, default='entailment', choices=['contradiction', 'entailment'], help='NLI score label')
    parser.add_argument('--nli-neutral-score', type=float, default=(1.0 / 3), help='An NLI entailment neutral score')
    parser.add_argument('--nli-prf', type=str, default='f', choices=['f', 'fh', 'fp', 'p', 'r'], help='NLI metric')
    parser.add_argument('--nltk-download', default=False, action='store_true', help='Download NLTK punkt data')
    parser.add_argument('--nthreads', type=int, default=2, help='Number of threads')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of background workers for data loader')
    parser.add_argument('--pin-memory', dest='pin_memory', default=False, action='store_true', help='Use pin-memory on data loaders')
    parser.add_argument('--sat-lstm-dim', type=int, default=1000, help='An LSTM dimension for SAT')
    parser.add_argument('--section', type=str, default='findings', help='Report section')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--sentsplitter', type=str, default='none', choices=['linebreak', 'nltk', 'none', 'stanford'], help='Sentence splitter name')
    parser.add_argument('--single-test', default=False, action='store_true', help='Test with a single image setting')
    parser.add_argument('--spice', default=False, action='store_true', help='SPICE evaluation')
    parser.add_argument('--splits', type=str, default=None, help='A path to a file defining splits')
    parser.add_argument('--stanza-download', default=False, action='store_true', help='Download Stanza clinical model')
    parser.add_argument('--textfilter', type=str, default='lower', help='Text filter')
    parser.add_argument('--tienet-labels', type=str, default=None, help='TieNet labels')
    parser.add_argument('--tokenfilter', type=str, default='none', help='Token filter')
    parser.add_argument('--tokenizer', type=str, default='nltk', choices=['nltk', 'none', 'stanford', 'whitespace'], help='Tokenizer name')
    parser.add_argument('--trans-enc-layers', type=int, default=None, help='Number of transformer encoder layers')
    parser.add_argument('--trans-layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--trans-layer-no-norm', dest='trans_layer_norm', default=True, action='store_false', help='Do not Layer normalize visual features in transformer models')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose outputs')
    parser.add_argument('--view-position', default=False, action='store_true', help='Include view position embeddings')
    return parser.parse_args()


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(cdir)
    args = parse_args()
    main(args)
