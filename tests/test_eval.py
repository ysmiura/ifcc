#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import torch
from clinicgen.data.image2text import PretrainedEmbeddings as PreEmb
from clinicgen.eval import EntityMatcher, RecoverWords


class TestEntityMatcher(unittest.TestCase):
    def setUp(self):
        sentences = {'1': {0: 'No pleural effusions.'}, '2': {0: 'Enlarged heart.'}}
        entities = {'1': {'pleural': [0], 'effusions': [0]}, '2': {'heart': [0]}}
        target_types = {'ANATOMY': True, 'OBSERVATION': True}
        self.matcher = EntityMatcher(sentences, entities, target_types)

    def test_score(self):
        rs = self.matcher.score(['1', '2'], ['No pleural effusion.', 'Normal heart size.'])
        self.assertEqual(rs[1][0], 0.5)
        self.assertEqual(rs[1][1], 1.0)


class TestRecoverWords(unittest.TestCase):
    def setUp(self):
        word_idxs = {'__PAD__': PreEmb.INDEX_PAD, '__START__': PreEmb.INDEX_START, '__UNK__': PreEmb.INDEX_UNKNOWN,
                     PreEmb.TOKEN_EOS: PreEmb.INDEX_EOS, 'Hello': 4, 'world': 5, '!': 6, 'NLP': 7}
        self.recover_words = RecoverWords(word_idxs)

    def test___call__(self):
        stops = torch.tensor([[-1.0, -0.2, 1.0], [-2.0, 0.1, 0.5]])
        samples = np.zeros((2, 3, 4))
        samples[0][0] = np.array([4, 5, 6, 0])
        samples[0][1] = np.array([4, 7, 6, 0])
        samples[1][0] = np.array([4, 5, 7, 6])
        samples = torch.tensor(samples).type(torch.long)
        rec, _ = self.recover_words(stops, samples)
        self.assertEqual(rec[0], 'Hello world !\nHello NLP !')
        self.assertEqual(rec[1], 'Hello world NLP !')
