#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
from clinicgen.nli import SimpleNLI


class TestSimpleNLI(unittest.TestCase):
    def setUp(self):
        resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources')
        model = os.path.join(resource_dir, SimpleNLI.RADNLI_STATES)
        model = SimpleNLI.load_model(model)
        self.nli = SimpleNLI(model, bert_score='distilbert-base-uncased')

    def test_predict(self):
        sent1s = ['No pleural effusions.', 'Enlarged heart.', 'Pulmonary edema.']
        sent2s = ['No pleural effusion.', 'Normal heart.', 'Clear lungs.']
        rs = self.nli.predict(sent1s, sent2s)
        self.assertEqual(rs[1][0], 'entailment')
        self.assertEqual(rs[1][1], 'contradiction')
        self.assertEqual(rs[1][2], 'neutral')
