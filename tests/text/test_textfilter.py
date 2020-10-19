#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from clinicgen.text.textfilter import LowerTextFilter


class TestLowerTextFilter(unittest.TestCase):
    def test_filter(self):
        text = 'Hello NLP!'
        tfilter = LowerTextFilter()
        self.assertEqual(tfilter.filter(text), 'hello nlp!')
