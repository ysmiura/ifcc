#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from clinicgen.text.sentsplit import LineBreakSplitter, NLTKSentenceSplitter, SpaCySentenceSplitter, StanzaSentenceSplitter


class TestLineBreakSplitter(unittest.TestCase):
    def test_split(self):
        splitter = LineBreakSplitter()
        text = 'Hello NLP! Running a test\nof sentence splitting. Line breaks are considered as sentence boundaries.'
        sents = splitter.split(text)
        self.assertEqual(len(sents), 2)
        self.assertTrue(sents[0].startswith('Hello'))
        self.assertTrue(sents[1].startswith('of'))


class TestNLTKSentenceSplitter(unittest.TestCase):
    def test_split(self):
        splitter = NLTKSentenceSplitter()
        text = 'Hello NLP! Running a test\nof sentence splitting. Line breaks are considered as sentence boundaries.'
        sents = splitter.split(text)
        self.assertEqual(len(sents), 4)
        self.assertTrue(sents[0].startswith('Hello'))
        self.assertTrue(sents[1].startswith('Running'))
        self.assertTrue(sents[2].startswith('of'))
        self.assertTrue(sents[3].startswith('Line'))


class TestSpaCySentenceSplitter(unittest.TestCase):
    def test_split(self):
        splitter = SpaCySentenceSplitter('en_core_web_sm')
        text = 'Hello NLP! Running a test\nof sentence splitting. Line breaks are considered as sentence boundaries.'
        sents = splitter.split(text)
        self.assertEqual(len(sents), 4)
        self.assertTrue(sents[0].startswith('Hello'))
        self.assertTrue(sents[1].startswith('Running'))
        self.assertTrue(sents[2].startswith('of'))
        self.assertTrue(sents[3].startswith('Line'))


class TestStanzaSentenceSplitter(unittest.TestCase):
    def test_split(self):
        splitter = StanzaSentenceSplitter()
        text = 'Hello NLP! Running a test\nof sentence splitting. Line breaks are considered as sentence boundaries.'
        sents = splitter.split(text)
        self.assertEqual(len(sents), 4)
        self.assertTrue(sents[0].startswith('Hello'))
        self.assertTrue(sents[1].startswith('Running'))
        self.assertTrue(sents[2].startswith('of'))
        self.assertTrue(sents[3].startswith('Line'))