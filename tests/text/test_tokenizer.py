#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from clinicgen.text.tokenizer import NLTKTokenizer, SpaCyTokenizer, StanzaTokenizer, WhiteSpaceTokenizer


class TestNLTKTokenizer(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = NLTKTokenizer()
        text = 'Hello NLP! Running a (tokenization) test.'
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 10)
        self.assertEqual(tokens[0], 'Hello')
        self.assertEqual(tokens[2], '!')
        self.assertEqual(tokens[3], 'Running')
        self.assertEqual(tokens[5], '(')
        self.assertEqual(tokens[6], 'tokenization')
        self.assertEqual(tokens[9], '.')


class TestSpaCyTokenizer(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = SpaCyTokenizer('en_core_web_sm')
        text = 'Hello NLP! Running a (tokenization) test.'
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 10)
        self.assertEqual(tokens[0], 'Hello')
        self.assertEqual(tokens[2], '!')
        self.assertEqual(tokens[3], 'Running')
        self.assertEqual(tokens[5], '(')
        self.assertEqual(tokens[6], 'tokenization')
        self.assertEqual(tokens[9], '.')


class TestStanzaTokenizer(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = StanzaTokenizer()
        text = 'Hello NLP! Running a (tokenization) test.'
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 10)
        self.assertEqual(tokens[0], 'Hello')
        self.assertEqual(tokens[2], '!')
        self.assertEqual(tokens[3], 'Running')
        self.assertEqual(tokens[5], '(')
        self.assertEqual(tokens[6], 'tokenization')
        self.assertEqual(tokens[9], '.')


class TestWhiteSpaceTokenizer(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = WhiteSpaceTokenizer()
        text = 'Hello NLP! Running a (tokenization) test.'
        tokens = tokenizer.tokenize(text)
        self.assertEqual(len(tokens), 6)
        self.assertEqual(tokens[0], 'Hello')
        self.assertEqual(tokens[1], 'NLP!')
        self.assertEqual(tokens[2], 'Running')
        self.assertEqual(tokens[4], '(tokenization)')
        self.assertEqual(tokens[5], 'test.')

