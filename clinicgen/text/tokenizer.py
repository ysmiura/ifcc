#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy
from nltk.tokenize import wordpunct_tokenize
from stanza import Pipeline


def get_tokenizer(name, port=None):
    if name == 'nltk':
        return NLTKTokenizer()
    elif name == 'scispacy':
        return SpaCyTokenizer('en_core_sci_md')
    elif name == 'spacy':
        return SpaCyTokenizer('en_core_web_sm')
    elif name == 'stanford':
        return StanzaTokenizer()
    elif name == 'whitespace':
        return WhiteSpaceTokenizer()
    else:
        return None


class NLTKTokenizer:
    @staticmethod
    def tokenize(text):
        return wordpunct_tokenize(text)


class SpaCyTokenizer:
    def __init__(self, model):
        self.nlp = spacy.load(model)

    def tokenize(self, text):
        toks = []
        for tok in self.nlp(text):
            toks.append(tok.text)
        return toks


class StanzaTokenizer:
    def __init__(self):
        self.nlp = Pipeline(lang='en', processors='tokenize')

    def tokenize(self, text):
        toks = []
        doc = self.nlp(text)
        for sentence in doc.sentences:
            for token in sentence.tokens:
                toks.append(token.text)
        return toks


class WhiteSpaceTokenizer:
    @staticmethod
    def tokenize(text):
        return text.split()
