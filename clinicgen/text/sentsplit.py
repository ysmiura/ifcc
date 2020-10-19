#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy
from nltk.tokenize import sent_tokenize
from stanza import Pipeline


def get_sentsplitter(name, linebreak=True):
    if name == 'linebreak':
        return LineBreakSplitter()
    elif name == 'nltk':
        return NLTKSentenceSplitter(linebreak)
    elif name == 'none':
        return NullSentenceSplitter()
    elif name == 'scispacy':
        return SpaCySentenceSplitter('en_core_sci_md', linebreak)
    elif name == 'spacy':
        return SpaCySentenceSplitter('en_core_web_sm', linebreak)
    elif name == 'stanford':
        return StanzaSentenceSplitter(linebreak)
    else:
        return None


class LineBreakSplitter:
    @staticmethod
    def split(text):
        return text.split('\n')


class NLTKSentenceSplitter:
    def __init__(self, linebreak=True):
        self.linebreak = linebreak

    def split(self, text):
        sents = []
        for sent in sent_tokenize(text):
            if self.linebreak:
                for sent2 in sent.split('\n'):
                    sents.append(sent2)
            else:
                sents.append(sent)
        return sents


class NullSentenceSplitter:
    @staticmethod
    def split(text):
        return [text]


class SpaCySentenceSplitter:
    def __init__(self, model, linebreak=True):
        self.nlp = spacy.load(model)
        self.linebreak = linebreak

    def split(self, text):
        sents = []
        doc = self.nlp(text)
        for sent in doc.sents:
            if self.linebreak:
                for sent2 in sent.text.split('\n'):
                    sents.append(sent2)
            else:
                sents.append(sent.text)
        return sents


class StanzaSentenceSplitter:
    def __init__(self, linebreak=True):
        self.nlp = Pipeline(lang='en', processors='tokenize')
        self.linebreak = linebreak

    def split(self, text):
        sents = []
        doc = self.nlp(text)
        for sent in doc.sentences:
            if self.linebreak:
                for sent2 in sent.text.split('\n'):
                    sents.append(sent2)
            else:
                sents.append(sent.text)
        return sents
