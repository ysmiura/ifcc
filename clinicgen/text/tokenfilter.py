#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import re


def get_tokenfilter(name):
    if name == 'alphanum':
        return AlphaNumFilter()
    elif name == 'none':
        return NoneTokenFilter()
    else:
        return None


class AlphaNumFilter:
    NUMBER_TOKEN = '__NUM__'
    ALPHANUM_PATTERN = re.compile('^[a-zA-Z0-9\\\.]+$')
    NUM_PATTERN = re.compile('^[0-9\\\.]+$')

    @classmethod
    def filter(cls, tokens):
        new_tokens = []
        for token in tokens:
            if token != '.':
                if cls.ALPHANUM_PATTERN.search(token) is not None:
                    new_tokens.append(token)
                elif cls.NUM_PATTERN.search(token) is not None:
                    new_tokens.append(cls.NUMBER_TOKEN)
        return new_tokens


class NoneTokenFilter:
    @staticmethod
    def filter(tokens):
        return tokens
