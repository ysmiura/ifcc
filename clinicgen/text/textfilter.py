#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_textfilter(name):
    if name == 'lower':
        return LowerTextFilter()
    elif name == 'none':
        return NullTextFilter()
    else:
        return None


class LowerTextFilter:
    @staticmethod
    def filter(text):
        return text.lower()


class NullTextFilter:
    @staticmethod
    def filter(text):
        return text
