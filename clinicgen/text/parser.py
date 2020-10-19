#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from nltk import Tree
from nltk.compat import unicode_repr
from six import string_types
from stanza.server import CoreNLPClient


class CoreNLPBinaryParser:
    DEFAULT_PORT = 9003

    def __init__(self, threads=1, port=None):
        sid = random.randint(0, 65535)
        if port is None:
            port = self.DEFAULT_PORT
        self.corenlp = CoreNLPClient(endpoint='http://localhost:{0}'.format(port), annotators=['parse'],
                                     output_format='json', properties={'ssplit.eolonly': 'true'}, timeout=300000,
                                     memory='8G', threads=threads, server_id='clinicgen{0}'.format(sid))
        self.corenlp.start()
        self.run = True

    def __del__(self):
        self.stop()

    @classmethod
    def _format(cls, tree):
        childstrs = []
        for child in tree:
            if isinstance(child, Tree):
                childstrs.append(cls._format(child))
            elif isinstance(child, tuple):
                childstrs.append("/".join(child))
            elif isinstance(child, string_types):
                childstrs.append('%s' % child)
            else:
                childstrs.append(unicode_repr(child))
        if len(childstrs) > 1:
            return '( %s )' % ' '.join(childstrs)
        else:
            return childstrs[0]

    @classmethod
    def binarize(cls, tree):
        # collapse
        t = Tree.fromstring(tree)
        # chomsky normal form transformation
        Tree.collapse_unary(t, collapsePOS=True, collapseRoot=True)
        Tree.chomsky_normal_form(t)
        s = cls._format(t)
        return s

    def parse(self, text):
        ann = self.corenlp.annotate(text)
        return self.binarize(ann['sentences'][0]['parse'])

    def stop(self):
        if self.run:
            self.corenlp.stop()
            self.run = False
