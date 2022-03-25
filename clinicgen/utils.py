#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torch import sigmoid
from torch.nn import DataParallel
from clinicgen.data.image2text import PretrainedEmbeddings


def data_cuda(*tensors, device='gpu', non_blocking=False):
    if device == 'gpu':
        cuda_tensors = []
        for tensor in tensors:
            cuda_tensors.append(tensor.cuda(device=0, non_blocking=non_blocking))
    else:
        cuda_tensors = tensors

    if len(cuda_tensors) > 1:
        return cuda_tensors
    else:
        return cuda_tensors[0]


class RecoverWords:
    def __init__(self, word_indexes):
        self.index_words = {}
        for word, index in word_indexes.items():
            self.index_words[index] = word

    def __call__(self, *inputs, normalized=False):
        reports = []

        if len(inputs) == 2:
            stops, samples = inputs
            if not normalized:
                stops = sigmoid(stops)
            stops = stops.detach().cpu().numpy()
            samples = samples.detach().cpu().numpy()
            masks = np.zeros((samples.shape[0], samples.shape[1], samples.shape[2]), dtype='float')

            for i in range(stops.shape[0]):
                stop_sent = False
                sentences = []
                for j in range(stops.shape[1]):
                    if not stop_sent:
                        if j > 0 and stops[i][j] >= 0.5:
                            stop_sent = True
                        else:
                            stop_word = False
                            words = []
                            for k in range(samples.shape[2]):
                                if not stop_word:
                                    index = samples[i][j][k]
                                    masks[i][j][k] = 1.0
                                    if index == PretrainedEmbeddings.INDEX_EOS:
                                        stop_word = True
                                    elif index != PretrainedEmbeddings.INDEX_PAD:
                                        words.append(self.index_words[index])
                            sentences.append(' '.join(words))
                reports.append('\n'.join(sentences))
        else:
            samples = inputs[0]
            samples = samples.detach().cpu().numpy()
            masks = np.zeros((samples.shape[0], samples.shape[1]), dtype='float')

            for i in range(samples.shape[0]):
                stop_word = False
                words = []
                for k in range(samples.shape[1]):
                    if not stop_word:
                        index = samples[i][k]
                        masks[i][k] = 1.0
                        if index == PretrainedEmbeddings.INDEX_EOS:
                            stop_word = True
                        elif index != PretrainedEmbeddings.INDEX_PAD:
                            words.append(self.index_words[index])
                reports.append(' '.join(words))
        return reports, masks

    def array(self, idxs):
        return list(map(lambda idx: self.index_words[idx], idxs))


class DataParallelSwitch(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelSwitch, self).__init__(module, device_ids, output_device, dim)
        self.parallel = False

    def forward(self, *inputs, **kwargs):
        if not self.parallel:
            return self.module(*inputs, **kwargs)
        return super(DataParallelSwitch, self).forward(inputs, kwargs)
