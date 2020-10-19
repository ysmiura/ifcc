#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from clinicgen.data.areport import AReportData
from clinicgen.data.chexpert import CheXpertData
from clinicgen.data.flickr30k import Flickr30kData
from clinicgen.data.mimiccxr import MIMICCXRData
from clinicgen.data.openi import OpenIData


class Data:
    @classmethod
    def get_datasets(cls, path, corpus, word_indexes, sentsplitter, tokenizer, textfilter, tokenfilter, max_sent,
                     max_word, multi_image=1, img_mode='center', img_augment=False, single_test=False, cache_data=None,
                     section='findings', anatomy=None, meta=None, ignore_blank=False, exclude_ids=None, a_labels=None,
                     filter_reports=True, test_only=False):
        datasets = {}
        if cache_data is not None:
            cache = True
            if not os.path.exists(cache_data):
                os.mkdir(cache_data)
            dump_dir = os.path.join(cache_data, corpus)
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)
            train_dump = os.path.join(dump_dir, 'train')
            val_dump = os.path.join(dump_dir, 'val')
            test_dump = os.path.join(dump_dir, 'test')
        else:
            cache = False
            train_dump, val_dump, test_dump = None, None, None

        if corpus == 'a':
            labels = cls.load_a_report_labels(a_labels)
            if not test_only:
                target_transform = AReportData.get_target_transform(word_indexes, 'train', sentsplitter, tokenizer,
                                                                    textfilter, tokenfilter, max_sent, max_word)
                datasets['train'] = AReportData(path, section=section, anatomy=anatomy, meta=meta, split='train',
                                                target_transform=target_transform, exclude_ids=exclude_ids,
                                                cache_image=cache, cache_text=cache, labels=labels,
                                                multi_image=multi_image, img_mode=img_mode, img_augment=img_augment,
                                                dump_dir=train_dump, filter_reports=filter_reports)
            target_transform = AReportData.get_target_transform(word_indexes, 'validation', sentsplitter, tokenizer,
                                                                textfilter, tokenfilter, max_sent, max_word)
            datasets['validation'] = AReportData(path, section=section, anatomy=anatomy, meta=meta, split='validation',
                                                 target_transform=target_transform, exclude_ids=exclude_ids,
                                                 cache_image=cache, cache_text=cache, labels=labels,
                                                 multi_image=multi_image, img_mode=img_mode,
                                                 single_image_doc=single_test, dump_dir=val_dump,
                                                 filter_reports=filter_reports)
            target_transform = AReportData.get_target_transform(word_indexes, 'test', sentsplitter, tokenizer,
                                                                textfilter, tokenfilter, max_sent, max_word)
            datasets['test'] = AReportData(path, section=section, anatomy=anatomy, meta=meta, split='test',
                                           target_transform=target_transform, exclude_ids=exclude_ids,
                                           cache_image=cache, cache_text=cache, labels=labels,
                                           multi_image=multi_image, img_mode=img_mode, single_image_doc=single_test,
                                           dump_dir=test_dump, filter_reports=filter_reports)
        elif corpus == 'chexpert':
            if not test_only:
                datasets['train'] = CheXpertData(path, split='train', cache_image=cache, multi_image=multi_image,
                                                 img_mode=img_mode, img_augment=img_augment,ignore_blank=ignore_blank,
                                                 dump_dir=train_dump)
            datasets['validation'] = CheXpertData(path, split='valid', cache_image=cache, multi_image=multi_image,
                                                  img_mode=img_mode, ignore_blank=ignore_blank, dump_dir=val_dump)
        elif corpus == 'flickr30k':
            if not test_only:
                target_transform = Flickr30kData.get_target_transform(word_indexes, 'train', sentsplitter, tokenizer,
                                                                      textfilter, tokenfilter, max_sent, max_word)
                datasets['train'] = Flickr30kData(path, meta=meta, split='train', target_transform=target_transform,
                                                  cache_image=cache, img_mode=img_mode, img_augment=img_augment,
                                                  cache_text=cache, dump_dir=train_dump)
            target_transform = Flickr30kData.get_target_transform(word_indexes, 'validation', sentsplitter, tokenizer,
                                                                  textfilter, tokenfilter, max_sent, max_word)
            datasets['validation'] = Flickr30kData(path, meta=meta, split='val', target_transform=target_transform,
                                                   cache_image=cache, img_mode=img_mode, cache_text=cache,
                                                   dump_dir=val_dump)
            target_transform = Flickr30kData.get_target_transform(word_indexes, 'test', sentsplitter, tokenizer,
                                                                  textfilter, tokenfilter, max_sent, max_word)
            datasets['test'] = Flickr30kData(path, meta=meta, split='test', target_transform=target_transform,
                                             cache_image=cache, img_mode=img_mode, cache_text=cache, dump_dir=test_dump)
        elif corpus == 'mimic-cxr':
            if not test_only:
                target_transform = MIMICCXRData.get_target_transform(word_indexes, 'train', sentsplitter, tokenizer,
                                                                     textfilter, tokenfilter, max_sent, max_word)
                datasets['train'] = MIMICCXRData(path, section=section, split='train',
                                                 target_transform=target_transform, cache_image=cache, cache_text=cache,
                                                 multi_image=multi_image, img_mode=img_mode, img_augment=img_augment,
                                                 dump_dir=train_dump, filter_reports=filter_reports)
            target_transform = MIMICCXRData.get_target_transform(word_indexes, 'validation', sentsplitter, tokenizer,
                                                                 textfilter, tokenfilter, max_sent, max_word)
            datasets['validation'] = MIMICCXRData(path, section=section, split='validate',
                                                  target_transform=target_transform, cache_image=cache,
                                                  cache_text=cache, multi_image=multi_image, img_mode=img_mode,
                                                  single_image_doc=single_test, dump_dir=val_dump,
                                                  filter_reports=filter_reports)
            target_transform = MIMICCXRData.get_target_transform(word_indexes, 'test', sentsplitter, tokenizer,
                                                                 textfilter, tokenfilter, max_sent, max_word)
            datasets['test'] = MIMICCXRData(path, section=section, split='test', target_transform=target_transform,
                                            cache_image=cache, cache_text=True, multi_image=multi_image,
                                            img_mode=img_mode, single_image_doc=single_test, dump_dir=test_dump,
                                            filter_reports=filter_reports)
        elif corpus == 'open-i':
            if not test_only:
                target_transform = OpenIData.get_target_transform(word_indexes, 'train', sentsplitter, tokenizer,
                                                                  textfilter, tokenfilter, max_sent, max_word)
                datasets['train'] = OpenIData(path, section=section, meta=meta, split='train',
                                              target_transform=target_transform, cache_image=cache, cache_text=cache,
                                              multi_image=multi_image, img_mode=img_mode, img_augment=img_augment,
                                              dump_dir=train_dump)
            target_transform = OpenIData.get_target_transform(word_indexes, 'validation', sentsplitter, tokenizer,
                                                              textfilter, tokenfilter, max_sent, max_word)
            datasets['validation'] = OpenIData(path, section=section, meta=meta, split='validation',
                                               target_transform=target_transform, cache_image=cache, cache_text=cache,
                                               multi_image=multi_image, img_mode=img_mode, single_image_doc=single_test,
                                               dump_dir=val_dump)
            target_transform = OpenIData.get_target_transform(word_indexes, 'test', sentsplitter, tokenizer, textfilter,
                                                              tokenfilter, max_sent, max_word)
            datasets['test'] = OpenIData(path, section=section, meta=meta, split='test',
                                         target_transform=target_transform, cache_image=cache, cache_text=cache,
                                         multi_image=multi_image, img_mode=img_mode, single_image_doc=single_test,
                                         dump_dir=test_dump)
        else:
            raise ValueError('Unknown corpus {0}'.format(corpus))
        return datasets

    @classmethod
    def load_a_report_labels(cls, path):
        labels = None
        if path is not None:
            labels = {}
            with open(path, encoding='utf-8') as f:
                for line in f:
                    entry = line.rstrip().split(' ')
                    label = int(entry[-1])
                    if label == AReportData.LABEL_GOOD or label == AReportData.LABEL_INVERT:
                        labels[entry[0]] = label
        return labels

    @classmethod
    def load_demo_data(cls, params, split, update_image=False, target_transform=False, dump_dir=None):
        cache = False if dump_dir is None else True
        if params['corpus'] == 'a':
            if split == 'val':
                split = 'validation'
            labels = Data.load_a_report_labels(params['a_labels'])
            transform = AReportData.get_target_transform({}, split) if target_transform else None
            data = AReportData(params['data'], params['section'], params['anatomy'], meta=params['splits'],
                               split=split, target_transform=transform, cache_text=cache, filter_reports=True,
                               labels=labels, multi_image=params['multi_image'], dump_dir=dump_dir,
                               update_image=update_image)
        elif params['corpus'] == 'flickr30k':
            transform = Flickr30kData.get_target_transform({}, split) if target_transform else None
            data = Flickr30kData(params['data'], meta=params['splits'], split=split, target_transform=transform,
                                 cache_text=True, dump_dir=dump_dir)
        elif params['corpus'] == 'mimic-cxr':
            if split == 'val':
                split = 'validate'
            transform = MIMICCXRData.get_target_transform({}, split) if target_transform else None
            data = MIMICCXRData(params['data'], params['section'], split=split, target_transform=transform,
                                filter_reports=True, multi_image=params['multi_image'], dump_dir=dump_dir)
        elif params['corpus'] == 'open-i':
            if split == 'val':
                split = 'validation'
            transform = OpenIData.get_target_transform({}, split) if target_transform else None
            data = OpenIData(params['data'], params['section'], meta=params['splits'], split=split,
                             target_transform=transform, cache_text=cache, multi_image=params['multi_image'],
                             dump_dir=dump_dir)
        else:
            raise ValueError('Unknown corpus {0}'.format(params['corpus']))
        return data
