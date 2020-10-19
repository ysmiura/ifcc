#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import time
import torch
from tqdm import tqdm
from clinicgen.data.image2text import _RadiologyReportData


class CheXpertData(_RadiologyReportData):
    IMAGE_NUM = {'train': 223414, 'valid': 234}
    LABEL_IGNORE = -100
    LABEL_NEGATIVE = 1
    NUM_CLASSES = 3
    NUM_LABELS = 14

    def __init__(self, root, split=None, cache_image=False, multi_image=1, img_mode='center', img_augment=False,
                 ignore_blank=False, dump_dir=None):
        super().__init__(root, section='findings', split=split, cache_image=cache_image, cache_text=True,
                         multi_image=multi_image, dump_dir=dump_dir)
        self.ignore_blank = ignore_blank
        pre_transform, self.transform = CheXpertData.get_transform(cache_image, img_mode, img_augment)
        self.target_transform = None

        if dump_dir is not None:
            t = time.time()
            if self.load():
                print('Loaded data dump from %s (%.2fs)' % (dump_dir, time.time() - t))
                self.pre_processes()
                return

        images = os.path.join(root, '{0}.csv'.format(split))
        with open(images, encoding='utf-8') as f:
            f.readline()
            reader = csv.reader(f)
            with tqdm(total=self.IMAGE_NUM[split]) as pbar:
                pbar.set_description('Data ({0})'.format(split))
                count = 0
                interval = 1000
                inc = 1 if split == 'train' else 0

                for entry in reader:
                    sub_paths = entry[0].split('/')
                    image_id = '/'.join(sub_paths[2:5])
                    doc_id = '/'.join(sub_paths[2:4])
                    self.ids.append(image_id)
                    self.doc_ids.append(doc_id)
                    # image
                    image = os.path.join(root, '/'.join(sub_paths[1:]))
                    if cache_image:
                        image = self.bytes_image(image, pre_transform)
                    # labels
                    labels = []
                    for label in entry[5:]:
                        if len(label) == 0:
                            labels.append(self.LABEL_IGNORE)
                        else:
                            labels.append(int(float(label)) + inc)
                    self.samples.append((image, labels))
                    self.targets.append(labels)
                    count += 1
                    if count >= interval:
                        pbar.update(count)
                        count = 0
                if count > 0:
                    pbar.update(count)

        if dump_dir is not None:
            self.dump()
        self.pre_processes()

    def __getitem__(self, index):
        rid, sample, target, vp = super().__getitem__(index)
        target = torch.tensor(target)
        return rid, sample, target, vp

    @classmethod
    def get_transform(cls, cache_image=False, mode='center', augment=False):
        return cls._transform(cache_image, 224, mode, augment)

    def convert_blank_labels(self, print_num=True):
        t = time.time()
        if print_num:
            print('Converting blank labels ... ', end='', flush=True)
        new_samples, new_targets = [], []
        for i in range(len(self.samples)):
            new_labels = []
            image, labels = self.samples[i]
            for label in labels:
                if label == self.LABEL_IGNORE:
                    new_labels.append(self.LABEL_NEGATIVE)
                else:
                    new_labels.append(label)
            new_samples.append((image, new_labels))
            new_targets.append(new_labels)
        self.samples = new_samples
        self.targets = new_targets
        if print_num:
            print('done (%.2fs)' % (time.time() - t), flush=True)

    def pre_processes(self):
        if not self.ignore_blank:
            self.convert_blank_labels()
        if self.multi_image > 1:
            self.convert_to_multi_images()
