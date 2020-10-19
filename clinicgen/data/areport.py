#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import re
import time
import numpy as np
import torch
from PIL import ImageFile, ImageOps
from torchvision.datasets.folder import has_file_allowed_extension
from tqdm import tqdm
from clinicgen.data.image2text import _RadiologyReportData


class AReportData(_RadiologyReportData):
    CHEXPERT_PATH = 'a-chexpert.csv.gz'
    ID_PATTERN = re.compile('^([0-9a-f]+)[-0-9a-f]*\\.(txt|png)$')
    IMAGE_ID_PATTERN = re.compile('/([0-9a-f]+-[0-9]+-[0-9]+)\\.png$')
    DIR_IMAGES = 'images'
    DIR_CORPUS = 'corpus'
    DIR_REPORTS = 'reports'
    PATH_IMAGES = 'all_images.csv'
    LABEL_INVERT = 0
    LABEL_GOOD = 2
    MODE_ALIGNED = 'aligned'
    MODE_TEXT = 'text'
    TARGET_IMAGE = 'image'
    TARGET_IMAGE_TEST = 'image-test'
    TARGET_REPORT = 'report'

    def __init__(self, root, section='findings', anatomy=None, meta=None, split=None, target='report',
                 target_transform=None, exclude_ids=None, cache_image=False, cache_text=False, labels=None,
                 multi_image=1, img_mode='center', img_augment=False, single_image_doc=False, dump_dir=None,
                 filter_reports=True, update_image=False):
        super().__init__(root, section, split, cache_image, cache_text, multi_image=multi_image,
                         single_image_doc=single_image_doc, dump_dir=dump_dir)
        self.labels = labels
        self.target = target
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        pre_transform, self.transform = AReportData.get_transform(cache_image, img_mode, img_augment)
        self.target_transform = target_transform
        anatomy_dic = {}
        if anatomy is not None:
            with open(os.path.join(root, 'meta.csv'), encoding='utf-8') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if row[-3].lower() != anatomy.lower():
                        anatomy_dic[row[0]] = True
        if meta is not None:
            self.chexpert_labels_path = os.path.join(os.path.dirname(meta), self.CHEXPERT_PATH)
        if dump_dir is not None:
            t = time.time()
            if self.load():
                print('Loaded data dump from %s (%.2fs)' % (dump_dir, time.time() - t))
                self.pre_processes(filter_reports, anatomy_dic, labels)
                return

        exclude_dic = {}
        if exclude_ids is not None:
            with open(exclude_ids, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        exclude_dic[line] = True

        meta_path = os.path.join(root, 'meta.csv') if meta is None else meta
        splits = {}
        with open(meta_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'id' and row[-1] == 'split':
                    continue
                splits[row[0]] = row[-1]
        paths = {}
        total = 0
        with open(os.path.join(root, self.PATH_IMAGES), encoding='utf-8') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if split is None or (row[1] in splits and splits[row[1]] == split):
                    total += 1
                    if row[1] not in paths:
                        paths[row[1]] = [row[0]]
                    else:
                        paths[row[1]].append(row[0])
        interval = 1 if int(total / 100) == 0 else int(total / 100)
        with tqdm(total=total) as pbar:
            pbar.set_description('Data ({0})'.format(split))
            count = 0
            for rid in sorted(paths.keys()):
                for image in paths[rid]:
                    if rid not in exclude_dic:
                        report = os.path.join(root, self.DIR_REPORTS, rid[0:2], rid[2:4], rid + '.txt')
                        if cache_text:
                            report = self.extract_text(report, compress=True)
                        m = self.IMAGE_ID_PATTERN.search(image)
                        self.ids.append(m.group(1))
                        self.doc_ids.append(rid)
                        if update_image:
                            image = self.update_image_path(root, image)
                        if cache_image:
                            image = self.bytes_image(image, pre_transform)
                        self.samples.append((image, report))
                        self.targets.append(report)
                    count += 1
                    if count >= interval:
                        pbar.update(count)
                        count = 0
            if count > 0:
                pbar.update(count)

        if len(self.samples) == 0 and len(self.targets) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        if dump_dir is not None:
            self.dump()
        self.pre_processes(filter_reports, anatomy_dic, labels)

    @classmethod
    def get_transform(cls, cache_image=False, mode='center', augment=False):
        return cls._transform(cache_image, 224, mode, augment)

    @classmethod
    def update_image_path(cls, root, image):
        check_dir = None
        check_i = 0
        root_dirs = root.split('/')
        for i, d in enumerate(root_dirs):
            if len(d) > 0:
                check_dir = d
                check_i = i
        image_dirs = image.split('/')
        image_i = 0
        for i, d in enumerate(image_dirs):
            if d == check_dir:
                image_i = i
        return os.path.join('/'.join(root_dirs[:check_i]), '/'.join(image_dirs[image_i:]))

    def __getitem__(self, index):
        rid, sample, target, vp = super().__getitem__(index)
        if self.target == self.TARGET_IMAGE:
            did = rid.split('-')[0]
            target = torch.Tensor(np.array([self.labels[did]])).type(torch.long)
        elif self.target == self.TARGET_IMAGE_TEST:
            target = 0
        return rid, sample, target, vp

    def _load_image(self, instance_id, path):
        img = super()._load_image(instance_id, path)
        if self.target != self.TARGET_IMAGE and self.target != self.TARGET_IMAGE_TEST:
            report_id = instance_id.split('-')[0]
            if report_id in self.labels and self.labels[report_id] == self.LABEL_INVERT:
                img = ImageOps.invert(img)
        return img

    def extract_section(self, text):
        if self.section is None:
            return text
        target = False
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 0:
                m = self.REPORT_PATTERN.search(line)
                if m is not None:
                    if m.group(1).lower() == self.section:
                        target = True
                        s = m.group(2)
                        if len(s) > 0:
                            lines.append(s)
                    else:
                        target = False
                elif target:
                    lines.append(line)
        if self.sub_section is not None:
            lines = list(filter(lambda ln: ln.lower().startswith(self.sub_section.lower()), lines))
        return '\n'.join(lines)

    def filter_with_dic(self, dic, label='anatomy', print_num=True):
        t = time.time()
        n = 0
        if print_num:
            print('Filtering reports by {0} ... '.format(label), end='', flush=True)
        new_ids, new_doc_ids, new_samples, new_targets = [], [], [], []
        for i in range(len(self.samples)):
            image_id = self.ids[i]
            rid = image_id.split('-')[0]
            image, report = self.samples[i]
            target = self.targets[i]
            if (label == 'anatomy' and rid not in dic) or (label == 'validity' and rid in dic):
                new_ids.append(image_id)
                new_doc_ids.append(self.doc_ids[i])
                new_samples.append((image, report))
                new_targets.append(target)
            else:
                n += 1
        self.ids = new_ids
        self.doc_ids = new_doc_ids
        self.samples = new_samples
        self.targets = new_targets
        if print_num:
            print('done %d (%.2fs)' % (n, time.time() - t), flush=True)

    def pre_processes(self, filter_reports, anatomy_dic, labels):
        if labels is not None:
            self.filter_with_dic(labels, 'validity')
        if filter_reports:
            self.filter_with_dic(anatomy_dic, 'anatomy')
            self.filter_empty_reports()
        if self.multi_image > 1:
            self.convert_to_multi_images()
        elif self.single_image_doc:
            self.convert_to_single_image()
        self.pre_transform_texts(self.split)

    def read_images(self, dir, extensions):
        images = {}
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    m = self.ID_PATTERN.search(fname)
                    if m is not None:
                        rid = m.group(1)
                        path = os.path.join(root, fname)
                        images[rid] = path
        return images

    def read_reports(self, dir, extensions):
        reports = {}
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    m = self.ID_PATTERN.search(fname)
                    if m is not None:
                        rid = m.group(1)
                        report = os.path.join(root, fname)
                        if self.cache_text:
                            self.extract_section(report)
                        reports[rid] = report
        return reports
