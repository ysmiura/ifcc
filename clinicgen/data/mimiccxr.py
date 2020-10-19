#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import gzip
import os
import pickle
import time
import torch
from tqdm import tqdm
from clinicgen.data.image2text import _RadiologyReportData


class MIMICCXRData(_RadiologyReportData):
    IMAGE_NUM = 377110
    LABEL_CHEXPERT = 'chexpert'
    CHEXPERT_MAP = [13, 4, 1, 7, 6, 3, 2, 9, 0, 10, 8, 11, 5, 12]

    CHEXPERT_PATH = 'mimic-cxr-2.0.0-chexpert.csv.gz'
    META_PATH = 'mimic-cxr-2.0.0-metadata.csv.gz'
    SECTIONED_PATH = 'mimic_cxr_sectioned.csv.gz'
    SPLITS_PATH = 'mimic-cxr-2.0.0-split.csv.gz'

    def __init__(self, root, section='findings', split=None, target_transform=None, cache_image=False, cache_text=True,
                 multi_image=1, img_mode='center', img_augment=False, single_image_doc=False, dump_dir=None,
                 filter_reports=True):
        if not cache_text:
            raise ValueError('MIMIC-CXR data only supports cached texts')
        super().__init__(root, section, split, cache_image, cache_text, multi_image=multi_image,
                         single_image_doc=single_image_doc, dump_dir=dump_dir)
        pre_transform, self.transform = MIMICCXRData.get_transform(cache_image, img_mode, img_augment)
        self.target_transform = target_transform
        self.chexpert_labels_path = os.path.join(root, 'mimic-cxr-jpg', '2.0.0', self.CHEXPERT_PATH)

        self.view_positions = {}
        doc_image_map = {}
        with gzip.open(os.path.join(root, 'mimic-cxr-resized', '2.0.0', self.META_PATH), 'rt', encoding='utf-8') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                self.view_positions[row[0]] = row[4]
                if row[2] in doc_image_map:
                    doc_image_map[row[2]].append(row[0])
                else:
                    doc_image_map[row[2]] = [row[0]]

        if dump_dir is not None:
            t = time.time()
            if self.load():
                print('Loaded data dump from %s (%.2fs)' % (dump_dir, time.time() - t))
                self.pre_processes(filter_reports)
                return

        with gzip.open(os.path.join(root, 'mimic-cxr-resized', '2.0.0', self.SECTIONED_PATH), 'rt',
                       encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            sections = {}
            reader = csv.reader(f)
            for row in reader:
                report = {}
                for i, sec in enumerate(header):
                    report[sec] = row[i]
                sections[row[0]] = gzip.compress(pickle.dumps(report))

        with gzip.open(os.path.join(root, 'mimic-cxr-resized', '2.0.0', self.SPLITS_PATH), 'rt',
                       encoding='utf-8') as f:
            f.readline()
            reader = csv.reader(f)
            interval = 1000
            with tqdm(total=self.IMAGE_NUM) as pbar:
                pbar.set_description('Data ({0})'.format(split))
                count = 0
                for row in reader:
                    if split is None or split == row[3]:
                        did = row[0]
                        sid = row[1]
                        pid = row[2]
                        self.ids.append(did)
                        self.doc_ids.append(sid)
                        # image
                        image = os.path.join(root, 'mimic-cxr-resized', '2.0.0', 'files', 'p{0}'.format(pid[:2]),
                                             'p' + pid, 's' + sid, did + '.png')
                        if cache_image:
                            image = self.bytes_image(image, pre_transform)
                        # report
                        report = os.path.join(root, 'mimic-cxr', '2.0.0', 'files', 'p{0}'.format(pid[:2]), 'p' + pid,
                                              's{0}.txt'.format(sid))
                        if cache_text:
                            sid = 's' + sid
                            report = sections[sid] if sid in sections else gzip.compress(pickle.dumps({}))
                        self.samples.append((image, report))
                        self.targets.append(report)
                    count += 1
                    if count >= interval:
                        pbar.update(count)
                        count = 0
                if count > 0:
                    pbar.update(count)

        if dump_dir is not None:
            self.dump()
        self.pre_processes(filter_reports)

    def __getitem__(self, index):
        rid, sample, target, _ = super().__getitem__(index)
        # View position features
        if self.multi_image > 1:
            vp = [self.view_position_embedding(self.view_positions[iid]) for iid in self.image_ids[index]]
            vp = [p.unsqueeze(dim=0) for p in vp]
            if len(vp) > self.multi_image:
                vp = vp[:self.multi_image]
            elif len(vp) < self.multi_image:
                first_vp = vp[0]
                for _ in range(self.multi_image - len(vp)):
                    vp.append(first_vp.new_zeros(first_vp.size()))
            vp = torch.cat(vp, dim=0)
        else:
            vp = self.view_position_embedding(self.view_positions[rid])
        return rid, sample, target, vp

    @classmethod
    def get_transform(cls, cache_image=False, mode='center', augment=False):
        return cls._transform(cache_image, 224, mode, augment)

    def compare_texts(self, text1, text2):
        if 'study' in text1 and 'study' in text2:
            return text1['study'] == text2['study']
        else:
            return True

    def decompress_text(self, text):
        return pickle.loads(gzip.decompress(text))

    def extract_section(self, text):
        if self.section in text:
            return text[self.section].replace('\n', ' ')
        else:
            return ''

    def pre_processes(self, filter_reports):
        if filter_reports:
            self.filter_empty_reports()
        if self.multi_image > 1:
            self.convert_to_multi_images()
        elif self.single_image_doc:
            self.convert_to_single_image()
        self.pre_transform_texts(self.split)

