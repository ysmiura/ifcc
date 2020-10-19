#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import sys
import time
import xml.etree.ElementTree as etree
from tqdm import tqdm
from clinicgen.data.image2text import _RadiologyReportData


class OpenIData(_RadiologyReportData):
    CHEXPERT_PATH = 'open-i-chexpert.csv.gz'
    IMAGES_DIR = 'NLMCXR_png'
    REPORTS_DIR = 'ecgen-radiology'

    def __init__(self, root, section='findings', meta=None, split=None, target_transform=None, cache_image=False,
                 cache_text=False, multi_image=1, img_mode='center', img_augment=False, single_image_doc=False,
                 dump_dir=None):
        super().__init__(root, section, split, cache_image, cache_text, multi_image=multi_image,
                         single_image_doc=single_image_doc, dump_dir=dump_dir)
        pre_transform, self.transform = OpenIData.get_transform(cache_image, img_mode, img_augment)
        self.target_transform = target_transform
        if meta is not None:
            self.chexpert_labels_path = os.path.join(os.path.dirname(meta), self.CHEXPERT_PATH)
        if dump_dir is not None:
            t = time.time()
            if self.load():
                print('Loaded data dump from %s (%.2fs)' % (dump_dir, time.time() - t))
                self.pre_processes()
                return

        splits = {}
        if meta is not None:
            with open(meta, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    splits[row[0]] = row[1]

        total = 3999
        interval = 100
        with tqdm(total=total) as pbar:
            pbar.set_description('Data ({0})'.format(split))
            count = 0
            for i in range(1, total + 1):
                path = os.path.join(root, OpenIData.REPORTS_DIR, '{0}.xml'.format(i))
                if os.path.exists(path):
                    tree = etree.parse(path)
                    rt = tree.getroot()
                    uid = None
                    for ele in rt.findall(".//uId"):
                        uid = ele.attrib['id']
                    if split is None or (uid in splits and splits[uid] == split):
                        image_ids = []
                        for ele in rt.findall(".//parentImage"):
                            image_ids.append(ele.attrib['id'])
                        findings = ''
                        for ele in rt.findall(".//AbstractText[@Label='{0}']".format('FINDINGS')):
                            findings = ele.text
                            findings = findings.strip() if findings is not None else ''
                        if len(findings) > 0 and len(image_ids) > 0:
                            for image_id in image_ids:
                                image = os.path.join(root, OpenIData.IMAGES_DIR, '{0}.png'.format(image_id))
                                report = self.extract_text(path, compress=True) if cache_text else path
                                if cache_image:
                                    image = self.bytes_image(image, pre_transform)
                                self.ids.append(image_id)
                                self.doc_ids.append(uid)
                                self.samples.append((image, report))
                                self.targets.append(report)
                count += 1
                if count >= interval:
                    pbar.update(count)
                    count = 0
            if count > 0:
                pbar.update(count)
        if len(self.samples) == 0 and len(self.targets) == 0:
            sys.stderr.write("WARNING: Found 0 files in subfolders of: " + root + "\n")

        if dump_dir is not None:
            self.dump()
        self.pre_processes()

    def extract_section(self, text):
        rt = etree.fromstring(text)
        report = ''
        if self.section is None:
            for ele in rt.findall(".//AbstractText"):
                s = ele.text
                if s is not None:
                    report += s.strip() + '/n'
        else:
            for ele in rt.findall(".//AbstractText[@Label='{0}']".format(self.section.upper())):
                report = ele.text
                if report is None:
                    report = ''
                else:
                    report = report.strip()
        return report

    def pre_processes(self):
        if self.multi_image > 1:
            self.convert_to_multi_images()
        elif self.single_image_doc:
            self.convert_to_single_image()
        self.pre_transform_texts(self.split)
