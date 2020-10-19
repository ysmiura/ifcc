#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import json
import os
import time
from tqdm import tqdm
from clinicgen.data.image2text import _CaptioningData


class Flickr30kData(_CaptioningData):
    IMAGE_NUM = 158915
    DIR_IMAGES = 'flickr30k-images'
    FILE_CAPTIONS = os.path.join('flickr30k', 'results_20130124.token')

    def __init__(self, root, meta=None, split=None, target_transform=None, cache_image=False, img_mode='center',
                 img_augment=False, cache_text=False, dump_dir=None):
        if not cache_text:
            raise ValueError('Flickr30k data only supports cached texts')
        super().__init__(root, split=split, cache_image=cache_image, cache_text=cache_text, dump_dir=dump_dir)
        pre_transform, self.transform = Flickr30kData.get_transform(cache_image, img_mode, img_augment)
        self.target_transform = target_transform
        self.multi_instance = True

        if dump_dir is not None:
            t = time.time()
            if self.load():
                print('Loaded data dump from %s (%.2fs)' % (dump_dir, time.time() - t))
                self.pre_processes()
                return

        splits = None
        if meta is not None:
            splits = {}
            with open(meta, encoding='utf-8') as f:
                meta_data = json.load(f)
                for entry in meta_data['images']:
                    splits[entry['filename']] = entry['split']

        captions = os.path.join(root, self.FILE_CAPTIONS)
        with open(captions, encoding='utf-8') as f:
            with tqdm(total=self.IMAGE_NUM) as pbar:
                pbar.set_description('Data ({0})'.format(split))
                count = 0
                interval = 1000
                prev_image, buffer = None, []

                for line in f:
                    entry = line.rstrip().split('\t')
                    image = entry[0].split('#')[0]

                    if split is None or (image in splits and splits[image] == split):
                        report = gzip.compress(entry[1].encode('utf-8'))
                        if prev_image is not None and image != prev_image:
                            count = self._append_image(prev_image, buffer, count, pre_transform)
                            buffer = []
                        prev_image = image
                        buffer.append((entry[0], report))
                    else:
                        count += 1
                    if count >= interval:
                        pbar.update(count)
                        count = 0

                if len(buffer) > 0:
                    count = self._append_image(prev_image, buffer, count, pre_transform)
                if count > 0:
                    pbar.update(count)

        if dump_dir is not None:
            self.dump()
        self.pre_processes()

    def _append_image(self, image_id, buffer, count, pre_transform):
        if len(buffer) > 0:
            image = os.path.join(self.root, self.DIR_IMAGES, image_id)
            if self.cache_image:
                image = self.bytes_image(image, pre_transform)
            if self.split == 'test' or self.split == 'val':
                buffer = [e[1] for e in buffer]
                self.ids.append(image_id)
                self.samples.append((image, buffer))
                self.targets.append(buffer)
                count += len(buffer)
            else:
                for entry_id, report in buffer:
                    self.ids.append(entry_id)
                    self.samples.append((image, report))
                    self.targets.append(report)
                    count += 1
        return count

    def pre_processes(self):
        self.pre_transform_texts(self.split)
