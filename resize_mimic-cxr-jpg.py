#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Resize


def main(args):
    in_files = os.path.join(args.jpgs, '2.0.0', 'files')
    resized_dir = os.path.join(os.path.dirname(args.jpgs), '..', 'mimic-cxr-resized')
    out_files = os.path.join(resized_dir, '2.0.0', 'files')
    resize = Resize(256)
    # Make mimic-cxr-resized directory
    if not os.path.exists(resized_dir):
        os.mkdir(resized_dir)
        os.mkdir(os.path.join(resized_dir, '2.0.0'))
        os.mkdir(out_files)
        for item in os.listdir(os.path.join(args.jpgs, '2.0.0')):
            if item.endswith('.csv.gz'):
                shutil.copy(os.path.join(args.jpgs, '2.0.0', item), os.path.join(resized_dir, '2.0.0', item))
    # Copy various MIMIC-CXR-JPG data
    shutil.copy(os.path.join(args.jpgs, '2.0.0', 'mimic-cxr-2.0.0-chexpert.csv.gz'), os.path.join(resized_dir, '2.0.0'))
    shutil.copy(os.path.join(args.jpgs, '2.0.0', 'mimic-cxr-2.0.0-metadata.csv.gz'), os.path.join(resized_dir, '2.0.0'))
    shutil.copy(os.path.join(args.jpgs, '2.0.0', 'mimic-cxr-2.0.0-split.csv.gz'), os.path.join(resized_dir, '2.0.0'))
    # Resize images
    count_resized, count_skipped = 0, 0
    for item1 in os.listdir(in_files):
        if item1.startswith('p') and os.path.isdir(os.path.join(in_files, item1)):
            print('Processing {0} ...'.format(item1))
            if not os.path.exists(os.path.join(out_files, item1)):
                os.mkdir(os.path.join(out_files, item1))
            for item2 in os.listdir(os.path.join(in_files, item1)):
                if item2.startswith('p'):
                    if not os.path.exists(os.path.join(out_files, item1, item2)):
                        os.mkdir(os.path.join(out_files, item1, item2))
                    for item3 in os.listdir(os.path.join(in_files, item1, item2)):
                        if item3.startswith('s'):
                            if not os.path.exists(os.path.join(out_files, item1, item2, item3)):
                                os.mkdir(os.path.join(out_files, item1, item2, item3))
                            for item4 in os.listdir(os.path.join(in_files, item1, item2, item3)):
                                if item4.endswith('.jpg'):
                                    out_image_path = os.path.join(out_files, item1, item2, item3,
                                                                  item4.replace('.jpg', '.png'))
                                    if not os.path.exists(out_image_path):
                                        image_path = os.path.join(in_files, item1, item2, item3, item4)
                                        image = default_loader(image_path)
                                        image = resize(image)
                                        with open(out_image_path, 'wb') as out:
                                            image.save(out, 'png')
                                        count_resized += 1
                                        if count_resized % 1000 == 0:
                                            print('Resized {0} images'.format(count_resized))
                                    else:
                                        count_skipped += 1
    print('Total {0} images'.format(count_resized + count_skipped))


def parse_args():
    parser = argparse.ArgumentParser(description='Resize MIMIC-CXR-JPG jpgs to 256 pixels pngs')
    parser.add_argument('jpgs', type=str, help='A path to MIMIC-CXR-JPG')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
