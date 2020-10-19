#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import io
import os
import pickle
import re
import numpy as np
import time
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop, ColorJitter, Compose, Normalize, RandomApply, RandomRotation, Resize, ToTensor
from torchvision.transforms.functional import pad
from clinicgen.text.sentsplit import get_sentsplitter
from clinicgen.text.textfilter import get_textfilter
from clinicgen.text.tokenfilter import get_tokenfilter
from clinicgen.text.tokenizer import get_tokenizer


class _CaptioningData(data.Dataset):
    LABEL_REPORT = 'report'
    TRANSFORM_CENTER = 'center'
    TRANSFORM_PAD = 'pad'

    def __init__(self, root, split, cache_image, cache_text, multi_image=1, single_image_doc=False, dump_dir=None):
        self.root = root
        self.split = split
        self.cache_image = cache_image
        self.cache_text = cache_text
        self.multi_image = multi_image
        self.single_image_doc = single_image_doc
        self.dump_dir = dump_dir

        self.ids = []
        self.doc_ids = []
        self.image_ids = None
        self.samples = []
        self.targets = []
        self.loader = default_loader
        self.transform = None
        self.target_transform = None
        self.multi_instance = False
        self.chexpert_labels_path = None

    @classmethod
    def _transform(cls, cache_image, resize_dim=256, mode='center', augment=False):
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if augment:
            rotate = RandomApply([RandomRotation(10.0, expand=True)], p=0.5)
            color = RandomApply([ColorJitter(brightness=0.1, contrast=0.1)], p=0.5)
            augs = [rotate, color]
            if mode == cls.TRANSFORM_CENTER:
                augs += [Resize(resize_dim), CenterCrop(224)]
            elif mode == cls.TRANSFORM_PAD:
                augs += [PadSquare(), Resize(224)]
            else:
                raise ValueError('Unknown transform mode {0}'.format(mode))
        else:
            augs = []
        if cache_image:
            if mode == cls.TRANSFORM_CENTER:
                return Compose([Resize(resize_dim), CenterCrop(224)]), Compose(augs + [ToTensor(), norm])
            elif mode == cls.TRANSFORM_PAD:
                return Compose([PadSquare(), Resize(224)]), Compose(augs + [ToTensor(), norm])
            else:
                raise ValueError('Unknown transform mode {0}'.format(mode))
        else:
            if mode == cls.TRANSFORM_CENTER:
                return None, Compose([Resize(resize_dim), CenterCrop(224)] + augs + [ToTensor(), norm])
            elif mode == cls.TRANSFORM_PAD:
                return None, Compose([PadSquare(), Resize(224)] + augs + [ToTensor(), norm])
            else:
                raise ValueError('Unknown transform mode {0}'.format(mode))

    @classmethod
    def extract_text(cls, path, compress=True):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        text = '\n'.join(lines)
        if compress:
            text = gzip.compress(text.encode('utf-8'))
        return text

    @classmethod
    def get_target_transform(cls, word_indexes, split='train', sentsplitter='linebreak', tokenizer='nltk',
                             textfilter='lower', tokenfilter='none', max_sent=6, max_word=50):
        if word_indexes is None:
            return None
        sentsplitter = get_sentsplitter(sentsplitter)
        tokenizer = get_tokenizer(tokenizer)
        textfilter = get_textfilter(textfilter)
        tokenfilter = get_tokenfilter(tokenfilter)
        if split == 'train':
            return ToWordIndexes(sentsplitter, tokenizer, textfilter, tokenfilter, word_indexes, max_sent, max_word)
        else:
            return ToTokenizedTexts(sentsplitter, tokenizer, textfilter, tokenfilter)

    @classmethod
    def get_transform(cls, cache_image=False, mode='center', augment=False):
        return cls._transform(cache_image, 256, mode, augment)

    def __getitem__(self, index):
        sample_path, target = self.samples[index]
        iid = self.ids[index]

        sample = []
        if self.multi_image == 1:
            sample_path = [sample_path]
        for spath in sample_path:
            sample.append(self._load_image(iid, spath))
        if self.transform is not None:
            sample = [self.transform(s) for s in sample]
            if self.multi_image > 1:
                sample = [s.unsqueeze(dim=0) for s in sample]
                if len(sample) > self.multi_image:
                    sample = sample[:self.multi_image]
                elif len(sample) < self.multi_image:
                    first_sample = sample[0]
                    for _ in range(self.multi_image - len(sample)):
                        sample.append(first_sample.new_zeros(first_sample.size()))
                sample = torch.cat(sample, dim=0)
            else:
                sample = sample[0]
        if not self.cache_text:
            target = self.extract_text(target, compress=False)
            target = self.extract_section(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return iid, sample, target, 0

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _load_image(self, instance_id, path):
        if self.cache_image:
            with io.BytesIO(path) as f:
                return Image.open(f).convert('RGB')
        else:
            return self.loader(path)

    def bytes_image(self, image, transform=None):
        image = self.loader(image)
        if transform is not None:
            image = transform(image)
        with io.BytesIO() as f:
            image.save(f, 'png')
            image = f.getvalue()
        return image

    def compare_texts(self, text1, text2):
        return text1 == text2

    def decompress_text(self, text):
        if not isinstance(text, list):
            text = [text]
        decomp_texts = []
        for rep in text:
            decomp_texts.append(gzip.decompress(rep).decode('utf-8'))
        if len(decomp_texts) == 1:
            return decomp_texts[0]
        else:
            return decomp_texts

    def dump(self):
        if not os.path.exists(self.dump_dir):
            os.mkdir(self.dump_dir)
        for attr in ['ids', 'doc_ids', 'samples', 'targets']:
            with open(os.path.join(self.dump_dir, '{0}.pkl'.format(attr)), 'wb') as out:
                obj = getattr(self, attr)
                pickle.dump(obj, out)

    def extract_section(self, text):
        return text

    def load(self):
        if not os.path.exists(self.dump_dir):
            return False
        for attr in ['ids', 'doc_ids', 'samples', 'targets']:
            with open(os.path.join(self.dump_dir, '{0}.pkl'.format(attr)), 'rb') as out:
                obj = pickle.load(out)
                setattr(self, attr, obj)
        return True

    def pre_transform_texts(self, split, print_time=True):
        if self.cache_text and self.target_transform is not None:
            if print_time:
                t = time.time()
                print('Pre-processing reports ... ', end='', flush=True)
            if split == 'train':
                target_transform = self.target_transform.to_tokenized_text
            else:
                target_transform = self.target_transform
            for i in range(len(self.samples)):
                image, text = self.samples[i]
                text = self.decompress_text(text)
                if not self.compare_texts(text, self.decompress_text(self.targets[i])):
                    raise ValueError('sample-target mismatch in {0}'.format(i))
                text = self.extract_section(text)
                text = target_transform(text)
                self.samples[i] = (image, text)
                self.targets[i] = text
            self.target_transform.pre_processed = True
            if print_time:
                print('done (%.2fs)' % (time.time() - t), flush=True)


class _RadiologyReportData(_CaptioningData):
    REPORT_PATTERN = re.compile('^(FINDINGS|IMPRESSION):(.*)$', flags=re.IGNORECASE)
    SECTION_SEPARATOR = '-'
    CHEXPERT_FEAT_DIM = 14

    def __init__(self, root, section, split, cache_image, cache_text, multi_image=1, single_image_doc=False,
                 dump_dir=None):
        super(_RadiologyReportData, self).__init__(root, split, cache_image, cache_text, multi_image, single_image_doc,
                                                   dump_dir)
        if section is not None:
            section = section.split(self.SECTION_SEPARATOR)
            self.sub_section = section[1] if len(section) > 1 else None
            self.section = section[0]
        else:
            self.section = None

    @classmethod
    def view_position_embedding(cls, view_position):
        if view_position == 'AP':
            return torch.tensor(np.array([1, 0, 0], dtype='float'))
        elif view_position == 'PA':
            return torch.tensor(np.array([0, 1, 0], dtype='float'))
        elif view_position == 'LATERAL' or view_position == 'LL':
            return torch.tensor(np.array([0, 0, 1], dtype='float'))
        elif view_position == '':
            return torch.tensor(np.array([0, 0, 0], dtype='float'))
        else:
            return torch.tensor(np.array([0, 0, 0], dtype='float'))

    def __getitem__(self, index):
        rid, sample, target, _ = super().__getitem__(index)
        return rid, sample, target, 0

    def convert_to_multi_images(self, print_num=True):
        t = time.time()
        n = 0
        if print_num:
            print('Converting to multiple image reports ... ', end='', flush=True)
        new_ids, image_ids, new_samples, new_targets = [], [], [], []
        total = len(self.samples)
        buffer = None
        for i in range(total):
            iid = self.ids[i]
            doc_id = self.doc_ids[i]
            image, report = self.samples[i]

            if doc_id == buffer:
                image_ids[-1].append(iid)
                new_samples[-1][0].append(image)
            else:
                new_ids.append(doc_id)
                image_ids.append([iid])
                new_samples.append(([image], report))
                new_targets.append(report)
                n += 1
            buffer = doc_id
        self.ids = new_ids
        self.doc_ids = new_ids
        self.image_ids = image_ids
        self.samples = new_samples
        self.targets = new_targets
        if print_num:
            print('done %d->%d (%.2fs)' % (total, n, time.time() - t), flush=True)

    def convert_to_single_image(self, print_num=True):
        t = time.time()
        n = 0
        if print_num:
            print('Converting to single image reports ... ', end='', flush=True)
        new_ids, new_doc_ids, new_samples, new_targets = [], [], [], []
        total = len(self.samples)
        buffer = None
        for i in range(total):
            iid = self.ids[i]
            doc_id = self.doc_ids[i]
            image, report = self.samples[i]

            if doc_id != buffer:
                new_ids.append(iid)
                new_doc_ids.append(doc_id)
                new_samples.append((image, report))
                new_targets.append(report)
                n += 1
            buffer = doc_id
        self.ids = new_ids
        self.doc_ids = new_doc_ids
        self.samples = new_samples
        self.targets = new_targets
        if print_num:
            print('done %d->%d (%.2fs)' % (total, n, time.time() - t), flush=True)

    def filter_empty_reports(self, print_num=True):
        t = time.time()
        n = 0
        if print_num:
            print('Filtering empty reports ... ', end='', flush=True)
        new_ids, new_doc_ids, new_samples, new_targets = [], [], [], []
        for i in range(len(self.samples)):
            iid = self.ids[i]
            doc_id = self.doc_ids[i]
            image, report = self.samples[i]
            if self.cache_text:
                rep = self.decompress_text(report)
                if not self.compare_texts(rep, self.decompress_text(self.targets[i])):
                    raise ValueError('sample-target mismatch in {0}'.format(i))
            else:
                if report != self.targets[i]:
                    raise ValueError('sample-target mismatch in {0}'.format(i))
                rep = self.extract_text(report, compress=False)
            text = self.extract_section(rep)
            if len(text) > 0:
                new_ids.append(iid)
                new_doc_ids.append(doc_id)
                new_samples.append((image, report))
                new_targets.append(report)
            else:
                n += 1
        self.ids = new_ids
        self.doc_ids = new_doc_ids
        self.samples = new_samples
        self.targets = new_targets
        if print_num:
            print('done %d (%.2fs)' % (n, time.time() - t), flush=True)


class PadSquare:
    def __call__(self, img):
        w, h = img.size
        diff = abs(w - h)
        p1 = diff // 2
        p2 = p1
        if diff % 2 == 1:
            p2 += 1
        if w > h:
            return pad(img, (0, p1, 0, p2))
        else:
            return pad(img, (p1, 0, p2, 0))

    def __repr__(self):
        return self.__class__.__name__


class PretrainedEmbeddings:
    GLOVE_UNK = '<unk>'
    INDEX_PAD = 0
    INDEX_START = 1
    INDEX_UNKNOWN = 2
    INDEX_EOS = 3
    TOKEN_EOS = '__EOS__'

    @classmethod
    def load_embeddings(cls, path):
        """
        Load pre-trained word embedding.
        :param path: A path to pre-trained word embeddings
        :return: A tensor of pre-trained word embeddings
        """
        f, embeddings = None, None
        word_idxs = {'__PAD__': cls.INDEX_PAD, '__START__': cls.INDEX_START, '__UNK__': cls.INDEX_UNKNOWN,
                     cls.TOKEN_EOS: cls.INDEX_EOS}
        try:
            if path.endswith('.gz'):
                f = gzip.open(path, 'rt', encoding='utf-8')
            else:
                f = open(path, encoding='utf-8')

            header = f.readline().split(' ')
            num = int(header[0])
            dim = int(header[1])
            embeddings = np.zeros((num + len(word_idxs), dim), dtype='float32')
            for idx in (cls.INDEX_START, cls.INDEX_UNKNOWN, cls.INDEX_EOS):
                embeddings[idx] = np.random.uniform(low=-1, high=1, size=dim)

            for line in f:
                entry = line.strip().split(' ')
                embed = np.array(list(map(lambda v: float(v), entry[1:])), dtype='float32')
                if entry[0] == cls.GLOVE_UNK:
                    embeddings[cls.INDEX_UNKNOWN] = embed
                else:
                    embeddings[len(word_idxs)] = embed
                    word_idxs[entry[0]] = len(word_idxs)
            embeddings = torch.tensor(embeddings)
        finally:
            if f is not None:
                f.close()
        return embeddings, word_idxs


class ToTokenizedTexts:
    INSTANCE_BREAK = '__IBR__'

    def __init__(self, sentsplitter, tokenizer, textfilter, tokenfilter):
        self.sentsplitter = sentsplitter
        self.tokenizer = tokenizer
        self.textfilter = textfilter
        self.tokenfilter = tokenfilter
        self.pre_processed = False

    def __call__(self, texts):
        """
        Convert a text to tokenized text.
        :param texts: Texts to tokenize.
        :return: A tokenized text.
        """
        if self.pre_processed:
            return texts

        if not isinstance(texts, list):
            texts = [texts]
        tokenized_texts = []

        for text in texts:
            sentences = []
            for sentence in filter(lambda sent: len(sent) > 0, self.sentsplitter.split(text)):
                if len(sentence) > 0:
                    sentence = self.textfilter.filter(sentence)
                    words = self.tokenizer.tokenize(sentence)
                    words = self.tokenfilter.filter(words)
                    sentences.append(' '.join(words))
            tokenized_texts.append('\n'.join(sentences))

        if len(tokenized_texts) == 1:
            return tokenized_texts[0]
        else:
            return self.INSTANCE_BREAK.join(tokenized_texts)


class ToWordIndexes:
    def __init__(self, sentsplitter, tokenizer, textfilter, tokenfilter, word_idxs, max_sent=6, max_word=50):
        self.sentsplitter = sentsplitter
        self.tokenizer = tokenizer
        self.textfilter = textfilter
        self.tokenfilter = tokenfilter
        self.to_tokenized_text = ToTokenizedTexts(sentsplitter, tokenizer, textfilter, tokenfilter)
        self.word_idxs = word_idxs
        self.max_sent = max_sent
        self.max_word = max_word
        self.pre_processed = False

    def __call__(self, text):
        """
        Convert a text to word indexes.
        :param text: A text to convert.
        :return: Word indexes of (sentence_num, word_num).
        """
        sent_n = 1 if self.max_sent is None else self.max_sent
        targ = np.zeros((sent_n, self.max_word), dtype='int64')

        if not self.pre_processed:
            sentences = list(filter(lambda sent: len(sent) > 0, self.sentsplitter.split(text)))
        else:
            sentences = text.split('\n')
        if self.max_sent is None:
            sentences = [' '.join(sentences)]
        else:
            if len(sentences) > self.max_sent:
                sentences = sentences[:self.max_sent]

        for i, sentence in enumerate(sentences):
            if not self.pre_processed:
                sentence = self.textfilter.filter(sentence)
                words = self.tokenizer.tokenize(sentence)
                words = self.tokenfilter.filter(words)
            else:
                words = sentence.split(' ')
            words.append(PretrainedEmbeddings.TOKEN_EOS)
            if len(words) > self.max_word:
                words = words[:self.max_word]
            idxs = np.zeros(self.max_word, dtype='int64')
            for j, word in enumerate(words):
                if word in self.word_idxs:
                    idxs[j] = self.word_idxs[word]
                else:
                    idxs[j] = PretrainedEmbeddings.INDEX_UNKNOWN
            targ[i] = idxs
        return targ

    def __repr__(self):
        return self.__class__.__name__ + '(size=({0},{1}))'.format(self.max_sent, self.max_word)
