#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR


class Optimizers:
    IMAGE = 'image'
    TEXT = 'text'
    TRANSFORMER = 'trans'

    @classmethod
    def get_optmizers(cls, model, lr, lr_img=None, lr_step=None, lr_scheduler='linear', lr_decay_rate=0.5, beta1=0.9,
                      beta2=0.999, train_steps=None, d_train=None, steps_per_epoch=None, warmup=None):
        if lr_scheduler == cls.TRANSFORMER:
            optimizer = Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
            optimizers = {cls.TEXT: optimizer}
            schedulers = {}
            batch_schedulers = {cls.TEXT: TransformerScheduler(optimizer, d_train, steps_per_epoch, warmup)}
        else:
            if lr_img is None or lr == lr_img:
                optimizers = {cls.TEXT: Adam(model.parameters(), lr=lr, betas=(beta1, beta2))}
                schedulers = {cls.TEXT: StepLR(optimizers[cls.TEXT], lr_step, lr_decay_rate)}
            else:
                if lr_img is None:
                    lr_img = lr
                text_params, img_params = [], []
                for name, param in model.named_parameters():
                    if name.startswith('image_feats'):
                        img_params.append(param)
                    else:
                        text_params.append(param)
                optimizers = {cls.TEXT: Adam(text_params, lr=lr, betas=(beta1, beta2)),
                              cls.IMAGE: Adam(img_params, lr=lr_img, betas=(beta1, beta2))}
                schedulers = {cls.TEXT: StepLR(optimizers[cls.TEXT], lr_step, lr_decay_rate),
                              cls.IMAGE: StepLR(optimizers[cls.IMAGE], lr_step, lr_decay_rate)}
            batch_schedulers = None
        return optimizers, schedulers, batch_schedulers


class TransformerScheduler(_LRScheduler):
    def __init__(self, optimizer, d_train, steps_per_epoch, warmup=4000, last_epoch=-1):
        self.epoch_step = 0
        self.d_train = d_train
        self.steps_per_epoch = steps_per_epoch
        self.warmup = warmup
        super(TransformerScheduler, self).__init__(optimizer, last_epoch)

    def batch_step(self):
        self.epoch_step += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        step = min(0, self.last_epoch * self.steps_per_epoch)
        step += self.epoch_step
        step = max(1, step)
        return [self.d_train ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)) for _ in self.base_lrs]



