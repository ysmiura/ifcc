#!/bin/sh
wget https://nlp.stanford.edu/ysmiura/ifcc/glove_mimic-cxr_train.512.txt.gz
wget https://nlp.stanford.edu/ysmiura/ifcc/model_medrad_19k.tar.gz
tar xvzf model_medrad_19k.tar.gz
rm model_medrad_19k.tar.gz
wget https://nlp.stanford.edu/ysmiura/ifcc/chexpert_auc14.dict.gz
