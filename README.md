# Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation

The reference code of [Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation](https://arxiv.org/abs/2010.10042).

## Implemented Models
* CNN-RNN-RNN ([Liu et al., 2019](https://arxiv.org/abs/1904.02633))
* Knowing When to Look ([Lu et al., 2017](https://arxiv.org/abs/1612.01887))
* Meshed-Memory Transformer ([Cornia et al., 2020](https://arxiv.org/abs/1912.08226))
* Show, Attend and Tell ([Xu et al., 2015](https://arxiv.org/abs/1502.03044))
* TieNet ([Wang et al., 2018](https://arxiv.org/abs/1801.04334))

## Supported Radiology Report Datasets
* MIMIC-CXR-JPG ([Johnson et al., 2019](https://doi.org/10.13026/8360-t248))
* Open-i ([Demner-Fushman et al., 2012](https://openi.nlm.nih.gov/))

## Radiology NLI Dataset

NOTE
: We are working to make the radiology NLI dataset publicly available.

## Prerequisites
* A Linux OS (tested on Ubuntu 16.04)
* Memory over 24GB
* A gpu with memory over 12GB (tested on NVIDIA Titan X and NVIDIA Titan XP) 

## Preprocesses

### Python Setup
Create a conda environment
```bash
$ conda env create -f environment.yml
```

NOTE
: `environment.yml` is set up for CUDA 10.1 and cuDNN 7.6.3. This may need to be changed depending on a runtime environment. 

### Resize MIMIC-CXR-JPG
1. Download [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
2. Make a resized copy of MIMIC-CXR-JPG using resize_mimic-cxr-jpg.py (MIMIC_CXR_ROOT is a dataset directory containing mimic-cxr)
    * `$ python resize_mimic-cxr-jpg.py MIMIC_CXR_ROOT`
3. Create the sections file of MIMIC-CXR (mimic_cxr_sectioned.csv.gz) with [create_sections_file.py](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt)
4. Move mimic_cxr_sectioned.csv.gz to MIMIC_CXR_ROOT/mimic-cxr-resized/2.0.0/

### Compute Document Frequencies
Pre-calculate document frequencies that will be used in CIDEr by:
```bash
$ python cider-df.py MIMIC_CXR_ROOT mimic-cxr_train-df.bin.gz
```

### Recognize Named Entities
Pre-recognize named entities in MIMIC-CXR by:
```bash
$ python ner_reports.py --stanza-download MIMIC_CXR_ROOT mimic-cxr_ner.txt.gz
```

### Pre-train CheXpert Image Weights
1. Download [CheXpert Dataset v1.0](https://stanfordmlgroup.github.io/competitions/chexpert/)
2. Train a CheXpert classification model by:
```bash
$ python train_image.py --cuda --epochs 12 --batch-size 16 --eval-interval 65000 --cache-data cache CheXpert-v1.0-small densenet chexpert_densenet
```

### Download Pre-trained Weights
Download pre-trained radiology NLI weights and GloVe embeddings
```bash
$ cd resources
$ ./download.sh
```

## Training a Report Generation Model
First, train the Meshed-Memory Transformer model with an NLL loss.
```bash
# NLL
$ python train.py --cuda --corpus mimic-cxr --cache-data cache --epochs 32 --batch-size 24 --cider-df mimic-cxr_train-df.bin.gz --entity-match mimic-cxr_ner.txt.gz --img-model densenet --img-pretrained chexpert_densenet/model_auc14.dict.gz --cider-df mimic-cxr_train-df.bin.gz --bert-score distilbert-base-uncased --corpus mimic-cxr --lr-scheduler trans MIMIC_CXR_ROOT resources/glove_mimic-cxr_train.512.txt.gz out_m2trans_nll
```

Second, further train the model a joint loss using the self-critical RL to achieve a better performance.
```bash
# RL with NLL + BERTScore + EntityMatchExact
$ python train.py --cuda --corpus mimic-cxr --cache-data cache --epochs 32 --batch-size 24 --rl-epoch 1 --rl-metrics BERTScore,EntityMatchExact --rl-weights 0.01,0.495,0.495 --entity-match resources/mimic-cxr_ner.txt.gz --baseline-model out_m2trans_nll/model_31-152173.dict.gz --img-model densenet --img-pretrained chexpert_densenet/chexpert_auc14.dict.gz --cider-df mimic-cxr_train-df.bin.gz --bert-score distilbert-base-uncased --lr 5e-6 MIMIC_CXR_ROOT resources/glove_mimic-cxr_train.512.txt.gz out_m2trans_nll-bs-emexact
```
```bash
# RL with NLL + BERTScore + EntityMatchNLI
$ python train.py --cuda --corpus mimic-cxr --cache-data cache --epochs 32 --batch-size 24 --rl-epoch 1 --rl-metrics BERTScore,EntityMatchNLI --rl-weights 0.01,0.495,0.495 --entity-match resources/mimic-cxr_ner.txt.gz --baseline-model out_m2trans_nll/model_31-152173.dict.gz --img-model densenet --img-pretrained chexpert_densenet/chexpert_auc14.dict.gz --cider-df mimic-cxr_train-df.bin.gz --bert-score distilbert-base-uncased --lr 5e-6 MIMIC_CXR_ROOT resources/glove_mimic-cxr_train.512.txt.gz out_m2trans_nll-bs-emnli
```

### Checking Result with TensorBoard
A training result can be checked with TensorBoard.
```bash
$ tensorboard --logdir out_m2trans_nll-bs-emnli/log
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.0.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

## Licence
See LICENSE and clinicgen/external/LICENSE_bleu-cider-rouge-spice for details.