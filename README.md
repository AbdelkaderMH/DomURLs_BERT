# DomURLs_BERT

This repo provides the implementation of **__{DomURLs_BERT}: Pre-trained BERT-based Model for Malicious Domains and URLs Detection and Classification__** experiments.


## Fine-Tuning DomURLs_BERT and other Pretrained Language Models for Malicious URL and Domain Name Detection

This guide provides instructions on how to fine-tune pretrained language models (PLMs) using the `main_plm.py` script for detecting malicious URLs and domain names.
### Requirements
- Python 3.8
- torch 2.2
- transformers 4.39.3
- lightning 2.1.3
- mlflow 2.14.2

the full list is in `requirements.txt`

### Usage
To fine-tune a model, run the script from the command line with the required parameters. Below is a description of all the parameters and how to use them:

- `--dataset`: Specify the dataset name. Default is `'Mendeley_AK_Singh_2020_phish'`.
- `--pretrained_path`: Path to the pretrained model. Default is `'amahdaouy/DomURLs_BERT'`.
- `--num_workers`: Number of workers for data loading. Default is `1`.
- `--dropout_prob`: Dropout probability for preventing overfitting. Default is `0.2`.
- `--lr`: Learning rate for the optimizer. Default is `1e-5`.
- `--weight_decay`: Weight decay for regularization. Default is `1e-3`.
- `--epochs`: Number of training epochs. Default is `10`.
- `--batch_size`: Size of each data batch. Default is `128`.
- `--experiment_type`: Specify the type of experiment, either `'url'` or `'domain'`. Default is `'url'`.
- `--label_column`: Name of the label column in your dataset. Default is `'label'`.
- `--seed`: Seed for random number generators to ensure reproducibility. Default is `3407`.
- `--device`: GPU device id if training with CUDA. Default is `0`.

Example command to start fine-tuning with default parameters:
```bash
python main_plm.py \
  --dataset Mendeley_AK_Singh_2020_phish \
  --pretrained_path amahdaouy/DomURLs_BERT \
  --num_workers 1 \
  --dropout_prob 0.2 \
  --lr 1e-5 \
  --weight_decay 1e-3 \
  --epochs 10 \
  --batch_size 128 \
  --experiment_type url \
  --label_column label \
  --seed 3407 \
  --device 0
```

NB. for URLBERT, you need to download the [urlBERT.pt]((https://drive.google.com/drive/folders/16pNq7C1gYKR9inVD-P8yPBGS37nitE-D?usp=drive_link)) model into `models\urlbert_model` folder.

## Training Character-Based Models for Malicious URL and Domain Name Detection

This guide provides instructions on how to train deep learning models using the `main_charnn.py` script to train character-based models for malicious URLs and domain names detection.

### Usage
To train a model, run the script from the command line with the required parameters. Below is a description of all the parameters and how to use them:

- `--dataset`: Specify the dataset name. Default is `'Mendeley_AK_Singh_2020_phish'`.
- `--model_name`: Choose the model type from the available options: `CharCNN`, `CharLSTM`, `CharGRU`, `CharBiLSTM`, `CharBiGRU`, `CharCNNBiLSTM`. Default is `'CharCNN'`.
- `--num_workers`: Number of workers for data loading. Default is `1`.
- `--dropout_prob`: Dropout probability for preventing overfitting. Default is `0.2`.
- `--lr`: Learning rate for the optimizer. Default is `1e-5`.
- `--weight_decay`: Weight decay for regularization. Default is `1e-3`.
- `--epochs`: Number of training epochs. Default is `20`.
- `--batch_size`: Size of each data batch. Default is `128`.
- `--experiment_type`: Specify the type of experiment, either `'url'` or `'domain'`. Default is `'url'`.
- `--label_column`: Name of the label column in your dataset. Default is `'label'`.
- `--seed`: Seed for random number generators to ensure reproducibility. Default is `3407`.
- `--device`: GPU device id if training with CUDA. Default is `0`.

Example command to start training with default parameters:
```bash
python main_charnn.py \
  --dataset Mendeley_AK_Singh_2020_phish \
  --model_name CharBiGRU \
  --num_workers 4 \
  --dropout_prob 0.25 \
  --lr 0.001 \
  --weight_decay 0.001 \
  --epochs 20 \
  --batch_size 256 \
  --experiment_type url \
  --label_column label \
  --seed 1234 \
  --device 0

```
## Abstract
 Detecting and classifying suspicious or malicious domain names and URLs is fundamental task in cybersecurity. To leverage such indicators of compromise, cybersecurity vendors and practitioners often maintain and update blacklists of known malicious domains and URLs. However, blacklists frequently fail to identify emerging and obfuscated threats. Over the past few decades, there has been significant interest in developing machine learning models that automatically detect malicious domains and URLs, addressing the limitations of blacklists maintenance and updates. In this paper, we introduce DomURLs_BERT, a pre-trained BERT-based encoder adapted for detecting and classifying suspicious/malicious domains and URLs. DomURLs_BERT is pre-trained using the Masked Language Modeling (MLM) objective on a large multilingual corpus of URLs, domain names, and Domain Generation Algorithms (DGA) dataset. In order to assess the performance of DomURLs_BERT, we have conducted experiments on several binary and multi-class classification tasks involving domain names and URLs, covering phishing, malware, DGA, and DNS tunneling. The evaluations results show that the proposed encoder outperforms state-of-the-art character-based deep learning models and cybersecurity-focused BERT models across multiple tasks and datasets. The pre-training dataset, the pre-trained DomURLs_BERT encoder, and the experiments source code are publicly available.

## Citation
To be updated
```bibtex
@article{domurlsbert2024,
  title={{DomURLs\_BERT}: Pre-trained BERT-based Model for Malicious Domains and URLs Detection and Classification},
  author={Abdelkader El Mahdaouy and Salima Lamsiyah and Meryem Janati Idrissi and Hamza Alami and Zakaria Yartaoui and Ismail Berrada},
  journal={arXiv preprint arXiv:},
  eprint={},
  eprinttype = {arXiv},
  url={https://arxiv.org/abs/},
  year={2024},
}
```
