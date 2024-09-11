import numpy as np
import sys
from torch.utils.data import Dataset
import torch

class CDataset(Dataset):
    """
    A PyTorch Dataset class for text classification tasks.

    Args:
        texts (list): List of input texts.
        feats (list): List of additional features.
        labels (list): List of target labels.
        max_length (int): Maximum length of input texts.

    Attributes:
        vocabulary (list): List of characters considered in the dataset.
        identity_mat (np.ndarray): Identity matrix for character encoding.
        texts (list): List of input texts.
        feats (list): List of additional features.
        labels (list): List of target labels.
        max_length (int): Maximum length of input texts.
        length (int): Number of samples in the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Retrieves a sample from the dataset at the given index.
    """
    def __init__(self, texts, labels, max_length=80):
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.vocabulary.index(i) + 2 if i in self.vocabulary else 1 for i in list(raw_text)],
                        dtype=np.int32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data)), dtype=np.int32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length), dtype=np.int32)
        label = self.labels[index]

        return data, label