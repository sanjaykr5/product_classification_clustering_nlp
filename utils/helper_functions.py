import re
import os
import torch
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features, lengths, targets = zip(*batch)
    features = pad_sequence(features, batch_first=True, padding_value=0)
    return features, torch.Tensor(lengths).int(), torch.Tensor(targets).long()


def tokenize(text):
    tokenized = text.split(' ')
    return tokenized


def return_idx(token, word2idx):
    try:
        return word2idx[token]
    except KeyError:
        return word2idx['<unk>']


def encode_sequence(tokenized, word2idx, max_length=30):
    encoded = np.array([return_idx(token, word2idx) for token in tokenized])
    length = min(max_length, len(encoded))
    return encoded[:length], length


def lower_text(text: str) -> str:
    """
    Convert text to Lowercase text
    :param text: input_text
    :return: output lowercase text
    """
    return text.lower()


def remove_stopwords(text: str) -> str:
    """
    Remove Stopwords from text 
    :param text: input_text
    :return: output text without stopwords
    """
    stop = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def remove_punctuation(text: str) -> str:
    """
    Remove Punctuation and replace it with white space
    :param text: input_text
    :return: output text without punctuation
    """
    return re.sub(r'[^\w\s]', ' ', text)


def preprocess_df(df, column):
    """
    Apply lower text, remove punctuation & remove stopwords on the given column and return dataframe
    :param df:
    :param column:
    :return:
    """
    df[column] = df[column].apply(lambda x: lower_text(x))
    df[column] = df[column].apply(lambda x: remove_punctuation(x))
    df[column] = df[column].apply(lambda x: remove_stopwords(x))
    return df


def read_dataframes():
    """
    Read & Preprocess DataFrames
    :return:
    """
    train_df = pd.read_csv(os.path.join('data/train_test', 'train_data.csv'))
    val_df = pd.read_csv(os.path.join('data/train_test', 'val_data.csv'))
    test_df = pd.read_csv(os.path.join('data/train_test', 'test_data.csv'))
    train_df = preprocess_df(train_df, 'product_title')
    val_df = preprocess_df(val_df, 'product_title')
    test_df = preprocess_df(test_df, 'product_title')
    train_df = preprocess_df(train_df, 'cluster_label')
    val_df = preprocess_df(val_df, 'cluster_label')
    test_df = preprocess_df(test_df, 'cluster_label')
    return train_df, val_df, test_df


def accuracy(preds, labels):
    """
    Computes Accuracy
    :param preds:
    :param labels:
    :return:
    """
    return torch.sum(preds == labels.data) / preds.size(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
