"""
This Script creates the Vocabulary, load glove embeddings, tokenize text & saves files for predictions
"""
import json
import numpy as np
from utils.helper_functions import read_dataframes


def load_glove_embeddings():
    embeddings_dict = {}
    with open("glove_embeddings/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


class CreateVocab:
    def __init__(self, first_run, load_embeddings):
        self.vocab = []
        self.word2idx = {}
        self.embedding_dim = 100
        self.embedding_matrix = None
        if first_run:
            self.create_vocab()
            self.save_mapping_and_embeddings()
        else:
            self.load_mapping_and_embeddings(load_embeddings)

    def create_vocab(self):
        """
        The Vocabulary is created using train set of product title and combined cluster label of train,
        test & val for product clustering

        :return list of tokens in dataset:
        """
        train_df, val_df, test_df = read_dataframes()
        cluster_name_vocab = train_df['cluster_label'].tolist() + val_df['cluster_label'].tolist() + test_df[
            'cluster_label'].tolist()
        cluster_name_vocab = [y for x in cluster_name_vocab for y in x.split(' ')]
        cluster_name_vocab = list(set(cluster_name_vocab))
        vocab = train_df['product_title'].tolist()
        vocab = [y for x in vocab for y in x.split(' ')]
        vocab = list(set(vocab))
        vocab = list(set(vocab + cluster_name_vocab))
        print('Final num of tokens in vocab ', len(vocab))
        self.vocab = vocab

    def load_mapping_and_embeddings(self, load_embeddings):
        """
        Load saved word to index file & embedding matrix
        """
        with open('model_files/word2idx.json', 'r') as f:
            self.word2idx = json.load(f)
        if load_embeddings:
            self.embedding_matrix = np.load('model_files/embeddings.npy')

    def save_mapping_and_embeddings(self):
        """
         This function iterates over the vocab and create a mapping of word and index;
         Also iterate over pretrained glove embeddings and save embeddings for words found otherwise initialize them
        :return saves dict of word and index as json:
        :return:
        """
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.embedding_matrix = np.zeros((len(self.vocab) + 2, self.embedding_dim), np.float32)
        glove_embeddings = load_glove_embeddings()
        words_found_in_glove = 0
        for i, token in enumerate(self.vocab):
            try:
                self.embedding_matrix[i + 2] = glove_embeddings[token]
                words_found_in_glove += 1
            except KeyError:
                self.embedding_matrix[i + 2] = np.random.normal(scale=1, size=(self.embedding_dim,))
            self.word2idx[token] = i + 2
        print('% of words found in glove', words_found_in_glove / len(self.vocab))
        with open('model_files/word2idx.json', 'w') as f:
            json.dump(self.word2idx, f)
        np.save('model_files/embeddings.npy', self.embedding_matrix)
