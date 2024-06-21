import os
import sys
import time
import warnings
import configparser
import numpy as np
from models.lstm import LSTM
from utils.helper_functions import AverageMeter, accuracy, tokenize, encode_sequence, collate_fn
from create_vocab import CreateVocab, read_dataframes

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


class TextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.df.loc[idx, 'encoded_x'][0])
        l = self.df.loc[idx, 'encoded_x'][1]
        y = self.df.loc[idx, 'category'].astype(np.uint)
        return x, l, y


def train(model, dataloader, criterion, optimizer, epoch, writer, train_steps, device):
    model.train()
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    for i, (text, text_lengths, labels) in enumerate(dataloader):
        text = text.to(device)
        labels = labels.to(device)
        outputs = model(text, text_lengths)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss.update(loss.item(), text.size(0))
        running_accuracy.update(accuracy(preds, labels), text.size(0))
        sys.stdout.write('\r')
        sys.stdout.write(f'Train Steps : {epoch} : {i} / {train_steps} \t'
                         f'Train Loss : {running_loss.avg:.2f} \t'
                         f'Train Accuracy : {running_accuracy.avg:.2f} \t')
        sys.stdout.flush()

        step_global = (epoch * train_steps) + i
        writer.add_scalar('Training Loss', running_loss.avg, step_global)
        writer.add_scalar('Training Accuracy', running_accuracy.avg, step_global)

    writer.add_scalar('Training Epoch Loss', running_loss.avg, epoch)
    writer.add_scalar('Training Epoch Accuracy', running_accuracy.avg, epoch)
    print('')


def val(model, dataloader, criterion, epoch, writer, val_steps, device):
    model.eval()
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    for i, (text, text_lengths, labels) in enumerate(dataloader):
        text = text.to(device)
        labels = labels.to(device)
        outputs = model(text, text_lengths)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss.update(loss.item(), text.size(0))
        running_accuracy.update(accuracy(preds, labels), text.size(0))
        sys.stdout.write('\r')
        sys.stdout.write(f'Val Steps : {epoch} : {i} / {val_steps} \t'
                         f'Val Loss : {running_loss.avg:.2f} \t'
                         f'Val Accuracy : {running_accuracy.avg:.2f} \t')
        sys.stdout.flush()
        step_global = (epoch * val_steps) + i
        writer.add_scalar('Validation Loss', running_loss.avg, step_global)
        writer.add_scalar('Validation Accuracy', running_accuracy.avg, step_global)

    writer.add_scalar('Validation Epoch Loss', running_loss.avg, epoch)
    writer.add_scalar('Validation Epoch Accuracy', running_accuracy.avg, epoch)


def main():
    save_path = os.path.join('model_files/lstm')
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'writer', str(time.time())))
    config = configparser.ConfigParser()
    config.read('utils/config.ini')

    train_df, val_df, test_df = read_dataframes()
    vocab = CreateVocab(first_run=True, load_embeddings=False)
    print('____________vocab created__________________')
    # Tokenize Product Title
    train_df['tokenized_x'] = train_df['product_title'].apply(lambda x: tokenize(x))
    val_df['tokenized_x'] = val_df['product_title'].apply(lambda x: tokenize(x))

    # Encode Product Title
    train_df['encoded_x'] = train_df['tokenized_x'].apply(lambda x: encode_sequence(x, vocab.word2idx))
    val_df['encoded_x'] = val_df['tokenized_x'].apply(lambda x: encode_sequence(x, vocab.word2idx))

    # Create Datasets
    train_dataset = TextDataset(train_df[['encoded_x', 'tokenized_x', 'category']])
    val_dataset = TextDataset(val_df[['encoded_x', 'tokenized_x', 'category']])

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=int(config['train']['batch_size']), collate_fn=collate_fn,
                                  pin_memory=True, num_workers=int(config['train']['num_workers']))
    val_dataloader = DataLoader(val_dataset, batch_size=int(config['train']['batch_size']), collate_fn=collate_fn,
                                pin_memory=True, num_workers=int(config['test']['num_workers']))

    train_steps, val_steps = len(train_dataloader), len(val_dataloader)
    print('Number of Training Steps: {}'.format(train_steps))
    print('Number of Validation Steps: {}'.format(val_steps))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTM(len(vocab.vocab)+2, vocab.embedding_dim, vocab.embedding_matrix,
                 int(config['train']['hidden_size']), num_output=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['train']['lr']))
    print('____________training started created__________________')
    for epoch in range(int(config['train']['epochs'])):
        train(model, train_dataloader, criterion, optimizer, epoch, writer, train_steps, device)
        print('')
        val(model, val_dataloader, criterion, epoch, writer, val_steps, device)
        print('')
        torch.save(model.state_dict(), os.path.join('model_files/lstm/', f'epoch_{epoch}.tar'))


if __name__ == '__main__':
    main()
