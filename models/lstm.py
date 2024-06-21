import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_size, num_output):
        super(LSTM, self).__init__()
        self.embeddings_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embeddings_layer.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(2 * hidden_size, num_output)

    def forward(self, text_index, text_lengths):
        embedded = self.embeddings_layer(text_index)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        output = self.dense(self.dropout(hidden))
        return output
