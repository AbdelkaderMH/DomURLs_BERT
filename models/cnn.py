import torch.nn as nn
import torch.nn.functional as F
import torch

class CharCNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim=None, n_layers= None, num_filters=100, kernel_sizes=[3, 4, 5], drop_prob=0.2):
        super(CharCNN, self).__init__()
        self.num_filters = num_filters
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs_1d = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)  # Reshape for Conv1d: (batch_size, embedding_dim, seq_length)
        
        x = [F.max_pool1d(F.relu(conv(x)), kernel_size=x.size(2)).squeeze(2) for conv in self.convs_1d]
        x = torch.cat(x, 1)  # Concatenate along the filter dimension
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
