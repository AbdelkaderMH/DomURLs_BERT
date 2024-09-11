import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNNBiLSTM(nn.Module):
 
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, num_filters=100, kernel_sizes=[3, 4, 5], drop_prob=0.2):
 
        super(CharCNNBiLSTM, self).__init__()
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs_1d = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.bilstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_dim, n_layers,
                              dropout=drop_prob, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim * 2, output_size)  # Multiply by 2 for bidirectional output
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Reshape for Conv1d: (batch_size, embedding_dim, seq_length)
        

        x = [F.relu(conv(x)) for conv in self.convs_1d]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, 1)  # Concatenate along the filter dimension
        
        x = x.unsqueeze(1)  # Add a sequence dimension (assumes each feature vector is a timestep)
        
        x, _ = self.bilstm(x)
        
        x = x[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
