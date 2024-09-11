import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class PositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )
        

class CharTransformerEncoder(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, num_heads=8, drop_prob=0.2):

        super(CharTransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.positional_encoding = PositionEmbedding(num_embeddings=512, embedding_dim=embedding_dim)  # Adjust the max sequence length as needed

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, drop_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(embedding_dim, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        
        x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x)
        
        x = x[:, -1, :]
        
        out = self.dropout(x)
        out = self.fc(out)
        
        return out
