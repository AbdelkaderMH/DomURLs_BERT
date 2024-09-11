import torch.nn as nn
import torch.nn.functional as F
import torch

class CharLSTM(nn.Module):


    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):

        super(CharLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):

        x = self.embedding(x)
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out[:, -1, :]  # get the output of the last time step
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        return out

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

class CharGRU(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):

        super(CharGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                          dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):

        x = self.embedding(x)
        
        gru_out, _ = self.gru(x)
        
        gru_out = gru_out[:, -1, :]  # get the output of the last time step
        
        out = self.dropout(gru_out)
        out = self.fc(out)
        
        return out

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        
        return hidden

class CharBiLSTM(nn.Module):


    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):

        super(CharBiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                              dropout=drop_prob, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_dim * 2, output_size) # multiply by 2 for bidirection
        
    def forward(self, x):
        x = self.embedding(x)
        
        bilstm_out, _ = self.bilstm(x)
        
        bilstm_out = bilstm_out[:, -1, :]  # get the output of the last time step
        
        out = self.dropout(bilstm_out)
        out = self.fc(out)
        
        return out

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(), # multiply by 2 for bidirection
                  weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        
        return hidden

class CharBiGRU(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):

        super(CharBiGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.bigru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_dim * 2, output_size) # multiply by 2 for bidirection
        
    def forward(self, x):
        x = self.embedding(x)
        
        bigru_out, _ = self.bigru(x)
        
        bigru_out = bigru_out[:, -1, :]  # get the output of the last time step
        
        out = self.dropout(bigru_out)
        out = self.fc(out)
        
        return out

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        hidden = weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_() # multiply by 2 for bidirection
        
        return hidden
