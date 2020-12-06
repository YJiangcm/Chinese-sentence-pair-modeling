import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, embeddings, max_length, num_labels=2, hidden_size=200, num_layer=2, device="gpu"):
        super(BiLSTM, self).__init__()
        self.device = device
        self.num_labels = num_labels
        self.max_length = max_length
        if(embeddings is not None):
            self.embeds_dim = embeddings.shape[1]
            self.word_emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        else:
            self.embeds_dim = 300
            self.word_emb = nn.Embedding(43000, 300)
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True
        self.word_emb.to(device)
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.bilstm = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0.to(device)
        self.pooling = nn.MaxPool1d(kernel_size=self.max_length, stride=None, padding=0)
        self.pred_fc = nn.Linear(self.hidden_size * 4, self.num_labels)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.bilstm(x)
        return output
    
    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)

    def forward(self, sent1, sent2):
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        p_encode = self.word_emb(sent1)
        h_endoce = self.word_emb(sent2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        
        encoding1 = self.forward_once(p_encode)
        encoding2 = self.forward_once(h_endoce)
        
        max_encoding1 = self.pooling(encoding1.permute(0, 2, 1)).squeeze(dim=-1) # batch_size * 2 hidden_size
        max_encoding2 = self.pooling(encoding2.permute(0, 2, 1)).squeeze(dim=-1) # batch_size * 2 hidden_size
        
        concat = torch.cat([max_encoding1,max_encoding2],axis=1) # batch_size * 4 hidden_size
        x = self.pred_fc(concat)
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities