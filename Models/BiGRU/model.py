import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, embeddings, num_labels, hidden_size=200, num_layer=2, device="gpu"):
        super(BiGRU, self).__init__()
        self.device = device
        self.num_labels = num_labels
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
        self.gru = nn.GRU(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0.to(device)
        self.pred_fc = nn.Linear(2 * self.hidden_size, self.num_labels)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.gru(x)
        return output.permute(1, 0, 2)[-1] # batchsize * (num_directions * hidden_size)
    
    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)

    def forward(self, sent1, sent2):
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        p_encode = self.word_emb(sent1)
        h_endoce = self.word_emb(sent2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        
        encoding1 = self.forward_once(p_encode) # batchsize * (num_directions * hidden_size)
        encoding2 = self.forward_once(h_endoce) # batchsize * (num_directions * hidden_size)
        
        element_wise_product = encoding1 * encoding2
        # sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True)) # batch_size * seq_len * 1
        # x = self.pred_fc(sim.squeeze(dim=-1)) # batch_size * seq_len
        x = self.pred_fc(element_wise_product)
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities