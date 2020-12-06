import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiLSTM(nn.Module):
    def __init__(self, embeddings, num_labels, hidden_size=200, num_layer=2, device="gpu"):
        super(BiLSTM, self).__init__()
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
        self.bilstm = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0.to(device)
        self.pred_fc = nn.Linear(self.hidden_size * 2, self.num_labels)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.bilstm(x)
        return output
    
    def dropout(self, v):
        return F.dropout(v, p=0.5, training=self.training)
        
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）
        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn 

    def forward(self, sent1, sent2):
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        p_encode = self.word_emb(sent1)
        h_endoce = self.word_emb(sent2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        
        encoding1 = self.forward_once(p_encode) # batch_size * seq_len * 2 hidden_size
        encoding2 = self.forward_once(h_endoce) # batch_size * seq_len * 2 hidden_size
        
        query1 = self.dropout(encoding1)
        query2 = self.dropout(encoding2)
        
        attn_output1, attention1 = self.attention_net(encoding1, query1)       # batch_size * 2 hidden_size
        attn_output2, attention2 = self.attention_net(encoding2, query2)       # batch_size * 2 hidden_size
        
        l1 = torch.abs(attn_output1 - attn_output2) # batch_size * 2 hidden_size
        # concat = torch.cat([attn_output1,attn_output2],axis=1) # batch_size * 6 hidden_size
        x = self.pred_fc(l1)
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities
        
        


