# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from modules import Module, ModuleList, ModuleDict
from modules.encoder import Encoder
from modules.alignment import registry as alignment
from modules.fusion import registry as fusion
from modules.connection import registry as connection
from modules.pooling import Pooling
from modules.prediction import registry as prediction

class RE2(Module):
    def __init__(self, args, embeddings, device="gpu"):
        super().__init__()
        self.dropout = args.dropout
        self.device = device
        if(embeddings is not None):
            self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings))
        else:
            self.embedding = nn.Embedding(43000, 300)
        self.embedding.float()
        self.embedding.weight.requires_grad = True
        self.embedding.to(device)
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, a, b):
        mask_a = torch.ne(a, 0).unsqueeze(2).to(self.device)
        mask_b = torch.ne(b, 0).unsqueeze(2).to(self.device)
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        logits = self.prediction(a, b)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities
