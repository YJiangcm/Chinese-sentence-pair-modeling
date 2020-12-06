# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: 31906
"""
import os
import torch
from torch import nn
# import SentencePiece
from transformers import (
    BertForSequenceClassification, 
    AlbertForSequenceClassification, 
    XLNetForSequenceClassification, 
    RobertaForSequenceClassification,
    DistilBertTokenizer,
    BertTokenizer, 
    AutoTokenizer, 
    XLNetTokenizer,
    AutoConfig,
    BertConfig,
    AutoModelForSequenceClassification
)

class AlbertModel(nn.Module):
    def __init__(self, requires_grad = True, num_labels = 2):
        super(AlbertModel, self).__init__()
        self.num_labels = num_labels
        self.albert = AlbertForSequenceClassification.from_pretrained('voidful/albert_chinese_base', num_labels = self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base', do_lower_case=True)
        # self.albert = AlbertForSequenceClassification.from_pretrained('albert-xxlarge-v2', num_labels = self.num_labels)
        # self.tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v2', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.albert.parameters():
            param.requires_grad = True  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
        
class BertModel(nn.Module):
    def __init__(self, requires_grad = True, num_labels = 2):
        super(BertModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels = self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class Bert_wwm_Model(nn.Module):
    def __init__(self, requires_grad = True, num_labels = 2):
        super(Bert_wwm_Model, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm-ext', num_labels = self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        

class DistilBertModel(nn.Module):
    def __init__(self, requires_grad = True, num_labels = 2):
        super(DistilBertModel, self).__init__()
        self.num_labels = num_labels
        self.distilbert  = BertForSequenceClassification.from_pretrained('adamlin/bert-distil-chinese', num_labels = self.num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained('adamlin/bert-distil-chinese', do_lower_case=True)
        
        # self.distilbert = BertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)
        # self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.distilbert.parameters():
            param.requires_grad = requires_grad  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.distilbert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
class RobertModel(nn.Module):
    def __init__(self, requires_grad = True, num_labels = 2):
        super(RobertModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels = self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
        
        # self.bert = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = self.num_labels)
        # self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
class XlnetModel(nn.Module):
    def __init__(self, requires_grad = True, num_labels = 2):
        super(XlnetModel, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetForSequenceClassification.from_pretrained('hfl/chinese-xlnet-base', num_labels = self.num_labels)
        self.tokenizer = XLNetTokenizer.from_pretrained('hfl/chinese-xlnet-base', do_lower_case=True)
        # self.xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', num_labels = self.num_labels)
        # self.tokenizer = AutoTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.xlnet.parameters():
            param.requires_grad = requires_grad  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
