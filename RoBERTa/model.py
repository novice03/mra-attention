
import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
import os
import sys

from encoders.backbone import BackboneWrapper

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embeddings = nn.Embedding(config["max_seq_len"] + 2, config["embedding_dim"])
        self.token_type_embeddings = nn.Embedding(config["num_sen_type"], config["embedding_dim"])

        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)
        torch.nn.init.normal_(self.token_type_embeddings.weight, std = 0.02)

        self.has_project = config["embedding_dim"] != config["dim"]
        if self.has_project:
            self.dense = nn.Linear(config["embedding_dim"], config["dim"])

        self.norm = nn.LayerNorm(config["dim"])
        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1) + 2
        type_ids = torch.zeros(input_ids.size(), dtype = torch.long, device = input_ids.device)

        print('input_ids', input_ids, 'position_ids', position_ids, 'type_ids', type_ids)

        X_token = self.word_embeddings(input_ids)
        print('input embed', X_token)
        X_pos = self.position_embeddings(position_ids)
        print('pos embed', X_pos)
        X_seq = self.token_type_embeddings(type_ids)
        print('token type', X_seq)
        X = X_token + X_pos + X_seq
        print('sum', X)

        if self.has_project:
            X = self.dense(X)

        X = self.norm(X)
        X = self.dropout(X)

        print('final emb', X)

        return X

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.backbone = BackboneWrapper(config)

    def forward(self, input_ids, sentence_mask = None):
        X = self.embeddings(input_ids)
        print('emebddings', X)
        if sentence_mask is None:
            sentence_mask = torch.ones_like(input_ids)
        X = self.backbone(X, sentence_mask)
        return X
