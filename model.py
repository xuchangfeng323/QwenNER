from transformers import BertModel
import torch

from utils import Arguments
import torch.nn as nn


class Bert4NER(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        self.bert = BertModel.from_pretrained(config.model_dir)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.embedding_dim, config.class_num)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.fc(last_hidden_state)
        return logits
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        return optimizer
