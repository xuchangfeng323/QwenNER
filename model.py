from transformers import AutoModel
import torch
from peft import LoraConfig, get_peft_model, TaskType
from utils import Arguments
import torch.nn as nn


class Qwen4NER(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        base_model = AutoModel.from_pretrained(config.model_dir, trust_remote_code=True)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.qwen = get_peft_model(base_model, lora_config)
        self.qwen.print_trainable_parameters()

        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.embedding_dim, config.class_num)
    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.fc(last_hidden_state)
        return logits
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
