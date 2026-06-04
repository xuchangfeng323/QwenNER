from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType
from utils import Arguments
import torch.nn as nn


class Qwen4NER(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        base_model = AutoModelForCausalLM.from_pretrained(config.model_dir, trust_remote_code=True)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.qwen = get_peft_model(base_model, lora_config)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        return outputs
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer