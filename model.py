from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, TaskType
from utils import Arguments
import torch.nn as nn


class Qwen4NER(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lr=config.lr
        self.max_new_tokens=config.max_new_tokens
        self.weight_decay=config.weight_decay
        base_model = AutoModelForCausalLM.from_pretrained(config.model_dir, trust_remote_code=True)
        base_model.config.pad_token_id = config.tokenizer.pad_token_id
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self.qwen = get_peft_model(base_model, lora_config)
    def generate(self,input_ids,attention_mask,use_cache=True):
        return self.qwen.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=self.max_new_tokens,pad_token_id=self.qwen.config.pad_token_id,use_cache=use_cache)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        return outputs
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer