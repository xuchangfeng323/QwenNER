from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from utils import Arguments
import torch.nn as nn


class Qwen4NER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lr = config.lr
        self.max_new_tokens = config.max_new_tokens
        self.weight_decay = config.weight_decay
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        if config.method == "lora":
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
                
            )
            base_model.config.pad_token_id = config.tokenizer.pad_token_id
            self.qwen = get_peft_model(base_model, lora_config)

        elif config.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
                quantization_config=bnb_config,
                
            )
            base_model.config.pad_token_id = config.tokenizer.pad_token_id
            base_model = prepare_model_for_kbit_training(base_model)
            self.qwen = get_peft_model(base_model, lora_config)

        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
                
            )
            base_model.config.pad_token_id = config.tokenizer.pad_token_id
            self.qwen = base_model
    def generate(self, input_ids, attention_mask, use_cache=True, **kwargs):
        return self.qwen.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.qwen.config.pad_token_id,
            eos_token_id=self.qwen.config.eos_token_id,
            use_cache=use_cache,
            do_sample=False,         
            num_beams=1,              
        )
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        return outputs
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    def get_trained_model(self,best_model_path):
        base_model = self.qwen.base_model
        model=PeftModel.from_pretrained(base_model, best_model_path)
        
