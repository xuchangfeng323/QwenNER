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
            self.model = get_peft_model(base_model, lora_config)

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
            self.model = get_peft_model(base_model, lora_config)

        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
                
            )
            base_model.config.pad_token_id = config.tokenizer.pad_token_id
            self.model = base_model
    
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    def get_model(self):
        return self.model
    def load_model(self, save_path):
        if config.method == "lora":
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
            )
            load_model = PeftModel(base_model, save_path)
            merge_model = load_model.merge_and_unload()
            return merge_model
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
            load_model = PeftModel(base_model, save_path)
            merge_model = load_model.merge_and_unload()
            return merge_model
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_dir,
                trust_remote_code=True,
            )
            return base_model
