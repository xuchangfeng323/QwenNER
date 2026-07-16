from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from utils import Arguments
import torch.nn as nn
import bitsandbytes as bnb

class Qwen4NER(nn.Module):
    def __init__(self, config,skip_load=False):
        super().__init__()
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.config = config
        self.model = None
        self.base_model = None
        if not skip_load:
            self.set_model()
    def get_optimizer(self):
        if self.config.method == "qlora":
            optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    def set_model(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        if self.config.method == "lora":
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_dir,
                trust_remote_code=True,
                

            )
            base_model.config.pad_token_id = self.config.tokenizer.pad_token_id
            base_model.config.eos_token_id = self.config.tokenizer.eos_token_id

            self.base_model = base_model
            self.model = get_peft_model(base_model, lora_config)

        elif self.config.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_dir,
                trust_remote_code=True,
                quantization_config=bnb_config,
                

            )
            base_model.config.pad_token_id = self.config.tokenizer.pad_token_id
            base_model.config.eos_token_id = self.config.tokenizer.eos_token_id
            base_model = prepare_model_for_kbit_training(base_model)
            self.base_model = base_model
            self.model = get_peft_model(base_model, lora_config)

        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_dir,
                trust_remote_code=True,

            )
            base_model.config.pad_token_id = self.config.tokenizer.pad_token_id
            base_model.config.eos_token_id = self.config.tokenizer.eos_token_id
            self.base_model = base_model
            self.model = base_model
        
    def load_adapter(self, save_path):
        del self.model
        if self.base_model is None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,

            )
            self.base_model.config.pad_token_id = self.config.tokenizer.pad_token_id
            
        self.model = PeftModel.from_pretrained(self.base_model, save_path)  
        return self.model
    def load_model(self, model_path):
        del self.model
        del self.base_model
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        return self.model
    def get_model(self):
        return self.model
    