import swanlab
from tqdm import tqdm
import torch
from model import Qwen4NER
import argparse
import torch.nn as nn
from transformers import  get_scheduler
from MyDataset import bc2gmDataset
from utils import get_next, write_log, Arguments, Metrics, EarlyStop
from peft import PeftModel
import os
class Trainer:
    def __init__(self, config, model_config):
        self.optimizer=None
        self.scheduler=None
        self.device=config.device
        self.num_epochs=config.num_epochs
        self.config=config
        self.model_config=model_config
        print(config.get_args_dict())
        self.metrics = Metrics(config)
        self.best_accuracy = 0.0
        self.save_dir = get_next(config.save_dir)
        self.early_stop = EarlyStop(config, self.save_dir)
        self.log_dir=os.path.join(self.save_dir,"log.jsonl")
        
        
    
    def train(self,traindataLoader, devdataLoader, testdataLoader, model,optimizer):
        self.optimizer=optimizer
        self.model=model
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.scheduler=get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.num_epochs * len(traindataLoader)
        )
        model.to(self.device)
        swanlab.init(
            project="qwen4ner",  
            name=f"{self.config.model_name}-{self.config.method}-{self.config.lora_target_modules}",
            config={
                "num_epochs": self.config.num_epochs,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "model": self.config.model_name,
                "weight_decay": self.config.weight_decay,
                "device": self.config.device,
                "embedding_dim": self.config.embedding_dim,
                "data_path": self.config.data_path,
                "max_length": self.config.max_length,
                "patience": self.config.patience,
                "monitor": self.config.monitor,
                "delta": self.config.delta,
                "dropout_rate": self.config.dropout_rate,
                "template_name": self.config.template_name,
                "method": self.config.method,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "lora_target_modules": self.config.lora_target_modules,
            }
        )
        write_log(self.log_dir, {"config": self.config.get_args_dict()})
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(traindataLoader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]", position=0, leave=True)
            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                texts = batch["text"]
                entities = batch["entities"]
                self.optimizer.zero_grad()
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss=outputs.loss
                del outputs
                del input_ids
                del labels
                del attention_mask
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_train_loss += loss.item()
                loss_record=loss.item()
                
                progress_bar.set_postfix({"Loss": loss_record})
                del loss
                if step % 50 == 0:
                    swanlab.log({
                        "train/loss_step": loss_record,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr']
                    }, step=epoch * len(traindataLoader) + step)
            avg_train_loss = total_train_loss / len(traindataLoader)
            swanlab.log({
                "train/loss_epoch": avg_train_loss
            }, step=epoch)
            results_dict = self.eval(epoch, devdataLoader)
            f1 = results_dict['micro_avg']['f1']
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "eval/f1": results_dict['micro_avg']['f1'],
                "eval/results": results_dict
            }
            write_log(self.log_dir, log_dict)

            if self.early_stop(epoch, avg_train_loss, None, f1, model, optimizer, self.scheduler):
                break

        self.eval(epoch,testdataLoader,is_test=True)
        swanlab.finish()
            
    
        
    def eval(self,epoch, dataLoader,is_test=False):

        if is_test:
            del self.model
            torch.cuda.empty_cache()
            self.model = self.model_config.load_adapter(self.early_stop.best_model_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        desc = "Evaluation" if not is_test else "Testing"
        progress_bar = tqdm(dataLoader, desc=desc, position=0, leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids=batch["input_ids"]
                attention_mask=batch["attention_mask"]
                labels=batch["labels"]
                texts=batch["text"]
                entities=batch["entities"]
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                generated_ids = self.model.generate(input_ids, attention_mask=attention_mask,use_cache=True,max_new_tokens=self.config.max_new_tokens,pad_token_id=self.model.config.pad_token_id,eos_token_id=self.model.config.eos_token_id)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)]
                response = self.config.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                if step == 0:
                    print(f"\n{'='*60}")
                    print(f"[DEBUG eval] 模型生成示例 (前3条):")
                    print(f"{'='*60}")
                    for i, r in enumerate(response[:3]):
                        print(f"  [{i}] 原始文本: {texts[i][:100]}...")
                        print(f"  [{i}] 模型输出: {repr(r)}")
                        print(f"  [{i}] 真实标签: {entities[i]}")
                        print()
                    print(f"{'='*60}\n")
                pred_entities_batch = [self.metrics.parse_json(r) for r in response]
                
                self.metrics.add_entities(pred_entities_batch, entities, texts)
        results = self.metrics.get_results()
        print(results)
        results_dict = self.metrics.get_result_dict()
        self.metrics.reset()
            
        print(f"{desc} F1 Score: {results_dict['micro_avg']['f1']:.4f}")
        
        swanlab.log({
            f"{desc}/f1": results_dict['micro_avg']['f1'],
        })
        log_dict = {
            f"{desc}/results": results_dict
        }
        write_log(self.log_dir, log_dict)
        if is_test:
            self.save_model()
            
        return results_dict
    
    def save_model(self):
        if isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
        self.model.save_pretrained(os.path.join(self.save_dir, "best_model"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=str, default='./args/arg1.json')
    args = parser.parse_args()
    args=Arguments(args.arg)
    train_dataset = bc2gmDataset(args, args.data_path + "train.json", is_train=True)
    dev_dataset = bc2gmDataset(args, args.data_path + "dev.json", is_train=False)
    test_dataset = bc2gmDataset(args, args.data_path + "test.json", is_train=False)
    dev_dataloader = dev_dataset.get_data_loader(batch_size=args.batch_size * 2, shuffle=False)
    test_dataloader = test_dataset.get_data_loader(batch_size=args.batch_size * 2, shuffle=False)
    train_dataloader = train_dataset.get_data_loader(batch_size=args.batch_size, shuffle=True)

    model4ner=Qwen4NER(args)
    model=model4ner.get_model()
    optimizer = model4ner.get_optimizer()
    trainer=Trainer(args, model4ner)
    trainer.train(train_dataloader, dev_dataloader, test_dataloader, model, optimizer)
        