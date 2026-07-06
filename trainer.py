import swanlab
from tqdm import tqdm
import torch
from model import Qwen4NER
import argparse
import torch.nn as nn
from transformers import  get_scheduler
from MyDataset import bc2gmDataset
from utils import get_next, write_log, Arguments, Metrics, EarlyStop, load_data
import os
class Trainer:
    def __init__(self,config):
        self.optimizer=None
        self.scheduler=None
        self.device=config.device
        self.num_epochs=config.num_epochs
        self.config=config
        print(config.get_args_dict())
        self.metrics = Metrics(config)
        self.best_accuracy = 0.0
        self.save_dir = get_next(config.save_dir)
        self.early_stop = EarlyStop(config, self.save_dir)
        self.log_dir=os.path.join(self.save_dir,"log.jsonl")
        
        
    
    def train(self,traindataLoader, devdataLoader, testdataLoader, model,optimizer):
        self.optimizer=optimizer
        
        self.model=model
        self.scheduler=get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.num_epochs * len(traindataLoader)
        )
        model.to(self.device)
        swanlab.init(
            project="qwen4ner",  
            name="qwen2.5-ner",
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
                loss.backward()
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
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

        self.test(testdataLoader)
        swanlab.finish()
            
    
        
    def eval(self,epoch, devdataLoader):
        self.model.eval()
        
        progress_bar = tqdm(devdataLoader, desc="Evaluation", position=0, leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids=batch["input_ids"]
                attention_mask=batch["attention_mask"]
                labels=batch["labels"]
                texts=batch["text"]
                entities=batch["entities"]
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                generated_ids = self.model.generate(input_ids, attention_mask,use_cache=True)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)]
                response = self.config.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                pred_entities_batch = [self.metrics.parse_json(r) for r in response]
                self.metrics.add_entities(pred_entities_batch, entities, texts)
        results = self.metrics.get_results()
        print(results)
        results_dict = self.metrics.get_result_dict()
        self.metrics.reset()
            
        print(f"Eval F1 Score: {results_dict['micro_avg']['f1']:.4f}")
        
        swanlab.log({
            "eval/f1": results_dict['micro_avg']['f1'],
        })

        
        return results_dict
    def test(self, testdataLoader):
        model=self.model.get_trained_model(self.early_stop.best_model_path)
        model.eval()
        model = model.to(self.device)

        progress_bar = tqdm(testdataLoader, desc="Testing", position=0, leave=True)
        with torch.no_grad():
            for  batch in progress_bar:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                texts = batch["text"]
                entities = batch["entities"]
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                generated_ids = self.model.generate(input_ids, attention_mask,use_cache=True)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)]
                response = self.config.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                pred_entities_batch = [self.metrics.parse_json(r) for r in response]
                self.metrics.add_entities(pred_entities_batch, entities, texts)

        results = self.metrics.get_results()
        results_dict = self.metrics.get_result_dict()
        
         
        
        print(f"Test F1 Score: {results_dict['micro_avg']['f1']:.4f}")
        self.metrics.reset()
        log_dict = {
            "test/results": results_dict
        }
        write_log(self.log_dir, {"test": log_dict})
        swanlab.log({
            "test/f1": results_dict['micro_avg']['f1'],
            
        })
        print(results)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=str, default='./args/arg1.json')
    args = parser.parse_args()
    args=Arguments(args.arg)
    train_dataset = bc2gmDataset(args, args.data_path + "train.json", is_train=True)
    dev_dataset = bc2gmDataset(args, args.data_path + "dev.json", is_train=False)
    test_dataset = bc2gmDataset(args, args.data_path + "test.json", is_train=False)
    dev_dataloader = dev_dataset.get_data_loader(batch_size=args.batch_size, shuffle=False)
    test_dataloader = test_dataset.get_data_loader(batch_size=args.batch_size, shuffle=False)
    train_dataloader = train_dataset.get_data_loader(batch_size=args.batch_size, shuffle=True)

    model=Qwen4NER(args)
    optimizer = model.get_optimizer()
    trainer=Trainer(args)
    trainer.train(train_dataloader, dev_dataloader, test_dataloader, model, optimizer)  
        