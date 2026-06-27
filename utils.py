import os
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer
from MyDataset import bc2gmDataset
import torch
import json
import numpy as np
from collections import Counter
global label2id, id2label
def get_next(prefix_dir):
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir+'/exp1')
        return prefix_dir+'/exp1'
    else:
        existing_nums = []
        for file in os.listdir(prefix_dir):
            if file.startswith('exp'):
                existing_nums.append(int(file[3:]))
        if len(existing_nums) == 0:
            next_num = 1
        else:        
            next_num = max(existing_nums) + 1
        os.makedirs(prefix_dir+'/exp'+str(next_num))
        return prefix_dir+'/exp'+str(next_num)

def build_label_mappings(labels, save_path=None):
    unique_labels = set()
    for seq in labels:
        for tag in seq:
            unique_labels.add(tag)
    unique_labels = sorted(unique_labels)
    
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    if save_path:
        mappings = {"label2id": label2id, "id2label": id2label}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=4)
        print(f"Label mappings saved to {save_path}")
    
    return label2id, id2label



def load_data(config):
    data_dir=config.data_path
    train_dataset = bc2gmDataset(os.path.join(data_dir, 'train.json'), config.tokenizer, config.max_length)
    test_dataset = bc2gmDataset(os.path.join(data_dir, 'test.json'), config.tokenizer, config.max_length)
    dev_dataset = bc2gmDataset(os.path.join(data_dir, 'dev.json'), config.tokenizer, config.max_length)
    
    train_dataLoader = train_dataset.get_data_loader(batch_size=config.batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    test_dataLoader = test_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    return train_dataLoader, dev_dataLoader, test_dataLoader

def write_log(log_jsonl_path, log_dict):

    with open(log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
class Metrics:
    def __init__(self,config,eps=1e-8):
        
        self.eps = eps
        self.entity_types = set()
        with open(os.path.join(config.data_path, 'labels.json'), 'r', encoding='utf-8') as f:
            data= json.load(f)
            self.entity_types = set(data)
        self.entity_types = sorted(self.entity_types)
        self.all_true_entities = set()
        self.all_pred_entities = set()
        self.seq_count = 0  
        self.result_df = None
    def parse_json(self,json_str):
        try:
            data=json.loads(json_str)
            return data['entities']
        except:
            pass
            return []
    
    def add_entities(self, batch_predictions, batch_labels,batch_texts):
        for preds, labels, texts in zip(batch_predictions, batch_labels, batch_texts):
            for true_entity in labels:
                entity_type = true_entity['type']
                entity_pos = true_entity['pos']
                if entity_type  in self.entity_types and entity_pos is not None:
                    self.all_true_entities.add((self.seq_count,entity_type,entity_pos[0],entity_pos[1]))
            occurrences = set()
            for pred_entity in preds:
                entity_type = pred_entity['type']
                entity_name = pred_entity['name']
                if not entity_name :
                    continue
                start = texts.find(entity_name)
                
                while start !=-1 and start in occurrences:
                    start = texts.find(entity_name,start+1)
                if start !=-1:
                    occurrences.add(start)
                    end = start + len(entity_name)
                    self.all_pred_entities.add((self.seq_count,entity_type,start,end))
            self.seq_count += 1
            
            
            



    def reset(self):
        self.all_true_entities = set()
        self.all_pred_entities = set()
        self.seq_count = 0  
        self.result_df = None

    
    
    def get_results(self):
       
        counts = {etype: {'tp': 0} for etype in self.entity_types}
        for pred in self.all_pred_entities:
            etype = pred[1]
            if etype not in counts:
                continue
            if pred in self.all_true_entities:
                counts[etype]['tp'] += 1
        pre_counts = Counter(item[1] for item in self.all_pred_entities)
        true_counts = Counter(item[1] for item in self.all_true_entities)
        results=[]
        for etype in self.entity_types:
            tp = counts[etype]['tp']
            precision = tp / ( pre_counts[etype] + self.eps)
            recall = tp / (true_counts[etype] + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            results.append({'precision':precision,'recall':recall,'f1':f1,"support":true_counts[etype]})
        df = pd.DataFrame(results,index=self.entity_types)
        if not df.empty:
            df.loc['macro_avg'] = df[['precision', 'recall', 'f1']].mean()
            df.loc['macro_avg', 'support'] = float('nan')
        total_tp = sum(c['tp'] for c in counts.values())
        total_pre = sum(pre_counts.values())
        total_true=sum(true_counts.values())
        micro_p = total_tp / ( total_pre + self.eps) 
        micro_r = total_tp / ( total_true + self.eps) 
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + self.eps)
        df.loc['micro_avg'] = [micro_p, micro_r, micro_f1, float('nan')]
        self.result_df = df
        return df
                
        
    
    def get_result_dict(self):
        
        if self.result_df is None:
            df = self.get_results()
        else:
            df = self.result_df
        
        result = {}
        for idx in df.index:
           
            if isinstance(idx, int) or isinstance(idx, str):
                key = str(idx)
            else:
                key = idx   
            row = df.loc[idx]
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = val.item()
                else:
                    row_dict[col] = val
            result[key] = row_dict
        return result

class EarlyStop():
    def __init__(self,config,save_dir=None):
        self.config=config
        self.monitor = config.monitor
        self.delta=config.delta
        self.best_score = None
        self.counter = 0
        self.patience = config.patience
        self.early_stop = False
        self.save_dir = save_dir
        self.best_model_path=None
        
    def __call__(self, epoch,loss,acc,f1_score, model,optimizer,scheduler,):
        if self.monitor == 'val_acc':
            if self.best_score is None :
                self.best_score = acc
                
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,True)
                return 
            

            if acc-self.best_score  < self.delta:
                self.counter += 1
                
                if self.counter > self.patience:
                    self.early_stop = True
                    self.save_checkpoint(model, optimizer, scheduler, epoch,acc,False)
            else:
                self.best_score = acc
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,True)
        elif self.monitor == 'val_loss':
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss,True)
                return
            if self.best_score - loss  < self.delta:
                self.counter += 1
                
                if self.counter > self.patience:
                    self.early_stop = True
                    self.save_checkpoint(model, optimizer, scheduler, epoch,loss,False)
            else:
                self.best_score = loss
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss,True)
        if self.monitor == 'val_f1':
            if self.best_score is None :
                self.best_score = f1_score
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,True)
                return 
            

            if f1_score-self.best_score  < self.delta:
                self.counter += 1
                
                if self.counter > self.patience:
                    self.early_stop = True
                    self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,False)
            else:
                self.best_score = f1_score
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,True)
        if epoch==self.config.num_epochs-1:
            if self.monitor == 'val_acc':
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,False)
            elif self.monitor == 'val_loss':
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss,False)
            elif self.monitor == 'val_f1':
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,False)
            return 
        return self.early_stop
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, dev_metrics,is_best):
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        checkpoint = {
            'epoch': epoch,
            'label2id': self.config.label2id,
            'id2label': self.config.id2label,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'peft_config': model.qwen.peft_config,
            'base_model_name': self.config.model_dir,
        }
        if is_best and self.best_model_path is not None:
            os.remove(self.best_model_path)
        torch.save(checkpoint, checkpoint_path)
        print(f"保存 epoch {epoch + 1} 的 checkpoint: {checkpoint_path}")
        if is_best:
            self.best_model_path = checkpoint_path
            print(f"更新最佳模型: {checkpoint_path} (监控指标 '{self.monitor}' = {dev_metrics:.6f})")
class Arguments:
    def __init__(self, config_path="arguments.json"):
        self.args_dict = self._load_json_config(config_path)
        self.class_num=None
        self.label2id=None
        self.id2label=None
        self.tokenizer=None
        for key, value in self.args_dict.items():
            setattr(self, key, value)
        self._set_seed()
        self._set_tokenizer()
        
    def _set_seed(self):
        seed = self.args_dict.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def _set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args_dict['model_dir'], trust_remote_code=True,use_fast=False,padding_side='left',)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def _load_json_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    def get_args_dict(self):
        return self.args_dict
    

