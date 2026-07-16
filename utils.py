import os
import argparse
import random
import pandas as pd
from transformers import AutoTokenizer, PreTrainedModel
from peft import PeftModel
from MyDataset import bc2gmDataset, template_dict
import torch
import json
import numpy as np
from collections import Counter
import shutil
from accelerate import Accelerator
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
                if not isinstance(pred_entity, dict):
                    continue
                entity_type = pred_entity.get('type')
                entity_name = pred_entity.get('name')
                if not entity_name or not entity_type:
                    continue
                if entity_type not in self.entity_types:
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
        print(counts)
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
                    self.save_checkpoint(model,False)
            else:
                self.best_score = acc
                self.counter = 0
                self.save_checkpoint(model,True)
        elif self.monitor == 'val_loss':
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(model,True)
                return
            if self.best_score - loss  < self.delta:
                self.counter += 1
                
                if self.counter > self.patience:
                    self.early_stop = True
                    self.save_checkpoint(model,False)
            else:
                self.best_score = loss
                self.counter = 0
                self.save_checkpoint(model,True)
        if self.monitor == 'val_f1':
            if self.best_score is None :
                self.best_score = f1_score
                self.save_checkpoint(model,True)
                return 
            

            if f1_score-self.best_score  < self.delta:
                self.counter += 1
                
                if self.counter > self.patience:
                    self.early_stop = True
                    self.save_checkpoint(model,False)
            else:
                self.best_score = f1_score
                self.counter = 0
                self.save_checkpoint(model,True)
        if epoch==self.config.num_epochs-1:
            if self.monitor == 'val_acc':
                self.save_checkpoint(model,False)
            elif self.monitor == 'val_loss':
                self.save_checkpoint(model,False)
            elif self.monitor == 'val_f1':
                self.save_checkpoint(model,False)
            return 
        return self.early_stop
        
    def save_checkpoint(self, model, is_best):
        if is_best and self.best_model_path is not None and os.path.exists(self.best_model_path):
            if os.path.isdir(self.best_model_path):
                shutil.rmtree(self.best_model_path)
            else:
                os.remove(self.best_model_path)
        if not isinstance(model, PreTrainedModel):
            if isinstance(model, PeftModel):
                model.save_pretrained(self.save_dir)
            elif isinstance(Accelerator.unwrap_model(model), PreTrainedModel):
                Accelerator.unwrap_model(model).save_pretrained(self.save_dir, state_dict=model.state_dict())
            else:
                torch.save(model.state_dict(), self.save_dir)
        else:
            model.save_pretrained(self.save_dir, state_dict=model.state_dict())
        if is_best:
            self.best_model_path = self.save_dir
            
                

            



class Arguments:
    def __init__(self, config_path="arguments.json"):
        self.args_dict = self._load_json_config(config_path)
        
        
        self.tokenizer=None
        self.template=None
        for key, value in self.args_dict.items():
            setattr(self, key, value)
        self._set_seed()
        self._set_tokenizer()
        self._set_template()
        
    def _set_seed(self):
        seed = self.args_dict.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def _set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args_dict['model_dir'], trust_remote_code=True,)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def _set_template(self):
        template_name = self.args_dict.get('template_name', 'qwen')
        self.template = template_dict.get(template_name, None)
    def _load_json_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    def get_args_dict(self):
        return self.args_dict
    

