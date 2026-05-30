import os
import random
import pandas as pd
from transformers import BertTokenizerFast
from MyDataset import WeiboNerDataset
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
    train_dataset = WeiboNerDataset(os.path.join(data_dir, 'train.txt'), config.tokenizer, config.max_length, config.align_type)
    test_dataset = WeiboNerDataset(os.path.join(data_dir, 'test.txt'), config.tokenizer, config.max_length, config.align_type)
    dev_dataset = WeiboNerDataset(os.path.join(data_dir, 'dev.txt'), config.tokenizer, config.max_length, config.align_type)

    label2id, id2label = build_label_mappings(train_dataset.label_list+test_dataset.label_list+dev_dataset.label_list, save_path=os.path.join(data_dir, 'label2id.json'))
    config.set_mapping(label2id,id2label)
    train_dataset.set_label2id(label2id)
    test_dataset.set_label2id(label2id)
    dev_dataset.set_label2id(label2id)
    
    
    train_dataLoader = train_dataset.get_data_loader(batch_size=config.batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    test_dataLoader = test_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    return train_dataLoader, dev_dataLoader, test_dataLoader,label2id,id2label

def write_log(log_jsonl_path, log_dict):

    with open(log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
class Metrics:
    def __init__(self,label2id,id2label,eps=1e-8):
        self.id2label = id2label
        self.label2id = label2id
        self.eps = eps
        self.entity_types = set()
        for idx, label in id2label.items():
            if label.startswith('B-'):
                self.entity_types.add(label[2:])  
        self.entity_types = sorted(self.entity_types)
        self.all_true_entities = set()
        self.all_pred_entities = set()
        self.seq_count = 0  
        self.result_df = None
        
    def add(self, predictions, labels):    
        predictions = predictions.tolist()
        labels = labels.tolist()
        
        for pred_seq, label_seq in zip(predictions, labels):
            
            pred_str = [self.id2label.get(p, 'O') for p, l in zip(pred_seq, label_seq) if l != -100]
            true_str = [self.id2label.get(l, 'O') for l in label_seq if l != -100]
            self.all_true_entities.update(self._extract_entities(true_str, self.seq_count))
            self.all_pred_entities.update(self._extract_entities(pred_str, self.seq_count))
            self.seq_count += 1

    def _extract_entities(self, tags,seq_id):
        entities =[]
        i=0
        while i<len(tags):
            if tags[i].startswith('B-'):
                entity_type = tags[i][2:]
                start=i
                i+=1
                while i<len(tags) and tags[i]=='I-'+entity_type:
                    i+=1
                end=i
                entities.append((seq_id,entity_type,start,end))
            else:
                i+=1
        return entities


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
        for key, value in self.args_dict.items():
            setattr(self, key, value)
        self._set_seed()
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)
        
    def _set_seed(self):
        seed = self.args_dict.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_json_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    def get_args_dict(self):
        return self.args_dict
    def set_mapping(self,label2id,id2label):
        self.label2id=label2id
        self.id2label=id2label
        self.class_num=len(label2id)
        self.args_dict['class_num']=self.class_num
        self.args_dict['label2id']=label2id
        self.args_dict['id2label']=id2label
        

