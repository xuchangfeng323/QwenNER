from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import json
from torch.utils.data import DataLoader

class bc2gmDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, max_length=1024):
        
        self.texts = []
        self.label_list = []
        self.get_sentences(data_path)
        self.label2id=None
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_list[idx]
        return {
            'text': text,
            'labels': label
        }
    def get_sentences(self,dir_path):
        with open(dir_path, 'r', encoding='utf-8') as f:
            data= json.load(f)
        self.texts = [item['sentence'] for item in data]
        self.label_list = [item['entities'] for item in data]
    def get_entities(self):
        
        all_entities = []
        for entities in self.label_list:
            all_entities.extend(entities['type'])
        return all_entities

    def set_label2id(self, label2id):
        self.label2id=label2id
    def collate_fn(self, batch):
        
        texts = [item['text'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            is_split_into_words=True,   
            padding='longest',
            max_length=self.max_length,
            return_tensors="pt"
        )

        targets = []
        targets = torch.tensor(targets, dtype=torch.long)
        
        return encodings['input_ids'], encodings['attention_mask'], targets
            
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    

