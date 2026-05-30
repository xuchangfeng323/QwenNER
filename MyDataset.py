from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

class WeiboNerDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, max_length=128,align_type='ignore'):
        self.align_type=align_type
        self.texts = []
        self.label_list = []
        self.get_sentences(data_path)
        self.label2id=None
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
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
        with open(dir_path, encoding='utf-8') as f:
            blocks = f.read().strip().split('\n\n')
        
        sentences_list, tags_list = [], []
        for block in blocks:
            pairs = [line.split() for line in block.split('\n') if len(line.split()) == 2]
            if pairs:
                words, tags = zip(*pairs)
                sentences_list.append(list(words))
                tags_list.append(list(tags))
            
        self.texts = sentences_list
        self.label_list = tags_list
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

        for i, labels in enumerate(labels_list):
            label_ids = []
            word_ids = encodings.word_ids(batch_index=i)  
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                   
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    
                    if word_idx >= len(labels):
                        
                        tag = 'O'
                    else:
                        tag = labels[word_idx]
                    label_ids.append(self.label2id.get(tag, self.label2id['O']))
                else:
                   
                    if self.align_type == 'ignore':
                        
                        label_ids.append(-100)
                    else:
                        
                        if word_idx >= len(labels):
                            tag = 'O'
                        else:
                            tag = labels[word_idx]
                            if tag.startswith('B-'):
                                tag = 'I-' + tag[2:]
                        label_ids.append(self.label2id.get(tag, self.label2id['O']))
                
            
                previous_word_idx = word_idx

            targets.append(label_ids)

    
        targets = torch.tensor(targets, dtype=torch.long)
        
        return encodings['input_ids'], encodings['attention_mask'], targets
            
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    

