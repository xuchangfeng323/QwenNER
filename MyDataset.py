from __future__ import annotations
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import json
from torch.utils.data import DataLoader
class bc2gmDataset(Dataset):
    def __init__(self, args: Arguments, data_path: str, is_train: bool = True):

        self.texts = []
        self.label_list = []
        self.get_sentences(data_path)
        self.label2id=None
        self.is_train=is_train
        self.tokenizer = args.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = args.max_length
        self.prompt = args.prompt
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        entities = self.label_list[idx]
        entity_sentence = ""
        entities_list = [{"name": e["name"], "type": e["type"]} for e in entities]
        entity_sentence = json.dumps({"entities": entities_list}, ensure_ascii=False)
        return {
            'text': text,
            'output':entity_sentence,
            'entities':entities
        }
    def get_sentences(self,dir_path):
        with open(dir_path, 'r', encoding='utf-8') as f:
            data= json.load(f)
        self.texts = [item['sentence'] for item in data]
        self.label_list = [item['entities'] for item in data]
    def get_entities(self,entities_list):

        all_entities = []
        for entity in entities_list:
            all_entities.append(entity['type'])
        return all_entities

    def set_label2id(self, label2id):
        self.label2id=label2id
    def _make_inst_text(self, item):
        
        return (
            f"<|im_start|>system\n{self.prompt}<|im_end|>\n"
            f"<|im_start|>user\n{item['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def collate_fn(self, batch):
        max_len = 0
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        for item in batch:
            instruct_text = self._make_inst_text(item)
            output_text = item["output"]
            instrct_enc = self.tokenizer(instruct_text, add_special_tokens=False)
            output_enc = self.tokenizer(output_text, add_special_tokens=False)
            input_ids = instrct_enc["input_ids"] + output_enc["input_ids"]
            attention_mask = instrct_enc["attention_mask"] + output_enc["attention_mask"]
            label = [-100] * len(instrct_enc["input_ids"]) + output_enc["input_ids"]
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            labels_list.append(label)
            max_len = max(max_len, len(input_ids))

        max_len = min(max_len, self.max_length)

        padded_input_ids, padded_masks, padded_labels = [], [], []
        for input_ids, attention_mask, label in zip(input_ids_list, attention_masks_list, labels_list):
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            label = label[:max_len]
            paddinglen = max_len - len(input_ids)

            if self.is_train:
               
                input_ids = input_ids.extend(paddinglen*[self.tokenizer.pad_token_id])
                attention_mask = attention_mask.extend(paddinglen*[0])
                label = label.extend(paddinglen*[-100])
            else:
                
                input_ids = [self.tokenizer.pad_token_id] * paddinglen + input_ids
                attention_mask = [0] * paddinglen + attention_mask
                label = [-100] * paddinglen + label

            padded_input_ids.append(torch.tensor(input_ids))
            padded_masks.append(torch.tensor(attention_mask))
            padded_labels.append(torch.tensor(label))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_masks),
            "labels": torch.stack(padded_labels),
            "text": [item["text"] for item in batch],
            "entities": [item["entities"] for item in batch],
        }


                
        
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)


