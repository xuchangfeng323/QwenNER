from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import json
from torch.utils.data import DataLoader
from utils import Arguments
class bc2gmDataset(Dataset):
    def __init__(self,args:Arguments):

        self.texts = []
        self.label_list = []
        self.get_sentences(args.data_path)
        self.label2id=None
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
        full_texts = []
        inst_lens = []
        for item in batch:
            inst_text = self._make_inst_text(item)
            full_texts.append(inst_text + item["output"])
            inst_lens.append(len(self.tokenizer(inst_text, add_special_tokens=False)["input_ids"]))

        enc = self.tokenizer(
            full_texts,
            padding_side="left",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        batch_len = enc["input_ids"].shape[1]
        labels = torch.full_like(enc["input_ids"], -100)
        for i, item in enumerate(batch):
            output_ids = self.tokenizer(item["output"], add_special_tokens=False)["input_ids"]
            avail_len = batch_len - inst_lens[i]
            output_len = min(len(output_ids), avail_len)
            if output_len > 0:
                labels[i, -output_len:] = torch.tensor(output_ids[:output_len])

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "text": [item["text"] for item in batch],
            "entities": [item["entities"] for item in batch],
        }
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)


