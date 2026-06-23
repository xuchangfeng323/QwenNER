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
    def process_func(self, item):
        input_ids, attention_mask, labels = [], [], []
        system_prompt = (
            "你是一个生物医学命名实体识别专家。"
            "你的任务是从给定的英文生物医学文本中识别基因和蛋白质实体（GENE）。\n\n"
            "实体类型定义：\n"
            "- GENE: 基因或蛋白质名称，包括基因产物、酶、受体、抗体、细胞因子等\n\n"
            "请严格按照以下JSON格式输出识别结果：\n"
            '{"entities": [{"name": "实体名称", "type": "实体类别"}]}\n\n'
            "要求：\n"
            "1. 必须输出合法的JSON字符串，不要包含任何额外解释\n"
            "2. 如果句子中没有找到任何实体，输出 {\"entities\": []}\n"
            "3. 实体名称必须与原文中的表述完全一致，不要修改或翻译\n"
            "4. 每个实体单独列出，不要合并不同的实体"
        )
        instruction = self.tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )   
        text = item['text']
        entities = item['entities']
        response = self.tokenizer(f"{item['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids)>self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            text = item['text'][:self.max_length]
            entities = item['entities']
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,"text":text,"entities":entities}
    def collate_fn(self, batch):
        processed_batch = [self.process_func(item) for item in batch]
        input_ids = torch.stack([torch.tensor(p["input_ids"]) for p in processed_batch])
        attention_mask = torch.stack([torch.tensor(p["attention_mask"]) for p in processed_batch])
        labels = torch.stack([torch.tensor(p["labels"]) for p in processed_batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,"text":[p["text"] for p in processed_batch],"entities":[p["entities"] for p in processed_batch]}
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    

