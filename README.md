# QwenNER — 基于 Qwen 大模型的命名实体识别

使用 Qwen2.5-7B-Instruct 模型，通过 QLoRA / LoRA 在 BC2GM数据集上进行命名实体识别（NER）。

## 项目结构

```
QwenNER/
├── data/                  # 数据集
│   ├── train.json         # 训练集 
│   ├── dev.json           # 验证集 
│   ├── test.json          # 测试集 
│   └── labels.json        # 实体类别
├── args/                  # 训练配置文件
│   ├── arg1.json          
├── model.py               # 模型定义 (Qwen4NER 类)
├── trainer.py             # 训练主入口 + Trainer 类
├── MyDataset.py           # 数据集与 collate 函数
├── utils.py               # 工具: Arguments, Metrics, EarlyStop
├── predict.py             # 推理脚本
├── analysis.py            # 序列长度分布分析
├── template.py            # Qwen 对话模板定义
├── requirements.txt       # 依赖列表

```



### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 下载基础模型

```bash
# 下载 Qwen2.5-7B-Instruct
python download_model.py
```

### 3. 配置训练参数

编辑 `args/arg1.json`，完整参数：

```json
{
    "num_epochs": 4,
    "batch_size": 8,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "eps": 1e-8,
    "warmup_steps": 100,
    "max_length": 380,
    "max_new_tokens": 300,
    "device": "cuda:0",
    "dropout_rate": 0.2,
    "model_name": "Qwen2.5-7B-Instruct",
    "model_dir": "../autodl-tmp/model/Qwen2.5-7B-Instruct",
    "data_path": "./data/",
    "method": "qlora",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "patience": 10,
    "monitor": "val_f1",
    "delta": 0.0001,
    "save_dir": "../autodl-tmp/checkpoint/",
    "template_name": "qwen",
    "prompt": "You are an expert in biomedical named entity recognition..."
}
```

### 4. 训练

```bash
python trainer.py --arg ./args/arg1.json
```


### 5. 推理

```bash
python predict.py \
    --arg ./args/arg1.json \
    --weight ./checkpoint/exp1/ \
    --text "Using the same approach we have shown that hFIRE binds the stimulatory proteins Sp1 and Sp3 in addition to CBF"
```

输出示例：

```json
[{"entities": [{"name": "hFIRE", "type": "GENE"}, {"name": "Sp1", "type": "GENE"}, {"name": "Sp3", "type": "GENE"}, {"name": "CBF", "type": "GENE"}]}]
```

## 数据集


数据格式：

```json
{
    "sentence": "Comparison with alkaline phosphatases and 5 - nucleotidase",
    "entities": [
        {"name": "alkaline phosphatases", "type": "GENE", "pos": [16, 37]}
    ]
}
```


## 📈 评估指标

在实体级别进行精确匹配评估：

- **Precision**：预测正确的实体数 / 预测实体总数
- **Recall**：预测正确的实体数 / 真实实体总数
- **F1**：精确率和召回率的调和平均
- 支持按实体类型单独计算和 micro/macro 平均

### 实验结果
#### 使用了LoRA，注入 "q_proj","k_proj","v_proj","o_proj" r=16，α=32
|            | precision | recall   | f1       | support |
|------------|-----------|----------|----------|---------|
| GENE       | 0.834507  | 0.824609 | 0.829528 | 6323.0  |
| macro_avg  | 0.834507  | 0.824609 | 0.829528 | —       |
| micro_avg  | 0.834507  | 0.824609 | 0.829528 | —       |

#### 使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj" r=16，α=32

|            | precision | recall   | f1       | support |
|------------|-----------|----------|----------|---------|
| GENE       | 0.825159  | 0.819548 | 0.822344 | 6323.0  |
| macro_avg  | 0.825159  | 0.819548 | 0.822344 | —       |
| micro_avg  | 0.825159  | 0.819548 | 0.822344 | —       |
#### 使用了QLoRA，注入"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj" r=16，α=32

|            | precision | recall   | f1       | support |
|------------|-----------|----------|----------|---------|
| GENE       | 0.827024  | 0.818915 | 0.82295  | 6323.0  |
| macro_avg  | 0.827024  | 0.818915 | 0.82295  | —       |
| micro_avg  | 0.827024  | 0.818915 | 0.82295  | —       |
