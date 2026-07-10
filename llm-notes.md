# 预训练
# 指令微调
## LoRA
![alt text](img/image.png)

LoRA将预训练的参数冻结，只训练新增的参数A和B。

$A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad r \ll \min(d, k)$


# 人类反馈强化学习RHLF