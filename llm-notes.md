# 预训练
# 指令微调
## LoRA
![alt text](img/image.png)

LoRA将预训练的参数冻结，只训练新增的参数A和B。

其中

$$
A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad r \ll \min(d, k)
$$

前向传播过程变为：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$
### QLoRA
QLoRA在微调之前，先对预训练模型进行量化，再进行微调。
对预训练模型进行量化。
首先分组将这个模型的参数分为r组，每组包含k个参数。找出每个组的参数的最大值，将参数放缩到【-1, 1】之间然后变化为量化后的参数。
$$ q_i = \frac{1}{2} \left( Q_X \left( \frac{i}{2^k + 1} \right) + Q_X \left( \frac{i+1}{2^k + 1} \right) \right) $$
其中 $Q_X$ 是分位数函数
# 人类反馈强化学习RHLF