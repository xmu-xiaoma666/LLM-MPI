## LLM-MPI: Multi Prompt Inference, touching the performance upper limit of large language models



### Introduction

In this post, we present a method called Multi-Prompt Inference, which allows users to interact with large language models (LLMs) without the need for extensive retraining of the model. This approach aims to lower the barrier for users in editing prompts while maximizing the effectiveness of their queries. For additional code examples and implementation details, see: [https://github.com/xmu-xiaoma666/LLM-MPI](https://github.com/xmu-xiaoma666/LLM-MPI).


### Background

When utilizing large language models, it has become evident that different prompts can lead to significant variations in performance. Crafting an optimal prompt is thus a critical step in querying these powerful models. However, for most users, LLMs function as a black box; they often lack insight into how to establish effective prompts that yield the best results. This situation presents a challenge: despite the robust capabilities of LLMs, ordinary users often struggle to achieve their full potential.

Given the intricacies involved in prompt engineering, we explore whether there exists a method that enables users—regardless of their technical expertise—to obtain improved outcomes when interacting with LLMs.


## Requirments

- Python >= 3.8
- PyTorch >= 1.12.0 (2.0.0 and above are recommended)
- Transformers >= 4.38


### Usage

**Single Prompt Inference:**

```
python generate_loigits_InternLM_sentence_cls.py \
	--user_question "如果一个人从房子里走出来，他是在室内还是在室外？"
```


**Multi Prompt Inference:**

```
python generate_loigits_InternLM_ensemble_cls.py \
	--user_question "如果一个人从房子里走出来，他是在室内还是在室外？"
```


### Experiments

#### LLM Inference

| 问题 | 单prompt推理 | 多prompt推理 |
| --- | --- | --- |
| **9.11和9.9哪个大？** | 9.11比9.9大。9.11是9.1后面再添加一个小数点和一个1，而9.9则是9后面再添加一个小数点和9。所以9.11比9.9大。 | 9.11和9.9都是小数，我们可以通过比较它们的整数部分和小数部分来确定哪个数更大。首先，观察整数部分，两个数都是9，所以整数部分相同。接下来，比较小数部分。9.11的小数部分是0.11，而9.9的小数部分是0.9。因为0.9大于0.11，所以9.9大于9.11。因此，9.9比9.11大。 |
| **如果一个人从房子里走出来，他是在室内还是在室外？** | 如果一个人从房子里走出来，他是在室外。这是因为“走出”意味着他已经从房子的内部位置转移到了外部位置。当一个人走出房子时，他不再处于房子内的空间，而是在房子的外部，即室外。 | 这个问题的答案是：这个人现在处于室外。解释：问题中提到“一个人从屋内走出”，这意味着他之前在室内，现在他“走出”，即离开了室内空间，进入了室外环境。因此，根据这个描述，可以确定他现在处于室外。 |
| **一张纸的厚度是0.1毫米，折叠10次后，纸的厚度是多少？** | 好的,让我们一步步来计算: 1. 初始厚度为0.1毫米。 2. 每次折叠后,厚度翻倍。 3. 第一次折叠后,厚度为0.1毫米 * 2 = 0.2毫米。 4. 第二次折叠后,厚度为0.2毫米 * 2 = 0.4毫米。 5. 以此类推,直到第10次折叠。所以,第10次折叠后,厚度为0.1毫米 * 2^10 = 0.1毫米 * 1024 = 102.4毫米。答案:折叠10次后,纸的厚度为102.4毫米 | 为了解决这个问题，我们可以使用指数增长的概念。每次折叠，纸张的厚度都会翻倍，这意味着厚度是按照2的幂次增长的。初始厚度是0.1毫米，折叠一次后厚度变为0.1 * 2 = 0.2毫米，折叠两次后厚度变为0.2 * 2 = 0.4毫米，以此类推。因此，折叠n次后的厚度可以表示为：厚度 = 初始厚度 * 2^n。将n=10代入上述公式，得到：厚度 = 0.1 * 2^10 = 0.1 * 1024 = 102.4毫米。因此，折叠10次后，纸张的厚度是102.4毫米。答案是102.4毫米。 |
| **一个长方形的周长为24厘米，宽是长的二分之一。这个长方形的长和宽分别是多少？** | 设长方形的长为$x$厘米，则宽为$0.5x$厘米。根据周长的计算公式，长方形的周长为$2 \times (长 + 宽)$。所以，$2 \times (x + 0.5x) = 24$。化简得 $2 \times 1.5x = 24$。进一步化简得 $3x = 24$。解方程得 $x = 24 / 3 = 8$。所以，长为$8$厘米，宽为$0.5 \times 8 = 4$厘米。答案：这个长方形的长是$8$厘米，宽是$4$厘米。 | 设长方形的长为$L$厘米，宽为$W$厘米。根据题意，我们有以下两个方程：1. 周长方程：$2(L + W) = 24$ 2. 宽度是长的一半：$W = \frac{1}{2}L$。我们可以将第二个方程代入第一个方程中求解。首先，将$W = \frac{1}{2}L$代入周长方程中：$$2(L + \frac{1}{2}L) = 24$$。化简得：$$2(\frac{3}{2}L) = 24$$。$$3L = 24$$。解得：$$L = \frac{24}{3} = 8$$。所以，长方形的长为8厘米。接下来，根据$W = \frac{1}{2}L$，我们可以求出宽：$$W = \frac{1}{2} \times 8 = 4$$。所以，长方形的宽为4厘米。答案是长为8厘米，宽为4厘米。 |
| **在一次考试中，学生们的平均分是80分。如果有一名新学生的分数是100分，新的平均分是什么？（假设总共有n名学生）** | 要计算新的平均分，我们需要先了解原始总分数和新的总分数。设原始总分数为 $S$，那么原始平均分为 $S/n$。已知原始平均分是80分，所以我们有：$$ \frac{S}{n} = 80 $$ $$ S = 80n $$。现在加入一名新学生，分数为100分，所以新的总分数为 $S + 100$，新的学生总数为 $n + 1$。新的平均分为：$$ \text{新平均分} = \frac{S + 100}{n + 1} $$。将 $S = 80n$ 代入上式，得：$$ \text{新平均分} = \frac{80n + 100}{n + 1} $$。因此，新的平均分是 $\boxed{\frac{80n + 100}{n + 1}}$。 | 解：设原来有 n 名学生，平均分为 80 分，则总分为 $80n$ 分。加入一名得分为 100 分的学生后，总人数变为 $n+1$，总分为 $80n + 100$。新的平均分为 $\frac{80n + 100}{n + 1}$。将 $80n + 100$ 分解为 $80n + 100 = 80n + 80 + 20 = 80(n + 1) + 20$，则新的平均分为 $\frac{80(n + 1) + 20}{n + 1} = 80 + \frac{20}{n + 1}$。因此，新的平均分为 $80 + \frac{20}{n + 1}$。 |


#### Sa2VA Inference on MeVIS

|  | J&F | F | J |
| --- | --- | --- | --- |
| 单prompt推理 | 0.6140498200029082 | 0.6618922969159698 | 0.5662073430898465 |
| 多prompt推理（11个prompt） | 0.6323577165160168 | 0.6761449317570541 | 0.5885705012749795 |
