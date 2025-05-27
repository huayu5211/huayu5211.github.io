---
layout: post
title: Transformer架构介绍
---

# Transformer架构介绍

## 1.整体结构
1. 通过tokenizer将输入文本转换为input_ids

2. 将input_ids转换为embedding并添加position encoding，得到最终的输入表示

3. 通过[MHA模块](https://zhida.zhihu.com/search?content_id=253530996&content_type=Article&match_order=1&q=MHA模块&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDA1MzYyNzMsInEiOiJNSEHmqKHlnZciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTM1MzA5OTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.N4P964mOrqEdzepLRiWG-_LSMYBQx4TbXBOdOMATvT4&zhida_source=entity)计算Q、K、V的注意力权重并加权求和

4. 通过[FFN模块](https://zhida.zhihu.com/search?content_id=253530996&content_type=Article&match_order=1&q=FFN模块&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDA1MzYyNzMsInEiOiJGRk7mqKHlnZciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTM1MzA5OTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.RnrfylUadonq4Nq5VAACRi9PTH1VZK2x2ZFr2DMj2-M&zhida_source=entity)进一步处理自注意力输出。

5. 添加残差模块并通过[LayerNorm](https://zhida.zhihu.com/search?content_id=253530996&content_type=Article&match_order=1&q=LayerNorm&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDA1MzYyNzMsInEiOiJMYXllck5vcm0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTM1MzA5OTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.v9QHIN8fJdtFSunfVQiLt2lxSl0cY2Cp5nPfVFqQI4U&zhida_source=entity)进行归一化

6. 堆叠多个transformer层。

7. 将最终hidden_states层映射到词表并计算概率分布。

### 第一步

获取输入句子的每一个单词的表示向量 **X**，**X**由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。

![image-20250527141635610](../assert/Transformer架构介绍.assets/image-20250527141635610.png)

### 第二步

将得到的单词表示向量矩阵传入 Encoder 中，经过 6个 Encoder block 后可以得到句子所有单词的编码信息矩阵 **C**，如下图。单词向量矩阵用 Xn×d 表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。

### 第三步

将 Encoder 输出的编码信息矩阵 **C**传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 **Mask (掩盖)** 操作遮盖住 i+1 之后的单词。

![image-20250527142611522](../assert/Transformer架构介绍.assets/image-20250527142611522.png)

## 2.输入

Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入。

## 3.自注意力机制（self-attn）

![image-20250527142756870](../assert/Transformer架构介绍.assets/image-20250527142756870.png)

**Multi-Head Attention**，是由多个 **Self-Attention**组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接用于防止网络退化，Norm 表示 [Layer Normalization](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=Layer+Normalization&zhida_source=entity)，用于对每一层的激活值进行归一化。

![image-20250527143222878](../assert/Transformer架构介绍.assets/image-20250527143222878.png)



在计算的时候需要用到矩阵Q（查询），K（键值），V（值）。实际中，self-attn接受的输入是(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而**Q,K,V**正是通过 Self-Attention 的输入进行线性变换得到的。

Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**(应该就是权重部分)计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

![image-20250527143754284](../assert/Transformer架构介绍.assets/image-20250527143754284.png)

计算矩阵**Q**和**K**每一行向量的内积，为了防止内积过大，因此除以 dk 的平方根。**Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以 KT ，1234 表示的是句子中的单词。

![image-20250527144117312](../assert/Transformer架构介绍.assets/image-20250527144117312.png)

得到QKT 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1.

得到 Softmax 矩阵之后可以和**V**相乘，得到最终的输出**Z**。

![image-20250527144210944](../assert/Transformer架构介绍.assets/image-20250527144210944.png)

Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 Z1 等于所有单词 i 的值 Vi 根据 attention 系数的比例加在一起得到，如下图所示：

![image-20250527144325792](../assert/Transformer架构介绍.assets/image-20250527144325792.png)



Multi-Head Attention 包含多个 Self-Attention 层，首先将输入**X**分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵**Z**。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵**Z**。

得到 8 个输出矩阵 Z1 到 Z8 之后，Multi-Head Attention 将它们拼接在一起 **(Concat)**，然后传入一个**Linear**层，得到 Multi-Head Attention 最终的输出**Z**。

![image-20250527144700809](../assert/Transformer架构介绍.assets/image-20250527144700809.png)



## 4.Encoder

负责处理输入序列，把输入“编码”成一组丰富的表示（向量），捕捉输入的语义和上下文。因此适用于机器翻译，**先翻译再生成**

Encoder block 结构，是由 Multi-Head Attention, **Add & Norm, Feed Forward, Add & Norm** 组成的。刚刚已经了解了 Multi-Head Attention 的计算过程，现在了解一下 Add & Norm 和 Feed Forward 部分。

### Add & Norm

![image-20250527145511901](../assert/Transformer架构介绍.assets/image-20250527145511901.png)



**X**表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(**X**) 和 FeedForward(**X**) 表示输出 (输出与输入 **X** 维度是一样的，所以可以相加)。

**Add**指 **X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分

**Norm**指 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛。

### Feed Forward

Feed Forward 层是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。

![image-20250527145637327](../assert/Transformer架构介绍.assets/image-20250527145637327.png)

**X**是输入，Feed Forward 最终得到的输出矩阵的维度与**X**一致。

### 组成encoder

Encoder block 接收输入矩阵 X(n×d) ，并输出一个矩阵 O(n×d) 。

第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block 输出的矩阵就是**编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。



## 5.Decoder



负责根据 Encoder 的输出和当前生成的部分序列，逐步生成目标序列。GPT 系列模型只使用 Decoder 结构

与 Encoder block 相似，但是存在一些区别：

- 包含两个 Multi-Head Attention 层。
- 第一个 Multi-Head Attention 层采用了 Masked 操作。
- 第二个 Multi-Head Attention 层的**K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 Decoder block 的输出计算。
- 最后有一个 Softmax 层计算下一个翻译单词的概率。

### 第一个 Multi-Head Attention

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。

![image-20250527150155094](../assert/Transformer架构介绍.assets/image-20250527150155094.png)

#### 第一步

是 Decoder 的输入矩阵和 **Mask** 矩阵，输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，**Mask** 是一个 5×5 的矩阵。在 **Mask** 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息![image-20250527150506830](../assert/Transformer架构介绍.assets/image-20250527150506830.png)

#### 第二步

接下来的操作和之前的 Self-Attention 一样，通过输入矩阵**X**计算得到**Q,K,V**矩阵。然后计算**Q**和 KT 的乘积 QKT 。

![image-20250527150525878](../assert/Transformer架构介绍.assets/image-20250527150525878.png)

#### 第三步

在得到 QKT 之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用**Mask**矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

![image-20250527150549823](../assert/Transformer架构介绍.assets/image-20250527150549823.png)

得到 **Mask** QKT 之后在 **Mask** QKT上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。

#### 第四步

使用 **Mask** QKT与矩阵 **V**相乘，得到输出 **Z**，则单词 1 的输出向量 Z1 是只包含单词 1 信息的。

![image-20250527150635576](../assert/Transformer架构介绍.assets/image-20250527150635576.png)

#### 第五步

通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Zi ，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出Zi 然后计算得到第一个 Multi-Head Attention 的输出**Z**，**Z**与输入**X**维度一样。

### 第二个 Multi-Head Attention

Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的 **K, V**矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 **Encoder 的编码信息矩阵 C** 计算的。

根据 Encoder 的输出 **C**计算得到 **K, V**，根据上一个 Decoder block 的输出 **Z** 计算 **Q** (如果是第一个 Decoder block 则使用输入矩阵 **X** 进行计算)，后续的计算方法与之前描述的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 **Mask**)。

Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如
## 训练和预测

训练时：第i个decoder的输入 = encoder输出 + ground truth embeding
预测时：第i个decoder的输入 = encoder输出 + 第(i-1)个decoder输出

训练时因为知道ground truth embeding，相当于知道正确答案，网络可以一次训练完成。
预测时，首先输入start，输出预测的第一个单词 然后start和新单词组成新的query，再输入decoder来预测下一个单词，循环往复 直至end

## Decoder—Only

​	1，decoder-only，采用**自回归方法**来逐个token生成输出；先prefill初始化第一个输入token的QKV-cache。然后进入decode阶段，根据当前的输入token，复用kv-cache逐个生成下一个token，	并使用新得qkv-cache更新；

## deepseek架构

1，模型层面主要是：模型结构上，采用MLA代替MHA, 减少了KV-cache的大小，减少了内存占用；采用MoE架构，在与dense模型的同等参数量级下，减少计算量。 模型压缩上，主要是采用FP8量化权重，同时对KV-cache也进行了量化，配合支持FP8计算的GPU架构，可以减少内存占用的同时，提升计算效率。

系统层面主要是：训练MTP模块用于投机采样，能够提升1.8倍decoding效率；采用PD分离的架构，配合一系列并行策略、通信优化、专家负载均衡、overlap计算与通性等手段，共同提升系统的推理效率。

​		2，MOE架构构成：共享专家（总机接待员（Gate）会根据来电者的需求将电话路由到相应的专业部门）和路由专家（负责处理特定类型的输入），

​				MOE流程：1，路由，计算每个词与各个专家的得分。

​								    2，选择，gate网格为每个词选择top-k个最合适专家；

​									3，专家处理，被选中的路由专家以及共享专家会对输入进行处理，提取特										征

​									4，加权与聚合， 每个路由专家的输出会根据 Gate 网络给出的权重进行加										权，然后与共享专家的输出进行聚合，形成 MoE 层的最终输出

​				案例：“我非常高兴地收到了一个礼物” ，“我”会被分配到实体专家，“非常高兴会被分配到情感专家”，收到会被分配到语义词义专家；

3，GRPO算法通过构建图结构的知识索引和采用检索-推理-剪枝机制，显著提升了多跳问答任务的性能，复杂的问答任务时，GRPO首先通过词汇或语义相似性检索初始文本片段，由图结构和LLM推理共同引导