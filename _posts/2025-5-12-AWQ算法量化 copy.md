---
layout: post
title: Minicpm导出onnx
---

总结了Minicpm导出onnx时的问题以及解决方法
### 1,torch.onnx.export:

特点：通用性强，适用于任何 PyTorch 模型，但需要手动配置输入、输出和动态轴。

### 2,optimum.onnxruntime.export_models:

特点：Hugging Face 提供的优化工具库，专为加速 Transformer 模型推理设计。

```python
#1,执行命令导出onnx：
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
#2，执行脚本推理：
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
outputs = model(**inputs)


```



### 3，Minicpm-o中的SiglipVisionTransformer 

- 它是一个视觉编码器，基于 Transformer 架构，用于将输入图像转换为特征嵌入，embeddings

  1获取设备和数据类型：

  - 从语言模型的嵌入层（self.llm.model.embed_tokens）获取 dtype 和 device。
  
  处理图像数据：
  
  - pixel_values_list 是图像列表，展平并转为 (batch_size, seq_len, channels) 的格式。
  - img_cnt 记录每个子列表中的图像数量。
  - tgt_sizes 转为张量并堆叠。
  
  填充图像序列：
  
  - 使用 torch.nn.utils.rnn.pad_sequence 将不同长度的图像序列填充到相同长度（max_patches）。
  
    导出ONNX报错，如下代码替换后解决：
  
- ```python
#这个算子不支持，使用如下代码替换后解决：
  def custom_pad_sequence(sequences, batch_first=False, padding_value=0):
    # 检查输入是否为空
      if not sequences:
          return torch.tensor([])
  
      # 计算最长序列的长度
      max_len = max([seq.size(0) for seq in sequences])
  
      # 获取除第一维外的其他维度
      trailing_dims = sequences[0].size()[1:]
  
      # 确定输出张量的形状
      if batch_first:
          out_dims = (len(sequences), max_len) + trailing_dims
      else:
          out_dims = (max_len, len(sequences)) + trailing_dims
  
      # 创建输出张量并用padding_value填充
      out_tensor = sequences[0].new_full(out_dims, padding_value)
  
      # 填充每个序列
      for i, tensor in enumerate(sequences):
          length = tensor.size(0)
          if batch_first:
              out_tensor[i, :length, ...] = tensor
          else:
              out_tensor[:length, i, ...] = tensor
  
      return out_tensor
  ```
  
  
  
- 重塑为 (batch_size, channels, height, width)。

生成注意力掩码：

- patch_attn_mask：布尔张量，标记有效 patch（根据 tgt_sizes 计算）。

分批处理图像：

- 如果图像数量超过 vision_batch_size，分批调用视觉模块 self.vpm（SiglipVisionTransformer）。
- 得到 vision_embedding，形状为 (batch_size, num_patches, hidden_size)。

投影到语言空间：

- 通过 self.resampler（多头注意力机制）将视觉特征投影到语言模型的嵌入维度。

分割视觉隐藏状态：

- 根据 img_cnt 将 vision_embedding 分割为每个图像组的隐藏状态，存入 vision_hidden_states。

无图像时的处理：

- 训练模式下生成虚拟图像特征（dummy_feature），推理模式下返回空列表。





### 4，Minicpm-o中Q,K,V计算过程

```python
past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]
#结构：

past_key_values 是一个列表，长度为 self.tts.config.num_hidden_layers（模型的隐藏层数，即解码器层数）。

每个元素是一个元组，包含两个张量：(key_tensor, value_tensor)。

每个张量由 torch.zeros 创建，表示全零张量，用于初始化该层的键和值缓存。
#张量形状：
[1, self.tts.config.num_attention_heads, condition_length - 1, self.tts.config.hidden_size // self.tts.config.num_attention_heads]




```

### 5，由于transformers版本问题需要past_key_values转DynamicCache

```
return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
```

### 6，onnx转化成TRT

```text
trtexec --onnx=/home/huayu/data/project/vision_test2.onnx \
        --shapes=pixel_values:1x3x14x14448 \
        --saveEngine=/home/huayu/data/project/vision_module2.trt \
        --fp32
```



### 7，问题总结

完成：vision_resampler模型转成成onnx以及.trt,但是在推理时有问题,报错：输入张量的形状与模型期望的形状不匹配；

​			使用自定义custom_pad_sequence解决onnx不支持Exporting the operator 'aten::pad_sequence'问题；

待办：audio模型转换，注：apm模型forward函数中包含EncoderDecoderCache对象，在切换时可能需要展开，比较困难；

​			TTS模型转换，ConditionalChatTTS：导出模型范围，从大到小，还是递归（）；导出dvae模型时问题

​			凯哥的qwen模型结构定义没有变，可以直接使用save_pretrained直接导出；

​			

​			拆分：llama，

​						dvae：包括（DVAEDecoder，GFSQ），

​				GFSQ的_embed的		self.quantizer.get_output_from_indices(x)操作，转换onnx不支持；

```python
		# 修改原始实现使用 einx.get_at
        # all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)
        # print("all_codes1:", all_codes)
        # 替换为 torch.gather
        batch_size, seq_len, num_quantizers = indices.shape
        codes = []
        for q in range(num_quantizers):
            codebook = self.codebooks[q]  # [codebook_size, dim]
            idx = indices[..., q]  # [batch, seq_len]
            code = codebook[idx]   # [batch, seq_len, dim]
            codes.append(code)
        all_codes = torch.stack(codes, dim=0)  # [num_quantizers, batch, seq_len, dim]
```

​						2, torch 矩阵乘法return torch.mul(dec_out, self.coef, out=dec_out)，onnx的Mul 算子只支持两个输入（A * B），没有原地操作（out）的概念。

​								修改为return torch.mul(dec_out, self.coef)解决；



```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huayu/Downloads/TensorRT-10.9.0.34/lib
```

vocos：Vocos.decode 的 ISTFTHead 使用了复数计算；torch.onnx.export无法支持；

### 15服务器export_llama

```
1,进入容器：
eafed8caed4a   yuanzhuo_docker:latest                                                                                                  "/opt/nvidia/nvidia_…"   6 weeks ago    Exited (255) 4 days ago     25011-25041/tcp                                                                                                              clever_sanderson
2，

```



输入参数：

ASR：

```
python3 wenet/bin/export_onnx_gpu.py --config=/home/huayu/data/project/asr_model/wenetspeech_u2pp_conformer_exp/20220506_u2pp_conformer_exp_wenetspeech/train.yaml --checkpoint=/home/huayu/data/project/asr_model/wenetspeech_u2pp_conformer_exp/20220506_u2pp_conformer_exp_wenetspeech/final.pt --cmvn_file=$model_dir/global_cmvn --ctc_weight=0.5 --output_onnx_dir=$onnx_model_dir --fp16
cp $model_dir/words.txt $model_dir/train.yaml $onnx_model_dir/
```

### 问题记录：

1，主要是ONNX 不直接支持嵌套结构作为输入。输入只能是张量或张量列表，对于llama这类大模型的输入包括past_key_values的情况，现在只能自定义实现把嵌套结构和类对象转换成张量列表；

```python
class DecoderLayersWrapperLlama(nn.Module):
    def __init__(self, layers, config, norm):
        super().__init__()
        self.layers = layers
        self.config = config
        self.norm = norm
        self.rotary_emb = LlamaRotaryEmbedding(config=config).to("cuda")
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values_in,  # 仍然是 [key0, value0, key1, value1, ...] 的列表
        output_attentions=False,
        use_cache=True,
        cache_position= None,
    ):
        # 将 past_key_values_in 转换为 DynamicCache
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        past_key_values = DynamicCache()
        
        for i in range(self.config.num_hidden_layers):
            past_key_values.key_cache.append(past_key_values_in[i * 2])
            past_key_values.value_cache.append(past_key_values_in[i * 2 + 1])
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        kv_caches_out = []
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                # # 更新 KV 缓存
                # past_key_values = layer_outputs[1]
                kv_caches_out.extend([past_key_values.key_cache[i], past_key_values.value_cache[i]])
        hidden_states = self.norm(hidden_states)
        return hidden_states, *kv_caches_out
def export_decoders(llama, decoder_layers, llama_config, tts_config, dtype, args, model_name, llama_norm):
    onnx_file_name = os.path.join(args.out_dir, f"llama_success_level1.onnx")
    hidden_size = llama_config.hidden_size  # 768
    layer_num = len(decoder_layers)
    head_num = llama_config.num_attention_heads
    hidden_size1 = hidden_size // head_num
    batch = 1
    N = 1
    sumN = 303
    lastN = sumN - N
    decoder_layers_wrapper = DecoderLayersWrapperLlama(decoder_layers, llama_config, llama_norm)
    hidden_in = torch.randn([batch, N, hidden_size], dtype=dtype).to(args.device)
    attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)

    condition_length = (
        1 + tts_config.use_speaker_embedding * tts_config.num_spk_embs + tts_config.streaming_text_reserved_len + 1
    )
    past_key_values = [
        (
            torch.zeros(
                1,
                llama_config.num_attention_heads,
                condition_length - 1,
                llama_config.hidden_size // llama_config.num_attention_heads,
                dtype=dtype,
                device=args.device,
            ),
            torch.zeros(
                1,
                llama_config.num_attention_heads,
                condition_length - 1,
                llama_config.hidden_size // llama_config.num_attention_heads,
                dtype=dtype,
                device=args.device,
            ),
        )
        for _ in range(llama_config.num_hidden_layers)
    ]

    # run llama
    outputs = llama(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=hidden_in,
            use_cache=True,
            past_key_values = past_key_values
    )

    in_names = ["hidden_in", "attention_mask", "position_ids"]
    dynamic_axes = {
        'hidden_in': {1: 'N'},
        'attention_mask': {2: 'N', 3: "sumN"},
        "position_ids": {1: 'N'},
    }
    kv_caches_in = convert_to_past_key_values_in(past_key_values)
    out_names = ["hidden_out"]
    kv_cache_in_shape = [batch, head_num, lastN, hidden_size1]
    kv_cache_dyn_axes = {2: "lastSum"}
    if args.kv_cache_format == 1:
        kv_cache_in_shape = [batch, lastN, head_num, hidden_size1]
        kv_cache_dyn_axes = {1: "lastSum"}
    for i in range(layer_num):
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])
        dynamic_axes[f"past_key_in{i}"] = kv_cache_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = kv_cache_dyn_axes
    
    input_datas = (hidden_in, attention_mask, position_ids, kv_caches_in)
    torch.onnx.export(
        decoder_layers_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=False,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )
```

