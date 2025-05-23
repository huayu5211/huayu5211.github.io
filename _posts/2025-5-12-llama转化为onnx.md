---
layout: post
title: llama转化为onnx
---

### 主要是ONNX 不直接支持嵌套结构作为输入。输入只能是张量或张量列表，对于llama这类大模型的输入包括past_key_values的情况，现在只能自定义实现把嵌套结构和类对象转换成张量列表；

```python
import os
import argparse
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModel
from transformers.cache_utils import DynamicCache
import numpy as np
from transformers.cache_utils import DynamicCache
from transformers import AutoModel
import torch
from typing import Tuple, Optional, Union, List
def export_llamas(llama, tts_config, dtype, args, model_name):
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}_test.onnx")
    hidden_size = tts_config.hidden_size  # 768
    layer_num = len(decoder_layers)
    head_num = tts_config.num_attention_heads
    hidden_size1 = hidden_size // head_num
    batch = 1
    N = 1
    sumN = 32
    lastN = sumN - N
    decoder_layers_wrapper = DecoderLayersWrapperLlama(decoder_layers, tts_config)
    hidden_in = torch.randn([batch, N, hidden_size], dtype=dtype).to(args.device)
    attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)

    # run llama
    outputs = llama(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=hidden_in,
            use_cache=True
    )


    in_names = ["hidden_in", "attention_mask", "position_ids"]
    dynamic_axes = {
        'hidden_in': {1: 'N'},
        'attention_mask': {2: 'N', 3: "sumN"},
        "position_ids": {1: 'N'},
    }
    kv_caches_in = []
    out_names = ["hidden_out"]
    kv_cache_in_shape = [batch, head_num, lastN, hidden_size1]
    kv_cache_dyn_axes = {2: "lastSum"}
    if args.kv_cache_format == 1:
        kv_cache_in_shape = [batch, lastN, head_num, hidden_size1]
        kv_cache_dyn_axes = {1: "lastSum"}
    for i in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
        kv_caches_in.extend([past_key_in, past_value_in])
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
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )
def get_true_length_past_key_values(past_key_values_in):
    """
    判断 past_key_values_in 的真实长度。
    真实长度定义为：从开头到第一个键值对（key 和 value 张量全为 0）的索引。
    如果没有全 0 的键值对，返回整个列表长度。

    Args:
        past_key_values_in: List[torch.Tensor], 格式为 [key0, value0, key1, value1, ...]

    Returns:
        int: past_key_values_in 的真实长度（键值对的数量）
    """
    # 检查输入长度是否为偶数
    if len(past_key_values_in) % 2 != 0:
        raise ValueError(f"past_key_values_in length must be even, got {len(past_key_values_in)}")
    
    num_layers = len(past_key_values_in) // 2
    
    # 遍历每层的键值对
    for i in range(num_layers):
        key_tensor = past_key_values_in[i * 2]
        value_tensor = past_key_values_in[i * 2 + 1]
        
        # 检查 key 和 value 是否全为 0
        if torch.all(key_tensor == 0) and torch.all(value_tensor == 0):
            return i  # 返回当前层的索引作为真实长度
    
    # 如果没有全 0 的键值对，返回总层数
    return num_layers

def convert_to_past_key_values_in(past_key_values):
    past_key_values_in = []
    
    # 遍历每层的键值对
    for key, value in past_key_values:
        # 验证键和值是张量
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected key and value to be torch.Tensor, got {type(key)} and {type(value)}")
        
        # 添加到扁平化列表
        past_key_values_in.extend([key, value])
    
    return past_key_values_in
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
    onnx_file_name = os.path.join(args.out_dir, f"llama_success_off.onnx")
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

import onnxruntime as ort

def To_onnx(args):
    device = args.device
    dtypes_config = {
        "fp32": False,
        "fp16": False,
        "bf16": False,
    }
    if args.dtype == "float32":
        dtype = torch.float32
        dtypes_config["fp32"] = True
    elif args.dtype == "float16":
        dtype = torch.float16
        dtypes_config["fp16"] = True
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
        dtypes_config["bf16"] = True

    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if not args.model_type:
        model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=dtype).eval().to(device)  # 移动到 GPU
    elif args.model_type == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device, **dtypes_config, trust_remote_code=True).eval()

    print(f"finish load model from {args.model_path}")
    # model.tts.model
    layers = model.tts.model.layers
    llama = model.tts.model
    llama_config = model.tts.model.config
    tts_config = model.tts.config
    llama_norm = model.tts.model.norm
    export_decoders(llama, layers, llama_config, tts_config, dtype, args, "decoders", llama_norm)
        # print("PyTorch outputs:", outputs)  # 假设输出是 hidden_states

def infer_llama(args):
    device = args.device
    dtypes_config = {
        "fp32": False,
        "fp16": False,
        "bf16": False,
    }
    if args.dtype == "float32":
        dtype = torch.float32
        dtypes_config["fp32"] = True
    elif args.dtype == "float16":
        dtype = torch.float16
        dtypes_config["fp16"] = True
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
        dtypes_config["bf16"] = True

    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if not args.model_type:
        model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=dtype).eval().to(device)  # 移动到 GPU
    elif args.model_type == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device, **dtypes_config, trust_remote_code=True).eval()

    print(f"finish load model from {args.model_path}")
    # model.tts.model
    config = model.tts.model.config
    tts_config = model.tts.config
    llama_config = model.tts.model.config
    decoder_layers = model.tts.model.layers
    if not args.model_type:
        # default configure for llama like models
        # lm_head_model = model.lm_head
        # embeding_model = model.model.embed_tokens
        # norm_model = model.model.norm
        model = model.tts.model
        # 获取模型配置
        num_layers = config.num_hidden_layers  # 获取层数
        hidden_size = config.hidden_size  # 获取隐藏维度
        num_heads = config.num_attention_heads  # 获取注意力头数
        head_dim = hidden_size // num_heads  # 计算每个头的维度

        # 构造 PyTorch 输入
        batch_size = 1
        seq_len = 1  # 假设单步推理
        past_seq_len = 302  # 过去序列长度（从 kv_cache 形状推测）

        hidden_in_torch = torch.randn([batch_size, seq_len, hidden_size], dtype=dtype, device=device)
        attention_mask_torch = torch.ones([batch_size, seq_len + past_seq_len], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        position_ids_torch = torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.int64, device=device).unsqueeze(0)
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
        kv_caches_in = convert_to_past_key_values_in(past_key_values)
        # 运行 PyTorch 模型推理
        if not args.model_type:
            llama = model  # 假设 model 直接是 Transformer 模型
            outputs = llama(
                inputs_embeds=hidden_in_torch,
                attention_mask=attention_mask_torch,
                position_ids=position_ids_torch,
                past_key_values=past_key_values,
            )
            decoder_layers_wrapper = DecoderLayersWrapperLlama(decoder_layers, llama_config, llama.norm)
            with torch.no_grad():
                pytorch_outputs = decoder_layers_wrapper(hidden_in_torch, attention_mask_torch, position_ids_torch, kv_caches_in)


            print("PyTorch outputs:", outputs)  # 假设输出是 hidden_states
            print("Output type:", type(outputs))
            print("Hidden states shape:", outputs[0].shape)
            print("Key-value shapes:", [(k.shape, v.shape) for k, v in outputs[1]])

            print("decoder_layers_wrapper outputs:", pytorch_outputs)  # 假设输出是 hidden_states
        print("ONNX session creation start")
        # 构造 ONNX 输入
        try:
            session = ort.InferenceSession(
                "/llama_success_off.onnx",
                providers=["CUDAExecutionProvider"],
                provider_options={"graph_optimization_level": "ORT_DISABLE_ALL"}  # 禁用图优化
            )
        except Exception as e:
            print("ONNX session creation failed:", e)
            exit(1)
        print("ONNX session creation success")
        input_names = [input.name for input in session.get_inputs()]
        # print("ONNX input names:", input_names)

        # 构造 ONNX 输入，匹配模型输入名称
        inputs_onnx = {
            "hidden_in": hidden_in_torch.cpu().numpy().astype(np.float16 if dtype == torch.float16 else np.float32),
            "attention_mask": attention_mask_torch.cpu().numpy().astype(np.float16 if dtype == torch.float16 else np.float32),
            "position_ids": position_ids_torch.cpu().numpy().astype(np.int64)
        }
       
        for i in range(num_layers):
            inputs_onnx[f"past_key_in{i}"] = past_key_values[i][0].cpu().numpy().astype(np.float16 if dtype == torch.float16 else np.float32)
            inputs_onnx[f"past_value_in{i}"] = past_key_values[i][1].cpu().numpy().astype(np.float16 if dtype == torch.float16 else np.float32)

        # 运行 ONNX 推理
        outputs_onnx = session.run(None, inputs_onnx)
        print("ONNX outputs:", outputs_onnx)

        # 比较输出
        pytorch_last_hidden_state_shape = outputs[0].shape
        onnx_last_hidden_state_shape = outputs_onnx[0].shape
        print(f"PyTorch last_hidden_state shape: {pytorch_last_hidden_state_shape}")
        print(f"ONNX last_hidden_state shape: {onnx_last_hidden_state_shape}")

        print(f"PyTorch last_hidden_state: {outputs[0]}")
        print(f"ONNX last_hidden_state: {outputs_onnx[0]}")

        pytorch_past_key_values = outputs[1]
        onnx_past_key_values = outputs_onnx[1]
        print(f"PyTorch pytorch_past_key_values: {pytorch_past_key_values}")
        print(f"ONNX onnx_past_key_values: {onnx_past_key_values}")

def export_llama(args):
    device = args.device
    dtypes_config = {
        "fp32": False,
        "fp16": False,
        "bf16": False,
    }
    if args.dtype == "float32":
        dtype = torch.float32
        dtypes_config["fp32"] = True
    elif args.dtype == "float16":
        dtype = torch.float16
        dtypes_config["fp16"] = True
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
        dtypes_config["bf16"] = True

    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if not args.model_type:
        model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=dtype,
            low_cpu_mem_usage=True).eval().to(device)  # 移动到 GPU
    elif args.model_type == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device, **dtypes_config, trust_remote_code=True).eval()

    print(f"finish load model from {args.model_path}")
    # model.tts.model
    config = model.tts.model.config
    decoder_layers = model.tts.model.layers
    session = ort.InferenceSession("data2/weiwenjie/yuanzhuo/MiniCPM-o/huayu_test/llama_finish.onnx", providers=["CUDAExecutionProvider"])
    if not args.model_type:
        # default configure for llama like models
        # lm_head_model = model.lm_head
        # embeding_model = model.model.embed_tokens
        # norm_model = model.model.norm
        model = model.tts.model
        
    elif args.model_type == "Qwen":
        # support alibaba Qwen
        lm_head_model = model.lm_head
        embeding_model = model.transformer.wte
        norm_model = model.transformer.ln_f
        decoder_layers = model.transformer.h
        args.kv_cache_format = 1
    else:
        raise ValueError("invalid model_type")

    # print(f"begin export_lm_head")
    # export_lm_head(lm_head_model, config, dtype, args, "lm_head")

    # print(f"begin export_embeding")
    # export_embeding(embeding_model, config, args, "embeding")

    # print(f"begin export_norm")
    # export_norm(norm_model, config, dtype, args, "norm")

    print(f"begin export_decoders")
    decoder_pack_size = args.decoder_pack_size
    if decoder_pack_size <= 0:
        # export decoders as one onnx models
        export_decoders(decoder_layers, config, dtype, args, "decoders")
    else:
        # export decoders to multiple onnx models
        decoder_num = len(decoder_layers)
        export_model_num = (decoder_num + decoder_pack_size - 1) // decoder_pack_size

        for i in range(export_model_num):
            layers = decoder_layers[i * decoder_pack_size:(i + 1) * decoder_pack_size]
            export_decoders(layers, config, dtype, args, f"decoders_{i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export llama',
    )
    parser.add_argument('-m', '--model_path', required=False, type=str, default="/home/huayu/data/root/model/MiniCPM-o-2_6")
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=17)
    parser.add_argument('-d', '--device', required=False, type=str, default="cuda")
    # supported dtype: ["float32", "float16", "bfloat16"]
    parser.add_argument('-p', '--dtype', required=False, type=str, default="float16")
    # 0: export all decoders into one onnx. >0: export multiple onnx files, and each onnx has decoder_pack_size layers
    parser.add_argument('--decoder_pack_size', required=False, type=int, default=0)
    # 0 means [batch, head, seq, hidden], 1 means [batch, seq, head, hidden]
    parser.add_argument('--kv_cache_format', required=False, type=int, default=0)
    # default model_type is llama_hf, other model_type such as Qwen, Baichuan ban be supported
    parser.add_argument('--model_type', required=False, type=str, default="")

    args = parser.parse_args()

    if args.dtype not in ["float32", "float16", "bfloat16"]:
        raise ValueError("dtype is invalid")
    infer_llama(args)
    # export_llama(args)
```

