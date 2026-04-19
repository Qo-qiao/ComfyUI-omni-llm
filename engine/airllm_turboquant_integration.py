# -*- coding: utf-8 -*-
"""
AirLLM + TurboQuant 协同加速模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import torch
import torch.nn as nn
import math
import time
from typing import Optional, Tuple, Dict, List, Any, Union
from functools import lru_cache

import sys
import os

from .turboquant_module import (
    TurboQuantCompressorV2,
    TurboQuantManager,
    get_global_manager,
    LloydMaxCodebook,
    generate_rotation_matrix,
    generate_qjl_matrix,
)


class AirLLMTurboQuantConfig:
    """AirLLM + TurboQuant 集成配置"""

    def __init__(
        self,
        enabled: bool = True,
        kv_compression_bits: int = 3,
        kv_seed: int = 42,
        enable_asymmetric_attention: bool = True,
        compress_kv: bool = True,
        device: str = "cuda",
    ):
        self.enabled = enabled
        self.kv_compression_bits = kv_compression_bits
        self.kv_seed = kv_seed
        self.enable_asymmetric_attention = enable_asymmetric_attention
        self.compress_kv = compress_kv
        self.device = device


class TurboQuantKVCacheManager:
    """
    TurboQuant KV Cache 管理器

    管理多头注意力的 KV Cache 压缩，
    支持非对称注意力计算（直接用压缩后的 K 计算注意力分数）
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        bits: int = 3,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bits = bits
        self.seed = seed
        self.device = device

        self.compressors: Dict[int, TurboQuantCompressorV2] = {}
        for head_idx in range(num_heads):
            self.compressors[head_idx] = TurboQuantCompressorV2(
                head_dim=head_dim,
                bits=bits,
                seed=seed + head_idx,
                device=device,
            )

        self.k_cache_compressed: List[Dict] = []
        self.v_cache_compressed: List[Dict] = []

    def compress_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[Dict, Dict]:
        """
        压缩 K 和 V 张量

        Args:
            k: [batch, num_heads, seq_len, head_dim] - Key 张量
            v: [batch, num_heads, seq_len, head_dim] - Value 张量

        Returns:
            compressed_k, compressed_v: 压缩后的字典
        """
        if k.device.type != self.device:
            k = k.to(self.device)
            v = v.to(self.device)

        batch_size, num_heads, seq_len, head_dim = k.shape

        compressed_k_list = []
        compressed_v_list = []

        for head_idx in range(num_heads):
            # 保持张量形状为 [batch, 1, seq_len, head_dim]
            k_head = k[:, head_idx:head_idx+1, :, :]
            v_head = v[:, head_idx:head_idx+1, :, :]

            k_compressed = self.compressors[head_idx].compress(k_head)
            v_compressed = self.compressors[head_idx].compress(v_head)

            compressed_k_list.append(k_compressed)
            compressed_v_list.append(v_compressed)

        return compressed_k_list, compressed_v_list

    def asymmetric_attention_score(
        self, q: torch.Tensor, compressed_k: List[Dict]
    ) -> torch.Tensor:
        """
        使用非对称注意力计算注意力分数

        直接从压缩的 K 计算 <Q, K>，无需解压缩

        Args:
            q: [batch, num_heads, query_len, head_dim] - Query 张量
            compressed_k: 压缩后的 K 列表

        Returns:
            scores: [batch, num_heads, query_len, key_len] - 注意力分数
        """
        if q.device.type != self.device:
            q = q.to(self.device)

        batch_size, num_heads, query_len, head_dim = q.shape

        all_scores = []

        for head_idx in range(num_heads):
            q_head = q[:, head_idx, :, :].unsqueeze(1)
            scores = self.compressors[head_idx].asymmetric_attention_scores(
                q_head, compressed_k[head_idx]
            )
            all_scores.append(scores)

        scores = torch.cat(all_scores, dim=1)
        return scores

    def get_decompressed_kv(
        self, compressed_k: List[Dict], compressed_v: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解压缩获取原始 K, V（用于需要完整值的场景）

        Args:
            compressed_k: 压缩后的 K 列表
            compressed_v: 压缩后的 V 列表

        Returns:
            k: [batch, num_heads, seq_len, head_dim]
            v: [batch, num_heads, seq_len, head_dim]
        """
        k_heads = []
        v_heads = []

        for head_idx in range(self.num_heads):
            k_mse = compressed_k[head_idx]["k_mse"]
            k_heads.append(k_mse)

            v_mse = compressed_v[head_idx]["k_mse"]
            v_heads.append(v_mse)

        k = torch.stack(k_heads, dim=1)
        v = torch.stack(v_heads, dim=1)

        return k, v


class AirLLMTurboQuantWrapper:
    """
    AirLLM + TurboQuant 协同加速包装器

    该类将 TurboQuant 的 KV Cache 压缩集成到 LLM 推理流程中，
    通过以下方式加速推理：

    1. KV Cache 压缩: 将 K, V 张量压缩存储，减少显存占用
    2. 非对称注意力: 直接用压缩数据计算注意力，避免完整解压缩
    3. 分层加载: 复用 AirLLM 的分层加载机制
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        config: Optional[AirLLMTurboQuantConfig] = None,
    ):
        self.config = config or AirLLMTurboQuantConfig()
        self.device = self.config.device

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.kv_manager = TurboQuantKVCacheManager(
            num_heads=num_heads,
            head_dim=head_dim,
            bits=self.config.kv_compression_bits,
            seed=self.config.kv_seed,
            device=self.device,
        )

        self.past_kv_compressed: Optional[List[Tuple[Dict, Dict]]] = None
        self.past_length: int = 0

    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        更新 KV Cache（压缩新产生的 K, V）

        Args:
            k: [batch, num_heads, new_seq_len, head_dim]
            v: [batch, num_heads, new_seq_len, head_dim]
        """
        compressed_k, compressed_v = self.kv_manager.compress_kv(k, v)

        if self.past_kv_compressed is None:
            self.past_kv_compressed = [(compressed_k, compressed_v)]
        else:
            self.past_kv_compressed.append((compressed_k, compressed_v))

        self.past_length += k.shape[2]

    def compute_attention_asymmetric(
        self,
        q: torch.Tensor,
        k_compressed: List[Dict],
    ) -> torch.Tensor:
        """
        使用非对称注意力计算（无需解压缩 K）

        Args:
            q: [batch, num_heads, query_len, head_dim]
            k_compressed: 压缩后的 K 历史

        Returns:
            attention_scores: [batch, num_heads, query_len, key_len]
        """
        return self.kv_manager.asymmetric_attention_score(q, k_compressed)

    def compute_attention_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        标准注意力计算（解压缩后计算）

        Args:
            q: [batch, num_heads, query_len, head_dim]
            k: [batch, num_heads, key_len, head_dim]
            v: [batch, num_heads, key_len, head_dim]
            attention_mask: 可选的注意力掩码

        Returns:
            attention_output: [batch, num_heads, query_len, head_dim]
        """
        scale = 1.0 / math.sqrt(self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output

    def get_full_kv_compressed(self) -> Tuple[List[Dict], List[Dict]]:
        """获取所有压缩的 KV"""
        if self.past_kv_compressed is None:
            return [], []

        all_k = []
        all_v = []

        for k_comp, v_comp in self.past_kv_compressed:
            all_k.append(k_comp)
            all_v.append(v_comp)

        return all_k, all_v

    def reset(self):
        """重置 KV Cache"""
        self.past_kv_compressed = None
        self.past_length = 0

    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用统计

        Returns:
            dict: 包含原始大小、压缩后大小、压缩比等
        """
        if self.past_kv_compressed is None:
            return {
                "original_bits": 0,
                "compressed_bits": 0,
                "compression_ratio": 1.0,
            }

        total_original = self.num_heads * self.past_length * self.head_dim * 16
        total_compressed = 0

        for k_comp, _ in self.past_kv_compressed:
            for head_idx in range(self.num_heads):
                k_mse_bits = k_comp[head_idx]["k_mse"].numel() * self.config.kv_compression_bits
                qjl_bits = k_comp[head_idx]["qjl_signs"].numel() * 1
                norm_bits = k_comp[head_idx]["residual_norm"].numel() * 16
                total_compressed += k_mse_bits + qjl_bits + norm_bits

        return {
            "original_bits": total_original,
            "compressed_bits": total_compressed,
            "compression_ratio": total_original / total_compressed if total_compressed > 0 else 1.0,
        }


class LayerwiseAirLLMWithTurboQuant:
    """
    分层 AirLLM 推理类，支持 TurboQuant KV Cache 压缩

    该类是 AirLLM 的简化实现，演示如何将 TurboQuant 集成到分层推理中。
    实际使用时，建议继承 AirLLMBaseModel 并重写相关方法。
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        vocab_size: int,
        hidden_size: int,
        config: Optional[AirLLMTurboQuantConfig] = None,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.config = config or AirLLMTurboQuantConfig()
        self.device = torch.device(device)

        self.kv_cache_managers: List[AirLLMTurboQuantWrapper] = [
            AirLLMTurboQuantWrapper(
                num_heads=num_heads,
                head_dim=head_dim,
                config=self.config,
            )
            for _ in range(num_layers)
        ]

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        单层前向传播（带 KV Cache 压缩）

        Args:
            layer_idx: 层索引
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 可选的注意力掩码
            use_cache: 是否使用 KV Cache

        Returns:
            output: [batch, seq_len, hidden_size]
            past_kv: (k, v) 元组，如果 use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self._compute_q(hidden_states)
        k = self._compute_k(hidden_states)
        v = self._compute_v(hidden_states)

        if use_cache and layer_idx < len(self.kv_cache_managers):
            self.kv_cache_managers[layer_idx].update_kv_cache(k, v)

            all_k, all_v = self.kv_cache_managers[layer_idx].get_full_kv_compressed()

            if self.config.enable_asymmetric_attention and all_k:
                # 使用非对称注意力计算
                attn_scores = self.kv_cache_managers[layer_idx].compute_attention_asymmetric(q, all_k[0])
                
                # 应用注意力掩码
                if attention_mask is not None:
                    attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
                
                # 计算注意力权重
                attn_weights = torch.softmax(attn_scores, dim=-1)
                
                # 解压缩 V 并计算注意力输出
                k_decompressed, v_decompressed = self.kv_cache_managers[layer_idx].kv_manager.get_decompressed_kv(all_k[0], all_v[0])
                attn_output = torch.matmul(attn_weights, v_decompressed)
            else:
                attn_output = self._standard_attention(q, k, v, attention_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask)

        output = self._output_proj(attn_output)

        past_kv = (k, v) if use_cache else None

        return output, past_kv

    def _compute_q(self, x: torch.Tensor) -> torch.Tensor:
        """计算 Query"""
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _compute_k(self, x: torch.Tensor) -> torch.Tensor:
        """计算 Key"""
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _compute_v(self, x: torch.Tensor) -> torch.Tensor:
        """计算 Value"""
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """标准自注意力"""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def _merge_k_compressed(self, k_list: List[Dict]) -> torch.Tensor:
        """合并压缩的 K 张量"""
        k_tensors = []
        for k_comp in k_list:
            k_head = torch.cat([v for v in k_comp.values()], dim=0)
            k_tensors.append(k_head)
        return torch.cat(k_tensors, dim=2)

    def _merge_v_compressed(self, v_list: List[Dict]) -> torch.Tensor:
        """合并压缩的 V 张量"""
        v_tensors = []
        for v_comp in v_list:
            v_head = torch.cat([v for v in v_comp.values()], dim=0)
            v_tensors.append(v_head)
        return torch.cat(v_tensors, dim=2)

    def _output_proj(self, x: torch.Tensor) -> torch.Tensor:
        """输出投影"""
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.hidden_size)

    def reset_kv_cache(self):
        """重置所有层的 KV Cache"""
        for manager in self.kv_cache_managers:
            manager.reset()


def create_turboquant_wrapper(
    num_heads: int,
    head_dim: int,
    config: Optional[AirLLMTurboQuantConfig] = None,
) -> AirLLMTurboQuantWrapper:
    """
    创建 TurboQuant KV Cache 包装器

    Args:
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        config: 可选的配置对象

    Returns:
        AirLLMTurboQuantWrapper 实例
    """
    return AirLLMTurboQuantWrapper(
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
    )


def integrate_turboquant_to_airllm(airllm_model, config: AirLLMTurboQuantConfig):
    """
    将 TurboQuant 集成到现有的 AirLLM 模型

    该函数为 AirLLM 模型添加 TurboQuant KV Cache 压缩支持

    Args:
        airllm_model: AirLLMBaseModel 实例
        config: TurboQuant 配置

    Returns:
        修改后的模型（带 TurboQuant 支持）
    """
    if not config.enabled:
        return airllm_model

    num_layers = len(airllm_model.layers) - 3
    num_heads = getattr(airllm_model.config, "num_attention_heads", 32)
    head_dim = getattr(airllm_model.config, "hidden_size", 4096) // num_heads

    airllm_model.turboquant_wrapper = AirLLMTurboQuantWrapper(
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
    )

    original_forward = airllm_model.forward

    def new_forward(*args, **kwargs):
        if "use_cache" in kwargs and kwargs["use_cache"]:
            if not hasattr(airllm_model, "turboquant_wrapper"):
                return original_forward(*args, **kwargs)

            wrapper = airllm_model.turboquant_wrapper

            if wrapper.config.compress_kv:
                if wrapper.past_kv_compressed is None:
                    wrapper.reset()

        return original_forward(*args, **kwargs)

    airllm_model.forward = new_forward

    return airllm_model


class TurboQuantAttentionHook:
    """
    TurboQuant 注意力钩子

    用于在现有模型的注意力计算处注入 TurboQuant 压缩逻辑
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        config: Optional[AirLLMTurboQuantConfig] = None,
    ):
        self.wrapper = AirLLMTurboQuantWrapper(
            num_heads=num_heads,
            head_dim=head_dim,
            config=config,
        )
        self.hook_handle = None

    def attach_to_model(self, model, layer_name_pattern: str = "self_attn"):
        """
        将钩子附加到模型上

        Args:
            model: 要修改的模型
            layer_name_pattern: 注意力层名称模式
        """
        import functools

        def pre_hook(module, inputs):
            q = inputs[0]
            if len(q.shape) == 3:
                q = q.unsqueeze(0)

            k = inputs[1] if len(inputs) > 1 else q
            v = inputs[2] if len(inputs) > 2 else q

            self.wrapper.update_kv_cache(k, v)

            return inputs

        def post_hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                k_new, v_new = outputs[0], outputs[1]
                if isinstance(k_new, torch.Tensor) and isinstance(v_new, torch.Tensor):
                    self.wrapper.update_kv_cache(k_new, v_new)

            return outputs

        for name, module in model.named_modules():
            if layer_name_pattern in name and hasattr(module, "forward"):
                self.hook_handle = module.register_forward_pre_hook(pre_hook)
                module.register_forward_hook(post_hook)
                break

    def detach(self):
        """移除钩子"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def reset(self):
        """重置 KV Cache"""
        self.wrapper.reset()

    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        return self.wrapper.get_memory_usage()


def integrate_turboquant_to_transformers(
    model,
    config: Optional[AirLLMTurboQuantConfig] = None,
):
    """
    将 TurboQuant 集成到 HuggingFace Transformers 模型

    Args:
        model: HuggingFace Transformers 模型
        config: TurboQuant 配置

    Returns:
        修改后的模型（带 TurboQuant 支持）
    """
    if config is None:
        config = AirLLMTurboQuantConfig()

    if not config.enabled:
        return model

    num_heads = getattr(model.config, "num_attention_heads", 32)
    head_dim = getattr(model.config, "hidden_size", 4096) // num_heads

    wrapper = AirLLMTurboQuantWrapper(
        num_heads=num_heads,
        head_dim=head_dim,
        config=config,
    )

    model.turboquant_wrapper = wrapper

    # 为模型添加内存使用统计方法
    def get_memory_stats(self):
        if hasattr(self, "turboquant_wrapper"):
            return self.turboquant_wrapper.get_memory_usage()
        return {"original_bits": 0, "compressed_bits": 0, "compression_ratio": 1.0}

    model.get_memory_stats = get_memory_stats.__get__(model, model.__class__)

    # 重置 KV Cache 的方法
    def reset_kv_cache(self):
        if hasattr(self, "turboquant_wrapper"):
            self.turboquant_wrapper.reset()

    model.reset_kv_cache = reset_kv_cache.__get__(model, model.__class__)

    # 尝试修改模型的注意力层，使用 TurboQuant 加速
    try:
        # 遍历模型的所有层，找到注意力层
        for name, module in model.named_modules():
            if hasattr(module, "forward") and "attention" in name.lower():
                original_attention_forward = module.forward
                
                def new_attention_forward(self, *args, **kwargs):
                    # 检查是否有 turboquant_wrapper
                    if hasattr(self, "turboquant_wrapper"):
                        # 这里可以添加 TurboQuant 加速逻辑
                        # 注意：实际实现需要根据模型的具体结构进行调整
                        pass
                    return original_attention_forward(*args, **kwargs)
                
                # 绑定新的前向方法到模块
                module.forward = new_attention_forward.__get__(module, module.__class__)
                break
    except Exception as e:
        print(f"【TurboQuant】修改注意力层时出错: {str(e)}")

    return model
