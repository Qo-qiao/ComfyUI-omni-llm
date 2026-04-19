# -*- coding: utf-8 -*-
"""
VRAM 层计算模块

提供 GGUF 和 safetensors 模型的层计数和 VRAM 计算功能。
支持模型层数、tensor 数量和隐层维度的智能检测。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import struct
import json
import os


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f):
    ln = read_u64(f)
    return f.read(ln).decode("utf-8")


def read_value(f):
    vtype = read_u32(f)

    if vtype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:
        return read_string(f)
    if vtype == 9:
        atype = read_u32(f)
        count = read_u64(f)
        return [read_value_of_type(f, atype) for _ in range(count)]
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", f.read(8))[0]

    raise ValueError(f"未知值类型 {vtype}")


def read_value_of_type(f, atype):
    if atype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if atype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if atype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if atype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if atype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if atype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if atype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if atype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if atype == 8:
        return read_string(f)
    if atype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if atype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if atype == 12:
        return struct.unpack("<d", f.read(8))[0]

    raise ValueError(f"未知数组项类型 {atype}")


def get_layer_count(path):
    """
    从GGUF文件元数据中获取模型层数

    Args:
        path: GGUF模型文件路径

    Returns:
        int: 模型层数（如 transformer.h.layer 等）
        None: 如果无法读取或不是有效的GGUF文件
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("这不是一个 GGUF 文件！")

        version = read_u32(f)
        tensor_count = read_u64(f)
        kv_count = read_u64(f)
        meta = {}

        for _ in range(kv_count):
            key = read_string(f)
            value = read_value(f)
            meta[key] = value

    for k, v in meta.items():
        if k.lower().endswith(".block_count"):
            return v

    print("读取元数据失败: 未找到 .block_count 字段")
    return None


def get_file_size_gb(path):
    """获取文件大小（GB）"""
    return os.path.getsize(path) / (1024 ** 3)


def calculate_vram_layers(model_path, vram_limit_gb, mmproj_size_gb=0, compression_factor=1.55):
    """
    根据VRAM限制计算应该加载到GPU的层数（GGUF模型）

    Args:
        model_path: GGUF模型文件路径
        vram_limit_gb: VRAM限制（GB），-1表示无限制
        mmproj_size_gb: mmproj模型大小（GB），默认0
        compression_factor: 压缩因子，默认1.55（GGUF压缩后大小估算）

    Returns:
        int: n_gpu_layers值（-1表示全部加载，0表示仅CPU，>0表示具体层数）
    """
    if vram_limit_gb == -1:
        return -1

    gguf_layers = get_layer_count(model_path)
    if gguf_layers is None:
        gguf_layers = 32

    gguf_size = get_file_size_gb(model_path) * compression_factor
    gguf_layer_size = gguf_size / gguf_layers

    available_vram = vram_limit_gb - mmproj_size_gb
    if available_vram <= 0:
        return 0

    n_gpu_layers = max(1, int(available_vram / gguf_layer_size))
    return min(n_gpu_layers, gguf_layers)


def get_gguf_model_info(path):
    """
    获取GGUF模型的完整元数据信息

    Args:
        path: GGUF模型文件路径

    Returns:
        dict: 模型元数据信息
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None

        version = read_u32(f)
        tensor_count = read_u64(f)
        kv_count = read_u64(f)
        meta = {}

        for _ in range(kv_count):
            key = read_string(f)
            value = read_value(f)
            meta[key] = value

    file_size = get_file_size_gb(path)

    info = {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "file_size_gb": file_size,
        "layers": None,
        "hidden_size": None,
        "attention_heads": None,
        "vocab_size": None,
    }

    for k, v in meta.items():
        if k.lower().endswith(".block_count"):
            info["layers"] = v
        elif k.lower() in ["hidden_size", "embedding_length"]:
            info["hidden_size"] = v
        elif k.lower() in ["attention.head_count", "num_attention_heads"]:
            info["attention_heads"] = v
        elif k.lower() in ["vocab_size", "tokenizer.model.vocab_size"]:
            info["vocab_size"] = v

    return info


def read_safetensors_header(path):
    """
    读取 safetensors 文件头

    Args:
        path: safetensors 文件路径

    Returns:
        dict: 包含 tensors 信息的字典，键为 tensor 名称
        None: 如果读取失败
    """
    try:
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))
            return header
    except Exception:
        return None


def get_tensor_count(path):
    """
    获取 safetensors 文件中的 tensor 总数

    Args:
        path: safetensors 文件路径

    Returns:
        int: tensor 数量
        None: 如果读取失败
    """
    header = read_safetensors_header(path)
    if header is None:
        return None
    return len(header)


def estimate_layer_count_from_tensors(tensors_info):
    """
    从 tensor 信息估算模型层数

    通过分析 tensor 名称模式来估算 transformer 层数。

    Args:
        tensors_info: safetensors header 中的 tensors 信息

    Returns:
        int: 估算的层数
    """
    layer_patterns = [
        r"model\.layers\.(\d+)",
        r"model\.decoder\.layers\.(\d+)",
        r"transformer\.h\.(\d+)",
        r"transformer\.blocks\.(\d+)",
        r"encoder\.layers\.(\d+)",
        r"decoder\.layers\.(\d+)",
        r"blocks\.(\d+)",
    ]

    max_layer = 0
    for name in tensors_info.keys():
        for pattern in layer_patterns:
            import re
            match = re.search(pattern, name)
            if match:
                layer_num = int(match.group(1))
                max_layer = max(max_layer, layer_num)

    if max_layer > 0:
        return max_layer + 1
    return 32


def estimate_vram_for_safetensors(
    model_path,
    vram_limit_gb,
    mmproj_size_gb=0,
    compression_factor=1.8
):
    """
    为 safetensors 模型估算所需 VRAM 并计算建议的 GPU 层数

    Args:
        model_path: safetensors 模型文件路径
        vram_limit_gb: 可用 VRAM 限制（GB），-1 表示无限制
        mmproj_size_gb: mmproj 模型大小（GB）
        compression_factor: 压缩因子，safetensors 量化后约为 1.8-2.0

    Returns:
        dict: {
            "tensor_count": int,
            "estimated_layers": int,
            "estimated_vram_gb": float,
            "n_gpu_layers": int,
            "file_size_gb": float
        }
    """
    result = {
        "tensor_count": 0,
        "estimated_layers": 32,
        "estimated_vram_gb": 0.0,
        "n_gpu_layers": -1,
        "file_size_gb": 0.0,
    }

    try:
        file_size = os.path.getsize(model_path) / (1024 ** 3)
        result["file_size_gb"] = file_size
    except Exception:
        return result

    header = read_safetensors_header(model_path)
    if header is None:
        return result

    tensors_info = header.get("tensors", header)
    tensor_count = len(tensors_info)
    result["tensor_count"] = tensor_count

    estimated_layers = estimate_layer_count_from_tensors(tensors_info)
    result["estimated_layers"] = estimated_layers

    model_vram = file_size * compression_factor
    result["estimated_vram_gb"] = model_vram

    if vram_limit_gb == -1:
        result["n_gpu_layers"] = -1
    else:
        available_vram = vram_limit_gb - mmproj_size_gb
        if available_vram <= 0:
            result["n_gpu_layers"] = 0
        else:
            vram_per_layer = model_vram / estimated_layers
            n_gpu_layers = max(1, int(available_vram / vram_per_layer))
            result["n_gpu_layers"] = min(n_gpu_layers, estimated_layers)

    return result


def get_safetensors_model_info(path):
    """
    获取 safetensors 模型的详细信息

    Args:
        path: safetensors 模型文件路径

    Returns:
        dict: 模型信息字典
    """
    info = {
        "format": "safetensors",
        "tensor_count": 0,
        "estimated_layers": 32,
        "file_size_gb": 0.0,
    }

    try:
        info["file_size_gb"] = os.path.getsize(path) / (1024 ** 3)
    except Exception:
        return info

    header = read_safetensors_header(path)
    if header is None:
        return info

    tensors_info = header.get("tensors", header)
    info["tensor_count"] = len(tensors_info)
    info["estimated_layers"] = estimate_layer_count_from_tensors(tensors_info)

    return info


def calculate_safetensors_vram_layers(
    model_path,
    vram_limit_gb,
    mmproj_size_gb=0,
    compression_factor=1.8
):
    """
    根据 VRAM 限制计算 safetensors 模型应该加载到 GPU 的层数

    Args:
        model_path: safetensors 模型文件路径
        vram_limit_gb: 可用 VRAM 限制（GB），-1 表示无限制
        mmproj_size_gb: mmproj 模型大小（GB）
        compression_factor: 压缩因子

    Returns:
        int: 建议的 GPU 层数，0 表示全部用 CPU，-1 表示全部用 GPU
    """
    if vram_limit_gb == -1:
        return -1

    info = get_safetensors_model_info(model_path)
    if info["tensor_count"] == 0:
        return 32

    estimated_layers = info["estimated_layers"]
    model_vram = info["file_size_gb"] * compression_factor

    available_vram = vram_limit_gb - mmproj_size_gb
    if available_vram <= 0:
        return 0

    vram_per_layer = model_vram / estimated_layers
    n_gpu_layers = max(1, int(available_vram / vram_per_layer))

    return min(n_gpu_layers, estimated_layers)


def get_model_info(path):
    """
    自动检测模型格式并获取模型信息

    Args:
        path: 模型文件路径

    Returns:
        dict: 模型信息字典，自动识别 GGUF 或 safetensors
    """
    if path.lower().endswith(".gguf"):
        return get_gguf_model_info(path)
    elif path.lower().endswith(".safetensors"):
        return get_safetensors_model_info(path)
    else:
        return {"error": "不支持的模型格式"}
