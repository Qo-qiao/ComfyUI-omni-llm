# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm 公共组件

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import io
import gc
import json
import base64
import random
import struct
import re
import torch
import inspect
import numpy as np
import psutil
import platform
import sys
import functools
import contextlib
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# -------------------------- 硬件检测（保留提速优化） --------------------------
try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
        NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler,
        Qwen3VLChatHandler
    )
    # Qwen25VLChatHandler 在较新版本中才有，单独导入
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    except ImportError:
        Qwen25VLChatHandler = None
        print("【提示】当前llama-cpp-python版本不支持Qwen25VLChatHandler，Qwen2.5-VL/Omni模型将使用备选方案")
except ImportError as e:
    print(f"【错误】缺少llama-cpp-python依赖，请运行：pip install llama-cpp-python")
    exit(1)

# 尝试导入音频处理库
try:
    import soundfile as sf
    _SOUNDFILE = True
except ImportError:
    _SOUNDFILE = False
    sf = None

try:
    import scipy.io.wavfile as wavfile
    _SCIPY_WAV = True
except ImportError:
    _SCIPY_WAV = False
    wavfile = None

# 尝试导入Qwen35ChatHandler
try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except:
    Qwen35ChatHandler = None

# 尝试导入Qwen25VLChatHandler（某些旧版本llama-cpp-python可能不支持）
try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
except:
    Qwen25VLChatHandler = None
    print("【警告】当前llama-cpp-python版本不支持Qwen25VLChatHandler，将使用备选方案")

# 尝试导入GLM系列ChatHandler
try:
    from llama_cpp.llama_chat_format import GLM46VChatHandler, GLM41VChatHandler, LFM2VLChatHandler
except:
    GLM46VChatHandler = None
    GLM41VChatHandler = None
    LFM2VLChatHandler = None

# 尝试导入Gemma3ChatHandler
try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
except:
    Gemma3ChatHandler = None

# 尝试导入Gemma4ChatHandler
try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
except:
    Gemma4ChatHandler = None

# 尝试导入LFM2.5VLChatHandler
try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
except:
    LFM25VLChatHandler = None

# 尝试导入GraniteDoclingChatHandler
try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
except:
    GraniteDoclingChatHandler = None

# MTMD支持检测
try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _has_mtmd = True
except ImportError:
    _has_mtmd = False
    MTMDChatHandler = None

try:
    import folder_paths
    import comfy.model_management as mm
    import comfy.utils
    _has_comfy = True
except ImportError as e:
    _has_comfy = False
    mm = None
    comfy = None
    print(f"【错误】未检测到ComfyUI环境，请将该文件放入ComfyUI/custom_nodes/ComfyUI-omni-llm/目录下")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -------------------------- 模型加载进度输出类 --------------------------
# -------------------------- 硬件检测（保留提速优化） --------------------------
def get_hardware_info():
    hardware_info = {
        "has_cuda": torch.cuda.is_available(),
        "has_rocm": False,
        "gpu_name": "未知显卡",
        "gpu_vram_total": 0.0,
        "gpu_vendor": "unknown",
        "cpu_cores": os.cpu_count() or 4,
        "is_high_perf": False,
        "is_low_perf": True
    }
    
    if hardware_info["has_cuda"]:
        try:
            device_prop = torch.cuda.get_device_properties(0)
            hardware_info["gpu_name"] = device_prop.name
            hardware_info["gpu_vram_total"] = round(device_prop.total_memory / (1024 ** 3), 2)
            hardware_info["gpu_vendor"] = "nvidia"
            
            # 按显存大小和显卡型号分级
            gpu_vram = hardware_info["gpu_vram_total"]
            gpu_name_lower = hardware_info["gpu_name"].lower()
            
            # 高性能显卡标志（24GB+显存）
            high_perf_flags = ["5090", "4090", "3090", "a100", "a10", "rtx 6000", "titan"]
            # 中高性能显卡标志（16GB显存）
            mid_high_perf_flags = ["4080", "3080 ti"]
            # 中性能显卡标志（12GB显存）
            mid_perf_flags = ["4070 ti", "3080", "3070 ti"]
            # 中低性能显卡标志（8GB显存）
            mid_low_perf_flags = ["4070", "3070", "2080 ti", "2080", "1660 ti", "1660 super"]
            
            if gpu_vram >= 24 or any(flag.lower() in gpu_name_lower for flag in high_perf_flags):
                hardware_info["is_high_perf"] = True
                hardware_info["is_low_perf"] = False
                hardware_info["perf_level"] = "high"  # 24GB+
            elif gpu_vram >= 16 or any(flag.lower() in gpu_name_lower for flag in mid_high_perf_flags):
                hardware_info["is_high_perf"] = False
                hardware_info["is_low_perf"] = False
                hardware_info["perf_level"] = "mid_high"  # 16GB
            elif gpu_vram >= 12 or any(flag.lower() in gpu_name_lower for flag in mid_perf_flags):
                hardware_info["is_high_perf"] = False
                hardware_info["is_low_perf"] = False
                hardware_info["perf_level"] = "mid"  # 12GB
            elif gpu_vram >= 4 or any(flag.lower() in gpu_name_lower for flag in mid_low_perf_flags):
                hardware_info["is_high_perf"] = False
                hardware_info["is_low_perf"] = False
                hardware_info["perf_level"] = "mid_low"  # 4-8GB
            else:
                hardware_info["is_high_perf"] = False
                hardware_info["is_low_perf"] = True
                hardware_info["perf_level"] = "low"  # <4GB
            print(f"【硬件检测】显卡：{hardware_info['gpu_name']}，显存：{hardware_info['gpu_vram_total']}GB")
        except Exception as e:
            print(f"【提示】显卡信息检测失败，自动使用兼容模式：{e}")
    else:
        # 尝试检测AMD显卡（ROCm）
        try:
            if hasattr(torch, 'hip') or hasattr(torch, 'rocm'):
                hardware_info["has_rocm"] = True
                hardware_info["gpu_vendor"] = "amd"
                
                # 尝试获取AMD显卡信息
                try:
                    if hasattr(torch, 'hip'):
                        device_prop = torch.hip.get_device_properties(0)
                    elif hasattr(torch, 'rocm'):
                        device_prop = torch.rocm.get_device_properties(0)
                    else:
                        raise AttributeError("No ROCm device properties available")
                    
                    hardware_info["gpu_name"] = device_prop.name
                    hardware_info["gpu_vram_total"] = round(device_prop.total_memory / (1024 ** 3), 2)
                    
                    # AMD显卡性能分级
                    gpu_vram = hardware_info["gpu_vram_total"]
                    gpu_name_lower = hardware_info["gpu_name"].lower()
                    
                    # 高性能AMD显卡（24GB+显存）
                    amd_high_perf_flags = ["mi300", "mi250", "instinct", "7900 xtx", "7900 xt"]
                    # 中高性能AMD显卡（16GB显存）
                    amd_mid_high_perf_flags = ["7900", "7800 xt", "6950 xt", "6900 xt", "6800 xt"]
                    # 中性能AMD显卡（12GB显存）
                    amd_mid_perf_flags = ["7800", "7700 xt", "6750 xt", "6700 xt", "6600 xt"]
                    # 中低性能AMD显卡（8GB显存）
                    amd_mid_low_perf_flags = ["7700", "7600", "6750", "6700", "6650 xt", "6600", "rx 7600", "rx 6600"]
                    
                    if gpu_vram >= 24 or any(flag.lower() in gpu_name_lower for flag in amd_high_perf_flags):
                        hardware_info["is_high_perf"] = True
                        hardware_info["is_low_perf"] = False
                        hardware_info["perf_level"] = "high"  # 24GB+
                    elif gpu_vram >= 16 or any(flag.lower() in gpu_name_lower for flag in amd_mid_high_perf_flags):
                        hardware_info["is_high_perf"] = False
                        hardware_info["is_low_perf"] = False
                        hardware_info["perf_level"] = "mid_high"  # 16GB
                    elif gpu_vram >= 12 or any(flag.lower() in gpu_name_lower for flag in amd_mid_perf_flags):
                        hardware_info["is_high_perf"] = False
                        hardware_info["is_low_perf"] = False
                        hardware_info["perf_level"] = "mid"  # 12GB
                    elif gpu_vram >= 4 or any(flag.lower() in gpu_name_lower for flag in amd_mid_low_perf_flags):
                        hardware_info["is_high_perf"] = False
                        hardware_info["is_low_perf"] = False
                        hardware_info["perf_level"] = "mid_low"  # 4-8GB
                    else:
                        hardware_info["is_high_perf"] = False
                        hardware_info["is_low_perf"] = True
                        hardware_info["perf_level"] = "low"  # <4GB
                    
                    print(f"【硬件检测】AMD显卡：{hardware_info['gpu_name']}，显存：{hardware_info['gpu_vram_total']}GB")
                except AttributeError:
                    # 如果无法获取设备属性，尝试使用默认设置
                    hardware_info["gpu_name"] = "AMD GPU (ROCm)"
                    hardware_info["gpu_vram_total"] = 16.0  # 默认值
                    hardware_info["is_high_perf"] = False
                    hardware_info["is_low_perf"] = False
                    hardware_info["perf_level"] = "mid_high"  # 默认中高性能
                    print(f"【硬件检测】检测到AMD ROCm环境，使用默认设置")
            else:
                # 尝试通过系统信息检测AMD显卡
                try:
                    import subprocess
                    if platform.system() == "Windows":
                        # Windows: 使用wmic命令
                        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                           capture_output=True, text=True, timeout=5)
                        gpu_names = result.stdout
                        if 'amd' in gpu_names.lower() or 'radeon' in gpu_names.lower():
                            hardware_info["gpu_vendor"] = "amd"
                            hardware_info["gpu_name"] = "AMD GPU"
                            hardware_info["is_high_perf"] = False
                            hardware_info["is_low_perf"] = False
                            hardware_info["perf_level"] = "mid"  # 默认中性能
                            print(f"【硬件检测】检测到AMD显卡（Windows）")
                    elif platform.system() == "Linux":
                        # Linux: 使用lspci命令
                        result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=5)
                        gpu_info = result.stdout
                        if 'amd' in gpu_info.lower() or 'radeon' in gpu_info.lower():
                            hardware_info["gpu_vendor"] = "amd"
                            hardware_info["gpu_name"] = "AMD GPU"
                            hardware_info["is_high_perf"] = False
                            hardware_info["is_low_perf"] = False
                            hardware_info["perf_level"] = "mid"  # 默认中性能
                            print(f"【硬件检测】检测到AMD显卡（Linux）")
                except Exception:
                    pass
                
                if hardware_info["gpu_vendor"] == "unknown":
                    try:
                        if platform.system() == "Windows":
                            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                               capture_output=True, text=True, timeout=5)
                            gpu_names = result.stdout
                            if 'intel' in gpu_names.lower() or 'uhd' in gpu_names.lower() or 'iris' in gpu_names.lower() or 'arc' in gpu_names.lower():
                                hardware_info["gpu_vendor"] = "intel"
                                hardware_info["gpu_name"] = "Intel GPU"
                                hardware_info["is_high_perf"] = False
                                hardware_info["is_low_perf"] = False
                                hardware_info["perf_level"] = "mid_low"
                                print(f"【硬件检测】检测到Intel显卡（Windows）")
                        elif platform.system() == "Linux":
                            result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=5)
                            gpu_info = result.stdout
                            if 'intel' in gpu_info.lower() or 'uhd' in gpu_info.lower():
                                hardware_info["gpu_vendor"] = "intel"
                                hardware_info["gpu_name"] = "Intel GPU"
                                hardware_info["is_high_perf"] = False
                                hardware_info["is_low_perf"] = False
                                hardware_info["perf_level"] = "mid_low"
                                print(f"【硬件检测】检测到Intel显卡（Linux）")
                    except Exception:
                        pass

                if hardware_info["gpu_vendor"] == "unknown":
                    print(f"【硬件检测】未检测到NVIDIA CUDA或AMD ROCm显卡，自动使用CPU/通用显卡兼容模式")
        except Exception as e:
            print(f"【提示】AMD显卡检测失败，自动使用兼容模式：{e}")
    
    print(f"【硬件检测】CPU核心数：{hardware_info['cpu_cores']}")
    return hardware_info

HARDWARE_INFO = get_hardware_info()

# -------------------------- GGUF/VRAM Module --------------------------
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
    raise ValueError(f"Unknown value type {vtype}")


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
    raise ValueError(f"Unknown array item type {atype}")


def get_layer_count(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file!")

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

    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)
        layer_count = reader.get_field("llama.block_count")
        if layer_count is None:
            for field in reader.fields.values():
                if field.name.endswith(".block_count"):
                    layer_count = field.parts[field.data[0]]
                    break
        if layer_count:
            return int(layer_count[0] if isinstance(layer_count, list) else layer_count)
    except Exception as e:
        print(f"[GGUFReader] Failed to get block_count: {e}")

    return None


def calculate_vram_layers(model_path, vram_limit_gb, mmproj_size_gb=0, compression_factor=1.55):
    if vram_limit_gb == -1:
        return -1

    gguf_layers = get_layer_count(model_path)
    if gguf_layers is None:
        gguf_layers = 32

    gguf_size = os.path.getsize(model_path) * compression_factor / (1024 ** 3)
    gguf_layer_size = gguf_size / gguf_layers

    available_vram = vram_limit_gb - mmproj_size_gb
    if available_vram <= 0:
        return 0

    n_gpu_layers = max(1, int(available_vram / gguf_layer_size))
    return min(n_gpu_layers, gguf_layers)


def get_gguf_model_info(path):
    file_size = os.path.getsize(path) / (1024 ** 3)
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

    info = {"version": version, "tensor_count": tensor_count, "kv_count": kv_count,
            "file_size_gb": file_size, "layers": None, "hidden_size": None,
            "attention_heads": None, "vocab_size": None}

    for k, v in meta.items():
        k_lower = k.lower()
        if k_lower.endswith(".block_count"):
            info["layers"] = v
        elif k_lower in ("hidden_size", "embedding_length"):
            info["hidden_size"] = v
        elif k_lower in ("attention.head_count", "num_attention_heads"):
            info["attention_heads"] = v
        elif k_lower in ("vocab_size", "tokenizer.model.vocab_size"):
            info["vocab_size"] = v

    return info


def estimate_vram_for_safetensors(model_path, gpu_vram, mmproj_size_gb=0):
    total_size = 0
    tensor_count = 0

    if os.path.isdir(model_path):
        for filename in os.listdir(model_path):
            if filename.endswith('.safetensors'):
                filepath = os.path.join(model_path, filename)
                total_size += os.path.getsize(filepath)
                try:
                    with open(filepath, 'rb') as f:
                        while True:
                            header_size_data = f.read(8)
                            if not header_size_data or len(header_size_data) < 8:
                                break
                            tensor_count += 1
                            header_size = int.from_bytes(header_size_data[:4], 'little')
                            f.read(header_size)
                except Exception:
                    pass
    elif model_path.endswith('.safetensors'):
        total_size = os.path.getsize(model_path)
        try:
            with open(model_path, 'rb') as f:
                while True:
                    header_size_data = f.read(8)
                    if not header_size_data or len(header_size_data) < 8:
                        break
                    tensor_count += 1
                    header_size = int.from_bytes(header_size_data[:4], 'little')
                    f.read(header_size)
        except Exception:
            pass

    total_size_gb = total_size / (1024 ** 3)
    estimated_layers = max(1, int(total_size_gb / 0.5))

    return {
        "tensor_count": tensor_count,
        "estimated_layers": estimated_layers,
        "total_size_gb": total_size_gb,
        "n_gpu_layers": -1
    }


def calculate_safetensors_vram_layers(model_path, vram_limit_gb, mmproj_size_gb=0):
    info = estimate_vram_for_safetensors(model_path, vram_limit_gb, mmproj_size_gb)
    
    if vram_limit_gb == -1:
        return info["n_gpu_layers"]
    
    available_vram = vram_limit_gb - mmproj_size_gb
    if available_vram <= 0:
        return 0
    
    layer_size_gb = info["total_size_gb"] / info["estimated_layers"] if info["estimated_layers"] > 0 else 0.5
    
    n_gpu_layers = max(1, int(available_vram / layer_size_gb))
    
    return min(n_gpu_layers, info["estimated_layers"])

# -------------------------- 模型和结果缓存 --------------------------
# 模型缓存：存储已加载的模型实例
MODEL_CACHE = {}
# 结果缓存：存储推理结果
RESULT_CACHE = {}
# 缓存大小限制
CACHE_SIZE = 2  # 模型缓存大小
RESULT_CACHE_SIZE = 20  # 结果缓存大小

# -------------------------- 设置进程高优先级 --------------------------
def set_high_priority():
    try:
        p = psutil.Process()
        if platform.system() == "Windows":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(max(-10, p.nice() - 5))
        print(f"【优化】已设置进程高优先级，提速更明显")
    except Exception as e:
        print(f"【提示】进程优先级设置失败（不影响核心功能）：{e}")

set_high_priority()

# -------------------------- 对话解析功能 --------------------------

class DialogueParser:
    """
    对话解析器 - 用于解析和验证对话内容
    """

    @staticmethod
    def parse_dialogue(dialogue_text, default_speaker="角色1", default_emotion="默认"):
        if not dialogue_text:
            return []
        lines = dialogue_text.strip().split('\n')
        dialogue = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 支持多种格式：
            # 1. 角色名: 文本
            # 2. [角色名] 文本
            # 3. 角色名(情感): 文本
            # 4. [角色名](情感) 文本
            match = re.match(r'^[\[\【]*(.*?)[\]\】]*(?:\((.*?)\))?(?:[:\s]+)(.*)$', line)
            if match:
                speaker = match.group(1).strip() or default_speaker
                emotion = match.group(2).strip() if match.group(2) else default_emotion
                text = match.group(3).strip()
                # 移除文本中可能包含的角色名前缀
                text = re.sub(r'^[\[\【]*' + re.escape(speaker) + r'[\]\】]*\s*', '', text)
                dialogue.append({"speaker": speaker, "emotion": emotion, "text": text})
            else:
                dialogue.append({"speaker": default_speaker, "emotion": default_emotion, "text": line})
        return dialogue

    @staticmethod
    def validate_dialogue(dialogue):
        errors = []
        if not dialogue:
            errors.append("对话为空")
            return False, errors
        for i, turn in enumerate(dialogue):
            if not isinstance(turn, dict):
                errors.append(f"第{i+1}轮对话不是有效的字典格式")
                continue
            if "speaker" not in turn or not turn["speaker"]:
                errors.append(f"第{i+1}轮对话缺少说话人")
            if "text" not in turn or not turn["text"]:
                errors.append(f"第{i+1}轮对话缺少文本内容")
            if "emotion" not in turn:
                errors.append(f"第{i+1}轮对话缺少情感标签")
        return len(errors) == 0, errors

    @staticmethod
    def format_dialogue_info(dialogue):
        if not dialogue:
            return "空对话"
        info_lines = []
        for i, segment in enumerate(dialogue):
            speaker = segment.get("speaker", "未知")
            emotion = segment.get("emotion", "默认")
            text = segment.get("text", "")
            info_lines.append(f"第{i+1}轮: [{speaker}]({emotion}) - {text}")
        return "\n".join(info_lines)


def parse_json(text, use_full_content=False):
    if not text:
        return None
    try:
        json_match = re.search(r'\{[\s\S]*?\}|\[[\s\S]*?\]', text)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            pass
        if use_full_content:
            return {"content": text}
        return None
    except Exception as e:
        print(f"【JSON解析】解析失败: {e}")
        return None


def extract_json_from_response(response):
    if not response:
        return None
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError:
        pass
    json_start = response.find('{')
    json_end = response.rfind('}')
    if json_start != -1 and json_end != -1 and json_start < json_end:
        try:
            json_str = response[json_start:json_end+1]
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            pass
    json_start = response.find('[')
    json_end = response.rfind(']')
    if json_start != -1 and json_end != -1 and json_start < json_end:
        try:
            json_str = response[json_start:json_end+1]
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            pass
    return None

# -------------------------- ChatHandler管理器（动态管理所有ChatHandler） --------------------------

class ChatHandlerManager:
    """
    ChatHandler管理器 - 动态管理llama-cpp-python中的所有ChatHandler
    支持自动检测、缓存和智能匹配
    """
    
    # 标准模型名称映射表（类级别常量，避免重复创建）
    _DISPLAY_MAP = {
        'qwen25vl': 'Qwen2.5-VL',
        'qwen3vl': 'Qwen3-VL',
        'qwen3vlchat': 'Qwen3-VL-Chat',
        'qwen3vlinstruct': 'Qwen3-VL-Instruct',
        'qwen3vlthinking': 'Qwen3-VL-Thinking',
        'qwen35': 'Qwen3.5',
        'qwen35thinking': 'Qwen3.5-Thinking',
        'qwen25omni': 'Qwen2.5-Omni',
        'qwen35gguf': 'Qwen3.5-GGUF',
        'glm46v': 'GLM-4.6V',
        'glm46vthinking': 'GLM-4.6V-Thinking',
        'glm41v': 'GLM-4.1V-Thinking',
        'minicpmv45': 'MiniCPM-v4.5',
        'minicpmv26': 'MiniCPM-v2.6',
        'minicpmo45': 'MiniCPM-O-4.5',
        'minicpmo26': 'MiniCPM-O-2.6',
        'minicpmogguf': 'MiniCPM-O-GGUF',
        'minicpmlama3v25': 'MiniCPM-Llama3-V 2.5',
        'moondream3': 'Moondream3',
        'moondream2': 'Moondream2',
        'llava15': 'LLaVA-1.5',
        'llava16': 'LLaVA-1.6',
        'nanollava': 'NanoLLaVA',
        'llama3visionalpha': 'Llama3-Vision-Alpha',
        'llama32visioninstruct': 'Llama-3.2-11B-Vision-Instruct',
        'llama31vision': 'LLaMA-3.1-Vision',
        'gemma3': 'Gemma-3',
        'granitedocling': 'Granite-DocLing',
        'lfm2vl': 'Lfm-2-VL',
        'paddleocr': 'PaddleOCR-VL-1.5',
        'obsidian': 'Obsidian',
        'cogvlm2': 'CogVLM2',
        'cogvlmmoe': 'CogVLM-MOE',
        'phi35vision': 'Phi-3.5-vision-instruct',
        'phi3vision128k': 'Phi-3-vision-128k-instruct',
        'internlmxcomposer2vl': 'InternLM-XComposer2-VL',
        'yutuvl4binstruct': 'Youtu-VL-4B-Instruct',
        'eraxvl7bv15': 'EraX-VL-7B-V1.5',
        'mimovl7brl': 'MiMo-VL-7B-RL',
        'asidcaptioner7b': 'ASID-Captioner-7B',
        'zen3vli1': 'zen3-vl-i1',
        'lightonocr21b': 'LightOnOCR-2-1B',
        'yivl6b': 'Yi-VL-6B',
        'yivl20': 'Yi-VL-2.0',
        'internvl15': 'InternVL-1.5',
        'internvl20': 'InternVL-2.0',
        'olmocr2': 'olmOCR-2',
    }
    
    def __init__(self):
        self._handlers = {}  # 缓存已加载的ChatHandler类
        self._handler_info = {}  # ChatHandler元数据
        self._model_mappings = {}  # 模型名称到ChatHandler的映射
        self._available_handlers = []  # 检测到的可用ChatHandler列表
        self._detect_handlers()
        self._build_mappings()
    
    def _detect_handlers(self):
        """自动检测llama_cpp.llama_chat_format中所有可用的ChatHandler"""
        try:
            import llama_cpp.llama_chat_format as chat_format
            
            # 扫描所有以ChatHandler结尾的类
            for attr_name in dir(chat_format):
                if attr_name.endswith('ChatHandler'):
                    try:
                        handler_cls = getattr(chat_format, attr_name)
                        # 验证是否是类且可以实例化
                        if isinstance(handler_cls, type):
                            self._available_handlers.append(attr_name)
                            self._handlers[attr_name] = handler_cls
                            
                            # 提取模型信息
                            model_info = self._extract_model_info(attr_name)
                            self._handler_info[attr_name] = model_info
                            
                            print(f"【ChatHandler检测】发现: {attr_name} -> {model_info['display_name']}")
                    except Exception as e:
                        print(f"【ChatHandler检测】跳过 {attr_name}: {e}")
            
            print(f"【ChatHandler检测】共发现 {len(self._available_handlers)} 个ChatHandler")
            
        except ImportError as e:
            print(f"【ChatHandler检测】无法导入llama_cpp.llama_chat_format: {e}")
        except Exception as e:
            print(f"【ChatHandler检测】检测过程出错: {e}")
    
    def _extract_model_info(self, handler_name):
        """从ChatHandler名称提取模型信息"""
        # 移除ChatHandler后缀
        base_name = handler_name.replace('ChatHandler', '')
        
        # 使用正则添加连字符
        model_name = re.sub(r'([a-z])([A-Z])', r'\1-\2', base_name)
        model_name = re.sub(r'([0-9])([A-Z])', r'\1-\2', model_name)
        model_name = re.sub(r'([A-Z])([0-9])', r'\1-\2', model_name)
        
        # 转换为小写
        model_key = model_name.lower().replace('-', '')
        
        # 获取显示名称
        display_name = self._get_display_name(model_key, model_name)
        
        # 检测是否是Qwen系列（需要特殊处理）
        is_qwen = 'qwen' in model_key
        
        # 检测参数类型
        param_type = self._detect_param_type(handler_name)
        
        return {
            'handler_name': handler_name,
            'model_key': model_key,
            'display_name': display_name,
            'is_qwen': is_qwen,
            'param_type': param_type,
            'base_name': base_name
        }
    
    def _get_display_name(self, model_key, default_name):
        """获取模型的显示名称"""
        return self._DISPLAY_MAP.get(model_key, default_name.title().replace('-', ' '))
    
    def _detect_param_type(self, handler_name):
        """检测ChatHandler的参数类型"""
        try:
            handler_cls = self._handlers.get(handler_name)
            if handler_cls is None:
                return 'unknown'
            
            sig = inspect.signature(handler_cls.__init__)
            params = list(sig.parameters.keys())
            
            if 'clip_model_path' in params:
                return 'clip_model_path'
            elif 'mmproj' in params:
                return 'mmproj'
            else:
                return 'none'
        except Exception:
            return 'unknown'
    
    def _build_mappings(self):
        """构建模型名称到ChatHandler的映射"""
        for handler_name, info in self._handler_info.items():
            # 多种键格式映射
            keys = [
                info['model_key'],
                info['model_key'].replace('-', ''),
                info['display_name'].lower().replace(' ', ''),
                info['display_name'].lower().replace(' ', '-'),
                info['base_name'].lower(),
            ]
            
            for key in keys:
                self._model_mappings[key] = handler_name
                self._model_mappings[key.replace('-', '')] = handler_name
    
    def get_handler(self, name):
        """
        获取ChatHandler类
        
        Args:
            name: ChatHandler名称（如'Qwen3VLChatHandler'）或模型名称（如'Qwen3-VL'）
        
        Returns:
            ChatHandler类或None
        """
        # 直接匹配
        if name in self._handlers:
            return self._handlers[name]
        
        # 添加ChatHandler后缀尝试
        if not name.endswith('ChatHandler'):
            handler_name = name + 'ChatHandler'
            if handler_name in self._handlers:
                return self._handlers[handler_name]
        
        # 通过模型名称映射查找
        key = name.lower().replace(' ', '').replace('-', '')
        if key in self._model_mappings:
            handler_name = self._model_mappings[key]
            return self._handlers.get(handler_name)
        
        return None
    
    def get_handler_for_model(self, model_name):
        """
        根据模型名称获取最佳匹配的ChatHandler
        
        Args:
            model_name: 模型名称或文件名
        
        Returns:
            (handler_name, handler_cls) 元组或 (None, None)
        """
        model_lower = model_name.lower()
        
        # 特殊规则匹配
        # 根据Qwen25VLChatHandler是否可用动态选择
        qwen25_handler = 'Qwen25VLChatHandler' if Qwen25VLChatHandler else 'Qwen3VLChatHandler'
        
        special_rules = [
            # (关键词, ChatHandler名称)
            ('qwen3-vl', 'Qwen3VLChatHandler'),
            ('qwen2.5-vl', qwen25_handler),  # 如果Qwen25VL不可用则使用Qwen3VL
            ('qwen2.5-omni', qwen25_handler),  # Omni系列模型
            ('qwen35', 'Qwen35ChatHandler'),
            ('qwen3.5', 'Qwen35ChatHandler'),  # 匹配带点的版本
            ('minicpm-v-4.5', 'MiniCPMv45ChatHandler'),
            ('minicpm-o-4.5', 'MiniCPMv45ChatHandler'),  # Omni版本
            ('minicpm-o-2.6', 'MiniCPMv26ChatHandler'),  # MiniCPM-O-2.6
            ('minicpmo2.6', 'MiniCPMv26ChatHandler'),  # MiniCPM-O-2.6变体
            ('minicpm-v-2.6', 'MiniCPMv26ChatHandler'),
            ('minicpm-llama3-v-2.5', 'MiniCPMv26ChatHandler'),
            ('glm-4.6v', 'GLM46VChatHandler'),
            ('glm-4.1v', 'GLM41VChatHandler'),
            ('gemma-3', 'Gemma3ChatHandler'),
            ('gemma-4', 'Gemma4ChatHandler'),
            ('granite-docling', 'GraniteDoclingChatHandler'),
            ('lfm-2-vl', 'LFM2VLChatHandler'),
            ('lfm-2.5-vl', 'LFM25VLChatHandler'),
            ('paddleocr', 'PaddleOCRChatHandler'),
            ('obsidian', 'ObsidianChatHandler'),
            ('llava-1.6', 'Llava16ChatHandler'),
            ('llava-1.5', 'Llava15ChatHandler'),
            ('nanollava', 'NanoLlavaChatHandler'),
            ('moondream2', 'MoondreamChatHandler'),
            ('moondream-2', 'MoondreamChatHandler'),
            ('llama3visionalpha', 'Llama3VisionAlphaChatHandler'),
            ('llama-3.2-11b-vision', 'Llama32VisionInstructChatHandler'),
            ('llama-3.1-vision', 'Llama31VisionChatHandler'),
            ('phi-3.5-vision', 'Phi35VisionChatHandler'),
            ('phi-3-vision', 'Phi3Vision128kChatHandler'),
            ('internvl', 'InternLMXComposer2VLChatHandler'),

            ('yutuvl', 'YutuVL4BInstructChatHandler'),
            ('erax-vl', 'EraXVL7BV15ChatHandler'),
            ('mimo-vl', 'MiMoVL7BRLChatHandler'),
            ('asid-captioner', 'ASIDCaptioner7BChatHandler'),
            ('zen3-vl', 'Zen3VLI1ChatHandler'),
            ('lightonocr', 'LightOnOCR21BChatHandler'),
            ('yivl', 'YiVL6BChatHandler'),
            ('cogvlm', 'CogVLM2ChatHandler'),
        ]
        
        for keyword, handler_name in special_rules:
            if keyword in model_lower:
                handler = self.get_handler(handler_name)
                if handler:
                    return handler_name, handler
        
        return None, None
    
    def get_all_handlers(self):
        """获取所有可用的ChatHandler名称"""
        return list(self._available_handlers)
    
    def get_all_models(self):
        """获取所有支持的模型显示名称"""
        return [info['display_name'] for info in self._handler_info.values()]
    
    def get_handler_info(self, handler_name):
        """获取ChatHandler的详细信息"""
        return self._handler_info.get(handler_name)


# 创建全局ChatHandler管理器实例
chat_handler_manager = ChatHandlerManager()

# -------------------------- 初始化ChatHandler（支持所有模型） --------------------------

# 基础模型列表（无需ChatHandler的模型 - 保留向后兼容）
base_models = ["LLaVA-1.6", "nanoLLaVA", "llama-joycaption", "moondream3-preview", "Moondream2", 
               "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking", "GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "InternLM-XComposer2-VL", "DreamOmni2", 
               "MiniCPM-Llama3-V 2.5", "Llama-3.2-11B-Vision-Instruct", "CogVLM2", 
               "CogVLM-MOE", "Phi-3.5-vision-instruct", "Phi-3-vision-128k-instruct", 
               "Qwen2.5-VL", "Qwen3-VL", "Qwen3-VL-Thinking", "Qwen3-VL-Chat", "Qwen3-VL-Instruct", 
               "Qwen3.5", "Qwen3.5-Thinking", "Qwen2.5-Omni", "MiniCPM-O-4.5",
               "LLaMA-3.1-Vision", "Zhipu-Vision", "智谱AI-Vision", "olmOCR-2", 
               "InternVL-1.5", "InternVL-2.0", "Yi-VL-2.0", "Gemma-3", "Granite-DocLing", 
               "Lfm-2-VL", "Llama3-Vision-Alpha", "LLaVA-1.5", "MiniCPM-v2.6", "Obsidian", 
               "Youtu-VL-4B-Instruct", "EraX-VL-7B-V1.5", "MiMo-VL-7B-RL", "Yi-VL-6B"]

# ChatHandler名称到标准模型名称的映射
CHAT_HANDLER_MODEL_MAP = {
    'qwen25vl': 'Qwen2.5-VL',
    'qwen3vl': 'Qwen3-VL',
    'qwen3vlchat': 'Qwen3-VL-Chat',
    'qwen3vlinstruct': 'Qwen3-VL-Instruct',
    'glm46v': 'GLM-4.6V',
    'glm41v': 'GLM-4.1V-Thinking',
    'minicpmv45': 'MiniCPM-v4.5',
    'minicpmv26': 'MiniCPM-v2.6',
    'minicpmlama3v25': 'MiniCPM-Llama3-V 2.5',
    'moondream3': 'moondream3-preview',
    'moondream2': 'Moondream2',
    'internlmxcomposer2vl': 'InternLM-XComposer2-VL',

    'llama32visioninstruct': 'Llama-3.2-11B-Vision-Instruct',
    'cogvlm2': 'CogVLM2',
    'cogvlmmoe': 'CogVLM-MOE',
    'phi35vision': 'Phi-3.5-vision-instruct',
    'phi3vision128k': 'Phi-3-vision-128k-instruct',
    'llama31vision': 'LLaMA-3.1-Vision',
    'zhipuvision': 'Zhipu-Vision',
    'zhipu-aivision': '智谱AI-Vision',
    'olmocr2': 'olmOCR-2',
    'internvl15': 'InternVL-1.5',
    'internvl20': 'InternVL-2.0',
    'yivl20': 'Yi-VL-2.0',
    'gemma3': 'Gemma-3',
    'granitedocling': 'Granite-DocLing',
    'lfm2vl': 'Lfm-2-VL',
    'llama3visionalpha': 'Llama3-Vision-Alpha',
    'llava15': 'LLaVA-1.5',
    'llava16': 'LLaVA-1.6',
    'obsidian': 'Obsidian',
    'yutuvl4binstruct': 'Youtu-VL-4B-Instruct',
    'eraxvl7bv15': 'EraX-VL-7B-V1.5',
    'mimovl7brl': 'MiMo-VL-7B-RL',
    'yivl6b': 'Yi-VL-6B',
    'lightonocr21b': 'LightOnOCR-2-1B',
    'minicpmo45': 'MiniCPM-O-4.5',
    'qwen25omni': 'Qwen2.5-Omni-7B',
    'asidcaptioner7b': 'ASID-Captioner-7B',
    'zen3vli1': 'zen3-vl-i1',
    'paddleocr': 'PaddleOCR-VL-1.5'
}

# -------------------------- Hook 工具模块 --------------------------

class StderrFilter:
    def __init__(self):
        self.original_stderr = sys.stderr
        self.buffer = io.StringIO()

    def write(self, text):
        if "find_slot" in text:
            match = re.search(
                r'find_slot: non-consecutive token position (\d+) after (\d+) for sequence (\d+) with (\d+) new tokens',
                text
            )
            if match:
                pos = match.group(1)
                prev_pos = match.group(2)
                seq = match.group(3)
                tokens = match.group(4)
                translated = f"find_slot: 序列 {seq} 的非连续token位置 {pos}（前一位置 {prev_pos}），使用 {tokens} 个新tokens"
                self.original_stderr.write(translated + "\n")
            else:
                translated = text.replace("find_slot: non-consecutive token position", "find_slot: 非连续token位置")
                translated = translated.replace("after", "在")
                translated = translated.replace("for sequence", "对于序列")
                translated = translated.replace("with", "使用")
                translated = translated.replace("new tokens", "新tokens")
                self.original_stderr.write(translated)
        else:
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()


@contextlib.contextmanager
def filter_stderr():
    filter = StderrFilter()
    sys.stderr = filter
    try:
        yield
    finally:
        sys.stderr = filter.original_stderr


class MultiLevelFilter:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self, level=INFO):
        self.original_stderr = sys.stderr
        self.level = level
        self._filter_patterns = [
            (r"find_slot", self.INFO),
            (r"kv cache", self.DEBUG),
            (r"loading model", self.INFO),
            (r"error|failed|exception", self.ERROR),
        ]

    def write(self, text):
        if self.level <= self.DEBUG:
            self.original_stderr.write(text)
            return
        should_print = True
        for pattern, min_level in self._filter_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if self._get_level(pattern) < self.level:
                    should_print = False
                    break
        if should_print:
            self.original_stderr.write(text)

    def _get_level(self, pattern):
        for p, lvl in self._filter_patterns:
            if p == pattern:
                return lvl
        return self.INFO

    def flush(self):
        self.original_stderr.flush()

    def set_level(self, level):
        self.level = level

    def add_filter(self, pattern, level):
        self._filter_patterns.append((pattern, level))


@contextlib.contextmanager
def filtered_stderr(level=MultiLevelFilter.INFO):
    filter = MultiLevelFilter(level)
    sys.stderr = filter
    try:
        yield
    finally:
        sys.stderr = filter.original_stderr


class ModelUnloadHook:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._llama_storage = None
        self._original_unload = None

    def register_hook(self, llama_storage):
        if not _has_comfy or hasattr(mm, "unload_all_models_backup"):
            return
        self._llama_storage = llama_storage
        self._original_unload = mm.unload_all_models

        def patched_unload_all_models(*args, **kwargs):
            if self._llama_storage is not None:
                try:
                    self._llama_storage.clean(all=True)
                    print("[llama-cpp] Model cleaned via unload hook")
                except Exception:
                    pass
            return self._original_unload(*args, **kwargs)

        mm.unload_all_models = patched_unload_all_models
        mm.unload_all_models_backup = self._original_unload
        print("[llama-cpp] Model unload hook applied")

    def uninstall_hook(self):
        if not _has_comfy or self._original_unload is None:
            return
        if hasattr(mm, "unload_all_models_backup"):
            mm.unload_all_models = mm.unload_all_models_backup
            delattr(mm, "unload_all_models_backup")
            print("[llama-cpp] Model unload hook removed")


def install_model_unload_hook(llama_storage):
    ModelUnloadHook().register_hook(llama_storage)


def uninstall_model_unload_hook():
    ModelUnloadHook().uninstall_hook()


def apply_acceleration_hooks(llama_storage):
    install_model_unload_hook(llama_storage)

# -------------------------- 进度条模块 --------------------------

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing", disable=False, **kwargs):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        if iterable is not None and total is None:
            try:
                self.total = len(iterable)
            except (TypeError, AttributeError):
                self.total = None
        if _has_comfy and comfy is not None:
            self.pbar = comfy.utils.ProgressBar(self.total) if self.total is not None else None
        else:
            self.pbar = None
        if tqdm is not None:
            self.tqdm = tqdm(
                iterable=self.iterable,
                total=self.total,
                desc=self.desc,
                disable=disable,
                dynamic_ncols=True,
                file=sys.stdout,
                **kwargs
            )
        else:
            self.tqdm = None

    def __iter__(self):
        if self.tqdm is None:
            return
        for item in self.tqdm:
            if self.pbar:
                self.pbar.update(1)
            yield item

    def update(self, n=1):
        if self.tqdm:
            self.tqdm.update(n)
        if self.pbar:
            self.pbar.update(n)

    def set_description(self, desc):
        if self.tqdm:
            self.tqdm.set_description(desc)

    def set_postfix(self, *args, **kwargs):
        if self.tqdm:
            self.tqdm.set_postfix(*args, **kwargs)

    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self.total


def create_progress_bar(total=None, desc="Processing", disable=False):
    return cqdm(total=total, desc=desc, disable=disable)

# 动态检测llama_cpp_python中的ChatHandler和模型支持（使用新的管理器）
def detect_available_chat_handlers():
    """
    自动检测llama_cpp.llama_chat_format中可用的ChatHandler
    现在使用ChatHandlerManager进行检测
    """
    # 从管理器获取已检测到的信息
    available_handlers = chat_handler_manager.get_all_handlers()
    detected_models = chat_handler_manager.get_all_models()
    
    print(f"【模型检测】发现{len(available_handlers)}个可用的ChatHandler")
    print(f"【模型检测】推断出{len(detected_models)}个模型")
    
    return available_handlers, detected_models

# 执行模型检测（使用新的管理器）
available_handlers, detected_models = detect_available_chat_handlers()

# 生成chat_handlers列表 - 合并基础模型和动态检测的模型
# 优先使用管理器检测到的模型，确保与llama-cpp-python版本匹配
manager_models = chat_handler_manager.get_all_models()
chat_handlers = ["None"] + base_models + manager_models

# 去重，保持顺序
seen = set()
chat_handlers = [x for x in chat_handlers if not (x in seen or seen.add(x))]

print(f"【模型列表】最终生成了{len(chat_handlers)}个模型选项")
print(f"【模型列表】前10个模型：{chat_handlers[:10]}")

# 构建模型信息映射表（供后续使用）
all_models = []
for handler_name in available_handlers:
    info = chat_handler_manager.get_handler_info(handler_name)
    if info:
        all_models.append({
            "handler": handler_name,
            "models": [info['display_name']],
            "is_qwen": info['is_qwen'],
            "param_type": info['param_type']
        })

# 动态导入所有ChatHandler（通过管理器已缓存，这里仅做兼容性处理）
for model_info in all_models:
    handler_name = model_info["handler"]
    models = model_info["models"]
    
    # 检查是否已在管理器中
    if chat_handler_manager.get_handler(handler_name):
        print(f"【模型支持】成功兼容{models[0]}模型（通过ChatHandlerManager）")
    else:
        print(f"【模型支持】{models[0]}模型暂未加载")


# -------------------------- 通用工具类 --------------------------
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    jpeg_quality = 70 if HARDWARE_INFO["is_low_perf"] else 75
    optimize = True if HARDWARE_INFO["is_high_perf"] else False
    img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=optimize)
    img_base64 = base64.b64encode(buffered.getbuffer()).decode('utf-8')
    return img_base64

def scale_image(image, max_size: int = 128):
    try:
        # 检查输入类型并进行相应处理
        if hasattr(image, 'cpu'):
            # PyTorch 张量
            img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        elif isinstance(image, np.ndarray):
            # numpy 数组
            img_np = np.clip(255.0 * image.squeeze(), 0, 255).astype(np.uint8)
        else:
            # 其他类型，尝试直接转换
            img_np = np.array(image)
        
        img_pil = Image.fromarray(img_np)
        w, h = img_pil.size
        scale = min(max_size / max(w, h), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resample_mode = Image.Resampling.LANCZOS if HARDWARE_INFO["cpu_cores"] >= 8 else Image.Resampling.BILINEAR
            img_pil = img_pil.resize((new_w, new_h), resample=resample_mode)
        img_np = np.array(img_pil)
        return img_np
    except Exception as e:
        print(f"【错误】图片缩放失败：{e}")
        # 安全回退：尝试直接转换为numpy数组
        try:
            if hasattr(image, 'cpu'):
                return np.array(Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)))
            elif isinstance(image, np.ndarray):
                return np.array(Image.fromarray(np.clip(255.0 * image.squeeze(), 0, 255).astype(np.uint8)))
            else:
                return np.array(image)
        except:
            # 最终回退：返回原始图像
            return image

def image_to_data_uri(image):
    try:
        if isinstance(image, str):
            image = Image.open(image)
        if hasattr(image, 'cpu'):
            img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            image = Image.fromarray(img_np)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{img_str}"
        return data_uri
    except Exception as e:
        print(f"【图像处理】转换data URI失败: {e}")
        return None

def parse_json(json_str):
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except Exception as e:
        print(f"【错误】解析JSON失败：{e}")
        return {}

# 模型文件名到ChatHandler的映射（用于自动检测）
MODEL_FILE_CHAT_HANDLER_MAP = {
    # Qwen 系列
    'qwen2.5-vl': 'Qwen2.5-VL',
    'qwen3-vl': 'Qwen3-VL',
    'qwen2.5-omni': 'Qwen2.5-Omni',
    # MiniCPM 系列
    'minicpm-v-2.6': 'MiniCPM-v2.6',
    'minicpm-v-4.5': 'MiniCPM-v4.5',
    'minicpm-o-4.5': 'MiniCPM-O-4.5',
    'minicpm-llama3-v-2.5': 'MiniCPM-Llama3-V 2.5',
    # GLM 系列
    'glm-4.6v': 'GLM-4.6V',
    'glm-4.1v': 'GLM-4.1V-Thinking',
    # LLaVA 系列
    'llava-1.6': 'LLaVA-1.6',
    'llava-1.5': 'LLaVA-1.5',
    'nanollava': 'nanoLLaVA',
    # 其他模型
    'gemma-3': 'Gemma-3',
    'gemma-4': 'Gemma-4',
    'moondream2': 'Moondream2',
    'moondream-2': 'Moondream2',
    'llama-3.2-11b-vision': 'Llama-3.2-11B-Vision-Instruct',
    'llama-3.1-vision': 'LLaMA-3.1-Vision',
    'phi-3.5-vision': 'Phi-3.5-vision-instruct',
    'phi-3-vision': 'Phi-3-vision-128k-instruct',
    'yi-vl': 'Yi-VL-6B',
    'internvl': 'InternVL-1.5',
    'internvl2': 'InternVL-2.0',
    'glm-4v': 'GLM-4.6V',
    'joycaption': 'llama-joycaption',
    'llama-joycaption': 'llama-joycaption',
    'olmocr': 'olmOCR-2',
    'lightonocr': 'LightOnOCR-2-1B',
    'paddleocr': 'PaddleOCR-VL-1.5',
    'granite-docling': 'Granite-DocLing',
    'lfm-2-vl': 'Lfm-2-VL',
    'lfm-2.5-vl': 'Lfm-2.5-VL',
    'erax-vl': 'EraX-VL-7B-V1.5',
    'mimo-vl': 'MiMo-VL-7B-RL',
    'asid-captioner': 'ASID-Captioner-7B',
    'zen3-vl': 'zen3-vl-i1',
    'youtu-vl': 'Youtu-VL-4B-Instruct',
    'obsidian': 'Obsidian',
    'cogvlm': 'CogVLM2',
    'cogvlm2': 'CogVLM2',
}

def detect_model_chat_handler(model_filename):
    """
    根据模型文件名自动检测应该使用的ChatHandler
    优先使用ChatHandlerManager进行智能匹配
    
    Args:
        model_filename: 模型文件名（如 'Qwen3-VL-8B-Instruct-Q4_K_M.gguf'）
    
    Returns:
        str: 推荐的ChatHandler显示名称，如果无法检测则返回None
    """
    if not model_filename:
        return None
    
    # 首先尝试使用ChatHandlerManager进行匹配
    handler_name, handler_cls = chat_handler_manager.get_handler_for_model(model_filename)
    if handler_name and handler_cls:
        info = chat_handler_manager.get_handler_info(handler_name)
        if info:
            print(f"【智能匹配】模型 {model_filename} -> {info['display_name']} ({handler_name})")
            return info['display_name']
    
    # 回退到原有的映射表匹配
    # 转换为小写以便匹配
    filename_lower = model_filename.lower()
    
    # 移除文件扩展名
    filename_lower = filename_lower.replace('.gguf', '').replace('.safetensors', '')
    
    # 先处理特殊情况（避免被通用规则覆盖）
    if 'nanollava' in filename_lower:
        return 'nanoLLaVA'
    
    # 尝试直接匹配
    for pattern, handler_display_name in MODEL_FILE_CHAT_HANDLER_MAP.items():
        if pattern in filename_lower:
            return handler_display_name
    
    # 特殊规则：如果包含特定关键词
    # 优先检测Omni模型（Omni模型同时支持音频和视觉）
    if 'qwen' in filename_lower and '2.5' in filename_lower:
        if 'omni' in filename_lower:
            return 'Qwen2.5-Omni' 
        elif 'vl' in filename_lower:
            return 'Qwen2.5-VL'
    elif 'qwen' in filename_lower and 'vl' in filename_lower:
        if '3' in filename_lower:
            return 'Qwen3-VL'
    
    # Qwen3.5 检测（非VL版本）
    if 'qwen' in filename_lower and '3.5' in filename_lower and 'vl' not in filename_lower:
        return 'Qwen3.5'
    
    if 'minicpm' in filename_lower:
        if 'llama3' in filename_lower:
            return 'MiniCPM-Llama3-V 2.5'
        elif 'o-4' in filename_lower or 'o_4' in filename_lower or 'o4' in filename_lower:
            return 'MiniCPM-O-4.5'
        elif 'v-4' in filename_lower or 'v_4' in filename_lower or 'v4' in filename_lower:
            return 'MiniCPM-v4.5'
        elif 'v-2' in filename_lower or 'v_2' in filename_lower or 'v2' in filename_lower:
            return 'MiniCPM-v2.6'
    
    if 'glm' in filename_lower and ('4v' in filename_lower or '4.6v' in filename_lower or '46v' in filename_lower):
        return 'GLM-4.6V'
    
    if 'glm' in filename_lower and ('4.1v' in filename_lower or '41v' in filename_lower):
        return 'GLM-4.1V-Thinking'
    
    if 'llava' in filename_lower:
        if '1.6' in filename_lower:
            return 'LLaVA-1.6'
        elif '1.5' in filename_lower:
            return 'LLaVA-1.5'
    
    if 'gemma' in filename_lower and '3' in filename_lower:
        return 'Gemma-3'
    
    if 'moondream' in filename_lower:
        return 'Moondream2'
    
    if 'mimo' in filename_lower and 'vl' in filename_lower:
        return 'MiMo-VL-7B-RL'
    
    return None

def get_model_info(model_name):
    return {"model_name": model_name, "supported": False}

def qwen3bbox(image, json):
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    bboxes = []
    for item in json:
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json_data, mode):
    label_colors = {}
    img = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
    except Exception:
        pass

    for item in json_data:
        try:
            label = item.get("label", item.get("text_content", "bbox"))
        except Exception:
            label = "bbox"
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)

        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

# -------------------------- 模型存储类（单例模式） --------------------------
class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    model_path = None
    model_name = None
    messages = {}
    sys_prompts = {}
    
    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
            print(f"【会话管理】已清理所有会话状态")
        else:
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
            print(f"【会话管理】已清理会话状态 (id={id})")
    
    @classmethod
    def clean(cls, all=False):
        try:
            if cls.llm is not None:
                cls.llm.close()
                print(f"【资源释放】成功关闭LLM模型")
        except Exception as e:
            print(f"【提示】关闭LLM模型失败（忽略，继续释放资源）：{e}")
        
        if cls.chat_handler is not None:
            release_methods = [("_exit_stack", lambda x: x.close()), ("close", lambda x: x()), ("cleanup", lambda x: x())]
            for attr, func in release_methods:
                if hasattr(cls.chat_handler, attr):
                    try:
                        attr_value = getattr(cls.chat_handler, attr)
                        if attr_value is not None:
                            func(attr_value)
                            print(f"【资源释放】成功释放ChatHandler资源 ({attr})")
                        else:
                            print(f"【提示】ChatHandler资源 ({attr}) 为None，跳过释放")
                    except Exception as e:
                        print(f"【提示】释放ChatHandler资源失败 ({attr})（忽略）：{e}")
        
        # 清理结果缓存
        try:
            from .nodes.llama_cpp_unified_inference import llama_cpp_unified_inference
            if hasattr(llama_cpp_unified_inference, "_result_cache"):
                llama_cpp_unified_inference._result_cache.clear()
                print(f"【结果缓存】已清理所有推理结果缓存")
        except Exception as e:
            print(f"【提示】清理结果缓存失败（忽略）：{e}")
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        cls.model_path = None
        cls.model_name = None
        
        if all:
            cls.clean_state()
    
    @classmethod
    def get_chat_handler_cls(cls, chat_handler_name):
        if not chat_handler_name or chat_handler_name == "None":
            return None

        handler_map = {
            "Qwen3.5": Qwen35ChatHandler,
            "Qwen3-VL": Qwen3VLChatHandler,
            "Qwen2.5-VL": Qwen25VLChatHandler,
            "Qwen2.5-Omni": Qwen25VLChatHandler,
            "LLaVA-1.5": Llava15ChatHandler,
            "LLaVA-1.6": Llava16ChatHandler,
            "Moondream2": MoondreamChatHandler,
            "nanoLLaVA": NanoLlavaChatHandler,
            "llama3-Vision-Alpha": Llama3VisionAlphaChatHandler,
            "MiniCPM-v2.6": MiniCPMv26ChatHandler,
            "MiniCPM-v4.5": MiniCPMv26ChatHandler,
            "MiniCPM-O-4.5": MiniCPMv26ChatHandler,
            "Gemma3": Gemma3ChatHandler,
            "Gemma4": Gemma4ChatHandler,
            "GLM-4.6V": GLM46VChatHandler,
            "GLM-4.1V-Thinking": GLM41VChatHandler,
            "Lfm-2-VL": LFM2VLChatHandler,
            "LFM2-VL": LFM2VLChatHandler,
            "LFM2.5-VL": LFM25VLChatHandler,
            "Granite-Docling": GraniteDoclingChatHandler,
        }

        handler = handler_map.get(chat_handler_name)
        if handler:
            return handler
        
        if chat_handler_name.startswith("Qwen2.5-Omni-"):
            return Qwen25VLChatHandler
        if chat_handler_name.startswith("Qwen3.5"):
            return Qwen35ChatHandler
        if chat_handler_name.startswith("Qwen3-VL"):
            return Qwen3VLChatHandler
        if chat_handler_name.startswith("MiniCPM-v4.5") or chat_handler_name.startswith("MiniCPM-v2.6"):
            return MiniCPMv26ChatHandler
        if "Llama3" in chat_handler_name and "MiniCPM" in chat_handler_name:
            return MiniCPMv26ChatHandler
        if chat_handler_name.startswith("GLM-4.6V"):
            return GLM46VChatHandler
        if chat_handler_name in ["InternVL-1.5", "InternVL-2.0", "InternLM-XComposer2-VL"]:
            return None

        handler = getattr(chat_handler_manager, chat_handler_name, None)
        return handler if callable(handler) else None
    
    @classmethod
    def init_chat_handler(cls, handler_cls, mmproj_path, chat_handler_name, image_max_tokens, image_min_tokens, model_path=None, video_input=False, enable_thinking=False):
        """
        初始化ChatHandler - 参考 ComfyUI-llama-cpp_vlm 的简洁处理方式
        """
        if handler_cls is None:
            return None
        
        init_params = {"verbose": False}
        think_mode = enable_thinking or ("Thinking" in chat_handler_name)
        
        try:
            handler_name = handler_cls.__name__
            print(f"【ChatHandler初始化】开始初始化：{handler_name}")
            
            vl_handlers = ["Qwen3-VL", "Qwen3-VL-Thinking", "Qwen2.5-VL", "Qwen2.5-Omni"]
            
            if mmproj_path and chat_handler_name in vl_handlers:
                init_params["clip_model_path"] = mmproj_path
                if chat_handler_name in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                    init_params["force_reasoning"] = think_mode
                    if image_max_tokens > 0:
                        init_params["image_max_tokens"] = image_max_tokens
                    if image_min_tokens > 0:
                        init_params["image_min_tokens"] = image_min_tokens
            elif mmproj_path:
                init_params["clip_model_path"] = mmproj_path
            
            if chat_handler_name in ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking", "MiniCPM-O-4.5", "GLM-4.6V", "GLM-4.6V-Thinking", "Qwen3.5", "Qwen3.5-Thinking"]:
                init_params["enable_thinking"] = think_mode
            
            if _has_mtmd:
                if image_max_tokens > 0:
                    init_params["image_max_tokens"] = image_max_tokens
                if image_min_tokens > 0:
                    init_params["image_min_tokens"] = image_min_tokens
            
            print(f"【ChatHandler初始化】参数：{list(init_params.keys())}")
            
            cls.chat_handler = handler_cls(**init_params)
            print(f"【ChatHandler初始化】✅ 成功：{handler_name}")
            return cls.chat_handler
        except Exception as e:
            print(f"【ChatHandler初始化错误】{e}")
            print(f"【提示】请更新 llama-cpp-python：https://github.com/JamePeng/llama-cpp-python/releases")
            return None

    @classmethod
    def load_model(cls, config):
        try:
            # 首先释放旧的模型资源，避免资源冲突
            print(f"【模型加载】开始加载新模型，正在释放旧资源...")
            cls.clean()
            
            model = config["model"]
            enable_mmproj = config["enable_mmproj"]
            mmproj = config["mmproj"]
            chat_handler_name = config["chat_handler"]
            device_mode = config.get("device_mode", "GPU")  # 默认使用 GPU 模式
            n_ctx = config["n_ctx"]
            n_gpu_layers = config["n_gpu_layers"]
            vram_limit = config["vram_limit"]
            image_min_tokens = config["image_min_tokens"]
            image_max_tokens = config["image_max_tokens"]
            
            # 新增参数
            n_batch = config.get("n_batch", 2048)
            n_ubatch = 0
            n_threads = config.get("n_threads", 0)
            n_threads_batch = config.get("n_threads_batch", 0)
            cache_prompt = config.get("cache_prompt", False)
            enable_thinking = config.get("enable_thinking", False)
            
            # 自动计算高级参数（从UI中移除，改为后台自动调整）
            # Flash Attention：根据GPU类型自动决定
            flash_attention = "Auto"
            # K/Q/V卸载：默认启用（提升GPU性能）
            offload_kqv = True
            # 低显存模式：根据显存大小自动决定
            low_vram = False
            # 内存映射：默认启用（减少内存占用）
            use_mmap = True
            # 内存锁定：默认禁用（除非AMD GPU）
            use_mlock = False
            # 半精度KV缓存：默认启用（减少显存占用）
            f16_kv = True
            
            # 根据设备模式调整参数
            if device_mode == "CPU":
                # CPU 模式：忽略 GPU 相关参数，强制使用纯 CPU
                print(f"【设备模式】使用 CPU 模式")
                n_gpu_layers = 0
                vram_limit = -1
            else:
                # GPU 模式：使用用户设置的参数
                print(f"【设备模式】使用 GPU 模式（n_gpu_layers={n_gpu_layers}, vram_limit={vram_limit}GB）")
            
            # 检查是否是Qwen3-VL模型，如果是则强制设置image_min_tokens为1024
            is_qwen3vl = "qwen3-vl" in model.lower() or "qwen3vl" in model.lower()
            if is_qwen3vl and image_min_tokens < 1024:
                print(f"【Qwen3-VL优化】强制设置image_min_tokens为1024（当前：{image_min_tokens}）")
                image_min_tokens = 1024
            if is_qwen3vl and image_max_tokens < image_min_tokens:
                image_max_tokens = image_min_tokens
                print(f"【Qwen3-VL优化】强制设置image_max_tokens为{image_max_tokens}")
            
            # 构建模型路径
            if os.path.isabs(model) and os.path.exists(model):
                model_path = os.path.normpath(model)
            else:
                model_path = os.path.join(folder_paths.models_dir, 'LLM', model)

            # 检查模型路径是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在：{model_path}")
            
            # 获取模型格式
            model_ext = os.path.splitext(model)[1].lower()
            if model_ext != ".gguf":
                print(f"【提示】仅支持GGUF格式模型，当前格式：{model_ext}")

            # 构建MMProj路径
            mmproj_path = None
            if enable_mmproj and mmproj != "None":
                mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
                if not os.path.exists(mmproj_path):
                    raise FileNotFoundError(f"MMProj文件不存在：{mmproj_path}")

            # 只有在启用多模态功能时才初始化ChatHandler
            cls.chat_handler = None
            
            # 检测是否为纯文本模型（不需要VL支持的模型）
            model_lower = model.lower()
            is_text_only_model = (
                ("qwen35" in model_lower or "qwen3.5" in model_lower) and 
                "vl" not in model_lower and "vision" not in model_lower and 
                "omni" not in model_lower
            ) or (
                "llama" in model_lower and "vision" not in model_lower and
                "qwen" not in model_lower and "glm" not in model_lower and
                "minicpm" not in model_lower and "moondream" not in model_lower
            )
            
            # 对于 Qwen3.5，即使是纯文本模型也需要初始化 ChatHandler 以控制 thinking 模式
            need_chat_handler = enable_mmproj and not is_text_only_model or "qwen35" in model_lower or "qwen3.5" in model_lower
            
            if need_chat_handler:
                # 获取并初始化ChatHandler
                handler_cls = cls.get_chat_handler_cls(chat_handler_name)
                cls.chat_handler = cls.init_chat_handler(handler_cls, mmproj_path, chat_handler_name, image_max_tokens, image_min_tokens, model, video_input=False, enable_thinking=enable_thinking)
                
                # ChatHandler初始化失败时的处理
                if cls.chat_handler is None:
                    if chat_handler_name == "None":
                        print(f"【警告】启用了MMProj但未选择ChatHandler，ChatHandler将为None")
                    elif handler_cls is None:
                        print(f"【警告】无法找到ChatHandler：{chat_handler_name}，ChatHandler将为None")
                    else:
                        print(f"【警告】ChatHandler初始化失败：{chat_handler_name}，ChatHandler将为None")
                        print(f"【提示】模型将以基础模式加载，TTS音频功能仍可使用")
                else:
                    print(f"【成功】已启用MMProj并初始化ChatHandler：{type(cls.chat_handler).__name__}")
            else:
                if is_text_only_model:
                    print(f"【提示】{model} 是纯文本模型，跳过ChatHandler初始化")
                else:
                    print(f"【提示】未启用MMProj，ChatHandler为None，模型将以基础模式加载")
                print(f"【提示】TTS音频功能仍可正常使用")
            
            # 启用CUDA优化
            enable_cuda_optimizations()
            
            # 加载LLM模型前的智能检查
            print(f"【模型加载】开始加载LLM模型：{model}（上下文：{n_ctx}，GPU层数：{n_gpu_layers}）")

            # 计算推荐的最大GPU层数（仅在 GPU 模式下）
            recommended_gpu_layers = n_gpu_layers
            is_qwen35 = "qwen35" in model.lower() or "qwen3.5" in model.lower()

            if device_mode == "GPU" and (HARDWARE_INFO["has_cuda"] or HARDWARE_INFO["has_rocm"]) and n_gpu_layers > 0:
                gpu_vram = HARDWARE_INFO["gpu_vram_total"]
                gpu_vendor = HARDWARE_INFO["gpu_vendor"]

                if model_ext == ".gguf":
                    mmproj_size_gb = 0
                    if enable_mmproj and mmproj != "None" and mmproj_path:
                        mmproj_size_gb = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)

                    recommended_gpu_layers = calculate_vram_layers(model_path, gpu_vram, mmproj_size_gb)
                    print(f"【VRAM计算】推荐GPU层数={recommended_gpu_layers}")
                    
                    # Qwen3.5模型特殊内存优化：降低GPU层数以确保推理成功
                    is_qwen35_model = "qwen35" in model.lower() or "qwen3.5" in model.lower()
                    if is_qwen35_model and recommended_gpu_layers > 16:
                        original_layers = recommended_gpu_layers
                        recommended_gpu_layers = min(recommended_gpu_layers, 16)
                        print(f"【Qwen3.5优化】降低GPU层数从{original_layers}到{recommended_gpu_layers}以确保推理成功")
            elif device_mode == "CPU":
                print(f"【CPU模式】跳过GPU层数计算")

            # 构建模型参数
            gpu_vendor = HARDWARE_INFO["gpu_vendor"]

            # 根据设备模式和GPU厂商设置参数
            if device_mode == "CPU":
                n_batch = 1024
                n_threads = os.cpu_count() or 8
                n_threads_batch = os.cpu_count() or 8
                low_vram = True
                use_mmap = True
                use_mlock = False
                f16_kv = True
            elif gpu_vendor == "amd":
                n_batch = 1024
                n_threads = os.cpu_count() or 8
                n_threads_batch = os.cpu_count() or 8
                low_vram = HARDWARE_INFO["is_low_perf"]
                use_mmap = False
                use_mlock = True
                f16_kv = True
            else:
                n_batch = 2048
                n_threads = os.cpu_count() or 8
                n_threads_batch = os.cpu_count() or 16
                low_vram = HARDWARE_INFO["is_low_perf"]
                use_mmap = True
                use_mlock = False
                f16_kv = True
            
            # Qwen3.5模型特殊内存优化：降低n_batch以确保推理成功
            if is_qwen35 and device_mode == "GPU":
                original_batch = n_batch
                n_batch = min(n_batch, 512)
                if original_batch > n_batch:
                    print(f"【Qwen3.5优化】降低n_batch从{original_batch}到{n_batch}以确保推理成功")
                # Qwen3.5模型需要至少4096的上下文长度
                if n_ctx < 4096:
                    print(f"【Qwen3.5优化】提高n_ctx从{n_ctx}到4096以满足模型最小要求")
                    n_ctx = 4096

            llama_kwargs = {
                "model_path": model_path,
                "chat_handler": cls.chat_handler,
                "n_gpu_layers": recommended_gpu_layers if device_mode == "GPU" else 0,
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "verbose": False,
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
                "low_vram": low_vram,
                "use_mmap": use_mmap,
                "use_mlock": use_mlock,
                "f16_kv": f16_kv,
                "cache_prompt": cache_prompt,
            }

            # Flash Attention 配置（仅 NVIDIA GPU）
            if device_mode == "GPU" and gpu_vendor == "NVIDIA":
                attention_type = config.get("attention_type", "Auto")
                try:
                    # 检测 Flash Attention 是否可用
                    flash_available, flash_version = check_flash_attention()
                    llama_sig = inspect.signature(llama_cpp.Llama.__init__)
                    if 'flash_attn' in llama_sig.parameters:
                        if attention_type == "Flash" or (attention_type == "Auto" and flash_available):
                            llama_kwargs["flash_attn"] = True
                            print(f"【Flash Attention】已启用{flash_version if flash_version else ''}")
                except Exception as e:
                    print(f"【Flash Attention】配置失败: {e}")
                    pass
            
            # Qwen3.5模型启用低显存模式
            if is_qwen35 and device_mode == "GPU":
                llama_kwargs["low_vram"] = True

            # 打印优化参数
            if device_mode == "CPU":
                print(f"【CPU模式】n_threads={n_threads}, n_batch={n_batch}")
            elif gpu_vendor == "amd":
                print(f"【AMD ROCm】GPU层数={recommended_gpu_layers}, n_batch={n_batch}")
            else:
                print(f"【NVIDIA CUDA】GPU层数={recommended_gpu_layers}, n_batch={n_batch}")
            
            # 尝试加载模型，失败时提供降级策略
            try:
                # 直接加载模型
                print(f"【模型加载】正在加载模型...")
                cls.llm = Llama(**llama_kwargs)

                # 确保chat_handler被正确设置到Llama对象
                if hasattr(cls.llm, 'chat_handler'):
                    if cls.chat_handler is not None and cls.llm.chat_handler is None:
                        cls.llm.chat_handler = cls.chat_handler
                        print(f"【模型加载】已设置chat_handler")
                else:
                    print(f"【模型加载】Llama对象不支持chat_handler属性")

                # 显示模型加载详细信息
                handler_name = type(cls.chat_handler).__name__ if cls.chat_handler else 'None'
                mmproj_status = "已启用" if enable_mmproj and mmproj != "None" else "未启用"
                gpu_info = f", GPU层数={recommended_gpu_layers}" if device_mode == "GPU" else ""
                print(f"【模型加载】✅ 成功！路径：{model_path}，格式：{model_ext}，上下文：{n_ctx}，设备：{device_mode}{gpu_info}，ChatHandler：{handler_name}，MMProj：{mmproj_status}，n_batch={n_batch}")

                # 禁用模型缓存
                # MODEL_CACHE[cache_key] = cls.llm
                # print(f"【模型缓存】已缓存模型：{model_path}")
            except Exception as e:
                error_msg = str(e)

                # 分析错误类型，提供针对性建议
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    print(f"【显存错误】{error_msg}")
                    print(f"【建议】降低n_gpu_layers({recommended_gpu_layers})或n_ctx({n_ctx})，或使用更小模型")
                    print(f"【尝试降级】使用纯CPU模式加载...")
                    llama_kwargs["n_gpu_layers"] = 0
                    llama_kwargs["low_vram"] = True
                    llama_kwargs["f16_kv"] = False

                    try:
                        # 直接加载模型（降级模式）
                        print(f"【模型加载】正在以CPU模式加载...")
                        cls.llm = Llama(**llama_kwargs)

                        # 确保chat_handler被正确设置到Llama对象
                        if hasattr(cls.llm, 'chat_handler'):
                            if cls.chat_handler is not None and cls.llm.chat_handler is None:
                                cls.llm.chat_handler = cls.chat_handler
                                print(f"【模型加载】(CPU) 已设置chat_handler")

                        # 显示CPU模式模型加载详细信息
                        handler_name = type(cls.chat_handler).__name__ if cls.chat_handler else 'None'
                        mmproj_status = "已启用" if enable_mmproj and mmproj != "None" else "未启用"
                        print(f"【模型加载】✅ CPU模式成功！路径：{model_path}，格式：{model_ext}，上下文：{n_ctx}，ChatHandler：{handler_name}，MMProj：{mmproj_status}，n_batch={n_batch}")
                        print(f"【提示】CPU模式推理速度较慢，建议使用更小的模型")

                        # 禁用模型缓存
                        # MODEL_CACHE[cache_key] = cls.llm
                        # print(f"【模型缓存】已缓存CPU模式模型：{model_path}")
                    except Exception as fallback_e:
                        raise RuntimeError(f"【错误】加载LLM模型失败（包括降级尝试）：{fallback_e}") from fallback_e
                else:
                    # 其他类型错误
                    raise RuntimeError(f"【错误】加载LLM模型失败：{e}") from e
            
            cls.current_config = config
            cls.model_path = model_path
            cls.model_name = model
        except Exception as e:
            print(f"【错误】加载模型失败：{e}")
            raise


# -------------------------- Flash Attention 检测 --------------------------
FLASH_ATTENTION_AVAILABLE = False
FLASH_ATTENTION_VERSION = None


def check_flash_attention():
    global FLASH_ATTENTION_AVAILABLE, FLASH_ATTENTION_VERSION
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False, None

        try:
            from flash_attn_3 import flash_attn_func
            FLASH_ATTENTION_AVAILABLE = True
            FLASH_ATTENTION_VERSION = "flash_attention_3"
            print("【Flash Attention】Flash Attention 3 可用")
            return True, "flash_attention_3"
        except ImportError:
            pass

        try:
            from flash_attn import flash_attn_func
            FLASH_ATTENTION_AVAILABLE = True
            FLASH_ATTENTION_VERSION = "flash_attention_2"
            print("【Flash Attention】Flash Attention 2 可用")
            return True, "flash_attention_2"
        except ImportError:
            pass

        return False, None
    except:
        return False, None


def enable_cuda_optimizations():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except:
            pass


# -------------------------- 音频处理函数 --------------------------
def convert_audio_to_wav_bytes(audio: Dict) -> Optional[bytes]:
    try:
        import io
        import wave

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)

        if waveform is None:
            return None

        if isinstance(waveform, torch.Tensor):
            if waveform.is_cuda:
                waveform = waveform.detach().contiguous().to("cpu", non_blocking=True)
            waveform = waveform.numpy()
        elif hasattr(waveform, 'numpy'):
            waveform = waveform.numpy()

        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()

        waveform = np.asarray(waveform, dtype=np.float32)
        waveform = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(waveform.tobytes())

        buffer.seek(0)
        return buffer.read()
    except:
        return None


def stream_audio_processing(audio: Dict, chunk_size: int = 1024) -> Optional[bytes]:
    try:
        import io
        import wave

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)

        if waveform is None:
            return None

        if isinstance(waveform, torch.Tensor):
            if waveform.is_cuda:
                waveform = waveform.cpu()
            waveform = waveform.numpy()

        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)

            total_samples = len(waveform)
            for i in range(0, total_samples, chunk_size):
                chunk = waveform[i:i+chunk_size]
                chunk = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(chunk.tobytes())

        buffer.seek(0)
        return buffer.read()
    except:
        return None


def convert_audio_to_format(audio: Dict, format: str = "wav") -> Optional[bytes]:
    try:
        import io
        import numpy as np

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)

        if waveform is None:
            return None

        if isinstance(waveform, torch.Tensor):
            if waveform.is_cuda:
                waveform = waveform.cpu()
            waveform = waveform.numpy()

        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()

        buffer = io.BytesIO()

        if format.lower() == "wav":
            import wave
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                waveform = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(waveform.tobytes())
        else:
            sf.write(buffer, waveform, sample_rate, format=format.lower())

        buffer.seek(0)
        return buffer.read()
    except:
        return None


def create_audio_data_uri(audio_bytes: bytes, format: str = "wav") -> Optional[str]:
    try:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return f"data:audio/{format};base64,{audio_base64}"
    except:
        return None


# -------------------------- 基础推理引擎 --------------------------
class BaseInferenceEngine:
    def __init__(self, model_info: Dict):
        self.model_info = model_info
        self.model_type = model_info.get("type", "vl")
        self.model_subtype = model_info.get("subtype", "default")
        self.supports_audio = model_info.get("supports_audio", False)
        self.supports_vision = model_info.get("supports_vision", True)
        self._cache_manager = None
        self._memory_threshold = 0.85

    @property
    def cache_manager(self):
        if self._cache_manager is None:
            class SimpleCacheManager:
                def __init__(self):
                    self._image_cache = {}
                    self._audio_cache = {}
                def clear_all(self):
                    self._image_cache.clear()
                    self._audio_cache.clear()
                def cache_image(self, key, value):
                    self._image_cache[key] = value
                def cache_audio(self, key, value):
                    self._audio_cache[key] = value
            self._cache_manager = SimpleCacheManager()
        return self._cache_manager

    @property
    def _image_cache(self):
        return self.cache_manager._image_cache

    @property
    def _audio_cache(self):
        return self.cache_manager._audio_cache

    def build_messages(self, system_prompt: str, user_content: Union[str, List],
                       history: List[Dict] = None) -> List[Dict]:
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def process_images_to_content(self, images: torch.Tensor, max_size: int,
                                  preset_prompt: str = "") -> List[Dict]:
        content = []
        if preset_prompt:
            content.append({"type": "text", "text": preset_prompt})

        if images is not None:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()
                img_np = scale_image(img, max_size)
                img_base64 = image2base64(img_np)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })

        return content

    def _generate_image_cache_key(self, image_tensor: torch.Tensor, max_size: int) -> str:
        import hashlib
        try:
            img_shape = tuple(image_tensor.shape)
            img_mean = image_tensor.mean().item()
            hash_input = f"{img_shape}_{img_mean:.2f}_{max_size}"
            return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        except:
            import time
            return f"fallback_{int(time.time() * 1000)}_{max_size}"

    def process_audio_to_content(self, audio: Dict, model_subtype: str = "default",
                                 audio_format: str = "wav") -> Optional[Dict]:
        if audio is None:
            return None

        try:
            if model_subtype in ["qwen35", "qwen25_omni"]:
                return None

            audio_bytes = convert_audio_to_format(audio, audio_format)
            if audio_bytes is None:
                return None

            audio_uri = create_audio_data_uri(audio_bytes, audio_format)
            if audio_uri is None:
                return None

            if model_subtype == "minicpm_o":
                audio_content = {"type": "audio", "audio": audio_uri}
            else:
                audio_content = {"type": "audio_url", "audio_url": {"url": audio_uri}}

            return audio_content
        except:
            return None

    def create_chat_completion(self, llm, messages: List[Dict], params: Dict) -> Dict:
        try:
            stop_words = params.get("stop", ["</s>", "<|im_end|>"])
            completion_params = {
                "messages": messages,
                "max_tokens": params.get("max_tokens", 512),
                "temperature": params.get("temperature", 0.7),
                "top_k": params.get("top_k", 40),
                "top_p": params.get("top_p", 0.9),
                "min_p": params.get("min_p", 0.05),
                "repeat_penalty": params.get("repeat_penalty", 1.0),
                "frequency_penalty": params.get("frequency_penalty", 0.0),
                "stream": False,
                "stop": stop_words
            }

            if "seed" in params:
                completion_params["seed"] = params["seed"]

            output = llm.create_chat_completion(**completion_params)
            return output
        except Exception as e:
            print(f"【API调用错误】{e}")
            raise

    def get_generation_params(self, perf_level: str = "balanced",
                              video_input: bool = False,
                              text_input: bool = True) -> Dict:
        """获取生成参数"""
        if perf_level == "quality":
            return {
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "min_p": 0.05,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
            }
        elif perf_level == "speed":
            return {
                "max_tokens": 512,
                "temperature": 0.8,
                "top_k": 30,
                "top_p": 0.9,
                "min_p": 0.05,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
            }
        else:
            return {
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "min_p": 0.05,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
            }

    def cleanup(self):
        try:
            self.cache_manager.clear_all()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


class InferenceEngineFactory:
    """推理引擎工厂"""

    @staticmethod
    def create_engine(model_info: Dict) -> BaseInferenceEngine:
        """
        创建推理引擎实例
        
        Args:
            model_info: 模型信息字典，包含模型路径、设备模式等配置
            
        Returns:
            BaseInferenceEngine: 推理引擎实例
        """
        # 可以根据模型类型、设备模式等参数创建不同的推理引擎
        # 目前返回基础推理引擎，未来可扩展为多种引擎实现
        return BaseInferenceEngine(model_info)


def clear_all_caches():
    """清理系统中的所有缓存"""
    print("【缓存清理】开始清理系统缓存...")
    
    # 清理Python内存
    collected = gc.collect()
    print(f"【缓存清理】Python垃圾回收: 清理了 {collected} 个对象")
    
    # 清理PyTorch缓存
    if mm is not None:
        try:
            mm.soft_empty_cache()
            print("【缓存清理】PyTorch缓存清理完成")
        except Exception as e:
            print(f"【缓存清理】PyTorch缓存清理失败: {e}")
    
    # 清理LLM模型缓存
    try:
        LLAMA_CPP_STORAGE.clean(all=True)
        print("【缓存清理】LLM模型缓存清理完成")
    except Exception as e:
        print(f"【缓存清理】LLM模型缓存清理失败: {e}")
    
    print("【缓存清理】系统缓存清理完成")
