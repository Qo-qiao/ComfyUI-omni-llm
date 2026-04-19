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
import torch
import inspect
import numpy as np
import psutil
import platform
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# 导入加速模块
try:
    from engine.vram_layers import (
        calculate_vram_layers,
        calculate_safetensors_vram_layers,
        get_layer_count,
        estimate_vram_for_safetensors,
    )
except ImportError:
    calculate_vram_layers = None
    calculate_safetensors_vram_layers = None
    get_layer_count = None
    estimate_vram_for_safetensors = None

# -------------------------- 自动导入依赖（缺失会提示） --------------------------
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

# 尝试导入MTMDChatHandler（用于检测MTMD支持）
try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    _MTMD = True
except:
    _MTMD = False
    MTMDChatHandler = None

# 尝试导入MTMD音频支持库
try:
    from llama_cpp import mtmd_cpp
    _MTMD_AUDIO = True
    print("【MTMD音频】MTMD音频支持已启用")
except ImportError:
    _MTMD_AUDIO = False
    mtmd_cpp = None
    print("【MTMD音频】MTMD音频支持不可用（需要llama-cpp-python支持MTMD）")

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

try:
    import folder_paths
    import comfy.model_management as mm
    import comfy.utils
except ImportError as e:
    print(f"【错误】未检测到ComfyUI环境，请将该文件放入ComfyUI/custom_nodes/ComfyUI-omni-llm/目录下")
    exit(1)

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
                    print(f"【硬件检测】未检测到NVIDIA CUDA或AMD ROCm显卡，自动使用CPU/通用显卡兼容模式")
        except Exception as e:
            print(f"【提示】AMD显卡检测失败，自动使用兼容模式：{e}")
    
    print(f"【硬件检测】CPU核心数：{hardware_info['cpu_cores']}")
    return hardware_info

HARDWARE_INFO = get_hardware_info()

# -------------------------- 音频处理工具函数 --------------------------
def convert_audio_to_wav_bytes(audio_input, sample_rate=22050):
    """
    将音频输入转换为WAV格式的字节流，用于MTMD音频处理
    
    Args:
        audio_input: 音频输入，支持以下格式：
            - dict: {"waveform": tensor, "sample_rate": int}
            - torch.Tensor: 音频波形张量
            - numpy.ndarray: 音频波形数组
            - str: 音频文件路径
            - bytes: 已编码的音频数据
        sample_rate: 默认采样率（当输入未指定时使用）
    
    Returns:
        bytes: WAV格式的音频字节流
    """
    try:
        waveform = None
        sr = sample_rate
        
        if isinstance(audio_input, dict):
            waveform = audio_input.get("waveform")
            sr = audio_input.get("sample_rate", sample_rate)
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input
        elif isinstance(audio_input, str):
            if os.path.exists(audio_input):
                if _SOUNDFILE and sf:
                    waveform, sr = sf.read(audio_input)
                elif _SCIPY_WAV and wavfile:
                    sr, waveform = wavfile.read(audio_input)
                else:
                    with open(audio_input, 'rb') as f:
                        return f.read()
            else:
                raise ValueError(f"音频文件不存在: {audio_input}")
        elif isinstance(audio_input, bytes):
            return audio_input
        else:
            raise ValueError(f"不支持的音频输入类型: {type(audio_input)}")
        
        if waveform is None:
            raise ValueError("无法从输入中提取音频波形")
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        waveform = np.asarray(waveform, dtype=np.float32)
        
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        
        wav_buffer = io.BytesIO()
        if _SOUNDFILE and sf:
            sf.write(wav_buffer, waveform, sr, format='WAV', subtype='PCM_16')
        elif _SCIPY_WAV and wavfile:
            waveform_int16 = (waveform * 32767).astype(np.int16)
            wavfile.write(wav_buffer, sr, waveform_int16)
        else:
            raise ImportError("需要soundfile或scipy来处理音频")
        
        wav_buffer.seek(0)
        return wav_buffer.read()
        
    except Exception as e:
        print(f"【音频转换错误】转换音频失败: {str(e)}")
        return None

def detect_audio_format_from_bytes(audio_bytes):
    """
    从字节流检测音频格式（基于魔数）
    
    Args:
        audio_bytes: 音频字节流
    
    Returns:
        str: 音频格式 ('wav', 'mp3', 'flac') 或 None
    """
    if not audio_bytes or len(audio_bytes) < 12:
        return None
    
    is_wav = audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE"
    is_mp3 = (
        audio_bytes.startswith(b"ID3") or
        (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0)
    )
    is_flac = audio_bytes.startswith(b"fLaC")
    
    if is_wav:
        return "wav"
    elif is_mp3:
        return "mp3"
    elif is_flac:
        return "flac"
    return None

def create_audio_data_uri(audio_bytes, audio_format="wav"):
    """
    创建音频的Data URI格式，用于传递给MTMDChatHandler
    
    Args:
        audio_bytes: 音频字节流
        audio_format: 音频格式 ('wav', 'mp3', 'flac')
    
    Returns:
        str: Data URI格式的音频数据
    """
    if not audio_bytes:
        return None
    
    detected_format = detect_audio_format_from_bytes(audio_bytes)
    if detected_format:
        audio_format = detected_format
    
    b64_data = base64.b64encode(audio_bytes).decode('utf-8')
    return f"data:audio/{audio_format};base64,{b64_data}"

# -------------------------- 多分段Safetensors模型检测 --------------------------
def detect_sharded_safetensors(model_path: str) -> tuple:
    """
    检测模型是否是多分段的safetensors文件
    
    Args:
        model_path: 模型文件路径（可能是单个文件或目录）
    
    Returns:
        tuple: (is_sharded, actual_path, shard_files)
            - is_sharded: 是否是多分段模型
            - actual_path: 实际传递给llama-cpp-python的路径
            - shard_files: 分段文件列表（如果是多分段）
    """
    model_path = os.path.normpath(model_path)
    
    # 如果是目录，检查是否包含safetensors文件
    if os.path.isdir(model_path):
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        
        # 检查是否有index.json文件
        index_json = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_json):
            # 有index.json，说明是多分段模型
            return True, model_path, safetensors_files
        
        # 检查文件名中是否包含"of"（如model-00001-of-00003.safetensors）
        sharded_files = [f for f in safetensors_files if "-of-" in f]
        if sharded_files:
            # 有分段的safetensors文件
            return True, model_path, sharded_files
        
        # 单个safetensors文件
        if len(safetensors_files) == 1:
            single_file = os.path.join(model_path, safetensors_files[0])
            return False, single_file, [safetensors_files[0]]
        
        # 多个safetensors文件但没有index.json，尝试使用目录
        if len(safetensors_files) > 1:
            return True, model_path, safetensors_files
    
    # 如果是文件，检查是否是safetensors文件
    elif os.path.isfile(model_path):
        if model_path.endswith('.safetensors'):
            # 检查是否是分段文件
            filename = os.path.basename(model_path)
            if "-of-" in filename:
                # 这是一个分段文件，返回目录
                model_dir = os.path.dirname(model_path)
                safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
                return True, model_dir, safetensors_files
            else:
                # 单个safetensors文件
                return False, model_path, [filename]
    
    # 默认情况
    return False, model_path, []

# -------------------------- 模型配套文件验证 --------------------------
def validate_model_support_files(model_path: str, model_type: str = "auto") -> tuple:
    """
    验证模型配套文件是否完整
    
    Args:
        model_path: 模型路径（文件或目录）
        model_type: 模型类型（auto/qwen/minicpm/glm/omni），auto表示自动检测
    
    Returns:
        tuple: (is_valid, missing_files, warnings)
            - is_valid: 是否验证通过
            - missing_files: 缺失的必需文件列表
            - warnings: 警告信息列表（可选文件缺失）
    """
    model_path = os.path.normpath(model_path)
    
    # 确定模型目录
    if os.path.isfile(model_path):
        model_dir = os.path.dirname(model_path)
    else:
        model_dir = model_path
    
    if not os.path.isdir(model_dir):
        return False, [model_path], []
    
    # 自动检测模型类型
    if model_type == "auto":
        model_name_lower = os.path.basename(model_dir).lower()
        if "qwen" in model_name_lower:
            if "omni" in model_name_lower or "vl" in model_name_lower:
                model_type = "qwen_omni"
            else:
                model_type = "qwen"
        elif "minicpm" in model_name_lower:
            model_type = "minicpm"
        elif "glm" in model_name_lower:
            model_type = "glm"
        else:
            model_type = "generic"
    
    # 定义必需文件和可选文件
    required_files = []
    optional_files = []
    
    if model_type in ["qwen", "qwen_omni", "omni"]:
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        optional_files = [
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.json",
            "generation_config.json",
            "preprocessor_config.json",
            "spk_dict.pt"
        ]
    elif model_type == "minicpm":
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        optional_files = [
            "tokenizer.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
            "preprocessor_config.json"
        ]
    elif model_type == "glm":
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        optional_files = [
            "tokenizer.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
            "preprocessor_config.json"
        ]
    else:
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        optional_files = [
            "tokenizer.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
            "preprocessor_config.json"
        ]
    
    # 检查必需文件
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    # 检查可选文件
    warnings = []
    for file in optional_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            warnings.append(f"可选文件缺失: {file}")
    
    # 特殊警告：音频相关文件
    if model_type in ["qwen_omni", "omni"]:
        spk_dict_path = os.path.join(model_dir, "spk_dict.pt")
        if not os.path.exists(spk_dict_path):
            warnings.append("【音频功能警告】缺少 spk_dict.pt，音频生成功能可能受限")
    
    return len(missing_files) == 0, missing_files, warnings

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

# -------------------------- ChatHandler管理器（动态管理所有ChatHandler） --------------------------

class ChatHandlerManager:
    """
    ChatHandler管理器 - 动态管理llama-cpp-python中的所有ChatHandler
    支持自动检测、缓存和智能匹配
    """
    
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
        import re
        
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
        # 标准模型名称映射表
        display_map = {
            'qwen25vl': 'Qwen2.5-VL',
            'qwen3vl': 'Qwen3-VL',
            'qwen3vlchat': 'Qwen3-VL-Chat',
            'qwen3vlinstruct': 'Qwen3-VL-Instruct',
            'qwen3vlthinking': 'Qwen3-VL-Thinking',
            'qwen35': 'Qwen3.5',
            'qwen35thinking': 'Qwen3.5-Thinking',
            'qwen25omni': 'Qwen2.5-Omni-7B',
            'qwen25omniaudio': 'Qwen2.5-Omni-Audio',
            'qwen25omnivl': 'Qwen2.5-Omni-VL',
            'qwen25omnigguf': 'Qwen2.5-Omni-GGUF',
            'qwen35gguf': 'Qwen3.5-GGUF',
            'glm46v': 'GLM-4.6V',
            'glm46vthinking': 'GLM-4.6V-Thinking',
            'glm41v': 'GLM-4.1V-Thinking',
            'minicpmv45': 'MiniCPM-V-4.5',
            'minicpmv26': 'MiniCPM-V-2.6',
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
        
        return display_map.get(model_key, default_name.title().replace('-', ' '))
    
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
            ('qwen2.5-omni', qwen25_handler),  # Omni使用VL的ChatHandler
            ('qwen2.5-omni-audio', qwen25_handler),  # Omni音频模型
            ('qwen2.5-omni-vl', qwen25_handler),  # Omni VL模型
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
               "MiniCPM-V-4.5", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking", "GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "InternLM-XComposer2-VL", "DreamOmni2", 
               "MiniCPM-Llama3-V 2.5", "Llama-3.2-11B-Vision-Instruct", "CogVLM2", 
               "CogVLM-MOE", "Phi-3.5-vision-instruct", "Phi-3-vision-128k-instruct", 
               "Qwen2.5-VL", "Qwen3-VL", "Qwen3-VL-Thinking", "Qwen3-VL-Chat", "Qwen3-VL-Instruct", 
               "Qwen3.5", "Qwen3.5-Thinking", "Qwen2.5-Omni-7B", "Qwen2.5-Omni-3B", "MiniCPM-O-4.5",
               "LLaMA-3.1-Vision", "Zhipu-Vision", "智谱AI-Vision", "olmOCR-2", 
               "InternVL-1.5", "InternVL-2.0", "Yi-VL-2.0", "Gemma-3", "Granite-DocLing", 
               "Lfm-2-VL", "Llama3-Vision-Alpha", "LLaVA-1.5", "MiniCPM-V-2.6", "Obsidian", 
               "Youtu-VL-4B-Instruct", "EraX-VL-7B-V1.5", "MiMo-VL-7B-RL", "Yi-VL-6B"]

# ChatHandler名称到标准模型名称的映射
CHAT_HANDLER_MODEL_MAP = {
    'qwen25vl': 'Qwen2.5-VL',
    'qwen3vl': 'Qwen3-VL',
    'qwen3vlchat': 'Qwen3-VL-Chat',
    'qwen3vlinstruct': 'Qwen3-VL-Instruct',
    'glm46v': 'GLM-4.6V',
    'glm41v': 'GLM-4.1V-Thinking',
    'minicpmv45': 'MiniCPM-V-4.5',
    'minicpmv26': 'MiniCPM-V-2.6',
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
    'qwen2.5-omni-3b': 'Qwen2.5-Omni-3B',
    'qwen2.5-omni': 'Qwen2.5-Omni-7B',
    # MiniCPM 系列
    'minicpm-v-2.6': 'MiniCPM-V-2.6',
    'minicpm-v-4.5': 'MiniCPM-V-4.5',
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
            if '3b' in filename_lower:
                return 'Qwen2.5-Omni-3B'
            return 'Qwen2.5-Omni-7B'  # Omni模型支持音频生成
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
            return 'MiniCPM-V-4.5'
        elif 'v-2' in filename_lower or 'v_2' in filename_lower or 'v2' in filename_lower:
            return 'MiniCPM-V-2.6'
    
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

def qwen3bbox(image, json_data):
    bbox_list = []
    img_np = np.array(image)
    h, w, _ = img_np.shape
    
    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and "bbox" in item:
                    bbox = item["bbox"]
                    label = item.get("label", "object")
                    x1, y1, x2, y2 = map(int, bbox)
                    bbox_list.append({"bbox_2d": [x1, y1, x2, y2], "label": label})
    except Exception as e:
        print(f"【错误】解析Qwen3边界框失败：{e}")
    
    return bbox_list

def draw_bbox(image, json_data, mode):
    try:
        img_np = np.array(image)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        if mode == "Qwen3-VL":
            bbox_list = qwen3bbox(img_pil, json_data)
        elif mode == "simple":
            bbox_list = parse_json(json_data)
            if not isinstance(bbox_list, list):
                return img_np
        else:
            bbox_list = parse_json(json_data)
            if not isinstance(bbox_list, list):
                return img_np
        
        for bbox_item in bbox_list:
            if isinstance(bbox_item, dict) and "bbox_2d" in bbox_item:
                bbox = bbox_item["bbox_2d"]
                label = bbox_item.get("label", "object")
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 20), label, fill="red", font=None)
    except Exception as e:
        print(f"【错误】绘制边界框失败：{e}")
    
    return np.array(img_pil)

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
        """
        获取ChatHandler类 - 使用ChatHandlerManager进行智能匹配
        
        Args:
            chat_handler_name: ChatHandler显示名称或类名
            
        Returns:
            ChatHandler类或None
        """
        try:
            if chat_handler_name == "None" or not chat_handler_name:
                return None
            
            # 特殊处理Qwen3-VL，确保使用正确的ChatHandler
            if chat_handler_name in ["Qwen3-VL", "Qwen3-VL-Chat", "Qwen3-VL-Instruct", "Qwen3-VL-Thinking"]:
                qwen3_handler = chat_handler_manager.get_handler("Qwen3VLChatHandler")
                if qwen3_handler:
                    print(f"【ChatHandler匹配】{chat_handler_name} -> Qwen3VLChatHandler (Qwen3-VL)")
                    return qwen3_handler
            
            # 特殊处理Qwen2.5-VL/Omni，确保使用正确的ChatHandler
            if chat_handler_name in ["Qwen2.5-VL", "Qwen2.5-Omni-7B", "Qwen2.5-Omni-3B"]:
                # 优先使用 Qwen25VLChatHandler，如果不可用则使用 Qwen3VLChatHandler
                if Qwen25VLChatHandler:
                    print(f"【ChatHandler匹配】{chat_handler_name} -> Qwen25VLChatHandler")
                    return Qwen25VLChatHandler
                else:
                    qwen3_handler = chat_handler_manager.get_handler("Qwen3VLChatHandler")
                    if qwen3_handler:
                        print(f"【ChatHandler匹配】{chat_handler_name} -> Qwen3VLChatHandler (备选)")
                        return qwen3_handler
            
            # 特殊处理Qwen3.5，确保使用正确的ChatHandler
            if chat_handler_name in ["Qwen3.5", "Qwen3.5-Thinking"]:
                # 尝试获取Qwen35ChatHandler
                qwen35_handler = chat_handler_manager.get_handler("Qwen35ChatHandler")
                if qwen35_handler:
                    print(f"【ChatHandler匹配】{chat_handler_name} -> Qwen35ChatHandler (Qwen3.5)")
                    return qwen35_handler
                else:
                    print(f"【警告】Qwen35ChatHandler 不可用，Qwen3.5 模型将使用备选方案")
                    # 尝试使用Qwen3VLChatHandler作为备选
                    qwen3vl_handler = chat_handler_manager.get_handler("Qwen3VLChatHandler")
                    if qwen3vl_handler:
                        print(f"【ChatHandler备选】{chat_handler_name} -> Qwen3VLChatHandler (备选方案)")
                        return qwen3vl_handler
                    # 尝试使用通用的MTMDChatHandler作为备选
                    try:
                        from llama_cpp.llama_chat_format import MTMDChatHandler
                        if MTMDChatHandler:
                            print(f"【ChatHandler备选】{chat_handler_name} -> MTMDChatHandler (通用备选)")
                            return MTMDChatHandler
                    except Exception:
                        pass
                    print(f"【提示】请更新 llama-cpp-python 到最新版本以获得最佳Qwen3.5支持")
            
            # 特殊处理MiniCPM-V-4.5，确保使用正确的ChatHandler
            if chat_handler_name in ["MiniCPM-V-4.5", "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]:
                minicpm_handler = chat_handler_manager.get_handler("MiniCPMv45ChatHandler")
                if minicpm_handler:
                    print(f"【ChatHandler匹配】{chat_handler_name} -> MiniCPMv45ChatHandler")
                    return minicpm_handler
                else:
                    print(f"【警告】MiniCPMv45ChatHandler 不可用，MiniCPM-V-4.5 模型可能无法正常工作")
                    print(f"【提示】请更新 llama-cpp-python 到最新版本以支持 MiniCPM-V-4.5")
            
            # 首先尝试使用ChatHandlerManager获取
            handler_cls = chat_handler_manager.get_handler(chat_handler_name)
            if handler_cls is not None:
                info = chat_handler_manager.get_handler_info(handler_cls.__name__)
                display_name = info['display_name'] if info else chat_handler_name
                print(f"【ChatHandler匹配】{chat_handler_name} -> {handler_cls.__name__} ({display_name})")
                return handler_cls
            
            # 备选规则映射表
            fallback_rules = {
                # 模型显示名称 -> 备选ChatHandler名称列表
                "MiMo-VL-7B-RL": ["Qwen3VLChatHandler", "Llava16ChatHandler"],
                "MiniCPM-V-4.5": ["MiniCPMv45ChatHandler", "Llava16ChatHandler"],
                "MiniCPM-v4.5": ["MiniCPMv45ChatHandler", "Llava16ChatHandler"],
                "MiniCPM-O-4.5": ["MiniCPMv45ChatHandler", "Llava16ChatHandler"],
                "Qwen2.5-Omni-7B": ["Qwen3VLChatHandler", "Llava16ChatHandler"],
                "Qwen2.5-Omni-3B": ["Qwen3VLChatHandler", "Llava16ChatHandler"],
                "Qwen2.5-VL": ["Qwen3VLChatHandler", "Llava16ChatHandler"],
                "DreamOmni2": ["Qwen3VLChatHandler", "Llava16ChatHandler"],
                "LightOnOCR-2-1B": ["Llava16ChatHandler"],
                "zen3-vl-i1": ["Llava16ChatHandler"],
                "ASID-Captioner-7B": ["Llava16ChatHandler"],
                "llama-joycaption": ["Llava16ChatHandler"],
            }
            
            # 检查是否有备选规则
            if chat_handler_name in fallback_rules:
                for fallback_name in fallback_rules[chat_handler_name]:
                    handler = chat_handler_manager.get_handler(fallback_name)
                    if handler:
                        print(f"【ChatHandler备选】{chat_handler_name} 使用 {fallback_name} 作为备选")
                        return handler
            
            # 尝试从模型名称推断
            handler_name, handler = chat_handler_manager.get_handler_for_model(chat_handler_name)
            if handler:
                print(f"【ChatHandler推断】从模型名称推断：{chat_handler_name} -> {handler_name}")
                return handler
            
            # 最后尝试使用Llava16ChatHandler作为通用备选
            llava16 = chat_handler_manager.get_handler("Llava16ChatHandler")
            if llava16:
                print(f"【ChatHandler默认】未找到匹配的ChatHandler：{chat_handler_name}，使用 Llava16ChatHandler 作为默认处理")
                return llava16
            
            print(f"【ChatHandler错误】无法找到任何可用的ChatHandler用于：{chat_handler_name}")
            return None
            
        except Exception as e:
            print(f"【ChatHandler错误】获取ChatHandler失败：{e}")
            return None
    
    @classmethod
    def init_chat_handler(cls, handler_cls, mmproj_path, chat_handler_name, image_max_tokens, image_min_tokens, video_input=False, enable_thinking=False):
        """
        初始化ChatHandler - 智能参数检测和初始化
        
        Args:
            handler_cls: ChatHandler类
            mmproj_path: MMProj模型路径
            chat_handler_name: ChatHandler名称
            image_max_tokens: 图片最大token数
            image_min_tokens: 图片最小token数
            video_input: 是否为视频输入
            enable_thinking: 是否启用thinking模式
            
        Returns:
            初始化的ChatHandler实例或None
        """
        if handler_cls is None:
            return None
        
        # 初始化变量
        params = {}
        init_params = {"verbose": False}
        handler_name = ""
        
        try:
            import inspect
            handler_name = handler_cls.__name__
            print(f"【ChatHandler初始化】开始初始化：{handler_name}")
            
            # 从管理器获取handler信息
            info = chat_handler_manager.get_handler_info(handler_name)
            param_type = info.get('param_type', 'unknown') if info else 'unknown'
            
            # 获取构造函数参数签名
            sig = inspect.signature(handler_cls.__init__)
            params = sig.parameters
            
            # 初始化MTMDChatHandler变量
            MTMDChatHandler = None
            
            # 根据参数类型添加模型路径参数
            if mmproj_path:
                if "clip_model_path" in params:
                    init_params["clip_model_path"] = mmproj_path
                    print(f"【ChatHandler初始化】使用 clip_model_path 参数")
                elif "mmproj" in params:
                    init_params["mmproj"] = mmproj_path
                    print(f"【ChatHandler初始化】使用 mmproj 参数")
            else:
                # 没有mmproj时，检查handler是否必需clip_model_path
                requires_clip_path = "clip_model_path" in params and params["clip_model_path"].default == inspect.Parameter.empty
                requires_mmproj = "mmproj" in params and params["mmproj"].default == inspect.Parameter.empty
                
                if requires_clip_path or requires_mmproj:
                    print(f"【ChatHandler初始化】{handler_name} 需要mmproj模型但未提供，跳过ChatHandler初始化")
                    print(f"【提示】模型将以基础模式加载，TTS音频功能仍可使用")
                    return None
                else:
                    # 为不需要路径参数的handler添加空值
                    if "clip_model_path" in params:
                        init_params["clip_model_path"] = ""
                    elif "mmproj" in params:
                        init_params["mmproj"] = ""
            
            # 特殊处理MTMDChatHandler及其子类（如Qwen3VLChatHandler、Qwen35ChatHandler）
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler
                if issubclass(handler_cls, MTMDChatHandler):
                    # 检查是否是Qwen35ChatHandler
                    is_qwen35_handler = "Qwen35" in handler_cls.__name__
                    # 检查是否是Qwen3VLChatHandler
                    is_qwen3vl_handler = "Qwen3VL" in handler_cls.__name__
                    
                    # 对于Qwen35ChatHandler和Qwen3VLChatHandler，即使没有mmproj_path也尝试初始化
                    if is_qwen35_handler or is_qwen3vl_handler:
                        print(f"【ChatHandler初始化】{handler_cls.__name__}特殊处理，即使没有mmproj_path也尝试初始化")
                        # 强制添加clip_model_path参数，使用空字符串作为默认值
                        init_params["clip_model_path"] = ""
                        print(f"【ChatHandler初始化】为{handler_cls.__name__}添加clip_model_path=''参数")
                        # 尝试捕获初始化异常，允许在没有mmproj的情况下初始化
                        try:
                            chat_handler = handler_cls(**init_params)
                            print(f"【ChatHandler初始化】成功：{handler_cls.__name__}")
                            return chat_handler
                        except Exception as e:
                            print(f"【ChatHandler初始化错误】{handler_cls.__name__}初始化失败：{e}")
                            # 不尝试其他备选，直接返回None，让模型以基础模式加载
                            # 这样可以避免后续的初始化尝试失败，同时保持TTS音频功能可用
                            print(f"【ChatHandler初始化】{handler_cls.__name__}初始化失败，模型将以基础模式加载")
                            print(f"【提示】虽然ChatHandler初始化失败，但TTS音频功能仍可使用")
                            return None
                    else:
                        # 其他MTMD模型：只有在提供了mmproj_path时才添加clip_model_path参数
                        if mmproj_path and "clip_model_path" not in init_params:
                            init_params["clip_model_path"] = mmproj_path
                            print(f"【ChatHandler初始化】为{handler_cls.__name__}添加clip_model_path参数")
                        elif not mmproj_path and "clip_model_path" in params and params["clip_model_path"].default == inspect.Parameter.empty:
                            # 如果handler需要clip_model_path但未提供，直接返回None
                            print(f"【ChatHandler初始化】{handler_cls.__name__} 需要clip_model_path但未提供，跳过初始化")
                            print(f"【提示】模型将以基础模式加载，TTS音频功能仍可使用")
                            return None
                    
                    # MTMD音频支持：添加use_gpu参数
                    if "use_gpu" in params:
                        use_gpu = HARDWARE_INFO.get("has_cuda", False) or HARDWARE_INFO.get("has_rocm", False)
                        init_params["use_gpu"] = use_gpu
                        print(f"【ChatHandler初始化】MTMD use_gpu={use_gpu}")
            except Exception as e:
                print(f"【ChatHandler初始化】MTMD检查失败：{e}")
                pass
            
            # 添加图片token参数（如果支持且用户设置了值）
            # 只在用户设置了值时才添加
            is_qwen3vl_handler = "Qwen3VL" in handler_name or "Qwen3-VL" in chat_handler_name
            
            # 检查是否是MTMDChatHandler或其子类（如Qwen3VLChatHandler）
            is_mtmd_handler = False
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler
                is_mtmd_handler = issubclass(handler_cls, MTMDChatHandler)
            except Exception:
                pass
            
            # 视频处理特殊优化：降低image_min_tokens和image_max_tokens以减少内存占用
            if video_input:
                # 视频处理时降低token数量以避免内存不足
                video_image_max_tokens = max(256, image_max_tokens // 2)
                video_image_min_tokens = max(256, image_min_tokens // 2)
                print(f"【视频优化】视频处理模式，降低image_max_tokens从{image_max_tokens}到{video_image_max_tokens}")
                print(f"【视频优化】视频处理模式，降低image_min_tokens从{image_min_tokens}到{video_image_min_tokens}")
                image_max_tokens = video_image_max_tokens
                image_min_tokens = video_image_min_tokens
            elif is_qwen3vl_handler:
                # 非视频模式下，Qwen3-VL模型强制设置image_min_tokens为1024
                if image_min_tokens < 1024:
                    image_min_tokens = 1024
                    print(f"【ChatHandler初始化】Qwen3-VL强制设置image_min_tokens=1024")
            
            # 对于MTMDChatHandler及其子类，即使在params中找不到这些参数，也要添加
            if ("image_max_tokens" in params or is_mtmd_handler) and (image_max_tokens > 0 or is_qwen3vl_handler):
                init_params["image_max_tokens"] = image_max_tokens
                print(f"【ChatHandler初始化】添加image_max_tokens={image_max_tokens}")
            if ("image_min_tokens" in params or is_mtmd_handler) and (image_min_tokens > 0 or is_qwen3vl_handler):
                init_params["image_min_tokens"] = image_min_tokens
                print(f"【ChatHandler初始化】添加image_min_tokens={image_min_tokens}")
            
            # 优先使用用户传入的enable_thinking参数，其次才基于chat_handler_name
            think_mode = enable_thinking or ("Thinking" in chat_handler_name)
            
            # 根据模型类型添加特殊参数
            if chat_handler_name in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                if "force_reasoning" in params:
                    init_params["force_reasoning"] = think_mode
                    print(f"【ChatHandler初始化】Qwen3-VL 启用 reasoning 模式: {think_mode}")
            elif chat_handler_name in ["Qwen2.5-VL", "Qwen2.5-Omni-7B"]:
                # Qwen2.5-VL 使用 Qwen25VLChatHandler，继承自 MTMDChatHandler
                # 注意：Qwen25VLChatHandler 不支持 force_reasoning 参数
                # 只支持: clip_model_path, verbose, use_gpu, image_min_tokens, image_max_tokens
                print(f"【ChatHandler初始化】Qwen2.5-VL/Omni 模型初始化")
                print(f"【ChatHandler初始化】{chat_handler_name} 是MTMD模型，支持音频推理")
                # 确保添加use_gpu参数
                if "use_gpu" in params:
                    use_gpu = HARDWARE_INFO.get("has_cuda", False) or HARDWARE_INFO.get("has_rocm", False)
                    init_params["use_gpu"] = use_gpu
                    print(f"【ChatHandler初始化】MTMD use_gpu={use_gpu}")
            elif chat_handler_name in ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking", "MiniCPM-O-4.5"]:
                # MiniCPM-V-4.5 和 MiniCPM-O-4.5 使用 MiniCPMv45ChatHandler，继承自 MTMDChatHandler
                if "enable_thinking" in params:
                    init_params["enable_thinking"] = think_mode
                    print(f"【ChatHandler初始化】{chat_handler_name} 启用 thinking 模式: {think_mode}")
                print(f"【ChatHandler初始化】{chat_handler_name} 是MTMD模型，支持音频推理")
            elif chat_handler_name in ["GLM-4.6V", "GLM-4.6V-Thinking"]:
                # GLM-4.6V 使用 GLM46VChatHandler，继承自 MTMDChatHandler
                if "enable_thinking" in params:
                    init_params["enable_thinking"] = think_mode
                    print(f"【ChatHandler初始化】{chat_handler_name} 启用 thinking 模式: {think_mode}")
                print(f"【ChatHandler初始化】{chat_handler_name} 是MTMD模型，支持音频推理")
            elif chat_handler_name in ["Qwen3.5", "Qwen3.5-Thinking"]:
                # Qwen3.5 使用 Qwen35ChatHandler，继承自 MTMDChatHandler
                # 只添加Qwen35ChatHandler支持的参数
                print(f"【ChatHandler初始化】{chat_handler_name} 是MTMD模型，支持音频推理")
                # 检测Qwen35ChatHandler支持的参数
                supported_params = list(params.keys())
                # 只添加支持的参数
                if "enable_thinking" in supported_params:
                    init_params["enable_thinking"] = think_mode
                    print(f"【ChatHandler初始化】{chat_handler_name} 启用 thinking 模式: {think_mode}")
                # 移除可能不支持的参数
                if "image_max_tokens" in init_params and "image_max_tokens" not in supported_params:
                    init_params.pop("image_max_tokens")
                    print(f"【ChatHandler初始化】移除不支持的image_max_tokens参数")
                if "image_min_tokens" in init_params and "image_min_tokens" not in supported_params:
                    init_params.pop("image_min_tokens")
                    print(f"【ChatHandler初始化】移除不支持的image_min_tokens参数")
            elif chat_handler_name == "DreamOmni2":
                # DreamOmni2 在 llama-cpp-python 中没有专用 ChatHandler
                # 使用备选方案（Qwen3VLChatHandler 或 Llava16ChatHandler）
                print(f"【ChatHandler初始化】DreamOmni2 使用备选ChatHandler，尝试支持音频推理")
            
            # MTMD支持：添加image_max_tokens和image_min_tokens
            if _MTMD and MTMDChatHandler is not None:
                if "image_max_tokens" in params:
                    init_params["image_max_tokens"] = image_max_tokens
                if "image_min_tokens" in params:
                    init_params["image_min_tokens"] = image_min_tokens
            
            print(f"【ChatHandler初始化】参数：{list(init_params.keys())}")
            
            # 初始化ChatHandler - 尝试不同的参数组合
            try:
                # 第一次尝试：完整参数
                chat_handler = handler_cls(**init_params)
                print(f"【ChatHandler初始化】成功：{handler_name}")
                return chat_handler
            except Exception as e1:
                error_msg1 = str(e1)
                print(f"【ChatHandler初始化错误】第一次尝试失败：{e1}")
                
                # 第二次尝试：移除image相关参数
                if "image_max_tokens" in init_params or "image_min_tokens" in init_params:
                    print(f"【ChatHandler初始化】尝试移除image相关参数...")
                    init_params_copy = init_params.copy()
                    init_params_copy.pop("image_max_tokens", None)
                    init_params_copy.pop("image_min_tokens", None)
                    try:
                        chat_handler = handler_cls(**init_params_copy)
                        print(f"【ChatHandler初始化】成功（无image参数）：{handler_name}")
                        return chat_handler
                    except Exception as e2:
                        error_msg2 = str(e2)
                        print(f"【ChatHandler初始化错误】第二次尝试失败：{e2}")
                
                # 第三次尝试：只使用必要参数
                print(f"【ChatHandler初始化】尝试只使用必要参数...")
                minimal_params = {"verbose": False}
                # 只保留必要的参数
                if "clip_model_path" in init_params:
                    minimal_params["clip_model_path"] = init_params["clip_model_path"]
                elif "mmproj" in init_params:
                    minimal_params["mmproj"] = init_params["mmproj"]
                # 添加enable_thinking参数（如果支持）
                if "enable_thinking" in params:
                    minimal_params["enable_thinking"] = think_mode
                try:
                    chat_handler = handler_cls(**minimal_params)
                    print(f"【ChatHandler初始化】成功（最小参数）：{handler_name}")
                    return chat_handler
                except Exception as e3:
                    error_msg3 = str(e3)
                    print(f"【ChatHandler初始化错误】第三次尝试失败：{e3}")
            
            # 所有尝试都失败，准备抛出异常
            error_msg = str(e1) if 'e1' in locals() else "未知错误"
            print(f"【ChatHandler初始化错误】所有尝试都失败：{error_msg}")
            
            # 参考原版逻辑：针对Qwen3-VL和Qwen2.5-VL的特殊处理
            is_qwen_vl = chat_handler_name in ["Qwen3-VL", "Qwen3-VL-Thinking", "Qwen2.5-VL", "Qwen2.5-Omni-7B"]
            if is_qwen_vl and (image_max_tokens > 0 or image_min_tokens > 0):
                # 检查是否是image参数相关的错误
                if "image" in error_msg.lower() or "token" in error_msg.lower():
                    if 'e1' in locals():
                        raise ValueError(
                            f'"image_min_tokens" and "image_max_tokens" are unavailable! '
                            f'Please update llama-cpp-python.\n'
                            f'Current error: {error_msg}'
                        ) from e1
                    else:
                        raise ValueError(
                            f'"image_min_tokens" and "image_max_tokens" are unavailable! '
                            f'Please update llama-cpp-python.\n'
                            f'Current error: {error_msg}'
                        )
            
            # 尝试使用备选方案
            if handler_cls.__name__ not in ["Llava16ChatHandler", "Llava15ChatHandler"]:
                print(f"【ChatHandler初始化】尝试使用Llava16ChatHandler作为备选...")
                try:
                    llava16 = chat_handler_manager.get_handler("Llava16ChatHandler")
                    if llava16:
                        # Llava16ChatHandler 不需要 mmproj_path
                        init_params = {"verbose": False}
                        try:
                            chat_handler = llava16(**init_params)
                            print(f"【ChatHandler初始化】备选方案成功：Llava16ChatHandler")
                            return chat_handler
                        except Exception as llava_e:
                            print(f"【ChatHandler初始化】备选方案也失败：{llava_e}")
                except Exception as e:
                    print(f"【ChatHandler初始化】Llava16ChatHandler备选失败：{e}")
            
            # 尝试使用MTMDChatHandler作为通用备选（仅适用于omni推理）
            print(f"【ChatHandler初始化】尝试使用MTMDChatHandler作为通用备选...")
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler
                if MTMDChatHandler:
                    # 检查是否是omni模型
                    is_omni_model = "omni" in model_path.lower() if 'model_path' in locals() else False
                    if not is_omni_model:
                        print(f"【ChatHandler初始化】MTMDChatHandler只适用于omni推理，VL推理跳过")
                        return None
                    
                    # MTMDChatHandler 需要 clip_model_path
                    if mmproj_path:
                        init_params = {"verbose": False, "clip_model_path": mmproj_path}
                        try:
                            chat_handler = MTMDChatHandler(**init_params)
                            print(f"【ChatHandler初始化】MTMDChatHandler备选成功")
                            return chat_handler
                        except TypeError as type_e:
                            # 检查是否是缺少 clip_model_path 参数
                            if 'clip_model_path' in str(type_e):
                                print(f"【ChatHandler初始化】MTMDChatHandler不需要clip_model_path参数，尝试其他初始化方式...")
                                try:
                                    # 尝试只使用 verbose 参数
                                    init_params = {"verbose": False}
                                    chat_handler = MTMDChatHandler(**init_params)
                                    print(f"【ChatHandler初始化】MTMDChatHandler备选成功（无clip_model_path）")
                                    return chat_handler
                                except Exception as alt_e:
                                    print(f"【ChatHandler初始化】MTMDChatHandler备选失败：{alt_e}")
                            else:
                                raise
                        except Exception as mtmd_e:
                            print(f"【ChatHandler初始化】MTMDChatHandler备选失败：{mtmd_e}")
                    else:
                        print(f"【ChatHandler初始化】MTMDChatHandler需要clip_model_path，跳过")
            except Exception as e:
                print(f"【ChatHandler初始化】MTMDChatHandler备选失败：{e}")
            
            print(f"【ChatHandler初始化错误】初始化过程中发生异常：{error_msg}")
            return None
        except Exception as e:
            print(f"【ChatHandler初始化错误】初始化过程中发生异常：{e}")
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
            
            # 视觉模型参数
            enable_vision = config.get("enable_vision", False)
            vision_model = config.get("vision_model", "None")
            
            # 音频模型参数
            enable_audio = config.get("enable_audio", False)
            audio_model = config.get("audio_model", "None")
            
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
                print(f"【设备模式】使用 CPU 模式（忽略 n_gpu_layers 和 vram_limit 参数）")
                n_gpu_layers = 0
                vram_limit = -1
                # CPU模式强制使用保守参数
                n_batch = 1024
                n_ubatch = 0
                n_threads = os.cpu_count() or 8
                n_threads_batch = os.cpu_count() or 8
                f16_kv = True
                low_vram = True
                print(f"【CPU模式】强制重置参数：n_batch={n_batch}")
                print(f"【CPU模式】线程配置：n_threads={n_threads}, n_threads_batch={n_threads_batch}")
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
                # 如果 model 是绝对路径但不存在，尝试修正路径
                if os.path.isabs(model):
                    alt_path = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model))
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"【模型加载】绝对路径不可用，改用候选路径: {model_path}")
                # 如果文件不存在，可能是多分段模型的显示名称
                # 尝试查找对应的目录
                model_dir = os.path.dirname(model_path)
                model_filename = os.path.basename(model)
                
                # 检查是否是多分段模型（文件名包含 "-of-"）
                if "-of-" in model_filename and model_filename.endswith('.safetensors'):
                    # 提取基础名称（去掉分段编号）
                    base_name = model_filename.split("-of-")[0]
                    # 尝试使用基础名称作为目录
                    potential_dir = os.path.join(model_dir, base_name)
                    if os.path.isdir(potential_dir):
                        model_path = potential_dir
                        print(f"【多分段模型】使用目录路径: {model_path}")
                    else:
                        # 如果目录不存在，使用原文件所在的目录
                        model_path = model_dir
                        print(f"【多分段模型】使用文件所在目录: {model_path}")
                else:
                    raise FileNotFoundError(f"模型文件不存在：{model_path}")
            
            # 检测是否是多分段的safetensors模型
            is_sharded, actual_model_path, shard_files = detect_sharded_safetensors(model_path)
            if is_sharded:
                print(f"【多分段模型】检测到多分段safetensors模型")
                print(f"【多分段模型】分段文件: {len(shard_files)}个")
                model_path = actual_model_path
                model_ext = ".safetensors"
            else:
                model_ext = os.path.splitext(model)[1].lower()
                if model_ext not in [".gguf", ".safetensors"]:
                    print(f"【提示】模型格式可能不支持：{model_ext}，将尝试加载")
            
            # 验证模型配套文件（仅对safetensors格式）
            if model_ext == ".safetensors":
                print(f"【配套文件验证】开始验证模型配套文件...")
                is_valid, missing_files, warnings = validate_model_support_files(model_path, model_type="auto")
                
                if not is_valid:
                    print(f"【配套文件验证】❌ 验证失败！缺少必需文件:")
                    for file in missing_files:
                        print(f"  - {file}")
                    print(f"【配套文件验证】⚠️  缺少这些文件可能导致模型无法正常加载或功能异常")
                    print(f"【配套文件验证】💡 请确保模型目录包含所有必需的配置文件")
                else:
                    print(f"【配套文件验证】✅ 验证通过！所有必需文件都存在")
                
                # 显示警告信息
                if warnings:
                    print(f"【配套文件验证】⚠️  发现以下警告:")
                    for warning in warnings:
                        print(f"  - {warning}")
            
            # 构建MMProj路径
            mmproj_path = None
            if enable_mmproj and mmproj != "None":
                mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
                if not os.path.exists(mmproj_path):
                    raise FileNotFoundError(f"MMProj文件不存在：{mmproj_path}")
            
            # 构建视觉模型路径
            vision_model_path = None
            if enable_vision and vision_model != "None":
                vision_model_path = os.path.join(folder_paths.models_dir, 'LLM', vision_model)
                if not os.path.exists(vision_model_path):
                    raise FileNotFoundError(f"视觉模型文件不存在：{vision_model_path}")
                print(f"【视觉模型】已启用视觉模型：{vision_model}")
            
            # 构建音频模型路径
            audio_model_path = None
            if enable_audio and audio_model != "None":
                audio_model_path = os.path.join(folder_paths.models_dir, 'LLM', audio_model)
                if not os.path.exists(audio_model_path):
                    raise FileNotFoundError(f"音频模型文件不存在：{audio_model_path}")
                print(f"【音频模型】已启用音频模型：{audio_model}")
            
            # 获取模型格式
            if not is_sharded:
                model_ext = os.path.splitext(model)[1].lower()
                if model_ext not in [".gguf", ".safetensors"]:
                    print(f"【提示】模型格式可能不支持：{model_ext}，将尝试加载")
            
            # 获取并初始化ChatHandler
            handler_cls = cls.get_chat_handler_cls(chat_handler_name)
            cls.chat_handler = cls.init_chat_handler(handler_cls, mmproj_path, chat_handler_name, image_max_tokens, image_min_tokens, video_input=False, enable_thinking=enable_thinking)
            
            # ChatHandler初始化失败时的处理
            if cls.chat_handler is None:
                if enable_mmproj:
                    if chat_handler_name == "None":
                        print(f"【警告】启用了MMProj但未选择ChatHandler，ChatHandler将为None")
                    elif handler_cls is None:
                        print(f"【警告】无法找到ChatHandler：{chat_handler_name}，ChatHandler将为None")
                    else:
                        print(f"【警告】ChatHandler初始化失败：{chat_handler_name}，ChatHandler将为None")
                        print(f"【提示】模型将以基础模式加载，TTS音频功能仍可使用")
                else:
                    print(f"【提示】未启用MMProj，ChatHandler为None，模型将以基础模式加载")
                    print(f"【提示】TTS音频功能仍可正常使用")
            else:
                if enable_mmproj:
                    print(f"【成功】已启用MMProj并初始化ChatHandler：{type(cls.chat_handler).__name__}")
            
            # 加载LLM模型前的智能检查
            print(f"【模型加载】开始加载LLM模型：{model}（上下文：{n_ctx}，GPU层数：{n_gpu_layers}）")
            
            # 计算推荐的最大GPU层数（仅在 GPU 模式下）
            recommended_gpu_layers = n_gpu_layers
            
            # 检查是否是Qwen3.5模型
            is_qwen35 = "qwen35" in model.lower() or "qwen3.5" in model.lower()
            is_qwen3 = "qwen3" in model.lower()

            # 检测模型格式
            model_ext = os.path.splitext(model_path)[1].lower() if model_path else ""
            is_safetensors = model_ext == ".safetensors"
            is_gguf = model_ext == ".gguf"

            if device_mode == "GPU" and (HARDWARE_INFO["has_cuda"] or HARDWARE_INFO["has_rocm"]) and n_gpu_layers > 0:
                # 根据显存大小计算推荐的GPU层数
                gpu_vram = HARDWARE_INFO["gpu_vram_total"]
                gpu_vendor = HARDWARE_INFO["gpu_vendor"]

                # 估算模型在GPU中占用的显存（包括上下文）
                estimated_vram_usage = 0
                try:
                    model_file_size = os.path.getsize(model_path) / (1024 ** 3)  # GB

                    # 尝试使用加速模块进行精确计算
                    if is_gguf and calculate_vram_layers is not None:
                        # GGUF模型：使用加速模块的精确层数计算
                        mmproj_size_gb = 0
                        if enable_mmproj and mmproj != "None":
                            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
                            if os.path.exists(mmproj_path):
                                mmproj_size_gb = os.path.getsize(mmproj_path) / (1024 ** 3)

                        recommended_gpu_layers = calculate_vram_layers(
                            model_path, gpu_vram, mmproj_size_gb
                        )
                        estimated_vram_usage = model_file_size * (1.9 if gpu_vendor == "amd" else 1.8)
                        print(f"【加速VRAM计算】使用GGUF精确层数计算: 推荐GPU层数={recommended_gpu_layers}")

                    elif is_safetensors and calculate_safetensors_vram_layers is not None:
                        # safetensors模型：使用safetensors加速模块
                        mmproj_size_gb = 0
                        if enable_mmproj and mmproj != "None":
                            mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
                            if os.path.exists(mmproj_path):
                                mmproj_size_gb = os.path.getsize(mmproj_path) / (1024 ** 3)

                        safetensors_info = estimate_vram_for_safetensors(
                            model_path, gpu_vram, mmproj_size_gb
                        )
                        recommended_gpu_layers = safetensors_info.get("n_gpu_layers", n_gpu_layers)
                        estimated_vram_usage = safetensors_info.get("estimated_vram_gb", model_file_size * 1.8)
                        print(f"【加速VRAM计算】safetensors模型: tensor数={safetensors_info.get('tensor_count', 0)}, 估算层数={safetensors_info.get('estimated_layers', 32)}, 推荐GPU层数={recommended_gpu_layers}")
                    else:
                        # 回退到原有计算方式
                        vram_multiplier = 1.9 if gpu_vendor == "amd" else 1.8
                        if is_qwen3:
                            vram_multiplier *= 1.2
                            print(f"【显存估算】Qwen3系列模型使用更高的显存倍数: {vram_multiplier}")
                        estimated_vram_usage = model_file_size * vram_multiplier
                    
                    if enable_mmproj and mmproj != "None":
                        mmproj_file_size = os.path.getsize(os.path.join(folder_paths.models_dir, 'LLM', mmproj)) / (1024 ** 3)
                        estimated_vram_usage += mmproj_file_size * 1.2
                    
                    if enable_vision and vision_model != "None":
                        vision_file_size = os.path.getsize(os.path.join(folder_paths.models_dir, 'LLM', vision_model)) / (1024 ** 3)
                        estimated_vram_usage += vision_file_size * 1.2
                    
                    # 预留显存给系统和其他进程（AMD需要更多预留）
                    reserved_vram = 2.0 if gpu_vendor == "amd" else 1.5
                    # Qwen3.5模型需要更多的系统预留
                    if is_qwen35:
                        reserved_vram += 1.0
                        print(f"【显存估算】Qwen3.5模型增加系统预留显存: {reserved_vram}GB")
                    
                    available_vram = gpu_vram - reserved_vram
                    
                    if estimated_vram_usage > available_vram and n_gpu_layers == -1:
                        # 如果模型需要的显存超过可用显存，自动降低GPU层数
                        recommended_gpu_layers = int(n_gpu_layers * (available_vram / estimated_vram_usage))
                        recommended_gpu_layers = max(0, recommended_gpu_layers)
                        
                        print(f"【显存警告】模型预计需要{estimated_vram_usage:.2f}GB显存，可用{available_vram:.2f}GB")
                        print(f"【智能建议】建议将GPU层数调整为{recommended_gpu_layers}层")
                        print(f"【提示】您可以通过降低n_ctx、减少max_tokens或使用更小的模型来解决显存问题")
                except Exception as e:
                    print(f"【提示】显存估算失败，使用默认设置：{e}")
                
                # Qwen3.5模型的特殊GPU参数优化
                if is_qwen35:
                    print(f"【GPU模式优化】Qwen3.5模型启用特殊GPU参数配置")
                    # 对于Qwen3.5模型，使用更保守的GPU层数
                    if recommended_gpu_layers == -1:
                        # 根据显存大小设置合理的GPU层数
                        if gpu_vram <= 8:
                            recommended_gpu_layers = 20
                            print(f"【GPU模式优化】Qwen3.5模型在{gpu_vram}GB显存下设置GPU层数为20")
                        elif gpu_vram <= 12:
                            recommended_gpu_layers = 30
                            print(f"【GPU模式优化】Qwen3.5模型在{gpu_vram}GB显存下设置GPU层数为30")
                        elif gpu_vram <= 16:
                            recommended_gpu_layers = 40
                            print(f"【GPU模式优化】Qwen3.5模型在{gpu_vram}GB显存下设置GPU层数为40")
                        else:
                            recommended_gpu_layers = 50
                            print(f"【GPU模式优化】Qwen3.5模型在{gpu_vram}GB显存下设置GPU层数为50")
            elif device_mode == "CPU":
                # CPU 模式：不需要显存估算
                print(f"【提示】CPU 模式：跳过显存估算")
            

            
            # 构建模型参数
            gpu_vendor = HARDWARE_INFO["gpu_vendor"]
            
            # 根据设备模式和GPU厂商调整参数
            if device_mode == "CPU":
                # CPU模式参数已在前面设置
                pass
            elif gpu_vendor == "amd":
                # AMD ROCm优化参数
                n_batch = 1024  # AMD ROCm的批处理大小较小
                n_threads = os.cpu_count() or 8
                n_threads_batch = os.cpu_count() or 8
                use_mmap = False  # AMD ROCm通常不需要mmap
                use_mlock = True  # AMD ROCm建议使用mlock
                f16_kv = True  # AMD ROCm支持f16 KV缓存
                low_vram = HARDWARE_INFO["is_low_perf"]
            else:
                # NVIDIA CUDA优化参数
                n_batch = 2048
                n_threads = os.cpu_count() or 8
                n_threads_batch = os.cpu_count() or 16
                use_mmap = True
                use_mlock = False
                f16_kv = True
                low_vram = HARDWARE_INFO["is_low_perf"]
            
            # 构建Llama参数
            llama_kwargs = {
                "model_path": model_path,
                "chat_handler": cls.chat_handler,
                "n_gpu_layers": recommended_gpu_layers if device_mode == "GPU" else 0,
                "n_ctx": n_ctx,
                "n_batch": n_batch,

                "verbose": False,
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
                "low_vram": low_vram if device_mode == "GPU" else True,  # CPU模式强制启用低显存模式
                "tensor_split": None,
                "use_mmap": use_mmap,
                "use_mlock": use_mlock,
                "f16_kv": f16_kv,
                "cache_prompt": cache_prompt,
            }

            # TurboQuant KV Cache 加速（仅GGUF格式支持 llama.cpp 原生TurboQuant）
            turboquant_kv_cache = config.get("turboquant_kv_cache", "None")
            if turboquant_kv_cache and turboquant_kv_cache != "None":
                model_ext_check = os.path.splitext(model_path)[1].lower() if model_path else ""
                if model_ext_check == ".gguf":
                    # GGUF格式：使用 llama.cpp 原生 TurboQuant
                    try:
                        # 使用 type_k 和 type_v 参数（新版本API）
                        from llama_cpp._ggml import GGMLType
                        
                        turbo_type_map = {
                            "f16 (无压缩)": GGMLType.GGML_TYPE_F16,
                            "q8_0 (8-bit)": GGMLType.GGML_TYPE_Q8_0,
                            "q6_k (6-bit)": GGMLType.GGML_TYPE_Q6_K,
                            "q5_k (5-bit)": GGMLType.GGML_TYPE_Q5_K,
                            "q5_0 (5-bit)": GGMLType.GGML_TYPE_Q5_0,
                            "q5_1 (5-bit)": GGMLType.GGML_TYPE_Q5_1,
                            "q4_k (4-bit)": GGMLType.GGML_TYPE_Q4_K,
                            "q4_0 (4-bit)": GGMLType.GGML_TYPE_Q4_0,
                            "q4_1 (4-bit)": GGMLType.GGML_TYPE_Q4_1,
                            "q3_k (3-bit)": GGMLType.GGML_TYPE_Q3_K,
                            "q2_k (2-bit)": GGMLType.GGML_TYPE_Q2_K,
                            "mxfp4 (4-bit)": GGMLType.GGML_TYPE_MXFP4,
                            "nvfp4 (4-bit)": GGMLType.GGML_TYPE_NVFP4,
                            "turbo3 (3-bit)": GGMLType.GGML_TYPE_TQ3_0 if hasattr(GGMLType, 'GGML_TYPE_TQ3_0') else GGMLType.GGML_TYPE_TQ1_0,
                            "turbo2 (2-bit)": GGMLType.GGML_TYPE_TQ2_0,
                            "turbo1 (1-bit)": GGMLType.GGML_TYPE_TQ1_0,
                        }
                        kv_type = turbo_type_map.get(turboquant_kv_cache, GGMLType.GGML_TYPE_TQ1_0)

                        # 检查 llama-cpp-python 是否支持 type_k 和 type_v 参数
                        import inspect
                        llama_sig = inspect.signature(llama_cpp.Llama.__init__)
                        if 'type_k' in llama_sig.parameters and 'type_v' in llama_sig.parameters:
                            llama_kwargs["type_k"] = kv_type
                            llama_kwargs["type_v"] = kv_type
                            print(f"【TurboQuant加速】GGUF模型启用 KV Cache 压缩: {turboquant_kv_cache} (type_k={kv_type}, type_v={kv_type})")
                        elif 'cache_type_k' in llama_sig.parameters:
                            # 回退到旧版本API
                            old_type_map = {
                                "f16 (无压缩)": "f16",
                                "q8_0 (8-bit)": "q8_0",
                                "q6_k (6-bit)": "q6_k",
                                "q5_k (5-bit)": "q5_k",
                                "q5_0 (5-bit)": "q5_0",
                                "q5_1 (5-bit)": "q5_1",
                                "q4_k (4-bit)": "q4_k",
                                "q4_0 (4-bit)": "q4_0",
                                "q4_1 (4-bit)": "q4_1",
                                "q3_k (3-bit)": "q3_k",
                                "q2_k (2-bit)": "q2_k",
                                "mxfp4 (4-bit)": "mxfp4",
                                "nvfp4 (4-bit)": "nvfp4",
                                "turbo3 (3-bit)": "turbo3",
                                "turbo2 (2-bit)": "turbo2",
                                "turbo1 (1-bit)": "turbo1",
                            }
                            cache_type_k_value = old_type_map.get(turboquant_kv_cache, "turbo3")
                            llama_kwargs["cache_type_k"] = cache_type_k_value
                            print(f"【TurboQuant加速】GGUF模型启用 KV Cache 压缩: {cache_type_k_value} (旧API)")
                        else:
                            # 如果不支持这些参数，尝试在模型加载后使用 set_kv_cache_compression 方法
                            print(f"【TurboQuant加速】当前llama-cpp-python版本不支持type_k/type_v参数，将在模型加载后尝试使用set_kv_cache_compression方法")
                            # 保存 turboquant_kv_cache 配置，以便在模型加载后使用
                            config["_turboquant_kv_cache"] = turboquant_kv_cache
                    except Exception as e:
                        print(f"【TurboQuant加速】启用失败: {e}")
                else:
                    # safetensors格式：使用 PyTorch 版本 TurboQuant（在模型加载后处理）
                    print(f"【TurboQuant加速】safetensors格式模型将在推理时使用PyTorch版本TurboQuant")
            
            # 处理attention_type参数（GPU模式下）
            attention_type = config.get("attention_type", "Auto")
            if device_mode == "GPU" and gpu_vendor == "NVIDIA":
                try:
                    import inspect
                    llama_sig = inspect.signature(llama_cpp.Llama.__init__)
                    
                    # 处理attention_type映射
                    if attention_type == "Flash":
                        # 启用Flash Attention
                        if 'flash_attn' in llama_sig.parameters:
                            llama_kwargs["flash_attn"] = True
                            print(f"【GPU加速】已启用Flash Attention加速推理")
                        elif hasattr(llama_cpp, 'llama_flash_attn_type'):
                            llama_kwargs["flash_attn_type"] = llama_cpp.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED
                            print(f"【GPU加速】已启用Flash Attention加速推理（新API）")
                        else:
                            print(f"【GPU加速】当前llama-cpp-python版本不支持Flash Attention")
                    elif attention_type == "Standard":
                        # 标准注意力，禁用Flash Attention
                        if 'flash_attn' in llama_sig.parameters:
                            llama_kwargs["flash_attn"] = False
                            print(f"【GPU加速】使用标准注意力（禁用Flash Attention）")
                        elif hasattr(llama_cpp, 'llama_flash_attn_type'):
                            llama_kwargs["flash_attn_type"] = llama_cpp.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED
                            print(f"【GPU加速】使用标准注意力（禁用Flash Attention）")
                    elif attention_type == "XFormers":
                        # XFormers（如果支持）
                        print(f"【GPU加速】XFormers注意力（实验性）")
                    else:  # Auto
                        # 自动模式：默认启用Flash Attention
                        if 'flash_attn' in llama_sig.parameters:
                            llama_kwargs["flash_attn"] = True
                            print(f"【GPU加速】Auto模式：已启用Flash Attention加速推理")
                        elif hasattr(llama_cpp, 'llama_flash_attn_type'):
                            llama_kwargs["flash_attn_type"] = llama_cpp.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED
                            print(f"【GPU加速】Auto模式：已启用Flash Attention加速推理（新API）")
                except Exception as e:
                    print(f"【GPU加速】Flash Attention启用失败: {e}")
            
            # Qwen3.5模型的特殊参数优化（简化版本）
            if is_qwen35 and device_mode == "GPU":
                print(f"【GPU模式优化】Qwen3.5模型应用特殊参数配置")
                # 启用低显存模式以减少显存使用
                llama_kwargs["low_vram"] = True
                print(f"【GPU模式优化】Qwen3.5模型启用low_vram模式")
            
            # 打印优化参数
            if device_mode == "CPU":
                print(f"【CPU模式】推理参数配置:")
                print(f"  - 线程: n_threads={n_threads}, n_threads_batch={n_threads_batch}")
                print(f"  - 批处理: n_batch={n_batch}")
                print(f"  - 内存: use_mmap={use_mmap}, use_mlock={use_mlock}, f16_kv={f16_kv}")
            elif gpu_vendor == "amd":
                print(f"【AMD ROCm】推理参数配置:")
                print(f"  - GPU层数: {recommended_gpu_layers}")
                print(f"  - 线程: n_threads={n_threads}, n_threads_batch={n_threads_batch}")
                print(f"  - 批处理: n_batch={n_batch}")
                print(f"  - 内存: use_mmap={use_mmap}, use_mlock={use_mlock}, f16_kv={f16_kv}, low_vram={low_vram}")
            else:
                # 检查Flash Attention状态
                flash_attn_status = "已启用" if ("flash_attn" in llama_kwargs and llama_kwargs["flash_attn"]) or \
                                                   ("flash_attn_type" in llama_kwargs and llama_kwargs.get("flash_attn_type") == llama_cpp.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED) else "未启用/标准"
                print(f"【NVIDIA CUDA】推理参数配置:")
                print(f"  - GPU层数: {recommended_gpu_layers}")
                print(f"  - 线程: n_threads={n_threads}, n_threads_batch={n_threads_batch}")
                print(f"  - 批处理: n_batch={n_batch}")
                print(f"  - 内存: use_mmap={use_mmap}, use_mlock={use_mlock}, f16_kv={f16_kv}, low_vram={low_vram}")
                print(f"  - Attention类型: {attention_type} (Flash Attention: {flash_attn_status})")
            
            # 尝试加载模型，失败时提供降级策略
            try:
                # 暂时重定向标准输出，每加载50个参数显示一条摘要
                import sys
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                # 创建一个自定义输出流，用于捕获和处理加载信息
                class LoadingOutput:
                    def __init__(self):
                        self.buffer = []
                        self.param_count = 0
                        self.last_percent = -1
                    
                    def write(self, text):
                        # 捕获所有输出
                        self.buffer.append(text)
                        
                        # 尝试提取参数加载信息
                        if "Loading weights:" in text:
                            try:
                                # 提取百分比和参数计数
                                if "|" in text:
                                    parts = text.split("|")
                                    # 查找包含参数计数的部分，如 "1/258"
                                    for part in parts:
                                        if "/" in part:
                                            count_str = part.strip()
                                            if count_str and count_str[0].isdigit():
                                                current = int(count_str.split("/")[0])
                                                total = int(count_str.split("/")[1])
                                                
                                                # 每50个参数或百分比变化时显示一条日志
                                                if current % 50 == 0 or (current > 0 and current % 10 == 0 and current < 100):
                                                    percent = int((current / total) * 100)
                                                    if percent != self.last_percent:
                                                        print(f"【模型加载】进度：{percent}%，参数：{current}/{total}")
                                                        self.last_percent = percent
                                    
                            except Exception:
                                pass
                    
                    def flush(self):
                        pass
                
                loading_output = LoadingOutput()
                sys.stdout = loading_output
                sys.stderr = loading_output
                
                try:
                    cls.llm = Llama(**llama_kwargs)
                finally:
                    # 恢复标准输出
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                
                # 确保chat_handler被正确设置到Llama对象
                print(f"【模型加载】模型加载后检查 - chat_handler: {type(cls.chat_handler).__name__ if cls.chat_handler else 'None'}")
                if hasattr(cls.llm, 'chat_handler'):
                    print(f"【模型加载】模型加载后检查 - Llama.chat_handler: {type(cls.llm.chat_handler).__name__ if cls.llm.chat_handler else 'None'}")
                    if cls.chat_handler is not None and cls.llm.chat_handler is None:
                        cls.llm.chat_handler = cls.chat_handler
                        print(f"【模型加载】已设置Llama对象的chat_handler")
                    elif cls.chat_handler is not None and cls.llm.chat_handler is not None:
                        print(f"【模型加载】Llama对象已有chat_handler，无需设置")
                else:
                    print(f"【模型加载】Llama对象不支持chat_handler属性")
                
                # 显示模型加载详细信息
                print(f"【模型加载】✅ LLM模型加载成功！")
                print(f"【模型加载】📁 模型路径：{model_path}")
                print(f"【模型加载】📊 模型格式：{model_ext}")
                print(f"【模型加载】⚙️  上下文长度：{n_ctx}")
                print(f"【模型加载】🖥️  设备模式：{device_mode}")
                if device_mode == "GPU":
                    print(f"【模型加载】🔄  GPU层数：{recommended_gpu_layers}")
                print(f"【模型加载】🤖  ChatHandler：{type(cls.chat_handler).__name__ if cls.chat_handler else 'None'}")
                if enable_mmproj and mmproj != "None":
                    print(f"【模型加载】🔍  MMProj：已启用")
                else:
                    print(f"【模型加载】🔍  MMProj：未启用")
                print(f"【模型加载】📈  推理参数：n_batch={n_batch}, n_threads={n_threads}")
                
                # 尝试启用 TurboQuant KV Cache 压缩（新版本API）
                if "_turboquant_kv_cache" in config:
                    try:
                        if hasattr(cls.llm, 'set_kv_cache_compression'):
                            # 启用 TurboQuant KV Cache 压缩
                            cls.llm.set_kv_cache_compression(True)
                            print(f"【TurboQuant加速】已启用 KV Cache 压缩加速（使用set_kv_cache_compression方法）")
                        elif hasattr(cls.llm, 'model') and hasattr(cls.llm.model, 'set_kv_cache_compression'):
                            # 对于包装的模型，尝试通过 model 属性启用
                            cls.llm.model.set_kv_cache_compression(True)
                            print(f"【TurboQuant加速】已启用 KV Cache 压缩加速（使用model.set_kv_cache_compression方法）")
                        else:
                            print(f"【TurboQuant加速】当前模型不支持set_kv_cache_compression方法")
                    except Exception as e:
                        print(f"【TurboQuant加速】启用失败: {e}")
                
                # 禁用模型缓存
                # MODEL_CACHE[cache_key] = cls.llm
                # print(f"【模型缓存】已缓存模型：{model_path}")
            except Exception as e:
                error_msg = str(e)
                
                # 分析错误类型，提供针对性建议
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    # 显存不足错误
                    print(f"【显存错误】加载模型失败：{error_msg}")
                    print(f"【智能建议】")
                    print(f"  1. 降低n_gpu_layers值（当前：{recommended_gpu_layers}）")
                    print(f"  2. 减少n_ctx值（当前：{n_ctx}）")
                    print(f"  3. 使用更小的模型或更高压缩率的量化版本")
                    print(f"  4. 关闭mmproj（如果不需要多模态功能）")
                    
                    # 尝试降级加载（纯CPU）
                    print(f"【尝试降级】尝试使用纯CPU模式加载模型...")
                    llama_kwargs["n_gpu_layers"] = 0
                    llama_kwargs["low_vram"] = True
                    llama_kwargs["f16_kv"] = False
                    
                    try:
                        # 再次重定向标准输出，使用相同的加载输出处理
                        import sys
                        original_stdout = sys.stdout
                        original_stderr = sys.stderr
                        
                        # 创建一个自定义输出流，用于捕获和处理加载信息
                        class LoadingOutput:
                            def __init__(self):
                                self.buffer = []
                                self.param_count = 0
                                self.last_percent = -1
                            
                            def write(self, text):
                                # 捕获所有输出
                                self.buffer.append(text)
                                
                                # 尝试提取参数加载信息
                                if "Loading weights:" in text:
                                    try:
                                        # 提取百分比和参数计数
                                        if "|" in text:
                                            parts = text.split("|")
                                            # 查找包含参数计数的部分，如 "1/258"
                                            for part in parts:
                                                if "/" in part:
                                                    count_str = part.strip()
                                                    if count_str and count_str[0].isdigit():
                                                        current = int(count_str.split("/")[0])
                                                        total = int(count_str.split("/")[1])
                                                        
                                                        # 每50个参数或百分比变化时显示一条日志
                                                        if current % 50 == 0 or (current > 0 and current % 10 == 0 and current < 100):
                                                            percent = int((current / total) * 100)
                                                            if percent != self.last_percent:
                                                                print(f"【模型加载】(CPU模式) 进度：{percent}%，参数：{current}/{total}")
                                                                self.last_percent = percent
                                    
                                    except Exception:
                                        pass
                            
                            def flush(self):
                                pass
                        
                        loading_output = LoadingOutput()
                        sys.stdout = loading_output
                        sys.stderr = loading_output
                        
                        try:
                            cls.llm = Llama(**llama_kwargs)
                        finally:
                            # 恢复标准输出
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr
                        
                        # 确保chat_handler被正确设置到Llama对象
                        if hasattr(cls.llm, 'chat_handler'):
                            print(f"【模型加载】(CPU模式) 模型加载后检查 - Llama.chat_handler: {type(cls.llm.chat_handler).__name__ if cls.llm.chat_handler else 'None'}")
                            if cls.chat_handler is not None and cls.llm.chat_handler is None:
                                cls.llm.chat_handler = cls.chat_handler
                                print(f"【模型加载】(CPU模式) 已设置Llama对象的chat_handler")
                        
                        # 显示CPU模式模型加载详细信息
                        print(f"【模型加载】✅ LLM模型已使用纯CPU模式加载成功！")
                        print(f"【模型加载】📁 模型路径：{model_path}")
                        print(f"【模型加载】📊 模型格式：{model_ext}")
                        print(f"【模型加载】⚙️  上下文长度：{n_ctx}")
                        print(f"【模型加载】🖥️  设备模式：CPU")
                        print(f"【模型加载】🤖  ChatHandler：{type(cls.chat_handler).__name__ if cls.chat_handler else 'None'}")
                        if enable_mmproj and mmproj != "None":
                            print(f"【模型加载】🔍  MMProj：已启用")
                        else:
                            print(f"【模型加载】🔍  MMProj：未启用")
                        print(f"【模型加载】📈  推理参数：n_batch={n_batch}, n_threads={n_threads}")
                        print(f"【提示】CPU模式推理速度会较慢，建议使用更小的模型以提高速度")
                        
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
            
            # 初始化视觉模型（如果启用）
            if enable_vision and vision_model != "None" and vision_model_path:
                try:
                    print(f"【视觉模型初始化】开始初始化视觉模型...")
                    # 视觉模型初始化逻辑将在后续实现
                    # 这里先保存视觉模型路径到配置中
                    cls.vision_model_path = vision_model_path
                    cls.vision_model_name = vision_model
                    print(f"【视觉模型初始化】视觉模型路径已保存：{vision_model}")
                except Exception as vision_e:
                    print(f"【视觉模型初始化错误】初始化视觉模型失败：{vision_e}")
            
            # 初始化音频模型（如果启用）
            if enable_audio and audio_model != "None" and audio_model_path:
                try:
                    print(f"【音频模型初始化】开始初始化音频模型...")
                    # 音频模型初始化逻辑将在后续实现
                    # 这里先保存音频模型路径到配置中
                    cls.audio_model_path = audio_model_path
                    cls.audio_model_name = audio_model
                    print(f"【音频模型初始化】音频模型路径已保存：{audio_model}")
                except Exception as audio_e:
                    print(f"【音频模型初始化错误】初始化音频模型失败：{audio_e}")
                    # 音频模型初始化失败不影响主模型
                    cls.audio_model_path = None
                    cls.audio_model_name = None
        except Exception as e:
            print(f"【错误】加载模型失败：{e}")
            raise
