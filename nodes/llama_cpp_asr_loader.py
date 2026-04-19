# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm ASR Model Loader Node

独立的 ASR 模型加载节点，支持各种语音识别模型

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import json
import torch
import numpy as np
import sys
import os
import requests
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自动检测并安装缺失的依赖
def _ensure_funasr_installed():
    """确保 funasr 已安装"""
    try:
        from funasr import AutoModel
        return True
    except ImportError:
        return False

# 初始化时检测
_funasr_available = _ensure_funasr_installed()

def _ensure_qwen_asr_installed():
    """确保 qwen-asr 已安装"""
    try:
        from qwen_asr import Qwen3ASRModel
        return True
    except ImportError:
        return False

# 初始化时检测
_qwen_asr_available = _ensure_qwen_asr_installed()

from common import HARDWARE_INFO, folder_paths


# ========== 模型文件检测和下载功能 ==========

def _check_network_connection():
    """检查网络连接"""
    try:
        response = requests.get("https://www.baidu.com", timeout=5)
        return response.status_code == 200
    except:
        try:
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except:
            return False

def _get_model_source():
    """根据网络连接选择模型源"""
    if _check_network_connection():
        # 国内网络使用魔搭社区
        return "modelscope"
    else:
        # 国外网络使用Hugging Face
        return "huggingface"

def _download_file(url, save_path):
    """下载文件"""
    try:
        print(f"【文件下载】开始下载: {url}")
        print(f"【文件下载】保存路径: {save_path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 下载文件
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        print(f"【文件下载】下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"【文件下载错误】下载失败: {str(e)}")
        return False

def _download_model_files(model_name, model_path):
    """下载模型必备文件"""
    source = _get_model_source()
    print(f"【模型下载】使用源: {source}")
    
    # 定义模型文件映射
    model_files = {
        # Qwen3-ASR模型文件
        "qwen3_asr": {
            "modelscope": {
                "model.safetensors": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ASR/snapshots/master/model.safetensors",
                "config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ASR/snapshots/master/config.json",
                "tokenizer.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ASR/snapshots/master/tokenizer.json",
                "preprocessor_config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ASR/snapshots/master/preprocessor_config.json"
            },
            "huggingface": {
                "model.safetensors": f"https://huggingface.co/Qwen/Qwen3-ASR/resolve/main/model.safetensors",
                "config.json": f"https://huggingface.co/Qwen/Qwen3-ASR/resolve/main/config.json",
                "tokenizer.json": f"https://huggingface.co/Qwen/Qwen3-ASR/resolve/main/tokenizer.json",
                "preprocessor_config.json": f"https://huggingface.co/Qwen/Qwen3-ASR/resolve/main/preprocessor_config.json"
            }
        },
        # Fun-ASR模型文件
        "fun_asr": {
            "modelscope": {
                "model.pt": f"https://modelscope.cn/api/v1/models/FunAudioLLM/Fun-ASR-Nano-2512/snapshots/master/model.pt",
                "config.yaml": f"https://modelscope.cn/api/v1/models/FunAudioLLM/Fun-ASR-Nano-2512/snapshots/master/config.yaml"
            },
            "huggingface": {
                "model.pt": f"https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/model.pt",
                "config.yaml": f"https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/main/config.yaml"
            }
        }
    }
    
    # 根据模型名称判断模型类型
    model_type = None
    if "qwen3" in model_name.lower() and "asr" in model_name.lower():
        model_type = "qwen3_asr"
    elif "fun" in model_name.lower() and "asr" in model_name.lower():
        model_type = "fun_asr"
    
    if model_type and model_type in model_files:
        files_to_download = model_files[model_type][source]
        success = True
        
        for filename, url in files_to_download.items():
            save_path = os.path.join(model_path, filename)
            if not os.path.exists(save_path):
                if not _download_file(url, save_path):
                    success = False
        
        return success
    else:
        print(f"【模型下载】未知模型类型: {model_name}")
        return False

def _check_asr_model_files(model_path, model_type):
    """检查ASR模型必备文件"""
    required_files = []
    
    if model_type == "qwen3_asr":
        required_files = [
            "model.safetensors",
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json"
        ]
    elif model_type == "fun_asr":
        required_files = [
            "model.pt",
            "config.yaml"
        ]
    elif model_type == "whisper":
        required_files = [
            "model.safetensors",
            "config.json"
        ]
    
    missing_files = []
    for filename in required_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    return missing_files

class llama_cpp_asr_loader:
    """
    ASR模型加载器
    用于加载自动语音识别模型，支持Whisper等ASR架构
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # 检查并添加LLM文件夹路径
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

        # 获取所有LLM文件夹路径
        llm_folders = folder_paths.get_folder_paths("LLM")
        
        # 筛选ASR模型（asr/whisper/stt相关）- 只加载文件夹，不加载文件
        asr_list = ["None"]
        model_set = set()
        model_set.add("None")
        
        for folder in llm_folders:
            try:
                root_tag = os.path.basename(folder) if len(llm_folders) > 1 else ""

                # 递归扫描所有子目录和文件
                for root, dirs, files in os.walk(folder):
                    # 计算相对于LLM文件夹的路径
                    rel_path = os.path.relpath(root, folder)
                    if rel_path == '.':
                        current_folder_name = ""
                    else:
                        current_folder_name = rel_path.replace(os.sep, '/')

                    # 获取目录名称
                    dir_name = os.path.basename(root).lower()

                    # 检查是否是包含ASR关键词的目录
                    is_asr_dir = any(keyword in dir_name for keyword in ["asr", "whisper", "stt", "speech-to-text", "speech2text"])

                    if is_asr_dir:
                        # 检查目录中是否包含模型文件
                        has_model_files = any(os.path.splitext(f)[1].lower() in [".safetensors", ".bin", ".gguf", ".pt", ".pth"] for f in files)

                        # 只有当目录包含模型文件时才添加目录
                        if has_model_files:
                            if current_folder_name:
                                folder_entry = current_folder_name
                            else:
                                folder_entry = dir_name

                            if root_tag:
                                folder_entry = f"{root_tag}/{folder_entry}"

                            folder_entry = folder_entry.rstrip('/') + '/'
                            
                            # 获取目录的绝对路径（去重依据）
                            folder_abs_path = os.path.normpath(os.path.join(folder, folder_entry.rstrip('/')))
                            if folder_abs_path not in model_set:
                                model_set.add(folder_abs_path)
                                asr_list.append(folder_entry)
                                print(f"【ASR模型检测】检测到ASR目录: {folder_entry} (绝对路径: {folder_abs_path})")

                        # 检查是否是分段模型文件夹
                        sharded_files1 = [f for f in files if "-of-" in f.lower() and f.endswith(".safetensors")]
                        sharded_files2 = [f for f in files if f.startswith("model.safetensors-") and "-of-" in f.lower()]
                        if sharded_files1 or sharded_files2:
                            # 分段模型文件夹，添加整个文件夹
                            if current_folder_name:
                                folder_entry = current_folder_name
                            else:
                                folder_entry = dir_name

                            if root_tag:
                                folder_entry = f"{root_tag}/{folder_entry}"

                            folder_entry = folder_entry.rstrip('/') + '/'
                            
                            # 获取目录的绝对路径（去重依据）
                            folder_abs_path = os.path.normpath(os.path.join(folder, folder_entry.rstrip('/')))
                            if folder_abs_path not in model_set:
                                model_set.add(folder_abs_path)
                                asr_list.append(folder_entry)
                                print(f"【ASR模型检测】检测到分段模型文件夹: {folder_entry} (绝对路径: {folder_abs_path})")

                    # 检查非ASR目录中是否包含模型文件（用于检测包含模型文件的文件夹）
                    has_model_files_in_dir = any(os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".pt", ".pth", ".bin"] for f in files)
                    if has_model_files_in_dir:
                        # 检查文件夹名称或路径中是否包含ASR关键词
                        folder_path_lower = root.lower()
                        folder_name_lower = dir_name.lower()
                        if any(keyword in folder_path_lower or keyword in folder_name_lower for keyword in ["asr", "whisper", "stt", "speech-to-text", "speech2text"]):
                            if current_folder_name:
                                folder_entry = current_folder_name
                            else:
                                folder_entry = dir_name

                            if root_tag:
                                folder_entry = f"{root_tag}/{folder_entry}"

                            folder_entry = folder_entry.rstrip('/') + '/'
                            
                            # 获取目录的绝对路径（去重依据）
                            folder_abs_path = os.path.normpath(os.path.join(folder, folder_entry.rstrip('/')))
                            if folder_abs_path not in model_set:
                                model_set.add(folder_abs_path)
                                asr_list.append(folder_entry)
                                print(f"【ASR模型检测】检测到包含ASR模型文件的目录: {folder_entry} (绝对路径: {folder_abs_path})")
            except Exception as e:
                print(f"【ASR模型检测】扫描文件夹 {folder} 失败: {e}")
        
        # 如果没有找到ASR模型，显示提示
        if len(asr_list) == 1:
            asr_list = ["None", "请将ASR语音识别模型放入models/LLM文件夹"]
        
        # 根据硬件性能推荐默认参数
        perf_level = HARDWARE_INFO.get("perf_level", "low")

        if perf_level == "high":  # 24GB+
            default_n_gpu_layers = -1
        elif perf_level == "mid_high":  # 16GB
            default_n_gpu_layers = -1
        elif perf_level == "mid":  # 12GB
            default_n_gpu_layers = -1
        elif perf_level == "mid_low":  # 8GB
            default_n_gpu_layers = -1
        else:  # <8GB
            default_n_gpu_layers = 20
        
        # Qwen3-ASR支持的52种语言和方言
        qwen3_languages = [
            "auto", "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", 
            "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl",
            "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"
        ]
        
        return {
            "required": {
                "asr_model": (asr_list, {"tooltip": "选择ASR模型文件（支持.gguf、.safetensors、.pt、.pth、.bin格式）"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载"}),
                "language": (qwen3_languages, {"default": "auto", "tooltip": "识别语言，auto=自动检测"}),
                "task": (["transcribe", "translate"], {"default": "transcribe", "tooltip": "任务类型：transcribe=转录，translate=翻译成英文"}),
            },
            "optional": {
                "enable_timestamps": ("BOOLEAN", {"default": False, "tooltip": "启用时间戳功能（需要Qwen3-ForcedAligner）"}),
                "flash_attention": ("BOOLEAN", {"default": False, "tooltip": "启用FlashAttention2优化（需要安装flash-attn）"}),
            }
        }
    
    RETURN_TYPES = ("ASRMODEL",)
    RETURN_NAMES = ("asr_model",)
    FUNCTION = "load_asr_model"
    CATEGORY = "llama-cpp-vlm"
    
    @classmethod
    def _resolve_asr_model_path(s, asr_model):
        key = asr_model.rstrip('/')
        if key == "None":
            return None

        print(f"【路径解析】开始解析: {asr_model} -> key: {key}")

        # 优先处理带root_tag的路径
        if "/" in key:
            root_name, inner = key.split("/", 1)
            print(f"【路径解析】带root_tag: root_name={root_name}, inner={inner}")
            for folder in folder_paths.get_folder_paths("LLM"):
                if os.path.basename(folder) == root_name:
                    candidate = os.path.join(folder, inner)
                    print(f"【路径解析】尝试: {candidate}")
                    if os.path.exists(candidate):
                        print(f"【路径解析】找到: {candidate}")
                        return os.path.normpath(candidate)

        # 尝试直接在所有LLM路径下查找
        for folder in folder_paths.get_folder_paths("LLM"):
            candidate = os.path.join(folder, key)
            print(f"【路径解析】尝试: {candidate}")
            if os.path.exists(candidate):
                print(f"【路径解析】找到: {candidate}")
                return os.path.normpath(candidate)

        # 尝试使用folder_paths的get_full_path方法
        full = folder_paths.get_full_path("LLM", key)
        if full and os.path.exists(full):
            print(f"【路径解析】通过get_full_path找到: {full}")
            return os.path.normpath(full)

        # 尝试去掉root_tag前缀后查找
        if "/" in key:
            root_name, inner = key.split("/", 1)
            print(f"【路径解析】尝试去掉root_tag: inner={inner}")
            for folder in folder_paths.get_folder_paths("LLM"):
                candidate = os.path.join(folder, inner)
                print(f"【路径解析】尝试: {candidate}")
                if os.path.exists(candidate):
                    print(f"【路径解析】找到: {candidate}")
                    return os.path.normpath(candidate)

        print(f"【路径解析】未找到: {asr_model}")
        return None

    def __init__(self):
        self.loaded_model = None
        self.current_config = None
    
    @classmethod
    def IS_CHANGED(s, asr_model, n_gpu_layers, language, task, enable_timestamps=False, flash_attention=False):
        resolved_path = s._resolve_asr_model_path(asr_model)
        config = {
            "asr_model": asr_model,
            "asr_model_path": resolved_path or "",
            "n_gpu_layers": n_gpu_layers,
            "language": language,
            "task": task,
            "enable_timestamps": enable_timestamps,
            "flash_attention": flash_attention
        }
        return json.dumps(config, sort_keys=True, ensure_ascii=False)
    
    def load_asr_model(self, asr_model, n_gpu_layers, language, task, enable_timestamps=False, flash_attention=False):
        """
        加载ASR模型
        """
        if asr_model == "None" or asr_model == "请将ASR语音识别模型放入models/LLM文件夹":
            print("【ASR加载器】未选择ASR模型")
            return (None,)

        resolved_model_path = self._resolve_asr_model_path(asr_model)
        if not resolved_model_path:
            raise RuntimeError(f"无法解析ASR模型路径: {asr_model}")

        try:
            print(f"【ASR加载器】正在加载ASR模型: {asr_model} (解析路径: {resolved_model_path})")
            print(f"【ASR加载器】配置: 时间戳={enable_timestamps}")
            
            # 构建配置
            config = {
                "asr_model": asr_model,
                "asr_model_path": resolved_model_path,
                "n_gpu_layers": n_gpu_layers,
                "language": language,
                "task": task,
                "enable_timestamps": enable_timestamps,
                "flash_attention": flash_attention
            }
            
            # 检查是否需要重新加载
            if self.loaded_model is not None and self.current_config == config:
                print("【ASR加载器】使用缓存的ASR模型")
                return (self.loaded_model,)
            
            # 直接使用_resolve_asr_model_path的结果（现在只返回文件夹路径）
            model_path = resolved_model_path
            
            if model_path is None or not os.path.exists(model_path):
                raise RuntimeError(f"ASR模型文件夹不存在: {asr_model} (解析路径：{resolved_model_path})")
            
            # 确保是文件夹路径
            if not os.path.isdir(model_path):
                # 如果是文件路径，获取其所在目录
                model_path = os.path.dirname(model_path)
                print(f"【ASR加载器】检测到文件路径，自动使用所在目录: {model_path}")
            
            print(f"【ASR加载器】模型路径: {model_path}")
            print(f"【ASR加载器】运行模式: GPU, GPU层数: {n_gpu_layers}")
            print(f"【ASR加载器】语言: {language}, 任务: {task}")
            
            # 检测模型类型并检查必备文件
            model_dir = model_path
            
            # 根据模型名称判断类型
            model_type = None
            if "qwen3" in asr_model.lower() and "asr" in asr_model.lower():
                model_type = "qwen3_asr"
            elif "fun" in asr_model.lower() and "asr" in asr_model.lower():
                model_type = "fun_asr"
            elif "whisper" in asr_model.lower():
                model_type = "whisper"
            
            # 检查必备文件
            if model_type:
                print(f"【ASR加载器】检测模型类型: {model_type}")
                missing_files = _check_asr_model_files(model_dir, model_type)
                
                if missing_files:
                    print(f"【ASR加载器】检测到缺失文件: {missing_files}")
                    print(f"【ASR加载器】尝试自动下载缺失文件...")
                    
                    # 尝试下载缺失文件
                    if _download_model_files(asr_model, model_dir):
                        print(f"【ASR加载器】缺失文件下载成功")
                    else:
                        print(f"【ASR加载器警告】部分文件下载失败，尝试继续加载")
            
            # 创建ASR模型包装器
            asr_wrapper = ASRModelWrapper(
                model_path=model_path,
                config=config,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            self.loaded_model = asr_wrapper
            self.current_config = config
            
            print("【ASR加载器】ASR模型加载成功")
            return (asr_wrapper,)
            
        except Exception as e:
            print(f"【ASR加载器错误】加载ASR模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return (None,)


class ASRModelWrapper:
    """
    ASR模型包装器
    提供统一的ASR接口
    """
    
    def __init__(self, model_path, config, device="cpu"):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model = None
        self.model_type = self._detect_model_type(model_path)
        self._audio_cache = {}
        self._cache_size_limit = 100
        
        print(f"【ASR包装器】检测到模型类型: {self.model_type}")
        print(f"【ASR包装器】设备: {device}")
        
        self._load_model()
    
    def _detect_model_type(self, model_path):
        """
        检测ASR模型类型
        支持从文件名和目录名检测，对大小写不敏感
        """
        # 获取文件名和目录名（都转小写）
        filename = os.path.basename(model_path).lower()
        dirname = os.path.basename(os.path.dirname(model_path)).lower()
        full_path_lower = model_path.lower()
        
        # 组合检测文本（文件名 + 目录名 + 完整路径）
        detection_text = f"{filename} {dirname} {full_path_lower}"
        
        if "whisper" in detection_text:
            return "whisper"
        elif "wav2vec" in detection_text or "wav2vec2" in detection_text:
            return "wav2vec2"
        elif "qwen3" in detection_text and "asr" in detection_text:
            return "qwen3_asr"
        elif "qwen" in detection_text and "asr" in detection_text:
            return "qwen_asr"
        elif "qwen" in detection_text and "audio" in detection_text:
            return "qwen_audio"
        elif "fun" in detection_text and "asr" in detection_text:
            return "fun_asr"
        elif filename.endswith(".gguf"):
            return "gguf_asr"
        elif filename.endswith(".safetensors") or filename.endswith(".pt") or filename.endswith(".pth"):
            return "pytorch_asr"
        else:
            return "unknown"
    
    def _load_model(self):
        """
        加载ASR模型
        """
        try:
            if self.model_type == "whisper":
                self._load_whisper_model()
            elif self.model_type == "wav2vec2":
                self._load_wav2vec2_model()
            elif self.model_type == "qwen_audio" or self.model_type == "qwen_asr" or self.model_type == "qwen3_asr":
                self._load_qwen_model()
            elif self.model_type == "fun_asr":
                self._load_fun_asr_model()
            elif self.model_type == "pytorch_asr":
                self._load_pytorch_model()
            elif self.model_type == "gguf_asr":
                self._load_gguf_model()
            else:
                print(f"【ASR警告】未知模型类型: {self.model_type}")
                
        except Exception as e:
            print(f"【ASR模型加载错误】{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_whisper_model(self):
        """
        加载Whisper模型
        """
        try:
            from whisper import load_model
            
            model_size = "base"
            filename = os.path.basename(self.model_path).lower()
            
            if "tiny" in filename:
                model_size = "tiny"
            elif "small" in filename:
                model_size = "small"
            elif "medium" in filename:
                model_size = "medium"
            elif "large" in filename or "large-v1" in filename or "large-v2" in filename:
                model_size = "large"
            elif "large-v3" in filename:
                model_size = "large-v3"
            
            device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
            self.model = load_model(model_size, device=device)
            
            print(f"【Whisper模型】已加载 {model_size} 模型到 {device}")
            
        except ImportError:
            print("【Whisper模型】未安装whisper库，请运行: pip install openai-whisper")
        except Exception as e:
            print(f"【Whisper模型加载错误】{str(e)}")
    
    def _load_wav2vec2_model(self):
        """
        加载Wav2Vec2模型
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            
            processor = Wav2Vec2Processor.from_pretrained(self.model_path)
            model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
            
            if self.device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
            
            self.model = {"processor": processor, "model": model}
            print(f"【Wav2Vec2模型】已加载到 {self.device}")
            
        except ImportError:
            print("【Wav2Vec2模型】未安装transformers库，请运行: pip install transformers")
        except Exception as e:
            print(f"【Wav2Vec2模型加载错误】{str(e)}")
    
    def _load_qwen_model(self):
        """
        加载Qwen ASR模型（使用官方qwen-asr包）
        """
        try:
            # 检查qwen-asr包是否可用
            if not _qwen_asr_available:
                print("【Qwen ASR模型】qwen-asr包未安装，无法加载Qwen3-ASR模型")
                return
                
            from qwen_asr import Qwen3ASRModel
            import torch
            
            # 根据模型类型选择不同的加载方式
            if self.model_type == "qwen3_asr":
                # Qwen3-ASR模型使用官方qwen-asr包
                print(f"【Qwen3-ASR模型】正在加载模型: {self.model_path}")
                
                # 获取配置参数
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                enable_timestamps = self.config.get("enable_timestamps", False)
                
                # 检查是否为文件路径，如果是则提取目录路径
                model_path_for_loading = self.model_path
                if os.path.isfile(self.model_path):
                    model_path_for_loading = os.path.dirname(self.model_path)
                    print(f"【Qwen3-ASR模型】检测到文件路径，使用目录路径: {model_path_for_loading}")
                
                # 使用transformers后端
                model_kwargs = {
                    "dtype": dtype,
                    "device_map": "cuda:0" if self.device == "cuda" else "cpu",
                    "max_inference_batch_size": 32,  # 固定值
                    "max_new_tokens": 256  # 固定值
                }
                
                # 如果启用FlashAttention2优化
                if self.config.get("flash_attention", False):
                    try:
                        import flash_attn
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                        print(f"【Qwen3-ASR模型】启用FlashAttention2优化")
                    except ImportError:
                        print(f"【Qwen3-ASR模型】FlashAttention2未安装，使用默认Attention实现")
                
                # 如果启用时间戳，添加ForcedAligner
                if enable_timestamps:
                    model_kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
                    aligner_kwargs = dict(
                        dtype=dtype,
                        device_map="cuda:0" if self.device == "cuda" else "cpu"
                    )
                    
                    # ForcedAligner也支持FlashAttention2优化
                    if self.config.get("flash_attention", False):
                        try:
                            import flash_attn
                            aligner_kwargs["attn_implementation"] = "flash_attention_2"
                        except ImportError:
                            pass
                    
                    model_kwargs["forced_aligner_kwargs"] = aligner_kwargs
                
                model = Qwen3ASRModel.from_pretrained(model_path_for_loading, **model_kwargs)
                
                self.model = model
                print(f"【Qwen3-ASR模型】已加载到 {self.device}")
                
            else:
                # 旧版Qwen Audio模型使用transformers
                from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
                
                processor = AutoProcessor.from_pretrained(self.model_path)
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                if self.device == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                
                self.model = {"processor": processor, "model": model}
                print(f"【Qwen Audio模型】已加载到 {self.device}")
                
        except ImportError:
            print("【Qwen ASR模型】未安装qwen-asr库，请运行: pip install -U qwen-asr")
        except Exception as e:
            print(f"【Qwen ASR模型加载错误】{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_pytorch_model(self):
        """
        加载PyTorch ASR模型
        """
        try:
            if self.model_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location="cpu")
            
            self.model = {"state_dict": state_dict}
            print(f"【PyTorch ASR模型】已加载模型状态字典")
            
        except Exception as e:
            print(f"【PyTorch ASR模型加载错误】{str(e)}")
    
    def _load_gguf_model(self):
        """
        加载GGUF ASR模型
        """
        try:
            from llama_cpp import Llama
            
            n_gpu_layers = self.config.get("n_gpu_layers", 0)
            n_ctx = 2048
            
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False
            )
            
            print(f"【GGUF ASR模型】已加载，GPU层数: {n_gpu_layers}")
            
        except ImportError:
            print("【GGUF ASR模型】未安装llama-cpp-python库")
        except Exception as e:
            print(f"【GGUF ASR模型加载错误】{str(e)}")
    
    def _load_fun_asr_model(self):
        """
        加载Fun-ASR模型
        支持 Fun-ASR-Nano-2512 和 Fun-ASR-MLT-Nano-2512 等模型
        """
        try:
            print(f"【Fun-ASR】正在加载模型: {self.model_path}")

            # 检查是否安装了funasr
            try:
                from funasr import AutoModel
            except ImportError:
                print("【Fun-ASR错误】未安装funasr库，请运行: pip install funasr")
                return False

            # 获取模型目录
            model_dir = os.path.dirname(self.model_path)

            # 检查模型目录结构
            config_yaml_path = os.path.join(model_dir, 'config.yaml')
            config_json_path = os.path.join(model_dir, 'configuration.json')
            model_file_path = os.path.join(model_dir, 'model.pt')

            print(f"【Fun-ASR】模型目录: {model_dir}")
            print(f"【Fun-ASR】检查配置文件: {os.path.exists(config_yaml_path) or os.path.exists(config_json_path)}")
            print(f"【Fun-ASR】检查模型文件: {os.path.exists(model_file_path)}")

            # 检查是否有Qwen3-0.6B子目录
            qwen_subdir = os.path.join(model_dir, 'Qwen3-0.6B')
            has_qwen_subdir = os.path.exists(qwen_subdir)

            if has_qwen_subdir:
                print(f"【Fun-ASR】检测到Qwen3-0.6B子目录")

            # 尝试加载模型
            try:
                # 方法1: 直接从模型目录加载
                print(f"【Fun-ASR】尝试从本地目录加载...")
                self.model = AutoModel(
                    model=model_dir,
                    trust_remote_code=True,
                    device=self.device,
                    disable_pbar=True,
                    disable_log=True
                )
                print(f"【Fun-ASR】模型加载成功（本地目录）")
                return True

            except Exception as e1:
                print(f"【Fun-ASR】本地加载失败: {e1}")

                # 方法2: 尝试从Hugging Face下载
                try:
                    print(f"【Fun-ASR】尝试从Hugging Face下载模型...")
                    # 根据模型名称推断HF模型ID
                    model_name = os.path.basename(model_dir).lower()
                    if "mlt" in model_name:
                        hf_model_id = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
                    else:
                        hf_model_id = "FunAudioLLM/Fun-ASR-Nano-2512"

                    print(f"【Fun-ASR】使用模型ID: {hf_model_id}")
                    self.model = AutoModel(
                        model=hf_model_id,
                        trust_remote_code=True,
                        device=self.device,
                        disable_pbar=True,
                        disable_log=True
                    )
                    print(f"【Fun-ASR】模型加载成功（Hugging Face）")
                    return True

                except Exception as e2:
                    print(f"【Fun-ASR】Hugging Face加载也失败: {e2}")
                    raise e1

        except Exception as e:
            print(f"【Fun-ASR加载错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return False


    
    def transcribe(self, audio_input, language=None):
        """
        执行语音识别
        
        Args:
            audio_input: 音频输入，支持以下格式：
                - dict: {"waveform": tensor, "sample_rate": int}
                - torch.Tensor: 音频波形张量
                - numpy.ndarray: 音频波形数组
                - str: 音频文件路径
            language: 语言代码（覆盖默认配置）
        
        Returns:
            dict: 包含识别结果的字典 {"text": "识别文本", "language": "检测语言", "segments": []}
        """
        import time
        import psutil
        
        if audio_input is None:
            return {"text": "", "language": "", "segments": []}
        
        language = language if language is not None else self.config.get("language", "auto")
        task = self.config.get("task", "transcribe")
        
        # 开始性能监控
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(audio_input, language, task)
            
            # 检查缓存
            if cache_key in self._audio_cache:
                return self._audio_cache[cache_key]
            
            # 根据模型类型调用不同的识别方法
            if self.model_type == "whisper":
                result = self._transcribe_whisper(audio_input, language, task)
            elif self.model_type == "qwen_audio" or self.model_type == "qwen_asr" or self.model_type == "qwen3_asr":
                result = self._transcribe_qwen(audio_input, language)
            elif self.model_type == "wav2vec2":
                result = self._transcribe_wav2vec2(audio_input, language)
            elif self.model_type == "fun_asr":
                result = self._transcribe_fun_asr(audio_input, language)
            elif self.model_type == "pytorch_asr":
                result = self._transcribe_pytorch(audio_input, language)
            elif self.model_type == "gguf_asr":
                result = self._transcribe_gguf(audio_input, language)
            else:
                result = self._transcribe_placeholder(audio_input, language)
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            # 结束性能监控
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            inference_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # 在结果中添加性能信息
            result["performance"] = {
                "inference_time": inference_time,
                "memory_used": memory_used
            }
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_used = end_gpu_memory - start_gpu_memory
                result["performance"]["gpu_memory_used"] = gpu_memory_used
            
            return result
                
        except Exception as e:
            print(f"【ASR识别错误】语音识别失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}
    
    def _generate_cache_key(self, audio_input, language, task):
        """
        生成缓存键
        """
        import hashlib
        
        try:
            waveform, sample_rate = self._extract_audio_data(audio_input)
            if waveform is None:
                return None
            
            # 使用音频数据的哈希作为缓存键
            audio_hash = hashlib.md5(waveform.tobytes()).hexdigest()
            cache_key = f"{audio_hash}_{language}_{task}"
            
            return cache_key
        except Exception as e:
            print(f"【ASR缓存错误】生成缓存键失败: {str(e)}")
            return None
    
    def _cache_result(self, cache_key, result):
        """
        缓存识别结果
        """
        if cache_key is None:
            return
        
        try:
            # 如果缓存已满，删除最旧的条目
            if len(self._audio_cache) >= self._cache_size_limit:
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]
            
            self._audio_cache[cache_key] = result
            
        except Exception as e:
            print(f"【ASR缓存错误】缓存结果失败: {str(e)}")
    
    def _transcribe_whisper(self, audio_input, language, task):
        """
        Whisper模型识别
        """
        print(f"【Whisper ASR】开始识别，语言: {language}, 任务: {task}")
        
        if self.model is None:
            print("【Whisper ASR】模型未加载，尝试重新加载")
            self._load_whisper_model()
            if self.model is None:
                return {"text": "模型加载失败", "language": language, "segments": []}
        
        # 提取音频数据
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}
        
        print(f"【Whisper ASR】音频长度: {len(waveform)/sample_rate:.2f}秒")
        
        try:
            # 转换音频格式为Whisper需要的格式
            import librosa
            
            # 重采样到16kHz
            if sample_rate != 16000:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            
            # 执行识别
            options = {
                "task": task,
                "language": None if language == "auto" else language,
                "fp16": self.device == "cuda" and torch.cuda.is_available()
            }
            
            result = self.model.transcribe(waveform, **options)
            
            # 提取结果
            text = result.get("text", "").strip()
            detected_language = result.get("language", language)
            segments = result.get("segments", [])
            
            print(f"【Whisper ASR】识别成功，文本长度: {len(text)}字符")
            
            return {
                "text": text,
                "language": detected_language,
                "segments": segments
            }
            
        except ImportError:
            print("【Whisper ASR】未安装librosa库，请运行: pip install librosa")
            return {"text": "请安装librosa库: pip install librosa", "language": language, "segments": []}
        except Exception as e:
            print(f"【Whisper ASR错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}
    
    def _transcribe_wav2vec2(self, audio_input, language):
        """
        Wav2Vec2模型识别
        """
        print(f"【Wav2Vec2 ASR】开始识别，语言: {language}")
        
        if self.model is None:
            print("【Wav2Vec2 ASR】模型未加载，尝试重新加载")
            self._load_wav2vec2_model()
            if self.model is None:
                return {"text": "模型加载失败", "language": language, "segments": []}
        
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}
        
        print(f"【Wav2Vec2 ASR】音频长度: {len(waveform)/sample_rate:.2f}秒")
        
        try:
            processor = self.model["processor"]
            model = self.model["model"]
            
            # 重采样到16kHz
            if sample_rate != 16000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            
            # 预处理音频
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # 移动到GPU
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # 推理
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # 解码
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            print(f"【Wav2Vec2 ASR】识别成功，文本长度: {len(transcription)}字符")
            
            return {
                "text": transcription,
                "language": language,
                "segments": []
            }
            
        except Exception as e:
            print(f"【Wav2Vec2 ASR错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}
    
    def _transcribe_qwen(self, audio_input, language):
        """
        Qwen音频模型识别
        """
        print(f"【Qwen ASR】开始识别，语言: {language}")
        
        if self.model is None:
            print("【Qwen ASR】模型未加载，尝试重新加载")
            self._load_qwen_model()
            if self.model is None:
                return {"text": "模型加载失败", "language": language, "segments": []}
        
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}
        
        print(f"【Qwen ASR】音频长度: {len(waveform)/sample_rate:.2f}秒")
        
        try:
            # 检查是否为Qwen3-ASR模型（使用qwen-asr包）
            if hasattr(self.model, 'transcribe'):
                # Qwen3ASRModel接口
                print(f"【Qwen3-ASR】使用官方qwen-asr包进行识别")
                
                # 准备音频输入
                audio_input_qwen = (waveform, sample_rate)
                
                # 执行识别
                results = self.model.transcribe(
                    audio=audio_input_qwen,
                    language=None if language == "auto" else language,
                    return_time_stamps=self.config.get("enable_timestamps", False)
                )
                
                # 提取结果
                if results and len(results) > 0:
                    result = results[0]
                    text = result.text if hasattr(result, 'text') else str(result)
                    detected_language = result.language if hasattr(result, 'language') else language
                    
                    # 处理时间戳
                    segments = []
                    if hasattr(result, 'time_stamps') and result.time_stamps:
                        for ts in result.time_stamps:
                            segments.append({
                                "start": ts.start_time,
                                "end": ts.end_time,
                                "text": ts.text
                            })
                    
                    print(f"【Qwen3-ASR】识别成功，文本长度: {len(text)}字符")
                    
                    return {
                        "text": text,
                        "language": detected_language,
                        "segments": segments
                    }
                else:
                    return {"text": "识别结果为空", "language": language, "segments": []}
                    
            else:
                # 旧版Qwen Audio模型（使用transformers）
                processor = self.model["processor"]
                model = self.model["model"]
                
                # 重采样到16kHz
                if sample_rate != 16000:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                
                # 预处理音频
                inputs = processor(audio=waveform, sampling_rate=16000, return_tensors="pt")
                
                # 移动到GPU
                if self.device == "cuda" and torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # 推理
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=256)
                
                # 解码
                transcription = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                
                print(f"【Qwen Audio模型】识别成功，文本长度: {len(transcription)}字符")
                
                return {
                    "text": transcription,
                    "language": language,
                    "segments": []
                }
            
        except Exception as e:
            print(f"【Qwen ASR错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}
    
    def _transcribe_pytorch(self, audio_input, language):
        """
        PyTorch格式ASR模型识别
        """
        print(f"【PyTorch ASR】开始识别，语言: {language}")
        
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}
        
        print(f"【PyTorch ASR】音频长度: {len(waveform)/sample_rate:.2f}秒")
        
        # TODO: 实现真正的PyTorch ASR推理
        return {
            "text": "[PyTorch ASR占位符] 请集成PyTorch ASR模型推理逻辑",
            "language": language if language != "auto" else "zh",
            "segments": []
        }
    
    def _transcribe_gguf(self, audio_input, language):
        """
        GGUF格式ASR模型识别
        """
        print(f"【GGUF ASR】开始识别，语言: {language}")
        
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}
        
        print(f"【GGUF ASR】音频长度: {len(waveform)/sample_rate:.2f}秒")
        
        # TODO: 实现真正的GGUF ASR推理
        return {
            "text": "[GGUF ASR占位符] 请集成GGUF ASR模型推理逻辑",
            "language": language if language != "auto" else "zh",
            "segments": []
        }
    
    def _transcribe_fun_asr(self, audio_input, language):
        """
        Fun-ASR模型识别
        支持 Fun-ASR-Nano-2512 和 Fun-ASR-MLT-Nano-2512 等模型
        """
        print(f"【Fun-ASR】开始识别，语言: {language}")

        if self.model is None:
            print("【Fun-ASR】模型未加载，尝试重新加载")
            if not self._load_fun_asr_model():
                return {"text": "模型加载失败", "language": language, "segments": []}

        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}

        print(f"【Fun-ASR】音频长度: {len(waveform)/sample_rate:.2f}秒, 采样率: {sample_rate}Hz")

        temp_path = None
        try:
            import tempfile
            import os

            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp_path = temp.name

            # 保存音频到临时文件
            try:
                import soundfile as sf
                sf.write(temp_path, waveform, sample_rate)
                print(f"【Fun-ASR】音频已保存到临时文件: {temp_path}")
            except Exception as e:
                print(f"【Fun-ASR错误】保存音频失败: {e}")
                return {"text": f"音频保存失败: {str(e)}", "language": language, "segments": []}

            # 准备识别参数
            language_map = {
                "auto": "auto",
                "zh": "zh",
                "en": "en",
                "ja": "ja",
                "ko": "ko",
                "fr": "fr",
                "de": "de",
                "es": "es"
            }
            lang_code = language_map.get(language, "auto")

            print(f"【Fun-ASR】执行识别，语言代码: {lang_code}")

            # 执行识别
            try:
                res = self.model.generate(
                    input=[temp_path],
                    cache={},
                    batch_size=1,
                    language=lang_code
                )
                print(f"【Fun-ASR】识别结果类型: {type(res)}")
            except Exception as e:
                print(f"【Fun-ASR错误】模型推理失败: {e}")
                # 尝试不使用语言参数
                try:
                    res = self.model.generate(
                        input=[temp_path],
                        cache={},
                        batch_size=1
                    )
                except Exception as e2:
                    print(f"【Fun-ASR错误】重试也失败: {e2}")
                    raise e

            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                temp_path = None

            # 提取结果
            text = ""
            if isinstance(res, list) and len(res) > 0:
                if isinstance(res[0], dict):
                    text = res[0].get("text", "")
                else:
                    text = str(res[0])
            elif isinstance(res, dict):
                text = res.get("text", "")
            else:
                text = str(res)

            print(f"【Fun-ASR】识别成功，文本长度: {len(text)}字符")

            return {
                "text": text,
                "language": language if language != "auto" else "zh",
                "segments": []
            }

        except Exception as e:
            # 确保清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

            print(f"【Fun-ASR错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}




    
    def _transcribe_placeholder(self, audio_input, language):
        """
        占位符ASR实现
        返回模拟的识别结果
        """
        print(f"【ASR占位符】模拟识别，语言: {language}")
        
        waveform, sample_rate = self._extract_audio_data(audio_input)
        duration = len(waveform) / sample_rate if waveform is not None else 0
        
        # 根据音频长度生成占位符文本
        placeholder_text = f"[ASR占位符] 检测到音频输入，时长{duration:.2f}秒。请集成真正的ASR模型以获得准确的语音识别结果。"
        
        return {
            "text": placeholder_text,
            "language": language if language != "auto" else "zh",
            "segments": [{"start": 0, "end": duration, "text": placeholder_text}]
        }
    
    def _extract_audio_data(self, audio_input):
        """
        从各种音频输入格式中提取波形数据和采样率
        
        Returns:
            tuple: (waveform_numpy_array, sample_rate)
        """
        try:
            waveform = None
            sample_rate = 16000  # 默认16kHz
            
            if isinstance(audio_input, dict):
                waveform = audio_input.get("waveform")
                sample_rate = audio_input.get("sample_rate", 16000)
            elif isinstance(audio_input, torch.Tensor):
                waveform = audio_input
            elif isinstance(audio_input, np.ndarray):
                waveform = audio_input
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                # 从文件加载音频
                try:
                    import soundfile as sf
                    waveform, sample_rate = sf.read(audio_input)
                except ImportError:
                    from scipy.io import wavfile
                    sample_rate, waveform = wavfile.read(audio_input)
            else:
                print(f"【ASR错误】不支持的音频输入类型: {type(audio_input)}")
                return None, 16000
            
            # 转换为numpy数组
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            # 处理维度
            if waveform.ndim > 1:
                if waveform.ndim == 3:  # [batch, channels, samples]
                    waveform = waveform.squeeze(0)
                if waveform.ndim == 2:  # [channels, samples]
                    waveform = waveform.mean(axis=0)  # 转为单声道
            
            # 确保是float32
            if waveform.dtype != np.float32:
                if waveform.dtype == np.int16:
                    waveform = waveform.astype(np.float32) / 32768.0
                else:
                    waveform = waveform.astype(np.float32)
            
            return waveform, sample_rate
            
        except Exception as e:
            print(f"【ASR错误】音频数据提取失败: {str(e)}")
            return None, 16000
    
    def clean(self):
        """
        清理模型资源
        """
        try:
            # 清理缓存
            self._audio_cache.clear()
            print("【ASR包装器】已清理音频缓存")
            
            # 清理模型
            if self.model is not None:
                if isinstance(self.model, dict):
                    for key, value in self.model.items():
                        if hasattr(value, 'cpu'):
                            value.cpu()
                    del self.model
                else:
                    del self.model
                self.model = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("【ASR包装器】已清理GPU缓存")
            
            print("【ASR包装器】ASR模型已卸载")
            
        except Exception as e:
            print(f"【ASR清理错误】{str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "llama_cpp_asr_loader": llama_cpp_asr_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_asr_loader": "Llama-cpp ASR模型加载器",
}
