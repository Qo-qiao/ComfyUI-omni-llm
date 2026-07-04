# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm ASR Model Loader Node
仅支持 Qwen3-ASR 模型

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import json
import torch
import numpy as np
import sys
import requests
import shutil
import time
import psutil
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

site_packages_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site-packages")
if os.path.exists(site_packages_path):
    sys.path.insert(0, site_packages_path)

from common import HARDWARE_INFO, folder_paths


ASR_MODEL_STORAGE = type('ASR_MODEL_STORAGE', (), {})()
asr_model_cache = {}


def _check_network_connection():
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
    if _check_network_connection():
        return "modelscope"
    else:
        return "huggingface"

def _download_file(url, save_path):
    try:
        print(f"【文件下载】开始下载: {url}")
        print(f"【文件下载】保存路径: {save_path}")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
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
    source = _get_model_source()
    print(f"【模型下载】使用源: {source}")
    
    model_files = {
        "qwen3_asr_1.7b": {
            "modelscope": {
                "model.safetensors": f"https://modelscope.cn/api/v1/models/eclipse005/Qwen3-ASR-1.7B/snapshots/master/model.safetensors",
                "config.json": f"https://modelscope.cn/api/v1/models/eclipse005/Qwen3-ASR-1.7B/snapshots/master/config.json",
                "tokenizer.json": f"https://modelscope.cn/api/v1/models/eclipse005/Qwen3-ASR-1.7B/snapshots/master/tokenizer.json",
                "preprocessor_config.json": f"https://modelscope.cn/api/v1/models/eclipse005/Qwen3-ASR-1.7B/snapshots/master/preprocessor_config.json"
            },
            "huggingface": {
                "model.safetensors": f"https://huggingface.co/eclipse005/Qwen3-ASR-1.7B/resolve/main/model.safetensors",
                "config.json": f"https://huggingface.co/eclipse005/Qwen3-ASR-1.7B/resolve/main/config.json",
                "tokenizer.json": f"https://huggingface.co/eclipse005/Qwen3-ASR-1.7B/resolve/main/tokenizer.json",
                "preprocessor_config.json": f"https://huggingface.co/eclipse005/Qwen3-ASR-1.7B/resolve/main/preprocessor_config.json"
            }
        },
        "qwen3_asr_0.6b": {
            "modelscope": {
                "model.safetensors": f"https://modelscope.cn/api/v1/models/Qwen/Qwen3-ASR-0.6B/snapshots/master/model.safetensors",
                "config.json": f"https://modelscope.cn/api/v1/models/Qwen/Qwen3-ASR-0.6B/snapshots/master/config.json",
                "tokenizer.json": f"https://modelscope.cn/api/v1/models/Qwen/Qwen3-ASR-0.6B/snapshots/master/tokenizer.json",
                "preprocessor_config.json": f"https://modelscope.cn/api/v1/models/Qwen/Qwen3-ASR-0.6B/snapshots/master/preprocessor_config.json"
            },
            "huggingface": {
                "model.safetensors": f"https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/model.safetensors",
                "config.json": f"https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/config.json",
                "tokenizer.json": f"https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/tokenizer.json",
                "preprocessor_config.json": f"https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/preprocessor_config.json"
            }
        }
    }
    
    model_type = None
    if "qwen3" in model_name.lower() and "asr" in model_name.lower():
        if "1.7b" in model_name.lower() or "17b" in model_name.lower():
            model_type = "qwen3_asr_1.7b"
        elif "0.6b" in model_name.lower() or "6b" in model_name.lower() and not "1.7b" in model_name.lower() and not "17b" in model_name.lower():
            model_type = "qwen3_asr_0.6b"
    
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

def _check_asr_model_files(model_path):
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json"
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    return missing_files


class llama_cpp_asr_loader:
    """
    Qwen3-ASR模型加载器
    """
    
    @classmethod
    def INPUT_TYPES(s):
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

        llm_folders = folder_paths.get_folder_paths("LLM")
        
        asr_list = ["None"]
        model_set = set()
        model_set.add("None")
        
        for folder in llm_folders:
            try:
                root_tag = os.path.basename(folder) if len(llm_folders) > 1 else ""

                for root, dirs, files in os.walk(folder):
                    rel_path = os.path.relpath(root, folder)
                    if rel_path == '.':
                        current_folder_name = ""
                    else:
                        current_folder_name = rel_path.replace(os.sep, '/')

                    dir_name = os.path.basename(root).lower()

                    if "qwen3" in dir_name and "asr" in dir_name:
                        has_model_files = any(os.path.splitext(f)[1].lower() in [".safetensors", ".bin"] for f in files)

                        if has_model_files:
                            if current_folder_name:
                                folder_entry = current_folder_name
                            else:
                                folder_entry = dir_name

                            if root_tag:
                                folder_entry = f"{root_tag}/{folder_entry}"

                            folder_entry = folder_entry.rstrip('/') + '/'
                            
                            folder_abs_path = os.path.normpath(os.path.join(folder, folder_entry.rstrip('/')))
                            if folder_abs_path not in model_set:
                                model_set.add(folder_abs_path)
                                asr_list.append(folder_entry)
                                print(f"【ASR模型检测】检测到Qwen3-ASR目录: {folder_entry} (绝对路径: {folder_abs_path})")

                        sharded_files1 = [f for f in files if "-of-" in f.lower() and f.endswith(".safetensors")]
                        sharded_files2 = [f for f in files if f.startswith("model.safetensors-") and "-of-" in f.lower()]
                        if sharded_files1 or sharded_files2:
                            if current_folder_name:
                                folder_entry = current_folder_name
                            else:
                                folder_entry = dir_name

                            if root_tag:
                                folder_entry = f"{root_tag}/{folder_entry}"

                            folder_entry = folder_entry.rstrip('/') + '/'
                            
                            folder_abs_path = os.path.normpath(os.path.join(folder, folder_entry.rstrip('/')))
                            if folder_abs_path not in model_set:
                                model_set.add(folder_abs_path)
                                asr_list.append(folder_entry)
                                print(f"【ASR模型检测】检测到分段Qwen3-ASR模型文件夹: {folder_entry} (绝对路径: {folder_abs_path})")

                    has_model_files_in_dir = any(os.path.splitext(f)[1].lower() in [".safetensors", ".bin"] for f in files)
                    if has_model_files_in_dir:
                        folder_path_lower = root.lower()
                        folder_name_lower = dir_name.lower()
                        if "qwen3" in folder_path_lower and "asr" in folder_path_lower:
                            if current_folder_name:
                                folder_entry = current_folder_name
                            else:
                                folder_entry = dir_name

                            if root_tag:
                                folder_entry = f"{root_tag}/{folder_entry}"

                            folder_entry = folder_entry.rstrip('/') + '/'
                            
                            folder_abs_path = os.path.normpath(os.path.join(folder, folder_entry.rstrip('/')))
                            if folder_abs_path not in model_set:
                                model_set.add(folder_abs_path)
                                asr_list.append(folder_entry)
                                print(f"【ASR模型检测】检测到包含Qwen3-ASR模型文件的目录: {folder_entry} (绝对路径: {folder_abs_path})")
            except Exception as e:
                print(f"【ASR模型检测】扫描文件夹 {folder} 失败: {e}")
        
        if len(asr_list) == 1:
            asr_list = ["None", "请将Qwen3-ASR模型放入models/LLM文件夹（支持Qwen3-ASR、Qwen3-ASR-1.7B、Qwen3-ASR-0.6B）"]
        
        perf_level = HARDWARE_INFO.get("perf_level", "low")

        if perf_level == "high":
            default_n_gpu_layers = -1
        elif perf_level == "mid_high":
            default_n_gpu_layers = -1
        elif perf_level == "mid":
            default_n_gpu_layers = -1
        elif perf_level == "mid_low":
            default_n_gpu_layers = -1
        else:
            default_n_gpu_layers = 20
        
        qwen3_languages = [
            "auto", "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", 
            "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl",
            "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro"
        ]
        
        return {
            "required": {
                "asr_model": (asr_list, {"tooltip": "选择Qwen3-ASR模型文件夹"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载"}),
                "language": (qwen3_languages, {"default": "auto", "tooltip": "识别语言，auto=自动检测"}),
                "task": (["transcribe", "translate"], {"default": "transcribe", "tooltip": "任务类型：transcribe=转录，translate=翻译成英文"}),
            },
            "optional": {
                "enable_timestamps": ("BOOLEAN", {"default": False, "tooltip": "启用时间戳功能（需要Qwen3-ForcedAligner）"}),
                "flash_attention": ("BOOLEAN", {"default": False, "tooltip": "启用FlashAttention优化（需安装flash-attn）"}),
            }
        }
    
    RETURN_TYPES = ("ASRMODEL",)
    RETURN_NAMES = ("asr_model",)
    FUNCTION = "load_asr_model"
    CATEGORY = "omni-llm"
    
    @classmethod
    def _resolve_asr_model_path(s, asr_model):
        key = asr_model.rstrip('/')
        if key == "None":
            return None

        print(f"【路径解析】开始解析: {asr_model} -> key: {key}")

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

        for folder in folder_paths.get_folder_paths("LLM"):
            candidate = os.path.join(folder, key)
            print(f"【路径解析】尝试: {candidate}")
            if os.path.exists(candidate):
                print(f"【路径解析】找到: {candidate}")
                return os.path.normpath(candidate)

        full = folder_paths.get_full_path("LLM", key)
        if full and os.path.exists(full):
            print(f"【路径解析】通过get_full_path找到: {full}")
            return os.path.normpath(full)

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
        if asr_model == "None" or asr_model == "请将Qwen3-ASR模型放入models/LLM文件夹":
            print("【ASR加载器】未选择ASR模型")
            return (None,)

        resolved_model_path = self._resolve_asr_model_path(asr_model)
        if not resolved_model_path:
            raise RuntimeError(f"无法解析ASR模型路径: {asr_model}")

        try:
            print(f"【ASR加载器】正在加载Qwen3-ASR模型: {asr_model} (解析路径: {resolved_model_path})")
            
            run_mode = "GPU"
            print(f"【ASR加载器】运行模式: {run_mode}, GPU层数: {n_gpu_layers}")
            
            config = {
                "asr_model": asr_model,
                "asr_model_path": resolved_model_path,
                "n_gpu_layers": n_gpu_layers,
                "language": language,
                "task": task,
                "enable_timestamps": enable_timestamps,
                "flash_attention": flash_attention
            }
            
            if self.loaded_model is not None and self.current_config == config:
                print("【ASR加载器】使用缓存的ASR模型")
                return (self.loaded_model,)
            
            model_path = resolved_model_path
            
            if model_path is None or not os.path.exists(model_path):
                raise RuntimeError(f"ASR模型文件夹不存在: {asr_model} (解析路径：{resolved_model_path})")
            
            if not os.path.isdir(model_path):
                model_path = os.path.dirname(model_path)
                print(f"【ASR加载器】检测到文件路径，自动使用所在目录: {model_path}")
            
            print(f"【ASR加载器】模型路径: {model_path}")
            print(f"【ASR加载器】语言: {language}, 任务: {task}")
            
            model_dir = model_path
            
            print("【ASR加载器】检测模型类型: qwen3_asr")
            missing_files = _check_asr_model_files(model_dir)
            
            if missing_files:
                print(f"【ASR加载器】检测到缺失文件: {missing_files}")
                print(f"【ASR加载器】尝试自动下载缺失文件...")
                
                if _download_model_files(asr_model, model_dir):
                    print(f"【ASR加载器】缺失文件下载成功")
                else:
                    print(f"【ASR加载器警告】部分文件下载失败，尝试继续加载")
            
            asr_wrapper = Qwen3ASRModelWrapper(
                model_path=model_path,
                config=config,
                device="cuda"
            )
            
            if asr_wrapper.model is None:
                print("【ASR加载器错误】模型包装器已创建，但内部模型为None，加载失败")
                return (None,)
            
            self.loaded_model = asr_wrapper
            self.current_config = config
            
            global asr_model_cache
            asr_model_cache[asr_model] = asr_wrapper
            print(f"【ASR加载器】Qwen3-ASR模型加载成功，已添加到缓存")
            return (asr_wrapper,)
            
        except Exception as e:
            print(f"【ASR加载器错误】加载ASR模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return (None,)


class Qwen3ASRModelWrapper:
    """
    Qwen3-ASR模型包装器
    """
    
    def __init__(self, model_path, config, device="cpu"):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model = None
        self._audio_cache = {}
        self._cache_size_limit = 100
        
        print(f"【ASR包装器】设备: {device}")
        
        self._load_model()
    
    def _load_model(self):
        try:
            from qwen_asr import Qwen3ASRModel
            import torch
            
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            enable_timestamps = self.config.get("enable_timestamps", False)
            
            model_path_for_loading = self.model_path
            if os.path.isfile(self.model_path):
                model_path_for_loading = os.path.dirname(self.model_path)
                print(f"【Qwen3-ASR模型】检测到文件路径，使用目录路径: {model_path_for_loading}")
            
            model_kwargs = {
                "dtype": dtype,
                "device_map": "cuda:0" if self.device == "cuda" else "cpu",
                "max_inference_batch_size": 32,
                "max_new_tokens": 256
            }
            
            if self.config.get("flash_attention", False):
                try:
                    try:
                        import flash_attn_3
                        model_kwargs["attn_implementation"] = "flash_attention_3"
                        print(f"【Qwen3-ASR模型】启用FlashAttention3优化")
                    except ImportError:
                        import flash_attn
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                        print(f"【Qwen3-ASR模型】启用FlashAttention2优化")
                except ImportError:
                    print(f"【Qwen3-ASR模型】FlashAttention未安装，使用默认Attention实现")
            
            if enable_timestamps:
                model_kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"
                aligner_kwargs = dict(
                    dtype=dtype,
                    device_map="cuda:0" if self.device == "cuda" else "cpu"
                )
                
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
            
        except ImportError:
            print("【Qwen3-ASR模型】未安装qwen-asr库，请运行: pip install -U qwen-asr")
        except Exception as e:
            print(f"【Qwen3-ASR模型加载错误】{str(e)}")
            import traceback
            traceback.print_exc()
    
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
        if audio_input is None:
            return {"text": "", "language": "", "segments": []}
        
        language = language if language is not None else self.config.get("language", "auto")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        try:
            cache_key = self._generate_cache_key(audio_input, language)
            
            if cache_key in self._audio_cache:
                return self._audio_cache[cache_key]
            
            result = self._transcribe_qwen(audio_input, language)
            
            self._cache_result(cache_key, result)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            inference_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            result["performance"] = {
                "inference_time": inference_time,
                "memory_used": memory_used
            }
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_used = end_gpu_memory - start_gpu_memory
                result["performance"]["gpu_memory_used"] = gpu_memory_used
            
            return result
                
        except Exception as e:
            print(f"【ASR识别错误】语音识别失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}
    
    def _generate_cache_key(self, audio_input, language):
        try:
            waveform, sample_rate = self._extract_audio_data(audio_input)
            if waveform is None:
                return None
            
            audio_hash = hashlib.md5(waveform.tobytes()).hexdigest()
            cache_key = f"{audio_hash}_{language}"
            
            return cache_key
        except Exception as e:
            print(f"【ASR缓存错误】生成缓存键失败: {str(e)}")
            return None
    
    def _cache_result(self, cache_key, result):
        if cache_key is None:
            return
        
        try:
            if len(self._audio_cache) >= self._cache_size_limit:
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]
            
            self._audio_cache[cache_key] = result
            
        except Exception as e:
            print(f"【ASR缓存错误】缓存结果失败: {str(e)}")
    
    def _transcribe_qwen(self, audio_input, language):
        """
        Qwen3-ASR模型识别
        """
        print(f"【Qwen ASR】开始识别，语言: {language}")
        
        if self.model is None:
            print("【Qwen ASR】模型未加载，尝试重新加载")
            self._load_model()
            if self.model is None:
                return {"text": "模型加载失败", "language": language, "segments": []}
        
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None:
            return {"text": "音频数据提取失败", "language": language, "segments": []}
        
        print(f"【Qwen ASR】音频长度: {len(waveform)/sample_rate:.2f}秒")
        
        try:
            if hasattr(self.model, 'transcribe'):
                print(f"【Qwen3-ASR】使用官方qwen-asr包进行识别")
                
                audio_input_qwen = (waveform, sample_rate)
                
                results = self.model.transcribe(
                    audio=audio_input_qwen,
                    language=None if language == "auto" else language,
                    return_time_stamps=self.config.get("enable_timestamps", False)
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    text = result.text if hasattr(result, 'text') else str(result)
                    detected_language = result.language if hasattr(result, 'language') else language
                    
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
                return {"text": "Qwen3-ASR模型不支持transcribe方法", "language": language, "segments": []}
            
        except Exception as e:
            print(f"【Qwen ASR错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language, "segments": []}
    
    def _extract_audio_data(self, audio_input):
        """
        从各种音频输入格式中提取波形数据和采样率
        
        Returns:
            tuple: (waveform_numpy_array, sample_rate)
        """
        try:
            waveform = None
            sample_rate = 16000
            
            if isinstance(audio_input, dict):
                waveform = audio_input.get("waveform")
                sample_rate = audio_input.get("sample_rate", 16000)
            elif isinstance(audio_input, torch.Tensor):
                waveform = audio_input
            elif isinstance(audio_input, np.ndarray):
                waveform = audio_input
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                try:
                    import soundfile as sf
                    waveform, sample_rate = sf.read(audio_input)
                except ImportError:
                    from scipy.io import wavfile
                    sample_rate, waveform = wavfile.read(audio_input)
            else:
                print(f"【ASR错误】不支持的音频输入类型: {type(audio_input)}")
                return None, 16000
            
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            
            if waveform.ndim > 1:
                if waveform.ndim == 3:
                    waveform = waveform.squeeze(0)
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=0)
            
            if waveform.dtype != np.float32:
                if waveform.dtype == np.int16:
                    waveform = waveform.astype(np.float32) / 32768.0
                else:
                    waveform = waveform.astype(np.float32)
            
            return waveform, sample_rate
            
        except Exception as e:
            print(f"【ASR错误】音频数据提取失败: {str(e)}")
            return None, 16000
    
    def release(self):
        """
        释放模型资源
        """
        try:
            if self.model is not None:
                # 将模型移到CPU
                if hasattr(self.model, 'to'):
                    try:
                        self.model = self.model.to('cpu')
                    except Exception:
                        pass
                
                del self.model
                self.model = None
            
            # 清空音频缓存
            self._audio_cache = {}
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 同步GPU操作并清空缓存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("【ASR包装器】已清理GPU缓存")
            else:
                print("【ASR包装器】CUDA不可用，仅清理CPU内存")
            
            print("【ASR包装器】ASR模型已卸载")
            
        except Exception as e:
            print(f"【ASR清理错误】{str(e)}")
    
    def __del__(self):
        """
        清理资源
        """
        self.release()


NODE_CLASS_MAPPINGS = {
    "llama_cpp_asr_loader": llama_cpp_asr_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_asr_loader": "Llama-cpp ASR模型加载器",
}