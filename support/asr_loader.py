# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm ASR Model Loader Node
基于 llama-cpp-python 0.4.0 MTMD 音频推理（Qwen3ASRChatHandler）

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import re
import json
import sys
import time
import base64
import wave
import io
import hashlib
import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 优先加载本地 site-packages（小型音频依赖安装位置），须在依赖检测之前完成
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_SITE_PACKAGES = os.path.join(_PROJECT_ROOT, "site-packages")
if os.path.isdir(_LOCAL_SITE_PACKAGES) and _LOCAL_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, _LOCAL_SITE_PACKAGES)

from common import HARDWARE_INFO, folder_paths, LLAMA_CPP_STORAGE, _has_mtmd, _llama_cpp

ASR_MODEL_STORAGE = type('ASR_MODEL_STORAGE', (), {})()
asr_model_cache = {}

LANGUAGE_ISO_TO_NAME = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
    "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}

# Qwen3-ASR 原生输出形如：language English<asr_text>识别文本
_ASR_OUTPUT_PATTERN = re.compile(
    r"^\s*language\s+(?P<language>[^<\r\n]+?)\s*<asr_text>\s*(?P<text>[\s\S]*)$",
    re.IGNORECASE,
)


def _normalize_language(language):
    if language and language != "auto" and language in LANGUAGE_ISO_TO_NAME:
        return LANGUAGE_ISO_TO_NAME[language]
    return language


def _scan_mmproj_files():
    """扫描所有 LLM 目录中的 mmproj 音频/视觉编码模型（gguf），返回相对路径列表"""
    if "LLM" not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

    mmproj_list = ["None"]
    mmproj_set = set()

    for folder in folder_paths.get_folder_paths("LLM"):
        try:
            for root, dirs, files in os.walk(folder):
                rel_path = os.path.relpath(root, folder)
                for f in files:
                    if not f.lower().endswith(".gguf"):
                        continue
                    if "mmproj" not in f.lower() and "vision" not in f.lower():
                        continue
                    file_abs_path = os.path.normpath(os.path.join(root, f))
                    if file_abs_path in mmproj_set:
                        continue
                    mmproj_set.add(file_abs_path)
                    if rel_path == '.':
                        mmproj_list.append(f)
                    else:
                        mmproj_list.append(f"{rel_path.replace(os.sep, '/')}/{f}")
        except Exception as e:
            print(f"【ASR模型检测】扫描mmproj文件夹 {folder} 失败: {e}")

    return mmproj_list


class omni_llm_asr_loader:
    """
    基于 llama-cpp-python 0.4.0 MTMD 的 ASR 模型加载器
    使用 Qwen3ASRChatHandler 走 llama.cpp 原生音频推理
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
                    dir_name = os.path.basename(root).lower()
                    folder_path_lower = root.lower()

                    is_asr_dir = (
                        "asr" in dir_name or "speech" in dir_name
                        or "asr" in folder_path_lower or "speech" in folder_path_lower
                    )
                    if not is_asr_dir:
                        continue

                    for f in files:
                        if not f.lower().endswith(".gguf"):
                            continue
                        # 跳过mmproj辅助文件
                        if f.lower().startswith("mmproj"):
                            continue

                        rel_path = os.path.relpath(root, folder)
                        if rel_path == '.':
                            file_entry = f
                        else:
                            file_entry = f"{rel_path.replace(os.sep, '/')}/{f}"

                        file_abs_path = os.path.normpath(os.path.join(root, f))
                        if file_abs_path in model_set:
                            continue

                        if root_tag:
                            file_entry = f"{root_tag}/{file_entry}"

                        model_set.add(file_abs_path)
                        asr_list.append(file_entry)
                        print(f"【ASR模型检测】检测到ASR模型文件: {file_entry}")
            except Exception as e:
                print(f"【ASR模型检测】扫描文件夹 {folder} 失败: {e}")

        if len(asr_list) == 1:
            asr_list = ["None", "请将ASR/GGUF模型放入models/LLM文件夹（支持GGUF格式模型）"]

        # ASR 强制 GPU/CUDA 推理：默认全部层卸载到 GPU
        default_n_gpu_layers = -1

        languages = [
            "auto", "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
            "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese",
            "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish",
            "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
            "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it",
            "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv",
            "da", "fi", "pl", "cs", "fil", "fa", "el", "ro", "hu", "mk",
        ]

        mmproj_list = _scan_mmproj_files()

        return {
            "required": {
                "asr_model": (asr_list, {"tooltip": "选择ASR GGUF模型文件（asr/speech目录内）"}),
                "mmproj": (mmproj_list, {"default": "None", "tooltip": "手动指定mmproj；None=自动从模型同目录检测"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载（强制GPU/CUDA推理）"}),
                "language": (languages, {"default": "auto", "tooltip": "识别语言，auto=自动检测（可传ISO代码或语言英文名）"}),
                "task": (["transcribe", "translate"], {"default": "transcribe", "tooltip": "任务类型：transcribe=转录，translate=翻译成英文"}),
            },
            "optional": {
                "enable_timestamps": ("BOOLEAN", {"default": False, "tooltip": "启用时间戳功能（需模型支持，当前Qwen3-ASR输出段为空）"}),
                "n_ctx": ("INT", {"default": 8192, "min": 512, "max": 32768, "step": 128, "tooltip": "上下文长度"}),
            }
        }

    RETURN_TYPES = ("ASRMODEL",)
    RETURN_NAMES = ("asr_model",)
    FUNCTION = "load_asr_model"
    CATEGORY = "omni-llm"

    @classmethod
    def _resolve_asr_model_path(s, asr_model):
        """解析选择的模型文件相对路径 -> 绝对路径"""
        key = asr_model.replace("\\", "/").rstrip('/')
        if key == "None" or not key:
            return None

        # 尝试 root_tag/inner 形式（多 LLM 文件夹）
        if "/" in key:
            root_name, inner = key.split("/", 1)
            for folder in folder_paths.get_folder_paths("LLM"):
                if os.path.basename(folder) == root_name:
                    candidate = os.path.join(folder, inner)
                    if os.path.isfile(candidate):
                        return os.path.normpath(candidate)

        for folder in folder_paths.get_folder_paths("LLM"):
            candidate = os.path.join(folder, key)
            if os.path.isfile(candidate):
                return os.path.normpath(candidate)

        full = folder_paths.get_full_path("LLM", key)
        if full and os.path.isfile(full):
            return os.path.normpath(full)

        return None

    @classmethod
    def _resolve_mmproj_path(s, mmproj):
        """解析手动选择的 mmproj 文件路径（必选）"""
        if not mmproj or mmproj == "None":
            return None

        key = mmproj.replace("\\", "/").rstrip('/')
        if os.path.isabs(key) and os.path.exists(key):
            return os.path.normpath(key)

        if "/" in key:
            root_name, inner = key.split("/", 1)
            for folder in folder_paths.get_folder_paths("LLM"):
                if os.path.basename(folder) == root_name:
                    candidate = os.path.join(folder, inner)
                    if os.path.exists(candidate):
                        return os.path.normpath(candidate)

        for folder in folder_paths.get_folder_paths("LLM"):
            candidate = os.path.join(folder, key)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)

        full = folder_paths.get_full_path("LLM", key)
        if full and os.path.exists(full):
            return os.path.normpath(full)

        return None

    @classmethod
    def _find_mmproj_in_dir(s, model_dir):
        """在模型同目录查找mmproj文件"""
        try:
            for f in sorted(os.listdir(model_dir)):
                if f.lower().endswith(".gguf") and f.lower().startswith("mmproj"):
                    return os.path.join(model_dir, f)
        except Exception:
            pass
        return None

    def __init__(self):
        self.loaded_model = None
        self.current_config = None

    @classmethod
    def IS_CHANGED(s, asr_model, n_gpu_layers, language, task, mmproj="None", enable_timestamps=False, n_ctx=8192):
        resolved_path = s._resolve_asr_model_path(asr_model)
        resolved_mmproj = s._resolve_mmproj_path(mmproj)
        config = {
            "asr_model": asr_model,
            "asr_model_path": resolved_path or "",
            "n_gpu_layers": n_gpu_layers,
            "language": language,
            "task": task,
            "enable_timestamps": enable_timestamps,
            "n_ctx": n_ctx,
            "mmproj": resolved_mmproj or "",
        }
        return json.dumps(config, sort_keys=True, ensure_ascii=False)

    def load_asr_model(self, asr_model, n_gpu_layers, language, task, mmproj="None", enable_timestamps=False, n_ctx=8192):
        if asr_model == "None" or "请将" in asr_model:
            print("【ASR加载器】未选择ASR模型")
            return (None,)

        model_file = self._resolve_asr_model_path(asr_model)
        if not model_file:
            raise RuntimeError(f"无法解析ASR模型路径: {asr_model}")

        # 解析mmproj：手动指定优先，否则自动从模型同目录检测
        manual_mmproj = self._resolve_mmproj_path(mmproj)
        model_dir = os.path.dirname(model_file)
        auto_mmproj = self._find_mmproj_in_dir(model_dir)
        mmproj_file = manual_mmproj or auto_mmproj

        if not mmproj_file:
            raise RuntimeError(
                f"未找到mmproj音频编码模型: {model_dir}\n"
                "【解决方法】将 mmproj-*.gguf 放入模型同目录，或在节点的 mmproj 下拉中手动指定"
            )

        try:
            print(f"【ASR加载器】正在加载llama.cpp ASR模型: {asr_model} (路径: {model_file})")

            config = {
                "asr_model": asr_model,
                "asr_model_path": model_file,
                "n_gpu_layers": n_gpu_layers,
                "language": language,
                "task": task,
                "enable_timestamps": enable_timestamps,
                "n_ctx": n_ctx,
                "mmproj": mmproj_file,
            }

            if self.loaded_model is not None and self.current_config == config:
                print("【ASR加载器】使用缓存的ASR模型")
                return (self.loaded_model,)

            if not os.path.isfile(model_file):
                raise RuntimeError(f"ASR模型文件不存在: {model_file}")

            print(f"【ASR加载器】模型文件: {model_file}")
            print(f"【ASR加载器】mmproj: {mmproj_file}")
            print(f"【ASR加载器】语言: {language}, 任务: {task}, 上下文: {n_ctx}")

            if self.loaded_model is not None:
                print("【ASR加载器】卸载旧模型")
                try:
                    self.loaded_model.release()
                except Exception:
                    pass
                self.loaded_model = None

            asr_wrapper = LlamaCppASRWrapper(
                model_file=model_file,
                mmproj_path=mmproj_file,
                config=config,
                n_ctx=n_ctx
            )

            if asr_wrapper.llm is None:
                print("【ASR加载器错误】模型包装器已创建，但内部模型为None，加载失败")
                return (None,)

            self.loaded_model = asr_wrapper
            self.current_config = config

            global asr_model_cache
            asr_model_cache[asr_model] = asr_wrapper
            print(f"【ASR加载器】llama.cpp ASR模型加载成功，已添加到缓存")
            return (asr_wrapper,)

        except Exception as e:
            print(f"【ASR加载器错误】加载ASR模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return (None,)


class LlamaCppASRWrapper:
    """
    基于 llama-cpp-python 0.4.0 的 ASR 模型包装器
    通过 Qwen3ASRChatHandler + Llama.create_chat_completion 走 MTMD 原生音频推理
    """

    def __init__(self, model_file, mmproj_path=None, config=None, n_ctx=8192):
        self.model_file = model_file
        self.mmproj_path = mmproj_path
        self.config = config or {}
        self.language = self.config.get("language", "auto")
        self.task = self.config.get("task", "transcribe")
        self.enable_timestamps = self.config.get("enable_timestamps", False)
        self.n_ctx = n_ctx
        # 强制 GPU/CUDA 推理：-1=全部层卸载到 GPU
        self.n_gpu_layers = self.config.get("n_gpu_layers", -1)
        self._audio_cache = {}
        self._cache_size_limit = 100
        self.llm = None
        self.chat_handler = None

        print(f"【ASR包装器】使用 llama-cpp-python MTMD 音频推理（Qwen3ASRChatHandler）")
        print(f"【ASR包装器】模型: {model_file}")
        print(f"【ASR包装器】mmproj: {mmproj_path or '未找到'}")

        self._load_model()

    def _build_chat_handler(self):
        """优先使用 Qwen3ASRChatHandler，不可用时回退到通用 MTMDChatHandler"""
        if not self.mmproj_path:
            return None
        try:
            from llama_cpp.llama_multimodal import Qwen3ASRChatHandler
            handler = Qwen3ASRChatHandler(
                mmproj_path=self.mmproj_path,
                use_gpu=True,
                verbose=False,
            )
            print(f"【ASR包装器】Qwen3ASRChatHandler 创建成功（mmproj: {self.mmproj_path}）")
            return handler
        except Exception as e:
            print(f"【ASR包装器】Qwen3ASRChatHandler 创建失败，回退 MTMDChatHandler: {e}")
            try:
                from llama_cpp.llama_multimodal import MTMDChatHandler
                handler = MTMDChatHandler(
                    mmproj_path=self.mmproj_path,
                    use_gpu=True,
                    verbose=False,
                )
                print(f"【ASR包装器】MTMDChatHandler 回退创建成功")
                return handler
            except Exception as e2:
                print(f"【ASR包装器】MTMDChatHandler 也创建失败: {e2}")
                return None

    def _load_model(self):
        try:
            from llama_cpp import Llama
            import llama_cpp

            print(f"【ASR包装器】llama-cpp-python 版本: {llama_cpp.__version__}")
            print(f"【ASR包装器】MTMD 支持: {_has_mtmd}")

            if not os.path.isfile(self.model_file):
                raise FileNotFoundError(f"ASR主模型文件不存在: {self.model_file}")

            self.chat_handler = self._build_chat_handler()

            llama_kwargs = {
                "model_path": self.model_file,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": False,
            }
            if self.chat_handler is not None:
                llama_kwargs["chat_handler"] = self.chat_handler

            try:
                self.llm = Llama(**llama_kwargs)
                print(f"【ASR包装器】llama.cpp ASR模型加载成功（GPU/CUDA，n_gpu_layers={self.n_gpu_layers}）")
            except Exception as e:
                # GPU 加载失败时降级为纯 CPU，保证功能可用
                print(f"【ASR包装器】GPU模式加载失败，尝试CPU基础模式: {e}")
                llama_kwargs["n_gpu_layers"] = 0
                self.llm = Llama(**llama_kwargs)
                print(f"【ASR包装器】llama.cpp ASR模型加载成功（CPU基础模式）")

        except ImportError as e:
            print(f"【ASR包装器错误】llama-cpp-python 导入失败: {e}")
        except Exception as e:
            print(f"【ASR包装器错误】模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def transcribe(self, audio_input, language=None, task=None):
        """
        执行语音识别（llama.cpp MTMD 原生音频推理）

        Args:
            audio_input: 音频输入，支持：
                - dict: {"waveform": tensor, "sample_rate": int}
                - torch.Tensor / numpy.ndarray: 音频波形
                - str: 音频文件路径（wav/flac/mp3）
            language: 语言代码或英文名（覆盖节点配置），auto=自动检测
            task: transcribe / translate（覆盖节点配置）

        Returns:
            dict: {"text": "识别文本", "language": "语言", "segments": []}
        """
        if audio_input is None:
            return {"text": "", "language": "", "segments": []}

        if self.llm is None:
            print("【ASR错误】模型未加载")
            return {"text": "模型未加载", "language": "", "segments": []}

        language = _normalize_language(language if language is not None else self.language)
        task = task or self.task

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            cache_key = self._generate_cache_key(audio_input, language, task)
            if cache_key and cache_key in self._audio_cache:
                return self._audio_cache[cache_key]

            result = self._transcribe_llm(audio_input, language, task)

            if cache_key:
                self._cache_result(cache_key, result)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            result["performance"] = {
                "inference_time": end_time - start_time,
                "memory_used": end_memory - start_memory,
            }

            return result

        except Exception as e:
            print(f"【ASR识别错误】语音识别失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"text": f"识别失败: {str(e)}", "language": language or "", "segments": []}

    def _generate_cache_key(self, audio_input, language, task):
        try:
            waveform, sample_rate = self._extract_audio_data(audio_input)
            if waveform is None:
                return None
            audio_hash = hashlib.md5(waveform.tobytes()).hexdigest()
            return f"{audio_hash}_{language}_{task}"
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

    def _build_system_prompt(self, language, task):
        """根据语言/任务构造系统指令（Qwen3ASRChatHandler 仅保留 system 文本，用户文本会被模板丢弃）"""
        if task == "translate":
            return ("Translate the speech audio into English. "
                    "Ignore background noise and output only the translated English text.")
        if language and language != "auto":
            return (f"Transcribe the speech audio into {language}. "
                    "Ignore background noise and output only the transcribed text.")
        # auto：返回 None，使用 Qwen3ASRChatHandler 的默认系统提示（按原语言转录）
        return None

    @staticmethod
    def _parse_asr_output(raw_text, fallback_language=""):
        """
        解析 Qwen3-ASR 原生输出：language English<asr_text>识别文本
        无标记时按纯文本处理
        """
        if raw_text is None:
            return "", fallback_language or ""
        text = str(raw_text).strip()

        match = _ASR_OUTPUT_PATTERN.match(text)
        if match:
            detected_language = match.group("language").strip()
            content = match.group("text")
            # 去掉识别文本末尾残留的结束符（<|im_end|> / ）
            content = re.sub(r"(?:<\|im_end\|>|)+[\s]*$", "", content).strip()
            return content, detected_language or fallback_language

        # 容错：仅去掉残留的特殊标记
        text = text.replace("<|im_end|>", "").replace("", "").strip()
        return text, fallback_language or ""

    def _transcribe_llm(self, audio_input, language, task):
        """
        使用 llama-cpp-python 0.4.0 标准接口执行语音识别：
        Qwen3ASRChatHandler 处理 input_audio（wav/mp3），create_chat_completion 生成文本
        """
        print(f"【llama.cpp ASR】开始识别，语言: {language}，任务: {task}")

        wav_bytes, sample_rate, audio_duration = self._build_wav_bytes(audio_input)
        if wav_bytes is None:
            return {"text": "音频数据提取失败", "language": language or "", "segments": []}

        print(f"【llama.cpp ASR】音频长度: {audio_duration:.2f}秒, 输入采样率: {sample_rate}, WAV字节: {len(wav_bytes)}")

        if not self.mmproj_path or self.chat_handler is None:
            return {
                "text": "ASR模型缺少mmproj音频编码文件，无法识别音频",
                "language": language or "",
                "segments": [],
            }

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            }
        ]

        messages = []
        system_prompt = self._build_system_prompt(language, task)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        completion = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )

        raw_text = ""
        try:
            raw_text = completion["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            print(f"【llama.cpp ASR错误】返回结构解析失败: {e}，原始返回: {completion}")

        text, detected_language = self._parse_asr_output(raw_text, language if language != "auto" else "")
        print(f"【llama.cpp ASR】识别成功，文本长度: {len(text)}字符，检测语言: {detected_language or '未知'}")

        segments = []
        if self.enable_timestamps:
            print(f"【llama.cpp ASR提示】Qwen3ASRChatHandler 暂不返回句级时间戳，segments 留空")

        return {
            "text": text,
            "language": detected_language,
            "segments": segments,
        }

    def _build_wav_bytes(self, audio_input, target_sr=16000):
        """
        将任意支持的音频输入转换为 16kHz 单声道 PCM16 WAV bytes（Qwen3-ASR 要求 16kHz）

        Returns:
            tuple: (wav_bytes, original_sample_rate, duration_seconds)
        """
        waveform, sample_rate = self._extract_audio_data(audio_input)
        if waveform is None or len(waveform) == 0:
            return None, sample_rate or target_sr, 0.0

        wav_tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).clone().clamp(-1.0, 1.0)

        # 重采样到 16kHz
        if sample_rate != target_sr:
            try:
                import torchaudio
                wav_tensor = torchaudio.functional.resample(wav_tensor, sample_rate, target_sr)
                print(f"【llama.cpp ASR】音频重采样: {sample_rate}Hz -> {target_sr}Hz")
            except Exception as e:
                print(f"【llama.cpp ASR警告】torchaudio重采样失败({e})，按原始采样率{sample_rate}Hz送入")
                target_sr = sample_rate

        pcm16 = (wav_tensor.numpy() * 32767.0).astype("<i2").tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            wf.writeframes(pcm16)

        duration = len(pcm16) / 2 / target_sr
        return buf.getvalue(), sample_rate, duration

    def _extract_audio_data(self, audio_input):
        """
        从各种音频输入格式中提取单声道浮点波形和采样率

        Returns:
            tuple: (waveform_numpy_float32_1d, sample_rate)
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
                    waveform, sample_rate = sf.read(audio_input, dtype="float32")
                except Exception:
                    import torchaudio
                    loaded, sr = torchaudio.load(audio_input)
                    waveform = loaded.numpy().mean(axis=0)
                    sample_rate = sr
            else:
                print(f"【ASR错误】不支持的音频输入类型: {type(audio_input)}")
                return None, 16000

            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()

            waveform = np.asarray(waveform, dtype=np.float32)

            # ComfyUI AUDIO 形状通常为 [1, channels, samples]，统一压成一维单声道
            while waveform.ndim > 1:
                if waveform.shape[0] == 1:
                    waveform = waveform.squeeze(0)
                else:
                    waveform = waveform.mean(axis=0)
            waveform = np.asarray(waveform, dtype=np.float32)

            if not np.isfinite(waveform).all():
                waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

            return waveform, int(sample_rate)

        except Exception as e:
            print(f"【ASR错误】音频数据提取失败: {str(e)}")
            return None, 16000

    def release(self):
        """释放模型资源"""
        try:
            if self.chat_handler is not None:
                try:
                    self.chat_handler.close()
                except Exception:
                    pass
                self.chat_handler = None

            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
                self.llm = None

            self._audio_cache = {}

            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("【ASR包装器】已清理GPU缓存")

            print("【ASR包装器】ASR模型已卸载")

        except Exception as e:
            print(f"【ASR清理错误】{str(e)}")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


NODE_CLASS_MAPPINGS = {
    "omni_llm_asr_loader": omni_llm_asr_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_asr_loader": "Omni LLM ASR Loader (llama-cpp)",
}
