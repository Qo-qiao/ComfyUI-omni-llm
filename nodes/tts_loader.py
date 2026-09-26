# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm TTS Model Loader Node
基于 llama-cpp-python MTMD 音频生成 (Qwen3-TTS / Pocket TTS)

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import HARDWARE_INFO, folder_paths, LLAMA_CPP_STORAGE, _has_mtmd, _llama_cpp

tts_model_cache = {}


def _scan_aux_files():
    """扫描所有 LLM 目录中的 mmproj/tokenizer/codec 辅助模型（gguf），返回相对路径列表"""
    if "LLM" not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

    aux_list = ["None"]
    aux_set = set()

    for folder in folder_paths.get_folder_paths("LLM"):
        try:
            for root, dirs, files in os.walk(folder):
                rel_path = os.path.relpath(root, folder)
                for f in files:
                    if not f.lower().endswith(".gguf"):
                        continue
                    lower = f.lower()
                    if not (lower.startswith("mmproj") or "tokenizer" in lower or "codec" in lower):
                        continue
                    file_abs_path = os.path.normpath(os.path.join(root, f))
                    if file_abs_path in aux_set:
                        continue
                    aux_set.add(file_abs_path)
                    if rel_path == '.':
                        aux_list.append(f)
                    else:
                        aux_list.append(f"{rel_path.replace(os.sep, '/')}/{f}")
        except Exception as e:
            print(f"【TTS模型检测】扫描辅助模型文件夹 {folder} 失败: {e}")

    return aux_list


class LlamaCppTTSWrapper:
    """基于 llama-cpp-python MTMD 的 TTS 模型包装器"""

    def __init__(self, model_path, aux_path, config):
        self.model_path = model_path
        self.aux_path = aux_path
        self.config = config
        self.n_gpu_layers = config.get("n_gpu_layers", -1)
        self.n_ctx = config.get("n_ctx", 4096)
        self.llama = None
        self.generator = None
        self.language = config.get("language", "auto")
        self.ref_audio_path = config.get("ref_audio_path", "")

        print(f"【TTS包装器】模型: {model_path}")
        print(f"【TTS包装器】辅助模型(mmproj/tokenizer): {aux_path or '未找到'}")
        if not aux_path:
            raise RuntimeError("未找到mmproj/tokenizer/codec文件，无法创建MTMDAudioGenerator")
        self._load_model()

    def _load_model(self):
        try:
            import llama_cpp
            from llama_cpp import Llama, LLAMA_POOLING_TYPE_NONE
            from llama_cpp.llama_multimodal import MTMDAudioGenerator

            print(f"【TTS包装器】llama-cpp-python 版本: {llama_cpp.__version__}")

            self.llama = Llama(
                model_path=self.model_path,
                embeddings=True,
                pooling_type=LLAMA_POOLING_TYPE_NONE,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            print("【TTS包装器】Llama 模型加载成功")

            self.generator = MTMDAudioGenerator(
                mmproj_path=self.aux_path,
                use_gpu=True,
                flash_attn=None,
            )
            print("【TTS包装器】MTMDAudioGenerator 创建成功")

        except Exception as e:
            print(f"【TTS包装器错误】模型加载失败: {e}")
            import traceback
            traceback.print_exc()

    def synthesize(self, text, language=None, speaker_reference=None,
                   seed=None, temperature=0.8, top_k=40, top_p=0.95,
                   min_p=0.05, repeat_penalty=1.05, max_frames=512):
        if self.llama is None or self.generator is None:
            raise RuntimeError("TTS 模型未加载")

        if language is None:
            language = getattr(self, "language", None)
        if language == "auto":
            language = None

        kwargs = {
            "llama": self.llama,
            "text": text,
            "response_format": "wav",
            "max_frames": max_frames,
        }

        if language:
            kwargs["language"] = language
        if speaker_reference is not None:
            kwargs["speaker_reference"] = speaker_reference
        if seed is not None:
            kwargs["seed"] = seed
        else:
            kwargs["seed"] = None
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_k is not None:
            kwargs["top_k"] = top_k
        if top_p is not None:
            kwargs["top_p"] = top_p
        if min_p is not None:
            kwargs["min_p"] = min_p
        if repeat_penalty is not None:
            kwargs["repeat_penalty"] = repeat_penalty

        return self.generator.create_speech(**kwargs)

    def release(self):
        if self.generator is not None:
            try:
                self.generator.close()
            except Exception:
                pass
            self.generator = None
        if self.llama is not None:
            try:
                self.llama.close()
            except Exception:
                pass
            self.llama = None


class omni_llm_tts_loader:
    """TTS 模型加载器（llama-cpp MTMD），选择目录内的具体模型文件"""

    @classmethod
    def INPUT_TYPES(s):
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

        llm_folders = folder_paths.get_folder_paths("LLM")

        tts_list = ["None"]
        model_set = set()
        model_set.add("None")

        for folder in llm_folders:
            try:
                root_tag = os.path.basename(folder) if len(llm_folders) > 1 else ""

                for root, dirs, files in os.walk(folder):
                    dir_name = os.path.basename(root).lower()
                    folder_path_lower = root.lower()

                    tts_keywords = ["tts", "voicedesign", "customvoice", "custom_voice", "voice"]
                    has_tts_keyword = any(kw in dir_name or kw in folder_path_lower for kw in tts_keywords)

                    # 对于Qwen/Omni系列，必须有明确的TTS关键词
                    if "qwen" in dir_name or "qwen" in folder_path_lower or "omni" in dir_name or "omni" in folder_path_lower:
                        if not any(kw in dir_name or kw in folder_path_lower for kw in tts_keywords):
                            continue

                    if not has_tts_keyword:
                        continue

                    for f in files:
                        if not f.lower().endswith(".gguf"):
                            continue
                        # 跳过辅助文件（mmproj/tokenizer/codec）
                        lower = f.lower()
                        if lower.startswith("mmproj") or "tokenizer" in lower or "codec" in lower:
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
                        tts_list.append(file_entry)
                        print(f"【TTS模型检测】检测到TTS模型文件: {file_entry}")
            except Exception as e:
                print(f"【TTS模型检测】扫描文件夹 {folder} 失败: {e}")

        if len(tts_list) == 1:
            tts_list = ["None", "请将TTS GGUF模型放入models/LLM/tts文件夹"]

        # TTS 强制 GPU/CUDA 推理：默认全部层卸载到 GPU
        default_n_gpu_layers = -1

        languages = [
            "auto", "zh", "en", "ja", "ko", "de", "fr", "ru", "es", "pt", "it",
        ]

        aux_list = _scan_aux_files()

        return {
            "required": {
                "tts_model": (tts_list, {"tooltip": "选择TTS GGUF模型文件（tts目录内）"}),
                "mmproj": (aux_list, {"default": "None", "tooltip": "手动指定辅助模型(mmproj/tokenizer)；None=自动从模型同目录检测"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载（强制GPU/CUDA推理）"}),
                "language": (languages, {"default": "auto", "tooltip": "合成语言（Qwen3-TTS支持，Pocket TTS自动忽略）"}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 128, "tooltip": "上下文长度"}),
            },
            "optional": {
                "ref_audio_path": ("STRING", {"default": "", "tooltip": "音色克隆参考音路径（wav格式，3~60秒最佳，单声道48000Hz）"}),
            }
        }

    RETURN_TYPES = ("TTSMODEL",)
    RETURN_NAMES = ("tts_model",)
    FUNCTION = "load_tts_model"
    CATEGORY = "omni-llm"

    @classmethod
    def _resolve_model_path(s, tts_model):
        """解析选择的模型文件相对路径 -> 绝对路径"""
        key = tts_model.replace("\\", "/").rstrip('/')
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
    def _resolve_aux_path(s, mmproj):
        """解析手动选择的辅助模型文件路径"""
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
    def _find_aux_in_dir(s, model_dir):
        """在模型同目录查找辅助文件（mmproj/tokenizer/codec）"""
        try:
            for f in sorted(os.listdir(model_dir)):
                if not f.lower().endswith(".gguf"):
                    continue
                lower = f.lower()
                if lower.startswith("mmproj") or "tokenizer" in lower or "codec" in lower:
                    return os.path.join(model_dir, f)
        except Exception:
            pass
        return None

    def __init__(self):
        self.loaded_model = None
        self.current_config = None

    @classmethod
    def IS_CHANGED(s, tts_model, n_gpu_layers, language, n_ctx=4096, mmproj="None",
                   ref_audio_path=""):
        resolved_path = s._resolve_model_path(tts_model)
        resolved_aux = s._resolve_aux_path(mmproj)
        config = {
            "tts_model": tts_model,
            "tts_model_path": resolved_path or "",
            "n_gpu_layers": n_gpu_layers,
            "language": language,
            "n_ctx": n_ctx,
            "mmproj": resolved_aux or "",
            "ref_audio_path": ref_audio_path,
        }
        return json.dumps(config, sort_keys=True, ensure_ascii=False)

    def load_tts_model(self, tts_model, n_gpu_layers, language, n_ctx=4096, mmproj="None",
                        ref_audio_path=""):
        if tts_model == "None" or "请将" in tts_model:
            print("【TTS加载器】未选择TTS模型")
            return (None,)

        model_file = self._resolve_model_path(tts_model)
        if not model_file:
            raise RuntimeError(f"无法解析TTS模型路径: {tts_model}")

        try:
            print(f"【TTS加载器】正在加载模型: {tts_model} (路径: {model_file})")

            manual_aux_path = self._resolve_aux_path(mmproj)
            if mmproj != "None" and not manual_aux_path:
                raise RuntimeError(f"无法解析手动指定的辅助模型: {mmproj}")

            config = {
                "tts_model": tts_model,
                "tts_model_path": model_file,
                "n_gpu_layers": n_gpu_layers,
                "language": language,
                "n_ctx": n_ctx,
                "mmproj": manual_aux_path or "",
                "ref_audio_path": ref_audio_path,
            }

            if self.loaded_model is not None and self.current_config == config:
                print("【TTS加载器】使用缓存的TTS模型")
                return (self.loaded_model,)

            # 在模型同目录查找辅助文件
            model_dir = os.path.dirname(model_file)
            auto_aux = self._find_aux_in_dir(model_dir)
            aux_file = manual_aux_path or auto_aux
            if manual_aux_path:
                print(f"【TTS加载器】使用手动指定的辅助模型: {manual_aux_path}")
            elif not aux_file:
                print(f"【TTS加载器警告】模型同目录未找到mmproj/tokenizer/codec文件: {model_dir}（TTS合成必须提供）")

            if self.loaded_model is not None:
                print("【TTS加载器】卸载旧模型")
                self.loaded_model.release()
                self.loaded_model = None

            wrapper = LlamaCppTTSWrapper(
                model_path=model_file,
                aux_path=aux_file,
                config=config,
            )

            if wrapper.llama is None:
                raise RuntimeError("TTS模型加载失败")

            wrapper.language = language
            wrapper.ref_audio_path = ref_audio_path
            self.loaded_model = wrapper
            self.current_config = config

            global tts_model_cache
            tts_model_cache[tts_model] = wrapper
            print("【TTS加载器】模型加载完成")
            return (wrapper,)

        except Exception as e:
            print(f"【TTS加载器错误】{e}")
            import traceback
            traceback.print_exc()
            raise
