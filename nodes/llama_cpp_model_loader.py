# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Model Loader Node

模型加载节点，支持加载各种GGUF格式的LLM模型，包括多模态和Omni模型

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import json
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import HARDWARE_INFO, chat_handlers, folder_paths, LLAMA_CPP_STORAGE

# 导入加速模块
from common import (
    apply_acceleration_hooks,
)

# 应用加速钩子
apply_acceleration_hooks(LLAMA_CPP_STORAGE)

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        # 动态导入chat_handlers，确保使用最新的列表
        from common import chat_handlers

        # 检查并添加LLM文件夹路径
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

        all_llms = []
        for folder in folder_paths.get_folder_paths("LLM"):
            for root, dirs, files in os.walk(folder):
                rel_path = os.path.relpath(root, folder)
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext == ".gguf":
                        if rel_path == '.':
                            all_llms.append(f)
                        else:
                            all_llms.append(f"{rel_path.replace(os.sep, '/')}/{f}")

     
        # 筛选.gguf格式的主模型，排除mmproj和语音相关模型
        model_list = ["None"]
        model_set = set()  # 使用绝对路径作为去重依据
        
        for f in all_llms:
            ext = os.path.splitext(f)[1].lower()
            if ext != ".gguf":
                continue
            if "mmproj" in f.lower():
                continue
            
            # 排除TTS/ASR等语音模型
            if any(keyword in f.lower() for keyword in ["tts", "asr", "speech", "voice", "audio"]):
                continue
            
            # 获取文件的绝对路径（去重依据）
            file_abs_path = None
            for folder in folder_paths.get_folder_paths("LLM"):
                candidate = os.path.join(folder, f)
                if os.path.exists(candidate):
                    file_abs_path = os.path.normpath(candidate)
                    break
            
            if file_abs_path and file_abs_path not in model_set:
                model_set.add(file_abs_path)
                model_list.append(f)
        
        # 筛选.gguf格式的MMProj模型
        # 支持mmproj和vision命名的视觉编码模型
        mmproj_list = ["None"]
        mmproj_set = set()  # 使用绝对路径作为去重依据
        
        for f in all_llms:
            ext = os.path.splitext(f)[1].lower()
            if ext != ".gguf":
                continue
            
            if "mmproj" in f.lower() or "vision" in f.lower():
                    # 获取文件的绝对路径（去重依据）
                    file_abs_path = None
                    for folder in folder_paths.get_folder_paths("LLM"):
                        candidate = os.path.join(folder, f)
                        if os.path.exists(candidate):
                            file_abs_path = os.path.normpath(candidate)
                            break
                    
                    if file_abs_path and file_abs_path not in mmproj_set:
                        mmproj_set.add(file_abs_path)
                        mmproj_list.append(f)
        


        # 根据硬件性能推荐默认参数
        perf_level = HARDWARE_INFO.get("perf_level", "low")
        
        # 统一设置 n_ctx 为 8192，确保预设模板的 1000 字内容生成不受影响
        default_n_ctx = 8192
        
        # 默认注意力类型
        default_attention_type = "Flash"  # 默认使用 Flash Attention 加速
        
        if perf_level == "high":  # 24GB+
            default_n_gpu_layers = -1  # 全部加载
            default_vram_limit = 24  # 24GB
        elif perf_level == "mid_high":  # 16GB
            default_n_gpu_layers = -1  # 全部加载
            default_vram_limit = 16  # 16GB
        elif perf_level == "mid":  # 12GB
            default_n_gpu_layers = -1  # 全部加载
            default_vram_limit = 12  # 12GB
        elif perf_level == "mid_low":  # 4-8GB
            default_n_gpu_layers = 20  # 部分加载
            default_vram_limit = 6  # 6GB
        else:  # <4GB
            default_n_gpu_layers = 10  # 部分加载
            default_vram_limit = -1  # 无限制
        
        return {
            "required": {
                "model": (model_list, {"tooltip": "选择要加载的LLM模型文件"}),
                "enable_mmproj": ("BOOLEAN", {"default": False, "tooltip": "启用多模态功能（需要选择mmproj模型）"}),
                "mmproj": (mmproj_list, {"default": "None", "tooltip": "选择对应的视觉编码模型文件"}),
                "enable_asr": ("BOOLEAN", {"default": False, "tooltip": "启用ASR语音识别功能（需配合ASR模型加载器使用）"}),
                "enable_tts": ("BOOLEAN", {"default": False, "tooltip": "启用TTS语音合成功能（需配合TTS模型加载器使用）"}),
                "n_ctx": ("INT", {"default": default_n_ctx, "min": 1024, "max": 327680, "step": 128, "tooltip": "上下文长度，影响可处理的文本长度"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载（GPU模式有效）"}),
                "vram_limit": ("INT", {"default": default_vram_limit, "min": -1, "max": 24, "step": 1, "tooltip": "显存限制（GB），-1=无限制（GPU模式有效）"}),
                "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最小编码token数"}),
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最大编码token数"}),
                "attention_type": (["Auto", "Standard", "Flash", "XFormers"], {"default": default_attention_type, "tooltip": "注意力类型：Auto=自动选择，Standard=标准，Flash=Flash Attention（NVIDIA GPU推荐），XFormers=XFormers（实验性）"}),
                "cache_prompt": ("BOOLEAN", {"default": False, "tooltip": "启用Prompt缓存，相同前缀的请求可复用KV Cache，提升批量推理速度"}),
            }
        }
    
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"

    @classmethod
    def _resolve_llm_model_path(s, model):
        key = model.rstrip('/')
        if key == "None":
            return None

        for folder in folder_paths.get_folder_paths("LLM"):
            candidate = os.path.join(folder, key)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)

        if "/" in key:
            root_name, inner = key.split("/", 1)
            for folder in folder_paths.get_folder_paths("LLM"):
                if os.path.basename(folder) == root_name:
                    candidate = os.path.join(folder, inner)
                    if os.path.exists(candidate):
                        return os.path.normpath(candidate)

        full = folder_paths.get_full_path("LLM", key)
        if full and os.path.exists(full):
            return os.path.normpath(full)

        return None
    
    @classmethod
    def IS_CHANGED(s, model, enable_mmproj, mmproj, enable_asr, enable_tts, n_ctx, n_gpu_layers, vram_limit, image_min_tokens, image_max_tokens, attention_type="Auto", cache_prompt=False):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        # 根据模型名称自动推断对话格式处理器（使用ChatHandlerManager）
        def get_auto_chat_handler(model_name):
            import common
            from common import chat_handler_manager, detect_model_chat_handler
            
            # 首先尝试使用ChatHandlerManager的智能匹配
            handler_name, handler_cls = chat_handler_manager.get_handler_for_model(model_name)
            if handler_name and handler_cls:
                info = chat_handler_manager.get_handler_info(handler_name)
                if info:
                    return info['display_name']
            
            # 回退到原有的检测函数
            detected = detect_model_chat_handler(model_name)
            if detected:
                return detected
            
            # 默认使用LLaVA-1.6
            return "LLaVA-1.6"
        
        # 自动选择对话格式处理器，多模态功能由用户控制
        chat_handler = get_auto_chat_handler(model)
        
        resolved_path = s._resolve_llm_model_path(model)
        custom_config = {
            "model": model,
            "model_path": resolved_path or "",
            "chat_handler": chat_handler,
            "enable_mmproj": enable_mmproj,
            "mmproj": mmproj,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "attention_type": attention_type,
            "cache_prompt": cache_prompt,
        }
        return json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
    
    def loadmodel(self, model, enable_mmproj, mmproj, enable_asr, enable_tts, n_ctx, n_gpu_layers, vram_limit, image_min_tokens, image_max_tokens, attention_type="Auto", cache_prompt=False):
        # 解析完整模型路径，避免同名冲突
        resolved_model_path = self._resolve_llm_model_path(model)
        if resolved_model_path:
            model = resolved_model_path

        # 处理"None"模型的情况
        if model == "None":
            print("【模型加载】未选择模型，仅启用ASR/TTS功能")
            # 清空当前模型状态
            LLAMA_CPP_STORAGE.clean()
            # 返回空的storage对象
            return (LLAMA_CPP_STORAGE,)

        model_dir_name = model
        
        # 根据硬件性能自动计算批处理和线程参数
        perf_level = HARDWARE_INFO.get("perf_level", "low")
        cpu_count = HARDWARE_INFO.get("cpu_count", 4)
        
        # 自动计算 n_batch (批处理大小)
        if perf_level == "high":
            n_batch = 4096
        elif perf_level == "mid_high":
            n_batch = 3072
        elif perf_level == "mid":
            n_batch = 2048
        elif perf_level == "mid_low":
            n_batch = 1536
        else:  # low
            n_batch = 1024
        
        # 自动计算 n_ubatch (物理批处理大小)
        n_ubatch = min(512, n_batch // 2)
        
        # 自动计算线程数
        if cpu_count <= 4:
            n_threads = max(2, cpu_count)
            n_threads_batch = cpu_count
        elif cpu_count <= 8:
            n_threads = cpu_count // 2
            n_threads_batch = cpu_count
        else:
            n_threads = min(8, cpu_count // 2)
            n_threads_batch = min(16, cpu_count)
        
        print(f"【自动参数】n_batch={n_batch}, n_ubatch={n_ubatch}, n_threads={n_threads}, n_threads_batch={n_threads_batch}")
        
        # 检查是否是Qwen3系列模型
        is_qwen3 = "qwen3" in model.lower() or "qwen35" in model.lower()
        is_qwen35 = "qwen35" in model.lower() or "qwen3.5" in model.lower()
        is_qwen3vl = "qwen3-vl" in model.lower() or "qwen3vl" in model.lower()
        

        

        # GPU模式下针对Qwen3系列模型的优化
        if is_qwen3:
            # Qwen3系列在GPU模式下需要合理的上下文长度（至少4096）
            if n_ctx < 4096:
                print(f"【GPU模式优化】Qwen3系列模型将上下文长度从 {n_ctx} 调整为 4096 以确保GPU推理成功")
                n_ctx = 4096
            
            # 针对Qwen3.5模型的特殊优化
            if is_qwen35:
                print(f"【GPU模式优化】Qwen3.5模型启用特殊GPU参数配置")
                # 确保使用正确的ChatHandler
                print(f"【提示】Qwen3.5模型将使用Qwen35ChatHandler")
            
            # Qwen3系列模型（包括Qwen3-VL和Qwen3.5）需要至少1024个image tokens
            if image_min_tokens < 1024:
                print(f"【GPU模式优化】Qwen3系列模型需要至少1024 image tokens，自动调整image_min_tokens从{image_min_tokens}到1024")
                image_min_tokens = 1024
            
            # 自动设置image_max_tokens（如果未设置或小于image_min_tokens）
            if image_max_tokens < image_min_tokens:
                image_max_tokens = image_min_tokens
                print(f"【GPU模式优化】自动设置image_max_tokens为{image_max_tokens}")
        
        # 根据模型名称自动推断对话格式处理器（使用ChatHandlerManager）
        def get_auto_chat_handler(model_name):
            import common
            from common import chat_handler_manager, detect_model_chat_handler
            
            # 首先尝试使用ChatHandlerManager的智能匹配
            handler_name, handler_cls = chat_handler_manager.get_handler_for_model(model_name)
            if handler_name and handler_cls:
                info = chat_handler_manager.get_handler_info(handler_name)
                if info:
                    return info['display_name']
            
            # 回退到原有的检测函数
            detected = detect_model_chat_handler(model_name)
            if detected:
                return detected
            
            # 默认使用LLaVA-1.6
            return "LLaVA-1.6"
        
        # 自动选择对话格式处理器，多模态功能由用户控制
        chat_handler = get_auto_chat_handler(model)
        
        mmproj_status = "已启用" if enable_mmproj else "已禁用"
        asr_status = "已启用" if enable_asr else "已禁用"
        tts_status = "已启用" if enable_tts else "已禁用"
        print(f"【自动配置】根据模型 {model} 选择对话格式处理器: {chat_handler}，多模态功能{mmproj_status}，ASR功能{asr_status}，TTS功能{tts_status}")

        custom_config = {
            "model": model, "enable_mmproj": enable_mmproj, "mmproj": mmproj,
            "enable_asr": enable_asr, "enable_tts": enable_tts,
            "chat_handler": chat_handler, "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers, "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens,
            "n_batch": n_batch, "n_ubatch": n_ubatch, "n_threads": n_threads,
            "n_threads_batch": n_threads_batch, "attention_type": attention_type,
            "cache_prompt": cache_prompt
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (LLAMA_CPP_STORAGE,)
