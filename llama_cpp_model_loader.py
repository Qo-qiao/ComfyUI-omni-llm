# -*- coding: utf-8 -*-
"""
Llama-cpp Model Loader Node
"""
import os
import json
from .common import HARDWARE_INFO, chat_handlers, folder_paths, LLAMA_CPP_STORAGE

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = folder_paths.get_filename_list("LLM")
        # 筛选.gguf和.safetensors格式的主模型
        model_list = [
            f for f in all_llms 
            if "mmproj" not in f.lower() 
            and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors"]
        ]
        # 筛选.gguf和.safetensors格式的MMProj模型
        mmproj_list = ["None"] + [
            f for f in all_llms 
            if "mmproj" in f.lower() 
            and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors"]
        ]
        
        # 根据硬件性能推荐默认参数
        default_n_ctx = 4096 if HARDWARE_INFO["is_low_perf"] else 8192
        default_n_gpu_layers = -1 if HARDWARE_INFO["is_high_perf"] else 0
        
        return {
            "required": {
                "model": (model_list, {"tooltip": "选择要加载的LLM模型文件"}),
                "enable_mmproj": ("BOOLEAN", {"default": False, "label": "启用MMProj模型（多模态）", "tooltip": "启用后可处理图片输入"}),
                "mmproj": (mmproj_list, {"default": "None", "tooltip": "选择对应的视觉编码模型文件"}),
                "chat_handler": (chat_handlers, {"default": "Qwen3-VL", "tooltip": "选择适合模型的对话格式处理器"}),
                "n_ctx": ("INT", {"default": default_n_ctx, "min": 1024, "max": 327680, "step": 128, "tooltip": "上下文长度，影响可处理的文本长度"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载"}),
                "vram_limit": ("INT", {"default": -1, "min": -1, "max": 24, "step": 1, "tooltip": "显存限制（GB），-1=无限制"}),
                "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最小编码token数"}),
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最大编码token数"}),
            }
        }
    
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    @classmethod
    def IS_CHANGED(s, model, enable_mmproj, mmproj, chat_handler, n_ctx, n_gpu_layers, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        custom_config = {
            "model": model, "enable_mmproj": enable_mmproj, "mmproj": mmproj,
            "chat_handler": chat_handler, "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit, "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens
        }
        return json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
    
    def loadmodel(self, model, enable_mmproj, mmproj, chat_handler, n_ctx, n_gpu_layers, vram_limit, image_min_tokens, image_max_tokens):
        custom_config = {
            "model": model, "enable_mmproj": enable_mmproj, "mmproj": mmproj,
            "chat_handler": chat_handler, "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit, "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (LLAMA_CPP_STORAGE,)
