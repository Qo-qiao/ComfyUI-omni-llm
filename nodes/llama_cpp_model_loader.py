# @亲卿于情 修改版本
# -*- coding: utf-8 -*-
"""
Llama-cpp Model Loader Node
"""
import os
import json
from ..common import HARDWARE_INFO, chat_handlers, folder_paths, LLAMA_CPP_STORAGE

class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        # 动态导入chat_handlers，确保使用最新的列表
        from ..common import chat_handlers
        
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
        perf_level = HARDWARE_INFO.get("perf_level", "low")
        
        # 统一设置 n_ctx 为 8192，确保预设模板的 1000 字内容生成不受影响
        default_n_ctx = 8192
        
        # 默认使用 GPU 模式
        default_device_mode = "GPU"
        
        if perf_level == "high":  # 24GB+
            default_n_gpu_layers = -1  # 全部加载
            default_vram_limit = 24  # 24GB
        elif perf_level == "mid_high":  # 16GB
            default_n_gpu_layers = -1  # 全部加载
            default_vram_limit = 16  # 16GB
        elif perf_level == "mid":  # 12GB
            default_n_gpu_layers = -1  # 全部加载
            default_vram_limit = 12  # 12GB
        elif perf_level == "mid_low":  # 8GB
            default_n_gpu_layers = 30  # 部分加载
            default_vram_limit = 8  # 8GB
        else:  # <8GB
            default_n_gpu_layers = 0  # 纯CPU
            default_vram_limit = -1  # 无限制
            default_device_mode = "CPU"  # 低性能硬件默认使用 CPU
        
        return {
            "required": {
                "model": (model_list, {"tooltip": "选择要加载的LLM模型文件"}),
                "auto_config": ("BOOLEAN", {"default": True, "label": "自动配置（多模态+对话格式）", "tooltip": "启用后自动选择对话格式处理器并启用多模态功能，提示词生成时建议关闭"}),
                "mmproj": (mmproj_list, {"default": "None", "tooltip": "选择对应的视觉编码模型文件"}),
                "device_mode": (["GPU", "CPU"], {"default": default_device_mode, "tooltip": "选择运行模式：GPU=使用显卡加速，CPU=纯CPU运行"}),
                "n_ctx": ("INT", {"default": default_n_ctx, "min": 1024, "max": 327680, "step": 128, "tooltip": "上下文长度，影响可处理的文本长度"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载（GPU模式有效）"}),
                "vram_limit": ("INT", {"default": default_vram_limit, "min": -1, "max": 24, "step": 1, "tooltip": "显存限制（GB），-1=无限制（GPU模式有效）"}),
                "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最小编码token数"}),
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最大编码token数"}),
            }
        }
    
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vlm"
    
    @classmethod
    def IS_CHANGED(s, model, auto_config, mmproj, device_mode, n_ctx, n_gpu_layers, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        # 根据模型名称自动推断对话格式处理器
        def get_auto_chat_handler(model_name):
            # 模型名称映射到对话格式处理器
            model_handler_map = {
                "Qwen2.5-VL": "Qwen2.5-VL",
                "Qwen3-VL-Instruct": "Qwen3-VL-Instruct",
                "olmOCR-2-7B-1025": "olmOCR-2",
                "llava-1.6-mistral-7b": "LLaVA-1.6",
                "nanoLLaVA-1.5": "nanoLLaVA",
                "MiniCPM-V-4.5": "MiniCPM-V-4.5",
                "MiniCPM-Llama3-V 2.5": "MiniCPM-Llama3-V 2.5",
                "GLM-4.6V": "GLM-4.6V",
                "llama-joycaption": "LLaVA-1.6",
                "Moondream2": "Moondream2",
                "gemma-3-12b": "Gemma-3",
                "Youtu-VL-4B-Instruct": "Youtu-VL-4B-Instruct",
                "EraX-VL-7B-V1.5": "EraX-VL-7B-V1.5",
                "MiMo-VL-7B-RL": "MiMo-VL-7B-RL",
                "DreamOmni2": "DreamOmni2",
                "Phi-3.5-vision-instruct": "Phi-3.5-vision-instruct",
                "Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
                "LLaMA-3.1-Vision": "LLaMA-3.1-Vision",
                "Yi-VL-6B": "Yi-VL-6B",
                "LightOnOCR-2-1B": "olmOCR-2"
            }
            
            # 尝试直接匹配
            if model_name in model_handler_map:
                return model_handler_map[model_name]
            
            # 尝试部分匹配
            model_name_lower = model_name.lower()
            for key, handler in model_handler_map.items():
                if key.lower() in model_name_lower:
                    return handler
            
            # 默认使用LLaVA-1.6
            return "LLaVA-1.6"
        
        # 确定使用的对话格式处理器和是否启用多模态
        if auto_config:
            chat_handler = get_auto_chat_handler(model)
            enable_mmproj = True
        else:
            chat_handler = "None"
            enable_mmproj = False
        
        custom_config = {
            "model": model, "enable_mmproj": enable_mmproj, "mmproj": mmproj,
            "chat_handler": chat_handler, "device_mode": device_mode, "n_ctx": n_ctx, 
            "n_gpu_layers": n_gpu_layers, "vram_limit": vram_limit, 
            "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens
        }
        return json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
    
    def loadmodel(self, model, auto_config, mmproj, device_mode, n_ctx, n_gpu_layers, vram_limit, image_min_tokens, image_max_tokens):
        # 根据模型名称自动推断对话格式处理器
        def get_auto_chat_handler(model_name):
            # 模型名称映射到对话格式处理器
            model_handler_map = {
                "Qwen2.5-VL": "Qwen2.5-VL",
                "Qwen3-VL-Instruct": "Qwen3-VL-Instruct",
                "olmOCR-2-7B-1025": "olmOCR-2",
                "llava-1.6-mistral-7b": "LLaVA-1.6",
                "nanoLLaVA-1.5": "nanoLLaVA",
                "MiniCPM-V-4.5": "MiniCPM-V-4.5",
                "MiniCPM-Llama3-V 2.5": "MiniCPM-Llama3-V 2.5",
                "GLM-4.6V": "GLM-4.6V",
                "llama-joycaption": "LLaVA-1.6",
                "Moondream2": "Moondream2",
                "gemma-3-12b": "Gemma-3",
                "Youtu-VL-4B-Instruct": "Youtu-VL-4B-Instruct",
                "EraX-VL-7B-V1.5": "EraX-VL-7B-V1.5",
                "MiMo-VL-7B-RL": "MiMo-VL-7B-RL",
                "DreamOmni2": "DreamOmni2",
                "Phi-3.5-vision-instruct": "Phi-3.5-vision-instruct",
                "Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
                "LLaMA-3.1-Vision": "LLaMA-3.1-Vision",
                "Yi-VL-6B": "Yi-VL-6B",
                "LightOnOCR-2-1B": "olmOCR-2"
            }
            
            # 尝试直接匹配
            if model_name in model_handler_map:
                return model_handler_map[model_name]
            
            # 尝试部分匹配
            model_name_lower = model_name.lower()
            for key, handler in model_handler_map.items():
                if key.lower() in model_name_lower:
                    return handler
            
            # 默认使用LLaVA-1.6
            return "LLaVA-1.6"
        
        # 确定使用的对话格式处理器和是否启用多模态
        if auto_config:
            chat_handler = get_auto_chat_handler(model)
            enable_mmproj = True
            print(f"【自动配置】根据模型 {model} 选择对话格式处理器: {chat_handler}，并启用多模态功能")
        else:
            chat_handler = "None"
            enable_mmproj = False
            print(f"【提示】自动配置已禁用，对话格式处理器和多模态功能均已关闭")
        
        custom_config = {
            "model": model, "enable_mmproj": enable_mmproj, "mmproj": mmproj,
            "chat_handler": chat_handler, "device_mode": device_mode, "n_ctx": n_ctx, 
            "n_gpu_layers": n_gpu_layers, "vram_limit": vram_limit, 
            "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (LLAMA_CPP_STORAGE,)
