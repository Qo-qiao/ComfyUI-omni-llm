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
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最大编码token数"}),
                "attention_type": (["Auto", "Standard", "Flash", "XFormers"], {"default": default_attention_type, "tooltip": "注意力类型：Auto=自动选择，Standard=标准，Flash=Flash Attention（NVIDIA GPU推荐），XFormers=XFormers（实验性）"}),
            },
            "optional": {
                "tensor_split": ("STRING", {"default": "", "tooltip": "多GPU tensor分割比例，格式：0.5,0.5（单GPU留空）"}),
            }
        }
    
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "omni-llm"

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
    def IS_CHANGED(s, model, enable_mmproj, mmproj, enable_asr, enable_tts, n_ctx, n_gpu_layers, vram_limit, image_max_tokens, attention_type="Auto", tensor_split=""):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        # 解析模型路径，确保与 loadmodel 中使用的值一致
        resolved_model_path = s._resolve_llm_model_path(model)
        actual_model = resolved_model_path if resolved_model_path else model
        
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
        
        # 使用解析后的路径来获取chat_handler，确保一致性
        chat_handler = get_auto_chat_handler(actual_model)
        
        # 注意：不包含model_path字段，因为它是动态计算的
        # 只包含用户可配置的参数，避免不必要的重新加载
        custom_config = {
            "model": actual_model,  # 使用解析后的路径，确保与loadmodel一致
            "chat_handler": chat_handler,
            "enable_mmproj": enable_mmproj,
            "mmproj": mmproj,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit,
            "image_max_tokens": image_max_tokens,
            "attention_type": attention_type,
            "tensor_split": tensor_split,
        }
        return json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
    
    def loadmodel(self, model, enable_mmproj, mmproj, enable_asr, enable_tts, n_ctx, n_gpu_layers, vram_limit, image_max_tokens, attention_type="Auto", tensor_split="", **kwargs):
        # 解析完整模型路径，避免同名冲突
        resolved_model_path = self._resolve_llm_model_path(model)
        if resolved_model_path:
            model = resolved_model_path
        resolved_mmproj_path = self._resolve_llm_model_path(mmproj)
        if resolved_mmproj_path:
            mmproj = resolved_mmproj_path

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
        is_qwen3 = "qwen3" in model.lower() or "qwen35" in model.lower() or "qwen36" in model.lower()
        is_qwen35 = "qwen35" in model.lower() or "qwen3.5" in model.lower()
        is_qwen36 = "qwen36" in model.lower() or "qwen3.6" in model.lower()
        is_qwen3vl = "qwen3-vl" in model.lower() or "qwen3vl" in model.lower()
        is_qwen36vl = "qwen36-vl" in model.lower() or "qwen3.6-vl" in model.lower()
        is_mimo_vl = "mimo-vl" in model.lower()
        
        # 特殊模型检测：Qwen3.5-9B-MTP、Qwen3.6-MTP和Qwen3.5-9B-DeepSeek-V4-Flash
        # DeepSeek-V4-Flash支持1M上下文长度，需要特殊处理
        is_deepseek_v4_flash = "deepseek-v4-flash" in model.lower() or "deepseek_v4_flash" in model.lower()
        # Qwen3.5-9B-MTP支持MTP推测解码，也需要较大的上下文
        is_qwen35_mtp = is_qwen35 and ("mtp" in model.lower() or "multitoken" in model.lower())
        # Qwen3.6-MTP支持MTP推测解码，也需要较大的上下文
        is_qwen36_mtp = is_qwen36 and ("mtp" in model.lower() or "multitoken" in model.lower())
        
        # 初始化 image_min_tokens，将在后续根据模型类型自动设置
        image_min_tokens = 0
        
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
                # Qwen3.5模型需要合理的image tokens（自动设置）
                if enable_mmproj:
                    image_min_tokens = 256
                    print(f"【GPU模式优化】Qwen3.5模型自动设置image_min_tokens为256")
                # 自动设置image_max_tokens（如果未设置或小于image_min_tokens）
                if enable_mmproj and image_max_tokens < image_min_tokens:
                    image_max_tokens = image_min_tokens
                    print(f"【GPU模式优化】自动设置image_max_tokens为{image_max_tokens}")
            
            # 针对Qwen3.6模型的特殊优化
            if is_qwen36:
                print(f"【GPU模式优化】Qwen3.6模型启用特殊GPU参数配置")
                # Qwen3.6模型需要更大的上下文长度
                if n_ctx < 8192:
                    print(f"【GPU模式优化】Qwen3.6模型需要至少8192的上下文长度，自动调整从{n_ctx}到8192")
                    n_ctx = 8192
                # 确保使用正确的ChatHandler
                if is_qwen36vl:
                    print(f"【提示】Qwen3.6-VL模型将使用Qwen3VLChatHandler")
                else:
                    print(f"【提示】Qwen3.6模型将使用Qwen35ChatHandler（兼容模式）")
                # Qwen3.6模型需要合理的image tokens（自动设置）
                if enable_mmproj:
                    image_min_tokens = 256
                    print(f"【GPU模式优化】Qwen3.6模型自动设置image_min_tokens为256")
                # 自动设置image_max_tokens（如果未设置或小于image_min_tokens）
                if enable_mmproj and image_max_tokens < image_min_tokens:
                    image_max_tokens = image_min_tokens
                    print(f"【GPU模式优化】自动设置image_max_tokens为{image_max_tokens}")
            
            # 针对Qwen3-VL模型的特殊优化
            if is_qwen3vl:
                print(f"【GPU模式优化】Qwen3-VL模型启用特殊GPU参数配置")
                # Qwen3-VL模型需要更大的上下文长度（视频处理时需要更多tokens）
                # 每个视频帧大约需要1024-2048 tokens，8帧需要约16384 tokens
                if n_ctx < 16384:
                    print(f"【GPU模式优化】Qwen3-VL模型需要至少16384的上下文长度以支持视频处理，自动调整从{n_ctx}到16384")
                    n_ctx = 16384
                # Qwen3-VL模型需要合理的image tokens（自动设置）
                if enable_mmproj:
                    image_min_tokens = 1024  # 提高到1024以支持视频帧
                    print(f"【GPU模式优化】Qwen3-VL模型自动设置image_min_tokens为1024")
                # 自动设置image_max_tokens（如果未设置或小于image_min_tokens）
                if enable_mmproj and image_max_tokens < image_min_tokens:
                    image_max_tokens = image_min_tokens
                    print(f"【GPU模式优化】自动设置image_max_tokens为{image_max_tokens}")
            
            # 针对DeepSeek-V4-Flash模型的特殊优化（支持1M上下文长度）
            if is_deepseek_v4_flash:
                print(f"【GPU模式优化】DeepSeek-V4-Flash模型启用特殊GPU参数配置")
                # DeepSeek-V4-Flash原生支持1M上下文长度(n_ctx_train=262144)
                # 当前设备显存有限，设置为训练上下文长度的一半以确保推理成功
                target_ctx = 131072  # 128K上下文长度
                if n_ctx < target_ctx:
                    print(f"【GPU模式优化】DeepSeek-V4-Flash模型需要至少{target_ctx}的上下文长度，自动调整从{n_ctx}到{target_ctx}")
                    n_ctx = target_ctx
                # DeepSeek-V4-Flash是MoE模型，需要降低n_batch以确保推理稳定
                n_batch = min(n_batch, 256)
                print(f"【GPU模式优化】DeepSeek-V4-Flash模型设置n_batch={n_batch}")
            
            # 针对Qwen3.5-MTP模型的特殊优化（支持MTP推测解码）
            if is_qwen35_mtp:
                print(f"【GPU模式优化】Qwen3.5-MTP模型启用特殊GPU参数配置")
                # MTP模型需要较大的上下文长度以支持推测解码
                if n_ctx < 32768:
                    print(f"【GPU模式优化】Qwen3.5-MTP模型需要至少32768的上下文长度，自动调整从{n_ctx}到32768")
                    n_ctx = 32768
            
            # 针对Qwen3.6-MTP模型的特殊优化（支持MTP推测解码）
            if is_qwen36_mtp:
                print(f"【GPU模式优化】Qwen3.6-MTP模型启用特殊GPU参数配置")
                # MTP模型需要较大的上下文长度以支持推测解码
                if n_ctx < 32768:
                    print(f"【GPU模式优化】Qwen3.6-MTP模型需要至少32768的上下文长度，自动调整从{n_ctx}到32768")
                    n_ctx = 32768
        
        # 针对MiMo-VL模型的特殊优化（基于Qwen2.5-VL架构）
        if is_mimo_vl:
            print(f"【GPU模式优化】MiMo-VL模型启用特殊GPU参数配置")
            # MiMo-VL模型需要合理的image tokens（自动设置）
            if enable_mmproj:
                image_min_tokens = 256
                print(f"【GPU模式优化】MiMo-VL模型自动设置image_min_tokens为256")
            # 自动设置image_max_tokens（如果未设置或小于image_min_tokens）
            if enable_mmproj and image_max_tokens < image_min_tokens:
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

        # 构建用于配置比较的配置字典（与IS_CHANGED返回的格式一致）
        # 只包含用户可配置的参数，不包含自动计算的参数
        compare_config = {
            "model": model,
            "chat_handler": chat_handler,
            "enable_mmproj": enable_mmproj,
            "mmproj": mmproj,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit,
            "image_max_tokens": image_max_tokens,
            "attention_type": attention_type,
            "tensor_split": tensor_split,
        }
        
        # 构建完整的配置字典（包含所有参数，用于实际加载模型）
        custom_config = {
            "model": model, "enable_mmproj": enable_mmproj, "mmproj": mmproj,
            "enable_asr": enable_asr, "enable_tts": enable_tts,
            "chat_handler": chat_handler, "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers, "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens,
            "n_batch": n_batch, "n_ubatch": n_ubatch, "n_threads": n_threads,
            "n_threads_batch": n_threads_batch, "attention_type": attention_type,
            "cache_prompt": False,
            "tensor_split": tensor_split,
        }
        
        # 使用compare_config进行比较，确保与IS_CHANGED返回的配置一致
        # 只有当配置真正改变时才重新加载模型
        current_config = LLAMA_CPP_STORAGE.current_config
        config_changed = True
        
        if LLAMA_CPP_STORAGE.llm and current_config:
            # 提取current_config中与compare_config对应的字段进行比较
            current_compare_config = {
                k: current_config.get(k) for k in compare_config.keys()
            }
            config_changed = current_compare_config != compare_config
        
        if not LLAMA_CPP_STORAGE.llm or config_changed:
            print(f"【模型加载】配置已变化，重新加载模型...")
            LLAMA_CPP_STORAGE.load_model(custom_config)
        else:
            print(f"【模型加载】配置未变化，跳过重新加载")
        
        return (LLAMA_CPP_STORAGE,)
