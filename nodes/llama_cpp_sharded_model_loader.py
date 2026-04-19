# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm 分段模型加载器节点
支持Qwen2.5-Omni系列分段模型的加载和推理

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import torch
import json
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import folder_paths

# 添加项目根目录到路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import HARDWARE_INFO, LLAMA_CPP_STORAGE
from engine.airllm_turboquant_integration import (
    integrate_turboquant_to_transformers,
    AirLLMTurboQuantConfig,
)


# 模型注册表 - 存储所有支持的模型版本信息
MODEL_REGISTRY = {
    # Qwen系列模型
    "Qwen2.5-Omni-3B": {
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json", 
            "spk_dict.pt", "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            "tokenizer_config.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ],
        "default": True
    },
    "Qwen2.5-Omni-7B": {
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json", 
            "spk_dict.pt", "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            "tokenizer_config.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ],
        "default": False
    },
    "Qwen2.5-VL": {
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json",
            "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            "tokenizer_config.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
        "test_file": "model-00004-of-00004.safetensors",
        "default": False
    },
    "Qwen3-VL": {
        "repo_id": {
            "huggingface": "Qwen/Qwen3-VL",
            "modelscope": "qwen/Qwen3-VL"
        },
        "required_files": [
            "added_tokens.json", "chat_template.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json",
            "tokenizer.json", "vocab.json", "config.json",
            "generation_config.json", "special_tokens_map.json",
            "tokenizer_config.json",
            "model-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors",
            "model-00004-of-00006.safetensors",
            "model-00005-of-00006.safetensors",
            "model-00006-of-00006.safetensors",
        ],
        "test_file": "model-00006-of-00006.safetensors",
        "default": False
    },
    
    # GLM系列模型
    "GLM-4.6V": {
        "repo_id": {
            "huggingface": "THUDM/glm-4.6v",
            "modelscope": "ZhipuAI/glm-4.6v"
        },
        "required_files": [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors.index.json", "preprocessor_config.json",
            "merges.txt", "vocab.json", "special_tokens_map.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ],
        "test_file": "model-00005-of-00005.safetensors",
        "default": False
    },
    "GLM-4.1V": {
        "repo_id": {
            "huggingface": "THUDM/glm-4.1v",
            "modelscope": "ZhipuAI/glm-4.1v"
        },
        "required_files": [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors.index.json", "preprocessor_config.json",
            "merges.txt", "vocab.json", "special_tokens_map.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
        "test_file": "model-00004-of-00004.safetensors",
        "default": False
    },
    
    # MiniCPM系列模型
    "MiniCPM-o-2_6": {
        "repo_id": {
            "huggingface": "OpenBMB/MiniCPM-o-2_6",
            "modelscope": "OpenBMB/MiniCPM-o-2_6"
        },
        "required_files": [
            "added_tokens.json", "config.json", "configuration.json",
            "model.safetensors.index.json", "preprocessor_config.json", 
            "tokenizer.json", "vocab.json", "merges.txt",
            "tokenizer_config.json", "special_tokens_map.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
        "test_file": "model-00004-of-00004.safetensors",
        "default": False
    },
    "MiniCPM-V-4.5": {
        "repo_id": {
            "huggingface": "OpenBMB/MiniCPM-V-4.5",
            "modelscope": "OpenBMB/MiniCPM-V-4.5"
        },
        "required_files": [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors.index.json", "preprocessor_config.json",
            "merges.txt", "vocab.json", "special_tokens_map.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ],
        "test_file": "model-00005-of-00005.safetensors",
        "default": False
    },
    
    # 其他VLM模型
    "Llama-3.2-11B-Vision": {
        "repo_id": {
            "huggingface": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "modelscope": "LLM-Research/Llama-3.2-11B-Vision-Instruct"
        },
        "required_files": [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors.index.json", "preprocessor_config.json",
            "merges.txt", "vocab.json", "special_tokens_map.json",
            "model-00001-of-00008.safetensors",
            "model-00002-of-00008.safetensors",
            "model-00003-of-00008.safetensors",
            "model-00004-of-00008.safetensors",
            "model-00005-of-00008.safetensors",
            "model-00006-of-00008.safetensors",
            "model-00007-of-00008.safetensors",
            "model-00008-of-00008.safetensors",
        ],
        "test_file": "model-00008-of-00008.safetensors",
        "default": False
    },
    "Phi-3.5-Vision": {
        "repo_id": {
            "huggingface": "microsoft/Phi-3.5-vision-instruct",
            "modelscope": "microsoft/Phi-3.5-vision-instruct"
        },
        "required_files": [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "model.safetensors.index.json", "preprocessor_config.json",
            "merges.txt", "vocab.json", "special_tokens_map.json",
            "model-00001-of-00006.safetensors",
            "model-00002-of-00006.safetensors",
            "model-00003-of-00006.safetensors",
            "model-00004-of-00006.safetensors",
            "model-00005-of-00006.safetensors",
            "model-00006-of-00006.safetensors",
        ],
        "test_file": "model-00006-of-00006.safetensors",
        "default": False
    }
}


def check_flash_attention():
    """检测Flash Attention 2支持（需Ampere架构及以上）"""
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except ImportError:
        return False


FLASH_ATTENTION_AVAILABLE = check_flash_attention()


def init_model_paths(model_name):
    """初始化模型路径"""
    # 获取LLM文件夹路径
    llm_folders = folder_paths.get_folder_paths("LLM")
    
    if not llm_folders:
        # 如果没有LLM文件夹，创建默认路径
        base_dir = Path(folder_paths.models_dir).resolve() / "LLM"
        base_dir.mkdir(parents=True, exist_ok=True)
        llm_folders = [str(base_dir)]
    
    # 使用第一个LLM文件夹
    model_dir = Path(llm_folders[0]) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"【分段模型】模型路径已初始化: {model_dir}")
    return str(model_dir)


def validate_model_path(model_path, model_name):
    """验证模型路径的有效性和模型文件是否齐全"""
    path_obj = Path(model_path)
    
    if not path_obj.is_absolute():
        print(f"【分段模型错误】{model_path} 不是绝对路径")
        return False
    
    if not path_obj.exists():
        print(f"【分段模型错误】模型目录不存在: {model_path}")
        return False
    
    if not path_obj.is_dir():
        print(f"【分段模型错误】{model_path} 不是目录")
        return False
    
    if not check_model_files_exist(model_path, model_name):
        print(f"【分段模型错误】模型文件不完整: {model_path}")
        return False
    
    return True


def check_model_files_exist(model_dir, model_name):
    """检查模型文件是否齐全"""
    # 检查是否是本地分段模型
    if model_name.startswith("Local: "):
        # 对于本地模型，检查关键文件是否存在
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "preprocessor_config.json"
        ]
        # 检查至少有一个分段模型文件（支持两种格式）
        sharded_files1 = list(Path(model_dir).glob("model-*-of-*.safetensors"))
        sharded_files2 = list(Path(model_dir).glob("model.safetensors-*-of-*.safetensors"))
        sharded_files = sharded_files1 + sharded_files2
        if not sharded_files:
            print(f"【分段模型错误】本地模型缺少分段文件")
            return False
        # 检查关键文件
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                print(f"【分段模型错误】缺少关键文件: {file}")
                return False
        return True
    
    # 对于注册表中的模型，使用标准检查
    if model_name not in MODEL_REGISTRY:
        print(f"【分段模型错误】未知模型版本 {model_name}")
        return False
    
    required_files = MODEL_REGISTRY[model_name]["required_files"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True


class llama_cpp_sharded_model_loader:
    """分段模型加载器"""
    
    @classmethod
    def INPUT_TYPES(s):
        # 检查并添加LLM文件夹路径
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))

        # 获取LLM文件夹路径
        llm_folders = folder_paths.get_folder_paths("LLM")
        
        # 扫描本地文件夹中的分段模型
        local_models = s._scan_local_sharded_models()
        
        # 过滤注册表中的模型，只保留本地实际存在的模型
        available_models = []
        
        # 检查注册表中的模型是否存在于本地
        for model_name in MODEL_REGISTRY.keys():
            model_exists = False
            for llm_folder in llm_folders:
                model_path = str(Path(llm_folder) / model_name)
                if os.path.exists(model_path) and check_model_files_exist(model_path, model_name):
                    model_exists = True
                    break
            if model_exists:
                available_models.append(model_name)
        
        # 添加本地扫描到的模型
        if local_models:
            available_models.extend(local_models)
        
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
            default_device_mode = "GPU"  # 默认使用GPU加速
        
        # 设置默认模型：如果available_models不为空，使用第一个；否则使用默认值
        default_model = None
        if available_models:
            # 尝试找到默认模型，如果不存在则使用第一个
            default_candidates = [name for name, info in MODEL_REGISTRY.items() if info.get("default", False)]
            for candidate in default_candidates:
                if candidate in available_models:
                    default_model = candidate
                    break
            if default_model is None:
                default_model = available_models[0]
        else:
            # 如果没有可用模型，使用注册表中的默认值
            default_model = next(name for name, info in MODEL_REGISTRY.items() if info.get("default", False))
        
        return {
            "required": {
                "model_name": (available_models, {"default": default_model}),
                "quantization": (["None", "4-bit (VRAM-friendly)", "8-bit (Balanced Precision)"], {"default": "None"}),
                "turboquant_kv_cache": (["None", "f16 (无压缩)", "q8_0 (8-bit)", "q6_k (6-bit)", "q5_k (5-bit)", "q5_0 (5-bit)", "q5_1 (5-bit)", "q4_k (4-bit)", "q4_0 (4-bit)", "q4_1 (4-bit)", "q3_k (3-bit)", "q2_k (2-bit)", "mxfp4 (4-bit)", "nvfp4 (4-bit)", "turbo3 (3-bit)", "turbo2 (2-bit)", "turbo1 (1-bit)"], {"default": "None", "tooltip": "TurboQuant KV Cache 压缩：None=禁用，f16=无压缩，其他=量化压缩（位数越低压缩率越高，显存占用越少）"}),
                "enable_asr": ("BOOLEAN", {"default": False, "tooltip": "启用ASR语音识别功能（需配合ASR模型加载器使用）"}),
                "enable_tts": ("BOOLEAN", {"default": False, "tooltip": "启用TTS语音合成功能（需配合TTS模型加载器使用）"}),
                "n_ctx": ("INT", {"default": default_n_ctx, "min": 1024, "max": 327680, "step": 128, "tooltip": "上下文长度，影响可处理的文本长度"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载（GPU模式有效）"}),
                "vram_limit": ("INT", {"default": default_vram_limit, "min": -1, "max": 24, "step": 1, "tooltip": "显存限制（GB），-1=无限制（GPU模式有效）"}),
                "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最小编码token数"}),
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "tooltip": "图片最大编码token数"}),
                "attention_type": (["Auto", "Standard", "Flash", "XFormers"], {"default": "Auto", "tooltip": "注意力类型：Auto=自动选择，Standard=标准，Flash=Flash Attention（NVIDIA GPU推荐），XFormers=XFormers（实验性）"}),
            }
        }
    
    @classmethod
    def _scan_local_sharded_models(s):
        """扫描本地文件夹中的分段模型"""
        local_models = []
        llm_folders = folder_paths.get_folder_paths("LLM")
        
        for llm_folder in llm_folders:
            llm_dir = Path(llm_folder)
            if llm_dir.exists():
                for item in llm_dir.iterdir():
                    if item.is_dir():
                        # 检查文件夹中是否包含分段模型文件（支持多种格式）
                        sharded_files1 = list(item.glob("model-*-of-*.safetensors"))
                        sharded_files2 = list(item.glob("model.safetensors-*-of-*.safetensors"))
                        sharded_files3 = list(item.glob("*.safetensors.index.json"))
                        sharded_files4 = list(item.glob("*model*-of-*.safetensors"))
                        sharded_files = sharded_files1 + sharded_files2 + sharded_files4
                        
                        # 检查是否有index.json文件（这也是分段模型的标志）
                        has_index_json = any(f.name.endswith('.safetensors.index.json') for f in item.iterdir())
                        
                        # 检查是否有多个safetensors文件（可能是分段模型）
                        has_multiple_safetensors = len([f for f in item.iterdir() if f.suffix == '.safetensors']) > 1
                        
                        if sharded_files or has_index_json or has_multiple_safetensors:
                            display_name = f"Local: {item.name}"
                            local_models.append(display_name)
        
        return local_models
    
    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "load_model"
    CATEGORY = "llama-cpp-vlm"

    @classmethod
    def IS_CHANGED(s, model_name, quantization, turboquant_kv_cache="None", enable_asr=False, enable_tts=False, 
                  n_ctx=8192, n_gpu_layers=-1, vram_limit=-1, 
                  image_min_tokens=0, image_max_tokens=0, attention_type="Auto"):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN") 
        
        # 根据模型名称自动推断对话格式处理器
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
        chat_handler = get_auto_chat_handler(model_name)
        
        custom_config = {
            "model": model_name,
            "quantization": quantization,
            "turboquant_kv_cache": turboquant_kv_cache,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "chat_handler": chat_handler,
            "device_mode": "GPU",
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "attention_type": attention_type
        }
        return json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device_mode = None
        self.model_path = None
    
    def load_model(self, model_name, quantization, turboquant_kv_cache="None", enable_asr=False, enable_tts=False, 
                  n_ctx=8192, n_gpu_layers=-1, vram_limit=-1, 
                  image_min_tokens=0, image_max_tokens=0, attention_type="Auto"):
        # 根据模型名称自动推断对话格式处理器
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
        chat_handler = get_auto_chat_handler(model_name)
        
        # 默认启用GPU模式
        device_mode = "GPU"
        
        # 处理本地分段模型
        if model_name.startswith("Local: "):
            model_folder_name = model_name.split("Local: ")[1]
            llm_folders = folder_paths.get_folder_paths("LLM")
            if llm_folders:
                model_path = str(Path(llm_folders[0]) / model_folder_name)
            else:
                model_path = str(Path(folder_paths.models_dir) / "LLM" / model_folder_name)
            print(f"【分段模型】加载本地分段模型: {model_path}")
        else:
            # 加载注册表中的模型
            model_path = init_model_paths(model_name)
        
        # 检查模型文件是否存在且完整
        if not validate_model_path(model_path, model_name):
            # 直接抛出错误，不尝试下载
            raise RuntimeError(f"【分段模型错误】模型文件不完整，请检查模型文件: {model_path}")
        
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError(f"【分段模型错误】运行 {model_name} 模型需要CUDA支持")
        print(f"【分段模型】使用GPU模式运行 {model_name} 模型")
        
        # 添加警告过滤
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
        
        quant_config = None
        # 设置计算精度为float16（GPU模式）
        compute_dtype = torch.float16
        
        if quantization == "4-bit (VRAM-friendly)":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8-bit (Balanced Precision)":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype
            )
        
        # 设置device_map为GPU
        device_map = {"": 0} if torch.cuda.device_count() > 0 else "auto"
        
        # 根据量化配置动态选择注意力实现
        if quant_config is not None:
            attn_impl = "sdpa"
            print("【分段模型】使用标准注意力实现 (sdpa) 替代FlashAttention，以兼容量化模式")
        else:
            attn_impl = "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "sdpa"
        
        # 设置模型精度（GPU模式）
        model_dtype = compute_dtype if quant_config else torch.float16
        precision_msg = "fp32" if model_dtype == torch.float32 else ("fp16" if model_dtype == torch.float16 else "bf16")
        print(f"【分段模型】使用精度: {precision_msg}")
        
        # 检查是否是Qwen3系列模型
        is_qwen3 = "qwen3" in model_name.lower() or "qwen35" in model_name.lower()
        is_qwen35 = "qwen35" in model_name.lower() or "qwen3.5" in model_name.lower()
        is_qwen3vl = "qwen3-vl" in model_name.lower() or "qwen3vl" in model_name.lower()
        
        # 检查是否是Omni模型
        is_omni = any(keyword in model_name.lower() for keyword in ["omni", "qwen3.5", "qwen35", "dreamomni", "dream-omni"])
        
        # 检查具体的Omni模型类型
        is_qwen35_omni = "qwen35" in model_name.lower() or "qwen3.5" in model_name.lower()
        is_qwen25_omni = "qwen2.5-omni" in model_name.lower() or "qwen25-omni" in model_name.lower()
        is_dreamomni = "dreamomni" in model_name.lower() or "dream-omni" in model_name.lower()
        
        # 显示Omni模型检测结果
        if is_qwen35_omni:
            print(f"【Omni模型检测】检测到Qwen3.5模型: {model_name}")
            print(f"【Omni模型检测】Qwen3.5模型，支持音频和视觉多模态，建议使用Qwen35ChatHandler")
        elif is_qwen25_omni:
            print(f"【Omni模型检测】检测到Qwen2.5-Omni模型: {model_name}")
            print(f"【Omni模型检测】Qwen2.5-Omni模型，支持音频和视觉多模态，建议使用Qwen2.5-VL ChatHandler")
        elif is_dreamomni:
            print(f"【Omni模型检测】检测到DreamOmni模型: {model_name}")
            print(f"【Omni模型检测】DreamOmni模型，支持音频和视觉多模态，建议使用LLaVA-1.6 ChatHandler")
        elif is_omni:
            print(f"【Omni模型检测】检测到Omni系列模型: {model_name}")
        
        # GPU模式下针对Qwen3系列模型的优化
        if is_qwen3:
            # Qwen3系列在GPU模式下也需要合理的上下文长度
            if n_ctx > 4096:
                print(f"【GPU模式优化】Qwen3系列模型将上下文长度从 {n_ctx} 调整为 4096 以确保GPU推理成功")
                n_ctx = 4096
            
            # 针对Qwen3.5模型的特殊优化
            if is_qwen35:
                print(f"【GPU模式优化】Qwen3.5模型启用特殊GPU参数配置")
                # 确保使用正确的ChatHandler
                print(f"【提示】Qwen3.5模型将使用Qwen35ChatHandler，启用thinking模式")
            
            # Qwen3系列模型（包括Qwen3-VL和Qwen3.5）需要至少1024个image tokens
            if image_min_tokens < 1024:
                print(f"【GPU模式优化】Qwen3系列模型需要至少1024 image tokens，自动调整image_min_tokens从{image_min_tokens}到1024")
                image_min_tokens = 1024
            
            # 自动设置image_max_tokens（如果未设置或小于image_min_tokens）
            if image_max_tokens < image_min_tokens:
                image_max_tokens = image_min_tokens
                print(f"【GPU模式优化】自动设置image_max_tokens为{image_max_tokens}")

        # GPU模式下针对Omni模型的优化
        elif is_omni:
            # Omni模型在GPU模式下的优化
            if n_ctx > 8192:
                print(f"【GPU模式优化】Omni模型将上下文长度从 {n_ctx} 调整为 8192 以确保GPU推理成功")
                n_ctx = 8192
            
            # Omni模型需要至少512个image tokens
            if image_min_tokens < 512:
                print(f"【GPU模式优化】Omni模型需要至少512 image tokens，自动调整image_min_tokens从{image_min_tokens}到512")
                image_min_tokens = 512
            
            if image_max_tokens < image_min_tokens:
                image_max_tokens = image_min_tokens
                print(f"【GPU模式优化】自动设置image_max_tokens为{image_max_tokens}")
            
            print(f"【GPU模式优化】Omni模型启用特殊GPU参数配置")
        
        # 根据模型名称选择不同的模型类和加载参数
        if model_name.startswith("Qwen"):
            model_class = Qwen2_5OmniForConditionalGeneration
            model_kwargs = {
                "enable_audio_output": True
            }
            processor_class = AutoProcessor
            processor_kwargs = {
                "trust_remote_code": True
            }
        elif model_name.startswith("MiniCPM"):
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
            model_kwargs = {
                "trust_remote_code": True
            }
            processor_class = AutoProcessor
            processor_kwargs = {
                "trust_remote_code": True
            }
        elif model_name.startswith("GLM"):
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
            model_kwargs = {
                "trust_remote_code": True
            }
            processor_class = AutoProcessor
            processor_kwargs = {
                "trust_remote_code": True
            }
        elif model_name.startswith("Llama"):
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
            model_kwargs = {
                "trust_remote_code": True
            }
            processor_class = AutoProcessor
            processor_kwargs = {
                "trust_remote_code": True
            }
        elif model_name.startswith("Phi"):
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
            model_kwargs = {
                "trust_remote_code": True
            }
            processor_class = AutoProcessor
            processor_kwargs = {
                "trust_remote_code": True
            }
        else:
            model_class = Qwen2_5OmniForConditionalGeneration
            model_kwargs = {
                "enable_audio_output": True
            }
            processor_class = AutoProcessor
            processor_kwargs = {
                "trust_remote_code": True
            }
        
        print(f"【分段模型】加载 {model_name} 模型，使用 {model_class.__name__}")
        
        # 需要重新加载，先释放现有资源
        self.clear_model_resources()
        
        # 更新当前模型名称和路径
        self.current_model_name = model_name
        self.current_quantization = quantization
        self.current_device_mode = device_mode
        self.model_path = model_path
        
        self.model = model_class.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=model_dtype,
            quantization_config=quant_config,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            offload_state_dict=True,
            **model_kwargs
        ).eval()

        num_heads = getattr(self.model.config, "num_attention_heads", 32)
        head_dim = getattr(self.model.config, "hidden_size", 4096) // num_heads
        
        turboquant_kv_cache_to_bits = {
            "f16 (无压缩)": 16,
            "q8_0 (8-bit)": 8,
            "q6_k (6-bit)": 6,
            "q5_k (5-bit)": 5,
            "q5_0 (5-bit)": 5,
            "q5_1 (5-bit)": 5,
            "q4_k (4-bit)": 4,
            "q4_0 (4-bit)": 4,
            "q4_1 (4-bit)": 4,
            "q3_k (3-bit)": 3,
            "q2_k (2-bit)": 2,
            "mxfp4 (4-bit)": 4,
            "nvfp4 (4-bit)": 4,
            "turbo3 (3-bit)": 3,
            "turbo2 (2-bit)": 2,
            "turbo1 (1-bit)": 1,
        }
        
        kv_compression_bits = turboquant_kv_cache_to_bits.get(turboquant_kv_cache, None)
        
        if turboquant_kv_cache != "None" and kv_compression_bits is not None:
            turboquant_config = AirLLMTurboQuantConfig(
                enabled=True,
                kv_compression_bits=kv_compression_bits,
                kv_seed=42,
                enable_asymmetric_attention=True,
                compress_kv=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"【TurboQuant】正在集成 KV Cache 压缩 (bits={turboquant_config.kv_compression_bits}, heads={num_heads}, head_dim={head_dim})")
            self.model = integrate_turboquant_to_transformers(self.model, turboquant_config)
            print(f"【TurboQuant】TurboQuant 集成完成，启用非对称注意力加速")
        else:
            if turboquant_kv_cache == "None":
                print(f"【TurboQuant】KV Cache 压缩已禁用，使用标准 KV Cache")
            else:
                print(f"【TurboQuant】未知的 KV Cache 压缩选项: {turboquant_kv_cache}，使用标准 KV Cache")

        # 编译优化（PyTorch 2.2+）
        if torch.__version__ >= "2.2":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # SDP优化
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        print(f"【分段模型】加载处理器，使用 {processor_class.__name__}")
        self.processor = processor_class.from_pretrained(model_path, **processor_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # MiniCPM模型特殊处理
        if model_name.startswith("MiniCPM"):
            print(f"【分段模型】为MiniCPM模型配置特殊参数")
            # 根据模型特性配置合适的参数
            self.model.config.use_cache = True
        
        # GLM模型特殊处理
        if model_name.startswith("GLM"):
            print(f"【分段模型】为GLM模型配置特殊参数")
            self.model.config.use_cache = True
        
        # 修复rope_scaling配置警告
        if hasattr(self.model.config, "rope_scaling"):
            print("【分段模型】修复ROPE缩放配置...")
            if "mrope_section" in self.model.config.rope_scaling:
                self.model.config.rope_scaling["mrope_section"] = "none"
        
        # 构建自定义配置，用于LLAMA_CPP_STORAGE
        custom_config = {
            "model": model_name,
            "model_path": model_path,
            "quantization": quantization,
            "turboquant_kv_cache": turboquant_kv_cache,
            "enable_asr": enable_asr,
            "enable_tts": enable_tts,
            "chat_handler": chat_handler,
            "device_mode": device_mode,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "attention_type": attention_type
        }
        
        # 更新LLAMA_CPP_STORAGE的配置
        LLAMA_CPP_STORAGE.current_config = custom_config
        LLAMA_CPP_STORAGE.model_path = model_path
        LLAMA_CPP_STORAGE.model_name = model_name
        
        print(f"【分段模型】模型 {model_name} 加载成功")
        asr_status = "已启用" if enable_asr else "已禁用"
        tts_status = "已启用" if enable_tts else "已禁用"
        print(f"【自动配置】根据模型 {model_name} 选择对话格式处理器: {chat_handler}，ASR功能{asr_status}，TTS功能{tts_status}")
        
        return (self,)
    

    
    def clear_model_resources(self):
        """释放当前模型占用的资源"""
        if self.model is not None:
            print("【分段模型】释放当前模型占用的资源...")
            del self.model, self.processor, self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {
    "llama_cpp_sharded_model_loader": llama_cpp_sharded_model_loader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_sharded_model_loader": "分段模型加载器"
}
