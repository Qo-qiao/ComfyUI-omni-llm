# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Image Inference Node

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import numpy as np
import torch
import tempfile
import os
import sys
import json
import base64
import re
import io
import soundfile as sf
import torchaudio
import time
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    HARDWARE_INFO, any_type, image2base64, scale_image,
    mm
)
from engine import (
    ModelTypeDetector,
    BaseInferenceEngine,
    GLM4VInferenceEngine,
    QwenOmniInferenceEngine,
    InferenceEngineFactory,
    convert_audio_to_wav_bytes,
    create_audio_data_uri
)
from engine.hook_utils import filter_stderr

class ErrorHandler:
    """错误处理类"""
    
    @staticmethod
    def handle_error(error, context=None):
        """
        处理错误并返回友好的错误提示
        
        Args:
            error: 错误对象
            context: 错误上下文
            
        Returns:
            友好的错误提示
        """
        error_message = str(error)
        
        # 分类错误类型
        error_type = ErrorHandler._classify_error(error_message)
        
        # 根据错误类型提供具体的错误提示
        if error_type == "memory":
            return ErrorHandler._handle_memory_error(error_message, context)
        elif error_type == "hardware":
            return ErrorHandler._handle_hardware_error(error_message, context)
        elif error_type == "model":
            return ErrorHandler._handle_model_error(error_message, context)
        elif error_type == "input":
            return ErrorHandler._handle_input_error(error_message, context)
        elif error_type == "network":
            return ErrorHandler._handle_network_error(error_message, context)
        else:
            return ErrorHandler._handle_generic_error(error_message, context)
    
    @staticmethod
    def _classify_error(error_message):
        """分类错误类型"""
        error_message_lower = error_message.lower()
        
        if any(keyword in error_message_lower for keyword in ["out of memory", "oom", "failed to find a memory slot"]):
            return "memory"
        elif any(keyword in error_message_lower for keyword in ["assertion failed", "ggml_assert", "cuda", "gpu"]):
            return "hardware"
        elif any(keyword in error_message_lower for keyword in ["model", "chat handler", "initialization", "load"]):
            return "model"
        elif any(keyword in error_message_lower for keyword in ["input", "invalid", "format", "size"]):
            return "input"
        elif any(keyword in error_message_lower for keyword in ["network", "connection", "timeout"]):
            return "network"
        else:
            return "generic"
    
    @staticmethod
    def _handle_memory_error(error_message, context):
        """处理内存错误"""
        suggestions = [
            "减少视频帧数",
            "降低max_tokens值",
            "降低image_max_tokens和image_min_tokens值",
            "使用更小的模型",
            "增加系统内存或显存"
        ]
        
        error_prompt = f"内存不足错误：{error_message}\n\n建议：\n"
        for i, suggestion in enumerate(suggestions, 1):
            error_prompt += f"{i}. {suggestion}\n"
        
        return error_prompt
    
    @staticmethod
    def _handle_hardware_error(error_message, context):
        """处理硬件错误"""
        suggestions = [
            "降低GPU层数或切换到CPU模式",
            "减少上下文长度",
            "检查显卡驱动是否最新",
            "确保GPU支持所需的CUDA版本"
        ]
        
        error_prompt = f"硬件错误：{error_message}\n\n建议：\n"
        for i, suggestion in enumerate(suggestions, 1):
            error_prompt += f"{i}. {suggestion}\n"
        
        return error_prompt
    
    @staticmethod
    def _handle_model_error(error_message, context):
        """处理模型错误"""
        suggestions = [
            "确保模型文件路径正确",
            "检查模型格式是否支持",
            "验证模型文件是否完整",
            "尝试使用其他模型"
        ]
        
        error_prompt = f"模型错误：{error_message}\n\n建议：\n"
        for i, suggestion in enumerate(suggestions, 1):
            error_prompt += f"{i}. {suggestion}\n"
        
        return error_prompt
    
    @staticmethod
    def _handle_input_error(error_message, context):
        """处理输入错误"""
        suggestions = [
            "检查输入格式是否正确",
            "确保输入大小在允许范围内",
            "验证输入文件是否完整",
            "尝试使用其他输入"
        ]
        
        error_prompt = f"输入错误：{error_message}\n\n建议：\n"
        for i, suggestion in enumerate(suggestions, 1):
            error_prompt += f"{i}. {suggestion}\n"
        
        return error_prompt
    
    @staticmethod
    def _handle_network_error(error_message, context):
        """处理网络错误"""
        suggestions = [
            "检查网络连接是否正常",
            "确保目标服务器可访问",
            "尝试增加超时时间",
            "检查代理设置"
        ]
        
        error_prompt = f"网络错误：{error_message}\n\n建议：\n"
        for i, suggestion in enumerate(suggestions, 1):
            error_prompt += f"{i}. {suggestion}\n"
        
        return error_prompt
    
    @staticmethod
    def _handle_generic_error(error_message, context):
        """处理通用错误"""
        suggestions = [
            "检查所有输入参数",
            "确保模型已正确加载",
            "查看控制台日志获取详细信息",
            "尝试重新启动ComfyUI"
        ]
        
        error_prompt = f"处理错误：{error_message}\n\n建议：\n"
        for i, suggestion in enumerate(suggestions, 1):
            error_prompt += f"{i}. {suggestion}\n"
        
        return error_prompt

# 导入预设提示词库
from support.prompt_enhancer_preset_zh import (
    NORMAL_DESCRIBE_TAGS_ZH,
    NORMAL_DESCRIBE_ZH,
    PROMPT_EXPANDER_ZH,
    ZIMAGE_TURBO_ZH,
    FLUX2_KLEIN_ZH,
    QWEN_IMAGE_2512_ZH,
    QWEN_IMAGE_EDIT_COMBINED_ZH,
    QWEN_IMAGE_LAYERED_ZH,
    LTX2_ZH,
    WAN_T2V_ZH,
    WAN_I2V_ZH,
    WAN_I2V_EMPTY_ZH,
    WAN_FLF2V_ZH,
    VIDEO_TO_PROMPT_ZH,
    VIDEO_DETAILED_SCENE_BREAKDOWN_ZH,
    VIDEO_SUBTITLE_FORMAT_ZH,
    AUDIO_SUBTITLE_CONVERT_ZH,
    VIDEO_TO_AUDIO_SUBTITLE_ZH,
    AUDIO_ANALYSIS_ZH,
    MULTI_SPEAKER_DIALOGUE_ZH,
    LYRICS_CREATION_ZH,
    OCR_ENHANCED_ZH,
    ULTRA_HD_IMAGE_REVERSE_ZH,
    VISION_BOUNDING_BOX_ZH,
)

from support.prompt_enhancer_preset_en import (
    NORMAL_DESCRIBE_TAGS_EN,
    NORMAL_DESCRIBE_EN,
    PROMPT_EXPANDER_EN,
    ZIMAGE_TURBO_EN,
    FLUX2_KLEIN_EN,
    QWEN_IMAGE_2512_EN,
    QWEN_IMAGE_EDIT_COMBINED_EN,
    QWEN_IMAGE_LAYERED_EN,
    LTX2_EN,
    WAN_T2V_EN,
    WAN_I2V_EN,
    WAN_I2V_EMPTY_EN,
    WAN_FLF2V_EN,
    VIDEO_TO_PROMPT_EN,
    VIDEO_DETAILED_SCENE_BREAKDOWN_EN,
    VIDEO_SUBTITLE_FORMAT_EN,
    AUDIO_SUBTITLE_CONVERT_EN,
    VIDEO_TO_AUDIO_SUBTITLE_EN,
    AUDIO_ANALYSIS_EN,
    MULTI_SPEAKER_DIALOGUE_EN,
    LYRICS_CREATION_EN,
    OCR_ENHANCED_EN,
    ULTRA_HD_IMAGE_REVERSE_EN,
    VISION_BOUNDING_BOX_EN,
)


class llama_cpp_unified_inference:
    """
    统一推理节点 - 支持VL模型
    自动检测模型类型并使用对应的推理方式
    支持ASR+VL+TTS组合处理音频
    """
    
    # 缓存变量
    _language_cache = {}
    _perf_params_cache = {}
    _model_type_cache = {}
    _model_cache = {}  # 模型缓存，避免重复加载
    _tts_model_cache = {}  # TTS模型缓存
    _inference_cache = {}  # 推理结果缓存，避免重复计算
    _cache_size = 1000  # 缓存大小限制
    
    @classmethod
    def _clean_cache(cls, cache, max_size=None):
        """
        清理缓存，保持缓存大小在限制范围内
        
        Args:
            cache: 要清理的缓存字典
            max_size: 最大缓存大小，默认使用 cls._cache_size
        """
        if max_size is None:
            max_size = cls._cache_size
        
        if len(cache) > max_size:
            # 按访问时间排序，删除最旧的条目
            items = sorted(cache.items(), key=lambda x: x[1].get('timestamp', 0))
            items_to_remove = len(cache) - max_size
            for key, _ in items[:items_to_remove]:
                del cache[key]
            print(f"【缓存管理】清理了 {items_to_remove} 个缓存条目，当前缓存大小: {len(cache)}")
    
    @classmethod
    def add_to_cache(cls, cache_name, key, value):
        """
        添加项到缓存，并自动清理超出限制的缓存
        
        Args:
            cache_name: 缓存名称（_language_cache, _perf_params_cache, 等）
            key: 缓存键
            value: 缓存值
        """
        cache = getattr(cls, cache_name)
        if not isinstance(cache, dict):
            return
        
        # 添加时间戳
        if not isinstance(value, dict):
            value = {"data": value, "timestamp": time.time()}
        else:
            value["timestamp"] = time.time()
        
        cache[key] = value
        cls._clean_cache(cache)
    
    @classmethod
    def get_from_cache(cls, cache_name, key):
        """
        从缓存中获取项，并更新访问时间
        
        Args:
            cache_name: 缓存名称
            key: 缓存键
            
        Returns:
            缓存值或 None
        """
        cache = getattr(cls, cache_name)
        if not isinstance(cache, dict):
            return None
        
        if key in cache:
            item = cache[key]
            # 更新时间戳
            if isinstance(item, dict):
                item["timestamp"] = time.time()
            else:
                cache[key] = {"data": item, "timestamp": time.time()}
                item = cache[key]
            
            # 返回实际数据
            return item.get("data", item)
        return None
    
    @classmethod
    def clear_all_caches(cls):
        """
        清理所有缓存
        """
        caches = ["_language_cache", "_perf_params_cache", "_model_type_cache", "_model_cache", "_tts_model_cache", "_inference_cache"]
        for cache_name in caches:
            cache = getattr(cls, cache_name)
            if isinstance(cache, dict):
                cache_size = len(cache)
                cache.clear()
                print(f"【缓存管理】清理了 {cache_size} 个 {cache_name} 缓存条目")
        
        # 清理 GPU 内存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("【内存管理】清理了 GPU 缓存")
        except Exception as e:
            print(f"【内存管理】清理 GPU 缓存失败: {e}")
    
    # 音色类型到 speaker_id 的映射
    VOICE_TYPE_TO_SPEAKER_ID = {
        "女声": 0, "男声": 1, "萝莉音": 2, "正太音": 3,
        "御姐音": 4, "大叔音": 5, "默认": 0,
        "Female": 0, "Male": 1, "Child Voice": 2,
        "Young Boy Voice": 3, "Mature Female Voice": 4,
        "Mature Male Voice": 5, "Default": 0,
        "default": 0, "female": 0, "male": 1, "loli": 2,
        "shota": 3, "mature_female": 4, "mature_male": 5,
        "young_female": 6, "young_male": 7, "elderly_female": 8,
        "elderly_male": 9, "dialect_female": 10,
        # Qwen3-TTS CustomVoice 音色映射
        "Vivian": 0,
        "Serena": 1,
        "Uncle_Fu": 2,
        "Dylan": 3,
        "Eric": 4,
        "Ryan": 5,
        "Aiden": 6,
        "Ono_Anna": 7,
        "Sohee": 8
    }
    
    # 初始化预设提示词字典
    preset_prompts = {}

    # 添加基础预设
    preset_prompts["Empty - Nothing"] = ""

    # 添加分类预设提示词（顺序与 prompt_enhancer_preset_zh.py / prompt_enhancer_preset_en.py 保持一致）
    preset_prompts["[Reverse] Tags"] = "NORMAL_DESCRIBE_TAGS"
    preset_prompts["[Reverse] Describe"] = "NORMAL_DESCRIBE"
    preset_prompts["[Normal] Expand"] = "PROMPT_EXPANDER"
    preset_prompts["[Portrait] ZIMAGE - Turbo"] = "ZIMAGE_TURBO"
    preset_prompts["[General] FLUX2 - Klein"] = "FLUX2_KLEIN"
    preset_prompts["[Poster] Qwen - Image 2512"] = "QWEN_IMAGE_2512"
    preset_prompts["[Image Edit] Qwen - Image Edit Combined"] = "QWEN_IMAGE_EDIT_COMBINED"
    preset_prompts["[Image Edit] Qwen - Image Layered"] = "QWEN_IMAGE_LAYERED"
    preset_prompts["[Text to Video] LTX-2"] = "LTX2"
    preset_prompts["[Text to Video] WAN - Text to Video"] = "WAN_T2V"
    preset_prompts["[Image to Video] WAN - Image to Video"] = "WAN_I2V"
    preset_prompts["[Image to Video] WAN - Image to Video Empty"] = "WAN_I2V_EMPTY"
    preset_prompts["[Image to Video] WAN - FLF to Video"] = "WAN_FLF2V"
    preset_prompts["[Video Analysis] Video - Reverse Prompt"] = "VIDEO_TO_PROMPT"
    preset_prompts["[Video Analysis] Video - Detailed Scene Breakdown"] = "VIDEO_DETAILED_SCENE_BREAKDOWN"
    preset_prompts["[Video Analysis] Video - Subtitle Format"] = "VIDEO_SUBTITLE_FORMAT"
    preset_prompts["[Audio] Audio ↔ Subtitle Convert"] = "AUDIO_SUBTITLE_CONVERT"
    preset_prompts["[Audio] Video to Audio & Subtitle"] = "VIDEO_TO_AUDIO_SUBTITLE"
    preset_prompts["[Audio] Audio Analysis"] = "AUDIO_ANALYSIS"
    preset_prompts["[Audio] Multi-Person Dialogue"] = "MULTI_SPEAKER_DIALOGUE"
    preset_prompts["[Music] Lyrics Creation"] = "LYRICS_CREATION"
    preset_prompts["[OCR] Enhanced OCR"] = "OCR_ENHANCED"
    preset_prompts["[HighRes] Ultra HD Image Reverse"] = "ULTRA_HD_IMAGE_REVERSE"
    preset_prompts["[Vision] Bounding Box"] = "VISION_BOUNDING_BOX"

    preset_tags = list(preset_prompts.keys())
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ========== 模型配置 ==========
                "llama_model": ("LLAMACPPMODEL", {"tooltip": "加载的VL/Omni模型，用于图像理解和文本生成"}),
                
                # ========== 推理模式 ==========
                "inference_mode": ([
                        "[基础] 文本生成 (Text Generation)",
                        "[基础] 图像理解 (Image Understanding)",
                        "[基础] 批量图像理解 (Batch Image Understanding)",
                        "[基础] 音频转文本 (Audio to Text)",
                        "[基础] 文本转音频 (Text to Audio)",
                        "[高级] 全模态整合 (Multimodal Integration)",
                        "[高级] 视频理解 (Video Understanding)"
                    ], {
                        "default": "[基础] 文本生成 (Text Generation)",
                        "tooltip": "选择推理模式：\n• [基础] 文本生成：使用语言模型生成文本内容\n• [基础] 图像理解：处理单张图像内容并生成描述\n• [基础] 批量图像理解：一次性处理多张图片，减少推理调用次数\n• [基础] 音频转文本：使用ASR模型将音频转换为文本\n• [基础] 文本转音频：生成文本后使用TTS模型转换为语音\n• [高级] 全模态整合：同时处理图像、音频和文本（Omni模型专用）\n• [高级] 视频理解：从视频中提取帧并进行分析"
                    }),
                
                # ========== 提示词配置 ==========
                "preset_prompt": (s.preset_tags, {"default": s.preset_tags[1], "tooltip": "选择预设提示词模板：\n• Empty - Nothing：无预设，完全自定义\n• [Normal] Tags：反推标签格式的描述\n• [Normal] Describe：反推详细描述文本\n• [Normal] Expand：扩展和丰富提示词\n• [Portrait] ZIMAGE - Turbo：人像生成优化\n• [General] FLUX2 - Klein：通用图像生成\n• [Poster] Qwen - Image 2512：海报风格图像\n• [Image Edit] Qwen - Image Edit Combined：图像编辑模板\n• [Image Edit] Qwen - Image Layered：分层图像编辑\n• [Text to Video] LTX-2：文本到视频生成\n• [Text to Video] WAN - Text to Video：WAN模型文本生视频\n• [Image to Video] WAN - Image to Video：图像生视频\n• [Image to Video] WAN - Image to Video Empty：图像生视频（无提示词）\n• [Image to Video] WAN - FLF to Video：首尾帧到视频\n• [Video Analysis] Video - Reverse Prompt：视频反推提示词\n• [Video Analysis] Video - Detailed Scene Breakdown：视频分镜详细拆解\n• [Video Analysis] Video - Subtitle Format：视频字幕格式\n• [Audio] Audio ↔ Subtitle Convert：音频与字幕互转\n• [Audio] Video to Audio & Subtitle：视频转音频和字幕\n• [Audio] Audio Analysis：音频内容分析\n• [Audio] Multi-Person Dialogue：多人对话处理\n• [Music] Lyrics & Audio Merge：歌词与音频合并\n• [OCR] Enhanced OCR：增强型文字识别\n• [HighRes] Ultra HD Image Reverse：超高清图像反推\n• [Vision] Bounding Box：视觉目标检测框"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "你是一位优秀的多模态助手。", "tooltip": "系统提示词，定义AI助手的角色和行为，可包含预设模板占位符#和自定义内容"}),
                "text_input": ("STRING", {"default": "", "multiline": True, "tooltip": "用户输入文本，作为对话的用户消息内容"}),
                
                # ========== 语言设置 ==========
                "prompt_language": (["中文", "English"], {"default": "中文", "tooltip": "预设提示词的语言"}),
                "response_language": (["自动检测", "中文", "English"], {"default": "自动检测", "tooltip": "AI回复的语言"}),
                
                # ========== 输出格式设置 ==========
                "output_format": (["JSON格式", "文本格式"], {
                    "default": "文本格式",
                    "tooltip": "输出格式控制：\n• JSON格式：输出JSON格式内容（遵循预设模板的output_format要求）\n• 文本格式：输出纯文本内容（使用input_template_text模板）"
                }),
                
                # ========== 视频处理参数 ==========
                "video_max_frames": ("INT", {"default": 16, "min": 2, "max": 1024, "step": 1, 
                                              "tooltip": "视频模式：最大提取帧数"}),
                "video_sampling": (["自动均匀采样", "手动指定帧索引"], 
                                  {"default": "自动均匀采样", 
                                   "tooltip": "视频帧采样方式：\n• 自动均匀采样：从视频中均匀抽取指定数量的帧\n• 手动指定帧索引：自定义要提取的帧号"}),
                "video_manual_indices": ("STRING", {"default": "", 
                                                     "placeholder": "例如: 0,10,20 或 0-10", 
                                                     "tooltip": "手动模式下的帧索引，仅在手动采样时生效"}),
                
                # ========== 图像处理参数 ==========
                "image_max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "step": 64,
                                           "tooltip": "图像处理的最大边长（像素），较大的值需要更多显存"}),
                
                # ========== 批量输出选项 ==========
                "batch_combination": (["逐个输出", "合并输出"], {
                    "default": "逐个输出",
                    "tooltip": "批量模式的结果处理方式：\n• 逐个输出：每张图片单独输出结果\n• 合并输出：所有结果合并为一个输出"
                }),
                "audio_output_mode": ([
                        "TTS音色美化输出",
                        "Omni原生音频输出"
                    ], {
                        "default": "TTS音色美化输出", 
                        "tooltip": "音频输出模式：\n• TTS音色美化输出：先生成文本，再使用TTS模型转换为音频\n• Omni原生音频输出：使用Omni模型直接生成音频输出"
                    }),
                "omni_speaker": ([
                        "Chelsie - 女声",
                        "Ethan - 男声"
                    ], {
                        "default": "Chelsie - 女声", 
                        "tooltip": "Omni模型音频输出音色（仅在Omni原生音频输出模式下生效）"
                    }),
                "tts_speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, 
                                        "display": "slider"}),
                
                # ========== 生成参数 ==========
                "seed": ("INT", {"default": 101, "min": 0, "max": 0xffffffffffffffff, "step": 1, "tooltip": "随机种子，用于复现结果"}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "强制卸载模型释放显存"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "parameters": ("LLAMACPPARAMS", {"tooltip": "额外的生成参数配置"}),
                "images": ("IMAGE", {"tooltip": "图像输入（用于图像理解模式）"}),
                "video": ("VIDEO", {"tooltip": "视频输入（用于视频理解模式）"}),
                "audio": ("AUDIO", {"tooltip": "音频输入（用于ASR识别）"}),
                "tts_model": ("TTSMODEL", {"tooltip": "TTS模型输入（用于语音合成）"}),
                "asr_model": ("ASRMODEL", {"tooltip": "ASR模型输入（用于语音识别）"}),
                "queue_handler": (any_type, {"tooltip": "队列处理器"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "AUDIO")
    RETURN_NAMES = ("output", "output_list", "state_uid", "audio")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def __init__(self):
        self.engine = None
        self.model_info = None
    
    def detect_model_type(self, llama_model) -> Dict:
        """检测模型类型"""
        model_path = llama_model.model_path
        model_config = getattr(llama_model, "config", None)
        
        # 检查缓存
        cache_key = f"{model_path}_{model_config}"
        if cache_key in self._model_type_cache:
            return self._model_type_cache[cache_key]
        
        # 处理"None"模型的情况
        if model_path is None or model_path == "None":
            model_info = {
                "key": "none",
                "type": "none",
                "subtype": "none",
                "supports_audio": False,
                "supports_vision": False,
                "file_formats": []
            }
            self._model_type_cache[cache_key] = model_info
            return model_info
        
        # 检测是否是分段模型
        is_sharded_model = self._is_sharded_model(llama_model)
        
        # 如果是分段模型，使用特殊的模型信息
        if is_sharded_model:
            model_info = {
                "key": "sharded_omni",
                "type": "omni",
                "subtype": "sharded",
                "supports_audio": True,
                "supports_vision": True,
                "file_formats": [".safetensors"]
            }
        else:
            # 检测模型类型
            model_info = ModelTypeDetector.detect_model_type(model_path, model_config)
            
            # 现在支持 VL 模型和 Omni 模型
            if model_info.get("type") not in ["vl", "omni"]:
                raise ValueError(f"此节点仅支持VL模型和Omni模型。当前模型类型: {model_info.get('type')}")
        
        # 缓存结果
        self._model_type_cache[cache_key] = model_info
        
        return model_info
    
    def _is_sharded_model(self, llama_model) -> bool:
        """检测是否是分段模型"""
        # 检查模型对象是否有transformers相关的属性
        has_transformers_model = hasattr(llama_model, 'model') and hasattr(llama_model.model, 'generate')
        has_processor = hasattr(llama_model, 'processor')
        
        # 检查模型名称是否包含特定标记
        model_name = getattr(llama_model, 'current_model_name', '')
        is_sharded_name = any(keyword in model_name.lower() for keyword in ['qwen', 'minicpm', 'sharded'])
        
        return has_transformers_model and has_processor and is_sharded_name
    
    def get_preset_text_by_language(self, preset_key, language, output_format="JSON格式"):
        """根据语言和输出格式获取预设提示词文本"""
        if language == "中文":
            preset_map = {
                "NORMAL_DESCRIBE_TAGS": NORMAL_DESCRIBE_TAGS_ZH,
                "NORMAL_DESCRIBE": NORMAL_DESCRIBE_ZH,
                "PROMPT_EXPANDER": PROMPT_EXPANDER_ZH,
                "ZIMAGE_TURBO": ZIMAGE_TURBO_ZH,
                "FLUX2_KLEIN": FLUX2_KLEIN_ZH,
                "QWEN_IMAGE_2512": QWEN_IMAGE_2512_ZH,
                "QWEN_IMAGE_EDIT_COMBINED": QWEN_IMAGE_EDIT_COMBINED_ZH,
                "QWEN_IMAGE_LAYERED": QWEN_IMAGE_LAYERED_ZH,
                "LTX2": LTX2_ZH,
                "WAN_T2V": WAN_T2V_ZH,
                "WAN_I2V": WAN_I2V_ZH,
                "WAN_I2V_EMPTY": WAN_I2V_EMPTY_ZH,
                "WAN_FLF2V": WAN_FLF2V_ZH,
                "VIDEO_TO_PROMPT": VIDEO_TO_PROMPT_ZH,
                "VIDEO_DETAILED_SCENE_BREAKDOWN": VIDEO_DETAILED_SCENE_BREAKDOWN_ZH,
                "VIDEO_SUBTITLE_FORMAT": VIDEO_SUBTITLE_FORMAT_ZH,
                "AUDIO_SUBTITLE_CONVERT": AUDIO_SUBTITLE_CONVERT_ZH,
                "VIDEO_TO_AUDIO_SUBTITLE": VIDEO_TO_AUDIO_SUBTITLE_ZH,
                "AUDIO_ANALYSIS": AUDIO_ANALYSIS_ZH,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_ZH,
                "LYRICS_CREATION": LYRICS_CREATION_ZH,
                "OCR_ENHANCED": OCR_ENHANCED_ZH,
                "ULTRA_HD_IMAGE_REVERSE": ULTRA_HD_IMAGE_REVERSE_ZH,
                "VISION_BOUNDING_BOX": VISION_BOUNDING_BOX_ZH,
            }
        else:
            preset_map = {
                "NORMAL_DESCRIBE_TAGS": NORMAL_DESCRIBE_TAGS_EN,
                "NORMAL_DESCRIBE": NORMAL_DESCRIBE_EN,
                "PROMPT_EXPANDER": PROMPT_EXPANDER_EN,
                "ZIMAGE_TURBO": ZIMAGE_TURBO_EN,
                "FLUX2_KLEIN": FLUX2_KLEIN_EN,
                "QWEN_IMAGE_2512": QWEN_IMAGE_2512_EN,
                "QWEN_IMAGE_EDIT_COMBINED": QWEN_IMAGE_EDIT_COMBINED_EN,
                "QWEN_IMAGE_LAYERED": QWEN_IMAGE_LAYERED_EN,
                "LTX2": LTX2_EN,
                "WAN_T2V": WAN_T2V_EN,
                "WAN_I2V": WAN_I2V_EN,
                "WAN_I2V_EMPTY": WAN_I2V_EMPTY_EN,
                "WAN_FLF2V": WAN_FLF2V_EN,
                "VIDEO_TO_PROMPT": VIDEO_TO_PROMPT_EN,
                "VIDEO_DETAILED_SCENE_BREAKDOWN": VIDEO_DETAILED_SCENE_BREAKDOWN_EN,
                "VIDEO_SUBTITLE_FORMAT": VIDEO_SUBTITLE_FORMAT_EN,
                "AUDIO_SUBTITLE_CONVERT": AUDIO_SUBTITLE_CONVERT_EN,
                "VIDEO_TO_AUDIO_SUBTITLE": VIDEO_TO_AUDIO_SUBTITLE_EN,
                "AUDIO_ANALYSIS": AUDIO_ANALYSIS_EN,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_EN,
                "LYRICS_CREATION": LYRICS_CREATION_EN,
                "OCR_ENHANCED": OCR_ENHANCED_EN,
                "ULTRA_HD_IMAGE_REVERSE": ULTRA_HD_IMAGE_REVERSE_EN,
                "VISION_BOUNDING_BOX": VISION_BOUNDING_BOX_EN,
            }
        
        preset = preset_map.get(preset_key, None)
        if preset is None:
            return preset_key
        
        if output_format == "文本格式" and "input_template_text" in preset:
            return preset.get("input_template_text", preset.get("input_template", preset_key))
        
        return preset.get("input_template", preset_key)
    
    def _analyze_audio_features(self, audio):
        """分析音频特征"""
        try:
            # 这里可以添加音频分析逻辑
            return {
                "voice_type": "默认",
                "emotion": "默认"
            }
        except Exception as e:
            print(f"音频分析失败: {str(e)}")
            return None
    
    def _generate_fallback_audio(self, text, voice_type, speed):
        """生成备用音频"""
        try:
            # 如果文本为空，则无法生成有效语音
            if not text:
                return None

            # 生成1秒静音波形作为安全回退，避免 downstream 保存音频时报错
            sample_rate = 24000
            duration_sec = 1.0
            length = int(sample_rate * duration_sec)
            waveform = torch.zeros(length, dtype=torch.float32)

            print(f"【TTS兼容】备用静音音频已生成（{duration_sec}s，sample_rate={sample_rate}）")
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"备用音频生成失败: {str(e)}")
            return None
    
    def _process_sharded_model(self, llama_model, final_prompt, text_input, mode, 
                            enable_tts, enable_asr, asr_text, audio_output_mode,
                            omni_speaker_type, tts_speed, 
                            has_images, has_audio, has_video, images, audio, tts_model, 
                            parameters, unique_id, seed, force_offload, output_format="文本格式"):
        """处理分段模型的推理请求"""
        try:
            print("【分段模型】使用分段模型推理方式")
            
            # 获取模型和处理器
            model = llama_model.model
            processor = llama_model.processor
            tokenizer = llama_model.tokenizer
            
            # 检查是否使用纯提示词模式（与GGUF格式模型行为一致）
            use_pure_prompt = output_format == "文本格式"
            
            if use_pure_prompt:
                print("【纯提示词模式】使用纯提示词输入，与GGUF格式模型行为一致")
                # 直接使用final_prompt作为输入，不添加系统提示词和对话结构
                input_text = final_prompt
                # 准备处理器参数
                processor_args = {
                    "text": input_text,
                    "return_tensors": "pt",
                    "padding": True
                }
                
                # 添加图像
                pil_images = []
                if has_images and images is not None:
                    try:
                        if len(images.shape) == 3:
                            images = images.unsqueeze(0)
                        
                        # 无论CPU还是GPU模式，都限制图像大小以避免内存问题
                        target_size = 512  # 适当的图像大小，平衡质量和性能
                        
                        for i in range(images.shape[0]):
                            try:
                                if hasattr(images[i], 'cpu'):
                                    img = images[i].cpu().numpy()
                                else:
                                    img = images[i]
                                
                                # 确保图像数据范围正确
                                if img.max() <= 1.0:
                                    img_np = (img * 255).astype(np.uint8)
                                else:
                                    img_np = img.astype(np.uint8)
                                
                                pil_img = Image.fromarray(img_np)
                                
                                # 调整图像大小
                                if target_size and (pil_img.size[0] > target_size or pil_img.size[1] > target_size):
                                    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                                    if i == 0:
                                        print(f"【图像优化】图像已调整大小至 {target_size}x{target_size}")
                                
                                pil_images.append(pil_img)
                            except Exception as e:
                                print(f"【图像处理错误】处理第{i}张图像时出错: {str(e)}")
                                continue
                        
                        if pil_images:
                            processor_args["images"] = pil_images
                            print(f"【图像优化】成功处理 {len(pil_images)} 张图像")
                        else:
                            print("【图像优化】没有成功处理任何图像")
                    except Exception as e:
                        print(f"【图像处理错误】{str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # 添加音频
                audio_path = None
                if has_audio and audio is not None:
                    try:
                        # 创建临时文件并保存音频
                        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
                            audio_path = f.name
                            if isinstance(audio, dict):
                                waveform = audio["waveform"].squeeze(0).cpu().numpy()
                                sample_rate = audio["sample_rate"]
                            else:
                                waveform = audio.squeeze(0).cpu().numpy()
                                sample_rate = 16000
                            sf.write(audio_path, waveform.T, sample_rate)
                        if audio_path:
                            processor_args["audio"] = audio_path
                    except Exception as e:
                        print(f"保存音频到临时文件时出错: {e}")
                        audio_path = None
            else:
                # 传统对话模式
                print("【对话模式】使用完整对话结构")
                # 构建对话
                SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                
                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": []}
                ]
                
                # 添加文本
                if final_prompt:
                    conversation[-1]["content"].append({"type": "text", "text": final_prompt})
                
                # 添加图像
                pil_images = []
                if has_images and images is not None:
                    try:
                        if len(images.shape) == 3:
                            images = images.unsqueeze(0)
                        
                        # 无论CPU还是GPU模式，都限制图像大小以避免内存问题
                        target_size = 512  # 适当的图像大小，平衡质量和性能
                        
                        for i in range(images.shape[0]):
                            try:
                                if hasattr(images[i], 'cpu'):
                                    img = images[i].cpu().numpy()
                                else:
                                    img = images[i]
                                
                                # 确保图像数据范围正确
                                if img.max() <= 1.0:
                                    img_np = (img * 255).astype(np.uint8)
                                else:
                                    img_np = img.astype(np.uint8)
                                
                                pil_img = Image.fromarray(img_np)
                                
                                # 调整图像大小
                                if target_size and (pil_img.size[0] > target_size or pil_img.size[1] > target_size):
                                    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                                    if i == 0:
                                        print(f"【图像优化】图像已调整大小至 {target_size}x{target_size}")
                                
                                pil_images.append(pil_img)
                                conversation[-1]["content"].append({"type": "image", "image": pil_img})
                            except Exception as e:
                                print(f"【图像处理错误】处理第{i}张图像时出错: {str(e)}")
                                continue
                        
                        if pil_images:
                            print(f"【图像优化】成功处理 {len(pil_images)} 张图像")
                        else:
                            print("【图像优化】没有成功处理任何图像")
                    except Exception as e:
                        print(f"【图像处理错误】{str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # 添加音频
                audio_path = None
                if has_audio and audio is not None:
                    try:
                        # 创建临时文件并保存音频
                        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
                            audio_path = f.name
                            if isinstance(audio, dict):
                                waveform = audio["waveform"].squeeze(0).cpu().numpy()
                                sample_rate = audio["sample_rate"]
                            else:
                                waveform = audio.squeeze(0).cpu().numpy()
                                sample_rate = 16000
                            sf.write(audio_path, waveform.T, sample_rate)
                        conversation[-1]["content"].append({"type": "audio", "audio": audio_path})
                    except Exception as e:
                        print(f"保存音频到临时文件时出错: {e}")
                        audio_path = None
                
                # 应用聊天模板
                input_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                
                # 准备处理器参数
                processor_args = {
                    "text": input_text,
                    "return_tensors": "pt",
                    "padding": True
                }
                
                # 添加图像和音频
                if pil_images:
                    processor_args["images"] = pil_images
                if audio_path:
                    processor_args["audio"] = audio_path
            
            # 参考ComfyUI-Qwen-Omni-main的处理方式：使用正确的多模态处理
            # 构建对话结构（即使在纯提示词模式下也使用对话结构）
            SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": []}
            ]
            
            # 添加文本
            if final_prompt:
                conversation[-1]["content"].append({"type": "text", "text": final_prompt})
            
            # 添加图像
            pil_images = []
            if has_images and images is not None:
                try:
                    if len(images.shape) == 3:
                        images = images.unsqueeze(0)
                    
                    # 无论CPU还是GPU模式，都限制图像大小以避免内存问题
                    target_size = 512  # 适当的图像大小，平衡质量和性能
                    
                    for i in range(images.shape[0]):
                        try:
                            if hasattr(images[i], 'cpu'):
                                img = images[i].cpu().numpy()
                            else:
                                img = images[i]
                            
                            # 确保图像数据范围正确
                            if img.max() <= 1.0:
                                img_np = (img * 255).astype(np.uint8)
                            else:
                                img_np = img.astype(np.uint8)
                            
                            pil_img = Image.fromarray(img_np)
                            
                            # 调整图像大小
                            if target_size and (pil_img.size[0] > target_size or pil_img.size[1] > target_size):
                                pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                                if i == 0:
                                    print(f"【图像优化】图像已调整大小至 {target_size}x{target_size}")
                            
                            pil_images.append(pil_img)
                            conversation[-1]["content"].append({"type": "image", "image": pil_img})
                        except Exception as e:
                            print(f"【图像处理错误】处理第{i}张图像时出错: {str(e)}")
                            continue
                    
                    if pil_images:
                        print(f"【图像优化】成功处理 {len(pil_images)} 张图像")
                    else:
                        print("【图像优化】没有成功处理任何图像")
                except Exception as e:
                    print(f"【图像处理错误】{str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # 添加音频
            audio_path = None
            if has_audio and audio is not None:
                try:
                    # 创建临时文件并保存音频
                    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
                        audio_path = f.name
                        if isinstance(audio, dict):
                            waveform = audio["waveform"].squeeze(0).cpu().numpy()
                            sample_rate = audio["sample_rate"]
                        else:
                            waveform = audio.squeeze(0).cpu().numpy()
                            sample_rate = 16000
                        sf.write(audio_path, waveform.T, sample_rate)
                    conversation[-1]["content"].append({"type": "audio", "audio": audio_path})
                except Exception as e:
                    print(f"保存音频到临时文件时出错: {e}")
                    audio_path = None
            
            # 应用聊天模板
            input_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            # 准备处理器参数
            processor_args = {
                "text": input_text,
                "return_tensors": "pt",
                "padding": True
            }
            
            # 添加图像和音频
            if pil_images:
                processor_args["images"] = pil_images
            if audio_path:
                processor_args["audio"] = audio_path
            
            # 将输入移至设备
            inputs = processor(**processor_args).to(model.device)
            
            # 清理临时文件
            if audio_path:
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"删除临时音频文件时出错: {e}")
            
            # 生成配置
            generate_config = {
                "max_new_tokens": 1024,  # 增加生成长度，获得更详细的内容
                "temperature": 0.8,  # 提高温度，增加输出多样性
                "do_sample": True,
                "use_cache": True,
                "return_audio": audio_output_mode != "TTS音色美化输出",
                "top_p": 0.95,  # 调整top_p，获得更丰富的输出
                "repetition_penalty": 1.0,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            
            # CPU模式特殊优化：平衡质量和速度
            if model.device.type == 'cpu':
                generate_config["max_new_tokens"] = 512  # 适当增加生成长度
                generate_config["temperature"] = 0.6  # 保持一定的多样性
                generate_config["do_sample"] = True  # 启用采样以获得更自然的输出
                generate_config["top_p"] = 0.9  # 保持一定的随机性
                print(f"【CPU模式优化】已调整生成参数: max_new_tokens={generate_config['max_new_tokens']}, temperature={generate_config['temperature']}, do_sample={generate_config['do_sample']}")
            
            # 视频处理特殊优化：调整参数以平衡质量和内存占用
            if has_video:
                # 针对视频反推等模式的优化
                generate_config["max_new_tokens"] = min(generate_config["max_new_tokens"], 768)  # 适当增加生成长度
                # 保持一定的温度以获得丰富的输出
                generate_config["temperature"] = max(generate_config["temperature"] - 0.05, 0.6)  # 保持较高的温度
                # 调整top_p以获得更丰富的输出
                generate_config["top_p"] = max(generate_config["top_p"] - 0.05, 0.9)  # 保持较高的top_p
                print(f"【视频优化】视频处理模式，调整max_new_tokens={generate_config['max_new_tokens']}, temperature={generate_config['temperature']}, top_p={generate_config['top_p']}")
            
            # 图像处理特殊优化：调整参数以减少推理时间
            elif has_images:
                # 针对图片反推等模式的优化
                generate_config["max_new_tokens"] = min(generate_config["max_new_tokens"], 512)  # 减少生成长度，加快推理速度
                # 降低温度以获得更稳定的输出
                generate_config["temperature"] = max(generate_config["temperature"] - 0.1, 0.5)  # 降低温度
                # 调整top_p以获得更准确的输出
                generate_config["top_p"] = max(generate_config["top_p"] - 0.05, 0.85)  # 降低top_p
                print(f"【图像优化】图像处理模式，调整max_new_tokens={generate_config['max_new_tokens']}, temperature={generate_config['temperature']}, top_p={generate_config['top_p']}")
            
            # 设置音频输出参数
            if generate_config["return_audio"]:
                # 根据omni_speaker参数设置speaker
                if "Chelsie" in omni_speaker_type:
                    generate_config["speaker"] = "Chelsie"
                elif "Ethan" in omni_speaker_type:
                    generate_config["speaker"] = "Ethan"
                else:
                    generate_config["speaker"] = "Chelsie"
            
            # 执行推理
            audio_output = None
            generated_text = ""
            
            try:
                # 动态获取设备类型，避免硬编码导致的设备不匹配错误
                device_type = 'cuda' if model.device.type == 'cuda' else 'cpu'
                
                # 内存优化：清理缓存
                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                    print("【内存优化】已清理GPU缓存")
                
                # 记录推理开始时间
                start_time = time.time()
                
                # 根据模式调整生成参数以平衡质量和内存使用
                if has_video:
                    # 视频处理模式：使用较大的生成长度
                    generate_config["max_new_tokens"] = min(generate_config["max_new_tokens"], 768)
                    print(f"【视频内存优化】已调整max_new_tokens为: {generate_config['max_new_tokens']}")
                elif has_images:
                    # 图像处理模式：使用适中的生成长度
                    generate_config["max_new_tokens"] = min(generate_config["max_new_tokens"], 384)  # 进一步减少生成长度，加快推理速度
                    print(f"【图像内存优化】已调整max_new_tokens为: {generate_config['max_new_tokens']}")
                else:
                    # 文本模式：使用较小的生成长度以节省内存
                    generate_config["max_new_tokens"] = min(generate_config["max_new_tokens"], 512)
                    print(f"【内存优化】已调整max_new_tokens为: {generate_config['max_new_tokens']}")
                
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                    outputs = model.generate(**inputs, **generate_config)
                
                # 记录推理结束时间
                end_time = time.time()
                inference_time = end_time - start_time
                print(f"【分段模型推理】推理完成，耗时: {inference_time:.2f} 秒")
                
                # 处理输出
                if generate_config["return_audio"]:
                    text_tokens = outputs[0] if outputs[0].dim() == 2 else outputs[0].unsqueeze(0)
                    audio_tensor = outputs[1]
                else:
                    text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
                    audio_tensor = torch.zeros(0, 0, device=model.device)
                
                # 截取新生成的token
                input_length = inputs["input_ids"].shape[1]
                text_tokens = text_tokens[:, input_length:]
                
                # 解码文本
                generated_text = tokenizer.decode(
                    text_tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                print(f"【分段模型推理】生成文本长度: {len(generated_text)}")
                
                # 处理音频输出
                if generate_config["return_audio"] and audio_tensor is not None:
                    # 参考ComfyUI-Qwen-Omni-main的处理方式：处理不同维度的音频张量
                    if audio_tensor.dim() == 1:
                        print(f"【分段模型】音频张量是1D，添加维度")
                        audio_tensor = audio_tensor.unsqueeze(0)
                    elif audio_tensor.dim() == 3:
                        print(f"【分段模型】音频张量是3D，平均处理")
                        audio_tensor = audio_tensor.mean(dim=1)
                    
                    assert audio_tensor.dim() == 2, f"Audio waveform must be 2D, got {audio_tensor.dim()}D"
                    
                    audio_output = {
                        "waveform": audio_tensor,
                        "sample_rate": 24000
                    }
                    
                    # 保存为WAV格式
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, audio_output["waveform"].cpu(), 24000, format="wav")
                    buffer.seek(0)
                    waveform, sample_rate = torchaudio.load(buffer)
                    audio_output = {
                        "waveform": waveform.unsqueeze(0),
                        "sample_rate": sample_rate
                    }
                    print(f"【分段模型】音频生成成功，shape={audio_output['waveform'].shape}")
                
                # 内存优化：清理缓存
                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                    print("【内存优化】推理完成后已清理GPU缓存")
            
            except RuntimeError as e:
                if "Allocation on device" in str(e) or "out of memory" in str(e).lower():
                    print(f"【分段模型推理错误】内存分配失败: {str(e)}")
                    print("【内存优化】尝试减少生成参数并重新推理...")
                    
                    # 内存优化：清理缓存
                    if device_type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # 进一步降低生成参数
                    generate_config["max_new_tokens"] = 128
                    generate_config["do_sample"] = False
                    generate_config["temperature"] = 0.3
                    print(f"【内存优化】已调整参数: max_new_tokens={generate_config['max_new_tokens']}, do_sample={generate_config['do_sample']}, temperature={generate_config['temperature']}")
                    
                    try:
                        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                            outputs = model.generate(**inputs, **generate_config)
                        
                        # 处理输出
                        if generate_config["return_audio"]:
                            text_tokens = outputs[0] if outputs[0].dim() == 2 else outputs[0].unsqueeze(0)
                        else:
                            text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
                        
                        # 截取新生成的token
                        input_length = inputs["input_ids"].shape[1]
                        text_tokens = text_tokens[:, input_length:]
                        
                        # 解码文本
                        generated_text = tokenizer.decode(
                            text_tokens[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        
                        print(f"【分段模型推理】内存优化后推理成功，生成文本长度: {len(generated_text)}")
                    except Exception as e2:
                        print(f"【分段模型推理错误】内存优化后仍然失败: {str(e2)}")
                        generated_text = f"推理失败: 内存不足，请尝试减少输入图像大小或降低生成参数"
                else:
                    print(f"【分段模型推理错误】{str(e)}")
                    generated_text = f"推理失败: {str(e)}"
            except Exception as e:
                print(f"【分段模型推理错误】{str(e)}")
                generated_text = f"推理失败: {str(e)}"
        
        except Exception as e:
            print(f"【分段模型处理错误】{str(e)}")
            return f"处理失败: {str(e)}", None
        
        finally:
            # 清理临时文件
            if 'audio_path' in locals() and audio_path:
                try:
                    os.remove(audio_path)
                except Exception as e:
                    print(f"删除临时音频文件时出错: {e}")
        
        return generated_text, audio_output
    
    def _process_batch_inference(self, llama_model, images, system_prompt, final_prompt,
                                  image_max_size, batch_combination, gen_params,
                                  audio_output_mode, engine, model_info, mode):
        """
        批量推理模式 - 优化多图推理性能
        一次性处理多张图片，减少推理调用次数
        """
        try:
            batch_results = []
            
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            num_images = images.shape[0]
            print(f"【批量推理】开始处理 {num_images} 张图片")
            
            # 预处理：批量编码所有图片
            print(f"【批量推理】预处理图片编码...")
            image_contents = []
            for i in range(num_images):
                mm.throw_exception_if_processing_interrupted()
                
                if hasattr(images[i], 'cpu'):
                    img = images[i].cpu().numpy()
                else:
                    img = images[i]
                img_np = scale_image(img, image_max_size)
                img_base64 = image2base64(img_np)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
                print(f"【批量推理】编码图片 {i+1}/{num_images}")
            
            # 批量推理
            print(f"【批量推理】开始批量推理...")
            
            if batch_combination == "逐个输出":
                # 逐个输出模式：每张图片单独推理
                for i in range(num_images):
                    mm.throw_exception_if_processing_interrupted()
                    print(f"【批量推理】推理图片 {i+1}/{num_images}")
                    
                    messages = []
                    if system_prompt and system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt})
                    
                    content = [{"type": "text", "text": final_prompt}]
                    content.append(image_contents[i])
                    
                    messages.append({"role": "user", "content": content})
                    
                    output = engine.create_chat_completion(llama_model.llm, messages, gen_params)
                    if output and 'choices' in output:
                        text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                        batch_results.append(text)
                        print(f"【批量推理】图片 {i+1} 完成: {text[:50]}...")
                    else:
                        batch_results.append("")
                        print(f"【批量推理】图片 {i+1} 失败")
            else:
                # 合并输出模式：所有图片一次性处理
                messages = []
                if system_prompt and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                
                content = [{"type": "text", "text": final_prompt}]
                content.extend(image_contents)
                
                messages.append({"role": "user", "content": content})
                
                output = engine.create_chat_completion(llama_model.llm, messages, gen_params)
                if output and 'choices' in output:
                    combined_text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                    # 尝试分割结果（假设模型按顺序返回结果）
                    # 可以根据输出格式进行分割
                    lines = combined_text.split('\n')
                    if len(lines) >= num_images:
                        batch_results = [line.strip() for line in lines[:num_images] if line.strip()]
                    else:
                        # 如果无法分割，将整个结果作为第一个输出
                        batch_results = [combined_text]
                    
                    print(f"【批量推理】合并模式完成: {combined_text[:50]}...")
                else:
                    batch_results = [""]
                    print(f"【批量推理】合并模式失败")
            
            return batch_results
            
        except Exception as e:
            print(f"【批量推理错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _check_audio_input(self, audio):
        """检查音频输入"""
        if audio is None:
            return False
        if isinstance(audio, dict):
            return "waveform" in audio or "text" in audio
        elif hasattr(audio, 'numel'):
            return audio.numel() > 0
        return False
    

    
    def _run_inference(self, llama_model, messages, gen_params, audio_output_mode, omni_speaker_type):
        """执行推理的内部方法，用于异步处理"""
        generated_text = ""
        audio_output = None
        
        # 生成缓存键
        model_path = getattr(llama_model, 'model_path', str(llama_model))
        cache_content = {
            "model_path": model_path,
            "messages": messages,
            "gen_params": {k: v for k, v in gen_params.items() if k != "seed"},  # 排除seed
            "audio_output_mode": audio_output_mode
        }
        import json
        cache_key = hash(json.dumps(cache_content, sort_keys=True, default=str))
        
        # 检查缓存
        cached_result = self.get_from_cache("_inference_cache", cache_key)
        if cached_result:
            return cached_result
        
        retry_count = 0
        max_retries = 2
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # 根据音频输出模式决定使用哪种方式
                if audio_output_mode == "Omni原生音频输出":
                    # 使用Omni模型原生音频输出
                    if self.model_info["type"] == "omni" and isinstance(self.engine, QwenOmniInferenceEngine):
                        generated_text, omni_audio = self.engine.generate_with_audio_output(
                            llama_model.llm, messages, gen_params, omni_speaker_type, None
                        )
                        if omni_audio:
                            audio_output = omni_audio
                            print(f"【Omni推理】使用原生音频输出")
                        else:
                            # 如果Omni模型未生成音频，回退到标准文本推理
                            output = self.engine.create_chat_completion(llama_model.llm, messages, gen_params)
                            if output and 'choices' in output:
                                generated_text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                                # 过滤掉Thinking Process内容

                            print(f"【提示】Omni无音频输出，回退至文本生成")
                    else:
                        # 非Omni模型或不支持原生音频输出，使用标准推理
                        output = self.engine.create_chat_completion(llama_model.llm, messages, gen_params)
                        if output and 'choices' in output:
                            generated_text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                            # 过滤掉Thinking Process内容
                            if "Thinking Process:" in generated_text:
                                generated_text = generated_text.split("Thinking Process:")[0].strip()
                        print(f"【提示】当前模型不支持Omni原生音频输出，已回退到文本输出")
                else:
                    # 标准文本推理（所有模型类型）
                    output = self.engine.create_chat_completion(llama_model.llm, messages, gen_params)
                    if output and 'choices' in output:
                        generated_text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")


                if generated_text and generated_text.strip():
                    success = True
                    print(f"【推理完成】生成文本长度: {len(generated_text)}")
                else:
                    retry_count += 1
                    print(f"【提示】推理结果为空，重试 {retry_count}/{max_retries}...")
                    # 调试：打印原始输出
                    if output and 'choices' in output and len(output['choices']) > 0:
                        raw_content = output['choices'][0]['message']['content']
                        print(f"【调试】原始返回内容: {raw_content[:100]}...")
                        print(f"【调试】原始内容长度: {len(raw_content)}")

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                # 分析错误类型，提供针对性建议
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower() or "failed to find a memory slot" in error_msg.lower():
                    print(f"【显存错误】推理失败：{error_msg}")
                    print(f"【智能建议】")
                    print(f"  1. 减少视频帧数")
                    print(f"  2. 降低max_tokens值（当前：{gen_params.get('max_tokens', 1024)}）")
                    print(f"  3. 降低image_max_tokens和image_min_tokens值")
                    print(f"  4. 使用更小的模型")
                    
                    # 尝试降低参数后重试
                    if retry_count < max_retries:
                        # 降低max_tokens重试
                        gen_params['max_tokens'] = max(256, gen_params.get('max_tokens', 1024) // 2)
                        print(f"【自动调整】已将max_tokens降低到{gen_params['max_tokens']}，重新尝试推理...")
                elif "assertion failed" in error_msg.lower() or "ggml_assert" in error_msg.lower():
                    print(f"【硬件错误】推理失败：{error_msg}")
                    print(f"【智能建议】")
                    print(f"  1. 降低GPU层数或切换到CPU模式")
                    print(f"  2. 减少上下文长度")
                    print(f"  3. 检查显卡驱动是否最新")
                else:
                    print(f"【提示】推理失败，重试 {retry_count}/{max_retries}：{e}")
        
        if not success:
            print(f"【错误】推理多次失败")
            generated_text = "推理失败"
        
        # 将结果添加到缓存
        result = (generated_text, audio_output)
        self.add_to_cache("_inference_cache", cache_key, result)
        
        return generated_text, audio_output
    
    def process(self, llama_model, inference_mode, preset_prompt, system_prompt, text_input,
                prompt_language, response_language, output_format, video_max_frames,
                video_sampling, video_manual_indices, image_max_size, batch_combination,
                audio_output_mode,
                omni_speaker, tts_speed, seed, force_offload,
                parameters=None, images=None, video=None, audio=None,
                tts_model=None, asr_model=None, queue_handler=None, unique_id=None):
        """处理推理请求"""
        try:
            # 检测模型类型
            self.model_info = self.detect_model_type(llama_model)
            
            # 处理推理模式
            mode_map = {
                "[基础] 文本生成 (Text Generation)": "text",
                "[基础] 图像理解 (Image Understanding)": "images",
                "[基础] 批量图像理解 (Batch Image Understanding)": "batch_images",
                "[基础] 音频转文本 (Audio to Text)": "audio",
                "[基础] 文本转音频 (Text to Audio)": "text_to_audio",
                "[高级] 全模态整合 (Multimodal Integration)": "multimodal",
                "[高级] 视频理解 (Video Understanding)": "video"
            }
            mode = mode_map.get(inference_mode, "text")
            
            # 处理Omni音频参数
            omni_speaker_type = omni_speaker.split(" - ")[0].strip()
            
            # 检查输入
            has_images = images is not None and (hasattr(images, 'numel') and images.numel() > 0)
            has_video = video is not None and (hasattr(video, 'numel') and video.numel() > 0)
            has_audio = self._check_audio_input(audio)
            
            # 处理自定义提示词
            custom_prompt = text_input
            
            # 检查是否启用ASR和TTS
            enable_asr = mode in ["audio", "text_to_audio"] or preset_prompt in ["[Audio] Audio to Text"]
            enable_tts = mode in ["text_to_audio"] or preset_prompt in ["[Audio] Text to Audio"]
            
            # 检查是否有ASR和TTS模型
            has_asr_model = asr_model is not None
            has_tts_model = tts_model is not None
            
            # 检查是否有音频输入
            has_audio_input = has_audio
            
            # 语言设置
            preset_prompts_language = prompt_language
            output_language = response_language
            
            # 获取预设提示词
            preset_key = self.preset_prompts.get(preset_prompt, "")
            preset_text = self.get_preset_text_by_language(preset_key, preset_prompts_language, output_format)
            
            # 构建最终提示词
            if preset_prompt == "Empty - Nothing":
                final_prompt = custom_prompt.strip() if custom_prompt.strip() else system_prompt.strip()
            else:
                final_prompt = preset_text
                if custom_prompt.strip():
                    if "下面是要优化的 Prompt：" in preset_text or "Below is the Prompt to optimize:" in preset_text:
                        final_prompt = preset_text + custom_prompt.strip()
                    else:
                        final_prompt = preset_text.replace("#", custom_prompt.strip()).replace("@", "video" if has_video else "image")
            
            # 添加语言指示
            if output_language == "中文":
                final_prompt += "\n\n请用中文回答。"
            elif output_language == "English":
                final_prompt += "\n\nPlease answer in English."
            
            # 执行ASR语音识别（如果启用）
            asr_text = ""
            if enable_asr and has_audio_input:
                print("【ASR】开始执行语音识别...")
                try:
                    if asr_model is not None and hasattr(asr_model, 'transcribe'):
                        # 使用独立的ASR模型
                        asr_result = asr_model.transcribe(audio)
                        asr_text = asr_result.get('text', '') if isinstance(asr_result, dict) else str(asr_result)
                        print(f"【ASR】识别结果: {asr_text[:100]}...")

                        # Audio 模式下直接将ASR结果作为最终输出文本
                        if mode == "audio" and asr_text:
                            print("【ASR模式】识别结果可用于输出文本")
                    else:
                        print("【ASR提示】未找到可用的ASR模型，跳过语音识别")
                except Exception as e:
                    print(f"【ASR错误】语音识别失败: {str(e)}")
            
            # 分析音频特征（用于TTS音色匹配）
            audio_features = None
            if audio is not None and enable_tts:
                audio_features = self._analyze_audio_features(audio)
                if audio_features:
                    print(f"【音频分析】音色: {audio_features['voice_type']}, 情感: {audio_features['emotion']}")
            
            # 将ASR结果合并到提示词中
            if asr_text:
                final_prompt = f"[音频内容: {asr_text}]\n{final_prompt}"

            # Audio 模式下：直接输出 ASR 文本
            generated_text = ""
            if mode == "audio" and asr_text:
                generated_text = asr_text
                print("【ASR模式】已将识别结果作为输出文本")
            
            # 处理无模型情况（音频转文本和文本转音频模式）
            if self.model_info["type"] == "none":
                if mode == "audio" and asr_text:
                    # 音频转文本模式，直接返回ASR结果
                    print("【无模型模式】音频转文本模式，仅返回ASR识别结果")
                    return (generated_text, [generated_text], seed, None)
                elif mode == "text_to_audio":
                    # 文本转音频模式，直接使用输入文本或ASR文本进行TTS合成
                    print("【无模型模式】文本转音频模式，直接进行TTS语音合成")
                    # 如果没有生成文本，使用输入文本或ASR文本
                    if not generated_text:
                        if custom_prompt.strip():
                            generated_text = custom_prompt.strip()
                            print(f"【TTS模式】使用输入文本进行音频合成: {generated_text[:60]}...")
                        elif asr_text:
                            generated_text = asr_text
                            print(f"【TTS模式】使用ASR识别文本进行音频合成: {generated_text[:60]}...")
                    # 处理TTS语音合成
                    audio_output = None
                    if generated_text:
                        print(f"【TTS】开始语音合成...")
                        # 尝试使用TTS模型
                        if tts_model is not None and hasattr(tts_model, 'synthesize'):
                            try:
                                # TTS模型的音色和情感由TTS节点控制
                                audio_output = tts_model.synthesize(
                                    text=generated_text,
                                    speed=tts_speed
                                )
                                print(f"【TTS】使用TTS模型合成成功")
                            except Exception as e:
                                print(f"【TTS】TTS模型合成失败: {str(e)}")
                                audio_output = None
                        # 如果TTS模型失败或未提供，使用备选方案
                        if audio_output is None:
                            audio_output = self._generate_fallback_audio(generated_text, omni_speaker_type, tts_speed)
                            if audio_output:
                                print(f"【TTS】使用备选方案生成音频")
                    # 兼容输出，确保传统Comfy音频保存器可以接受
                    if audio_output is None:
                        # 如果没有音频输出，使用1秒静音波形作为安全回退
                        audio_output = {"waveform": torch.zeros(int(24000 * 1.0), dtype=torch.float32), "sample_rate": 24000}
                        print("【TTS兼容】音频输出为空，已生成1秒静音回退音频")
                    
                    # 支持numpy->torch类型转换
                    if isinstance(audio_output, dict):
                        waveform = audio_output.get("waveform")
                        sample_rate = audio_output.get("sample_rate")

                        if isinstance(waveform, np.ndarray):
                            waveform = torch.from_numpy(waveform)
                            print(f"【TTS兼容】waveform由numpy.ndarray转换为torch.Tensor，shape={waveform.shape}")
                        elif isinstance(waveform, list):
                            waveform = torch.tensor(waveform)
                            print(f"【TTS兼容】waveform由list转换为torch.Tensor，长度={len(waveform)}")
                        elif isinstance(waveform, torch.Tensor):
                            # 已经符合要求
                            pass
                        else:
                            print(f"【TTS兼容】警告：waveform类型不支持 ({type(waveform)})")
                            waveform = None
                        
                        # 确保waveform是三维张量 [batch, channels, samples]
                        if waveform is not None:
                            if waveform.dim() == 1:
                                waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
                                print(f"【TTS兼容】waveform已reshape为三维，shape={waveform.shape}")
                            elif waveform.dim() == 2:
                                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
                                print(f"【TTS兼容】waveform已reshape为三维，shape={waveform.shape}")
                            audio_output["waveform"] = waveform

                        # 默认sample_rate
                        if sample_rate is None:
                            audio_output["sample_rate"] = 24000
                            print("【TTS兼容】sample_rate缺失，已默认设为24000")
                    
                    return (generated_text, [generated_text], seed, audio_output)

            # 文本转音频优先直接使用输入文本
            if mode != "audio" and mode != "text_to_audio":
                generated_text = ""
            if mode == "text_to_audio" or preset_prompt in ["[Audio] Text to Audio"]:
                if custom_prompt.strip():
                    generated_text = custom_prompt.strip()
                    print(f"【TTS模式】直接使用输入文本进行音频合成: {generated_text[:60]}...")
                elif asr_text:
                    generated_text = asr_text
                    print(f"【TTS模式】使用ASR识别文本进行音频合成: {generated_text[:60]}...")
            
            # 检查是否是分段模型
            is_sharded_model = self._is_sharded_model(llama_model)
            
            # 如果是分段模型，使用分段模型的推理方式
            if is_sharded_model:
                generated_text, audio_output = self._process_sharded_model(
                    llama_model, final_prompt, text_input, mode, 
                    enable_tts, enable_asr, asr_text, audio_output_mode,
                    omni_speaker_type, tts_speed, 
                    has_images, has_audio, has_video, images, audio, tts_model, 
                    parameters, unique_id, seed, force_offload, output_format
                )
            else:
                # 创建推理引擎（使用缓存）
                engine_key = f"{self.model_info['key']}_{self.model_info['type']}_{self.model_info['subtype']}"
                if engine_key not in self._model_cache:
                    self._model_cache[engine_key] = InferenceEngineFactory.create_engine(self.model_info)
                    print(f"【模型缓存】创建并缓存推理引擎: {engine_key}")
                self.engine = self._model_cache[engine_key]
                
                # 确定性能级别
                perf_level = "balanced"
                if has_video:
                    perf_level = "fast"
                elif has_images and image_max_size > 512:
                    perf_level = "quality"
                
                # 获取生成参数
                video_input = has_video
                text_input = bool(custom_prompt.strip())
                cache_key = f"{perf_level}_{video_input}_{text_input}_{self.model_info['subtype']}"
                
                if cache_key in self._perf_params_cache:
                    gen_params = self._perf_params_cache[cache_key].copy()
                else:
                    gen_params = self.engine.get_generation_params(perf_level, video_input, text_input)
                    # 视频处理特殊优化：调整参数以减少内存占用
                    if video_input:
                        # 降低批处理大小以避免内存不足
                        gen_params["max_tokens"] = min(gen_params["max_tokens"], 1024)
                        # 降低temperature以获得更稳定的输出
                        gen_params["temperature"] = max(gen_params["temperature"] - 0.1, 0.5)
                        print(f"【视频优化】视频处理模式，调整max_tokens={gen_params['max_tokens']}, temperature={gen_params['temperature']}")
                    self._perf_params_cache[cache_key] = gen_params.copy()
                
                # 应用用户自定义参数
                if parameters:
                    gen_params.update({k: v for k, v in parameters.items() if k != "state_uid"})
                
                gen_params["seed"] = seed
                
                if self.model_info["type"] == "omni":
                    # Omni 模型处理
                    if isinstance(self.engine, QwenOmniInferenceEngine):
                        # 处理音频输入：区分 ASR 输出（字典）和原始音频（tensor）
                        audio_input = None
                        if has_audio and audio is not None:
                            if isinstance(audio, dict):
                                # ASR 输出：检查是否包含波形数据
                                if "waveform" in audio:
                                    # 包含波形数据，可以用于多模态理解
                                    audio_input = audio
                                else:
                                    # 只有文本数据，不传递给多模态输入
                                    audio_input = None
                                    print("【Omni 音频】ASR 输出仅包含文本，已跳过音频多模态输入")
                            elif hasattr(audio, 'numel'):
                                # 原始音频 tensor，需要包装成字典
                                audio_input = {"waveform": audio, "sample_rate": 16000}
                        
                        # 文本生成模式下优化：跳过图像处理
                        if mode == "text":
                            messages = self.engine.build_messages(
                                system_prompt=system_prompt,
                                user_content=final_prompt
                            )
                            print("【Qwen3.5 优化】文本生成模式下使用简化消息格式")
                        else:
                            messages = self.engine.build_omni_messages(
                                system_prompt=system_prompt,
                                text=final_prompt,
                                images=images if has_images else None,
                                audio=audio_input,
                                max_size=image_max_size
                            )
                    else:
                        # 回退到标准处理
                        content = []
                        if final_prompt:
                            content.append({"type": "text", "text": final_prompt})
                        messages = self.engine.build_messages(system_prompt, content)
                elif self.model_info["subtype"] == "glm4v":
                    # GLM-4V系列 (仅视觉)
                    if isinstance(self.engine, GLM4VInferenceEngine):
                        messages = self.engine.build_vision_messages(
                            system_prompt, final_prompt, images, image_max_size
                        )
                else:
                    # 默认VL模型
                    content = []
                    if final_prompt:
                        content.append({"type": "text", "text": final_prompt})
                    
                    if images is not None and mode in ["images", "video"]:
                        if len(images.shape) == 3:
                            images = images.unsqueeze(0)
                        
                        for i in range(images.shape[0]):
                            # 处理PyTorch张量或numpy数组
                            if hasattr(images[i], 'cpu'):
                                img = images[i].cpu().numpy()
                            else:
                                img = images[i]
                            img_np = scale_image(img, image_max_size)
                            img_base64 = image2base64(img_np)
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            })
                    
                    messages = self.engine.build_messages(system_prompt, content)
                
                print(f"【推理】使用模型: {self.model_info['key']}, 消息数: {len(messages)}")
                
                # ========== 批量推理模式 ==========
                if mode == "batch_images" and has_images and images is not None and len(images.shape) >= 3:
                    num_images = images.shape[0] if len(images.shape) == 4 else 1
                    if num_images > 1:
                        print(f"【批量推理】启用批量模式，图片数量: {num_images}，组合方式: {batch_combination}")
                        
                        batch_results = self._process_batch_inference(
                            llama_model=llama_model,
                            images=images,
                            system_prompt=system_prompt,
                            final_prompt=final_prompt,
                            image_max_size=image_max_size,
                            batch_combination=batch_combination,
                            gen_params=gen_params,
                            audio_output_mode=audio_output_mode,
                            engine=self.engine,
                            model_info=self.model_info,
                            mode=mode
                        )
                        
                        if batch_results:
                            if batch_combination == "合并输出":
                                generated_text = " | ".join(batch_results)
                                output_list = batch_results
                            else:
                                generated_text = batch_results[0] if batch_results else ""
                                output_list = batch_results
                            
                            print(f"【批量推理】完成，生成 {len(batch_results)} 个结果")
                            
                            # 跳转到TTS处理部分
                            if audio_output_mode == "TTS音色美化输出" and enable_tts and audio_output is None and generated_text:
                                pass  # 继续到TTS处理
                            else:
                                # 直接返回结果
                                _uid = parameters.get("state_uid", None) if parameters else None
                                uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
                                return (generated_text, output_list, int(uid), None)
                
                # 执行推理（异步）
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                audio_output = None
                if not generated_text:
                    # 使用线程池执行推理
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self._run_inference,
                            llama_model=llama_model,
                            messages=messages,
                            gen_params=gen_params,
                            audio_output_mode=audio_output_mode,
                            omni_speaker_type=omni_speaker_type
                        )
                        generated_text, audio_output = future.result()
            
            # 处理TTS语音合成（独立TTS模型）- 仅在TTS音色美化输出模式下启用
            if audio_output_mode == "TTS音色美化输出" and enable_tts and audio_output is None and generated_text:
                print(f"【TTS】开始语音合成...")
                
                # 尝试使用TTS模型
                if tts_model is not None and hasattr(tts_model, 'synthesize'):
                    try:
                        # TTS模型缓存
                        tts_model_key = f"tts_{id(tts_model)}"
                        if tts_model_key not in self._tts_model_cache:
                            self._tts_model_cache[tts_model_key] = tts_model
                            print(f"【TTS缓存】缓存TTS模型")
                        cached_tts_model = self._tts_model_cache[tts_model_key]
                        
                        # 异步执行TTS合成
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(cached_tts_model.synthesize,
                                text=generated_text,
                                speed=tts_speed
                            )
                            audio_output = future.result()
                        
                        print(f"【TTS】synthesize返回: {type(audio_output)}, 内容: {str(audio_output)[:200]}")
                        if audio_output:
                            if isinstance(audio_output, dict) and "waveform" in audio_output and "sample_rate" in audio_output:
                                print(f"【TTS】音频格式正确: waveform={type(audio_output['waveform'])}, sample_rate={audio_output['sample_rate']}")
                            else:
                                print(f"【TTS】警告：音频格式不符合预期（需要dict包含waveform和sample_rate）")
                        print(f"【TTS】使用TTS模型合成成功")
                    except Exception as e:
                        print(f"【TTS】TTS模型合成失败: {str(e)}")
                        audio_output = None
                
                # 如果TTS模型失败或未提供，使用备选方案
                if audio_output is None:
                    audio_output = self._generate_fallback_audio(generated_text, omni_speaker_type, tts_speed)
                    if audio_output:
                        print(f"【TTS】使用备选方案生成音频")
            
            # 兼容输出，确保传统Comfy音频保存器可以接受
            if audio_output is None:
                # 如果没有音频输出，使用1秒静音波形作为安全回退
                audio_output = {"waveform": torch.zeros(int(24000 * 1.0), dtype=torch.float32), "sample_rate": 24000}
                print("【TTS兼容】音频输出为空，已生成1秒静音回退音频")

            if isinstance(audio_output, dict):
                waveform = audio_output.get("waveform")
                sample_rate = audio_output.get("sample_rate")

                # 支持numpy->torch类型转换
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                    print(f"【TTS兼容】waveform由numpy.ndarray转换为torch.Tensor，shape={waveform.shape}")
                elif isinstance(waveform, list):
                    waveform = torch.tensor(waveform)
                    print(f"【TTS兼容】waveform由list转换为torch.Tensor，长度={len(waveform)}")
                elif isinstance(waveform, torch.Tensor):
                    # 已经符合要求
                    pass
                else:
                    print(f"【TTS兼容】警告：waveform类型不支持 ({type(waveform)})")
                    waveform = None
                
                # 确保waveform是三维张量 [batch, channels, samples]
                if waveform is not None:
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
                        print(f"【TTS兼容】waveform已reshape为三维，shape={waveform.shape}")
                    elif waveform.dim() == 2:
                        waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
                        print(f"【TTS兼容】waveform已reshape为三维，shape={waveform.shape}")
                    audio_output["waveform"] = waveform

                # 默认sample_rate
                if sample_rate is None:
                    audio_output["sample_rate"] = 24000
                    print("【TTS兼容】sample_rate缺失，已默认设为24000")

            # 强制卸载
            if force_offload:
                mm.soft_empty_cache()
                # 清理所有缓存
                self.clear_all_caches()
            
            # 处理UID
            _uid = parameters.get("state_uid", None) if parameters else None
            uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
            
            return (generated_text, [generated_text], int(uid), audio_output)
        
        except Exception as e:
            error_message = ErrorHandler.handle_error(e, context={"mode": mode, "model_type": self.model_info.get("type", "unknown")})
            print(f"【处理错误】{str(e)}")
            return (error_message, [error_message], seed, None)
    
    @classmethod
    def _run_parallel_inference(cls, llm, tasks, params):
        """
        并行执行多个推理任务
        
        Args:
            llm: LLM 模型实例
            tasks: 任务列表，每个任务包含 messages 和其他参数
            params: 推理参数
            
        Returns:
            推理结果列表
        """
        results = []
        max_workers = min(4, os.cpu_count() or 4)  # 限制并发数，避免资源过度使用
        
        print(f"【并行处理】开始并行执行 {len(tasks)} 个推理任务，使用 {max_workers} 个线程")
        
        def inference_task(task):
            """单个推理任务"""
            try:
                messages = task.get('messages', [])
                task_params = {**params, **task.get('params', {})}
                
                # 生成缓存键
                cache_key = hash(str(messages) + str(task_params))
                
                # 尝试从缓存获取结果
                cached_result = cls.get_from_cache("_inference_cache", cache_key)
                if cached_result:
                    print("【并行处理】从缓存获取推理结果")
                    return cached_result
                
                # 执行推理
                result = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=task_params.get('max_tokens', 1024),
                    temperature=task_params.get('temperature', 0.7),
                    top_p=task_params.get('top_p', 0.9),
                    top_k=task_params.get('top_k', 40),
                    repeat_penalty=task_params.get('repeat_penalty', 1.1),
                    stop=task_params.get('stop', []),
                    stream=False
                )
                
                # 处理结果
                if result and 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']

                    
                    # 缓存结果
                    cls.add_to_cache("_inference_cache", cache_key, generated_text)
                    return generated_text
                return ""
            except Exception as e:
                print(f"【并行处理】任务执行失败: {e}")
                return f"处理失败: {str(e)}"
        
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(inference_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"【并行处理】获取结果失败: {e}")
                    results.append(f"处理失败: {str(e)}")
        
        print(f"【并行处理】完成 {len(results)} 个推理任务")
        return results
    
    @classmethod
    def get_recommended_model(cls, task_type, input_content=None):
        """
        根据任务类型和输入内容推荐合适的模型
        
        Args:
            task_type: 任务类型，如 "text_generation", "image_understanding", "audio_to_text", 等
            input_content: 输入内容，用于进一步分析任务需求
            
        Returns:
            推荐的模型类型和配置
        """
        # 任务类型到模型推荐的映射
        model_recommendations = {
            "text_generation": {
                "model_type": "general",
                "recommended_models": ["Qwen3.5", "Llama-3.1", "Gemma-3"],
                "params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024
                }
            },
            "image_understanding": {
                "model_type": "vision",
                "recommended_models": ["Qwen3-VL", "LLaVA-1.6", "MiniCPM-V-4.5"],
                "params": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                    "image_min_tokens": 1024,
                    "image_max_tokens": 1024
                }
            },
            "batch_image_understanding": {
                "model_type": "vision",
                "recommended_models": ["Qwen3-VL", "LLaVA-1.6"],
                "params": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 512,
                    "image_min_tokens": 512,
                    "image_max_tokens": 512
                }
            },
            "audio_to_text": {
                "model_type": "audio",
                "recommended_models": ["Qwen3-ASR", "Whisper"],
                "params": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 2048
                }
            },
            "text_to_audio": {
                "model_type": "audio",
                "recommended_models": ["Qwen3-TTS", "ElevenLabs"],
                "params": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": 512
                }
            },
            "multimodal_integration": {
                "model_type": "omni",
                "recommended_models": ["Qwen3.5", "Qwen2.5-Omni", "MiniCPM-O-4.5"],
                "params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024,
                    "image_min_tokens": 1024,
                    "image_max_tokens": 1024
                }
            },
            "video_understanding": {
                "model_type": "vision",
                "recommended_models": ["Qwen3-VL", "LLaVA-1.6"],
                "params": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 1536,
                    "image_min_tokens": 512,
                    "image_max_tokens": 512
                }
            }
        }
        
        # 根据任务类型获取推荐
        if task_type in model_recommendations:
            recommendation = model_recommendations[task_type]
            
            # 根据输入内容进一步调整推荐
            if input_content:
                # 检查输入长度
                input_length = len(str(input_content))
                if input_length > 1000:
                    # 长输入，推荐更大的模型
                    if "Qwen3.5" in recommendation["recommended_models"]:
                        recommendation["recommended_models"].insert(0, recommendation["recommended_models"].pop(recommendation["recommended_models"].index("Qwen3.5")))
                
                # 检查是否包含代码
                if "```" in str(input_content) or "import " in str(input_content):
                    # 代码相关任务，推荐支持代码的模型
                    if "Qwen3.5" in recommendation["recommended_models"]:
                        recommendation["recommended_models"].insert(0, recommendation["recommended_models"].pop(recommendation["recommended_models"].index("Qwen3.5")))
        else:
            # 默认推荐
            recommendation = {
                "model_type": "general",
                "recommended_models": ["Qwen3.5", "Llama-3.1"],
                "params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024
                }
            }
        
        print(f"【模型推荐】为任务类型 '{task_type}' 推荐模型: {recommendation['recommended_models'][0]}")
        return recommendation


# 节点映射
NODE_CLASS_MAPPINGS = {
    "llama_cpp_unified_inference": llama_cpp_unified_inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_unified_inference": "Llama CPP Unified Inference (VL)",    
}
