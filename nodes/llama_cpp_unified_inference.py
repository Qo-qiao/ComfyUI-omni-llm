# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Unified Inference Node

统一推理节点，支持VL模型的多模态推理
支持文本生成、图像理解、音频处理和视频分析等多种模式

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
    HARDWARE_INFO, image2base64, scale_image,
    mm, BaseInferenceEngine, InferenceEngineFactory,
    convert_audio_to_wav_bytes, create_audio_data_uri,
    filter_stderr
)


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
        if error_message is None:
            return "unknown"
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

from support.preset_universal import (
    PROMPT_EXPANDER_ZH,
    PROMPT_EXPANDER_EN,
    EDIT_COMBINED_ZH,
    EDIT_COMBINED_EN,
    IDEOGRAM4_ZH,
    IDEOGRAM4_EN,
)

from support.preset_image_reverse import (
    IMAGE_REVERSE_TAGS_ZH,
    IMAGE_REVERSE_DESCRIBE_ZH,
    IMAGE_REVERSE_TAGS_EN,
    IMAGE_REVERSE_DESCRIBE_EN,
)

from support.preset_anime import (
    ILLUSTRIOUS_ZH,
    ILLUSTRIOUS_EN,
    ANIME_PROMPT_ZH,
    ANIME_PROMPT_EN,
    THICKPAINT_ROLE_ZH,
    THICKPAINT_ROLE_EN,
)

from support.preset_asia_portrait import (
    REALISTIC_FEMALE_ZH,
    REALISTIC_FEMALE_EN,
    REALISTIC_MALE_ZH,
    REALISTIC_MALE_EN,
)
from support.preset_western_portrait import (
    WESTERN_FEMALE_ZH,
    WESTERN_FEMALE_EN,
    WESTERN_MALE_ZH,
    WESTERN_MALE_EN,
)

from support.preset_hyper_realistic import (
    HYPER_REALISTIC_FEMALE_ZH,
    HYPER_REALISTIC_FEMALE_EN,
    HYPER_REALISTIC_MALE_ZH,
    HYPER_REALISTIC_MALE_EN, 
)

from support.preset_elderly_portrait import (
    YOUNG_BOY_PORTRAIT_ZH,
    YOUNG_BOY_PORTRAIT_EN,
    MIDDLE_ELDERLY_FEMALE_PORTRAIT_ZH,
    MIDDLE_ELDERLY_FEMALE_PORTRAIT_EN,
    MIDDLE_ELDERLY_MALE_PORTRAIT_ZH,
    MIDDLE_ELDERLY_MALE_PORTRAIT_EN,
)

from support.preset_building import (
    SCENE_DESIGN_ZH,
    SCENE_DESIGN_EN,
    INTERIOR_DESIGN_ZH,
    INTERIOR_DESIGN_EN,
    ARCHITECTURE_RENDERING_ZH,
    ARCHITECTURE_RENDERING_EN,
)

from support.preset_design import (
    ART_ILLUSTRATION_ZH,
    ART_ILLUSTRATION_EN,
    POSTER_DESIGN_ZH,
    POSTER_DESIGN_EN,
    ECOMMERCE_ZH,
    ECOMMERCE_EN,
    FOOD_PHOTOGRAPHY_ZH,
    FOOD_PHOTOGRAPHY_EN,
)

from support.preset_video import (
    UNIVERSAL_VIDEO_ZH,
    UNIVERSAL_VIDEO_EN,
    CONTINUING_I2V_ZH,
    CONTINUING_I2V_EN,
    CONTINUING_FLF2V_ZH,
    CONTINUING_FLF2V_EN,
    CONTINUING_MULTI_STORYBOARD_ZH,
    CONTINUING_MULTI_STORYBOARD_EN,
)

from support.preset_video_reverse import (
    VIDEO_FRAME_SEQUENCE_ZH,
    VIDEO_FRAME_SEQUENCE_EN,
    VIDEO_TO_PROMPT_ZH,
    VIDEO_TO_PROMPT_EN,
    VIDEO_SCENE_BREAKDOWN_ZH,
    VIDEO_SCENE_BREAKDOWN_EN,
    VIDEO_SUBTITLE_FORMAT_ZH,
    VIDEO_SUBTITLE_FORMAT_EN,
)

from support.preset_audio import (
    MULTI_SPEAKER_DIALOGUE_ZH,
    MULTI_SPEAKER_DIALOGUE_EN,
    LYRICS_CREATION_ZH,
    LYRICS_CREATION_EN,
)

class VideoProcessor:
    """视频处理器 - 处理视频输入、视频帧提取和视频理解功能"""

    def __init__(self):
        """初始化视频处理器"""
        self.use_torchcodec = False
        self.torchcodec = None
        self._try_import_torchcodec()

    def _try_import_torchcodec(self):
        """尝试导入torchcodec作为备选视频处理库"""
        try:
            import torchcodec
            self.torchcodec = torchcodec
            self.use_torchcodec = True
            print("【视频处理】使用torchcodec进行视频处理")
        except (ImportError, RuntimeError):
            self.use_torchcodec = False
            print("【视频处理】torchcodec不可用，使用默认处理方式")

    def _check_video_input(self, video) -> bool:
        """检查视频输入是否有效"""
        if video is None:
            return False
        
        # 检查是否为 tensor（直接的图像序列）
        if hasattr(video, 'numel'):
            return video.numel() > 0
        
        # 检查是否为 VideoOutput 对象（包含 frames 属性）
        if hasattr(video, 'frames'):
            frames = video.frames
            if hasattr(frames, 'numel'):
                return frames.numel() > 0
        
        # 检查是否为字典格式
        if isinstance(video, dict) and 'frames' in video:
            frames = video['frames']
            if hasattr(frames, 'numel'):
                return frames.numel() > 0
        
        return False

    def get_video_optimization_params(self, base_params: Dict) -> Dict:
        """获取视频处理优化参数"""
        optimized_params = base_params.copy()
        
        # 视频处理需要更多显存，降低参数以避免内存不足
        optimized_params["max_new_tokens"] = min(optimized_params.get("max_new_tokens", 1024), 768)
        optimized_params["temperature"] = max(optimized_params.get("temperature", 0.7) - 0.05, 0.6)
        optimized_params["top_p"] = max(optimized_params.get("top_p", 0.95) - 0.05, 0.9)
        
        # 关键优化：降低n_batch以节省显存（视频处理需要更多显存）
        current_n_batch = optimized_params.get("n_batch", 512)
        optimized_params["n_batch"] = max(64, current_n_batch // 4)  # 降低到原来的1/4
        
        print(f"【视频优化】视频处理模式，调整max_new_tokens={optimized_params['max_new_tokens']}, temperature={optimized_params['temperature']}, top_p={optimized_params['top_p']}, n_batch={optimized_params['n_batch']}")
        return optimized_params

    def get_video_perf_level(self) -> str:
        """获取视频处理的性能级别"""
        return "fast"

    def process_video_input(self, video, max_frames: int = 16, sampling_method: str = "auto", manual_indices: str = "") -> Optional[Dict]:
        """处理视频输入，返回处理后的视频数据"""
        if not self._check_video_input(video):
            return None

        try:
            frames = self._extract_frames(video, max_frames, sampling_method, manual_indices)
            if frames:
                frames = self._resize_frames(frames, max_size=384)
            video_output = {
                "frames": frames,
                "frame_count": len(frames),
                "max_frames": max_frames
            }
            print(f"视频处理成功，提取了 {len(frames)} 帧")
            return video_output
        except Exception as e:
            print(f"处理视频输入时出错: {e}")
            return None

    def _extract_frames(self, video, max_frames: int, sampling_method: str, manual_indices: str) -> List:
        """提取视频帧"""
        frames = []
        max_frames = min(max_frames, 15)

        # 获取实际的视频帧数据（处理 VideoOutput 对象或直接 tensor）
        video_frames = video
        
        # 如果是 VideoOutput 对象，获取其 frames 属性
        if hasattr(video, 'frames'):
            video_frames = video.frames
        
        # 如果是字典格式，获取 frames 字段
        if isinstance(video, dict) and 'frames' in video:
            video_frames = video['frames']

        if hasattr(video_frames, 'shape') and video_frames.ndim == 4:
            total_frames = video_frames.shape[0]
            if sampling_method == "auto":
                step = max(1, total_frames // max_frames)
                indices = list(range(0, total_frames, step))[:max_frames]
            else:
                try:
                    indices = [int(i) for i in manual_indices.split(',') if i.strip()]
                    indices = [i for i in indices if 0 <= i < total_frames][:max_frames]
                except Exception:
                    step = max(1, total_frames // max_frames)
                    indices = list(range(0, total_frames, step))[:max_frames]

            if video_frames.is_cuda:
                for idx in indices:
                    frame = video_frames[idx].detach().cpu()
                    frames.append(frame)
            else:
                for idx in indices:
                    frames.append(video_frames[idx])

        if not frames and hasattr(video_frames, 'shape') and video_frames.ndim == 4 and video_frames.shape[0] > 0:
            frames.append(video_frames[0])

        return frames

    def _resize_frames(self, frames: List, max_size: int = 256) -> List:
        """调整帧大小以减少计算量和显存使用"""
        resized = []
        for frame in frames:
            if hasattr(frame, 'shape'):
                if frame.shape[0] > max_size or frame.shape[1] > max_size:
                    scale = max_size / max(frame.shape[0], frame.shape[1])
                    new_h, new_w = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
                    frame = self._resize_tensor(frame, new_h, new_w)
            resized.append(frame)
        return resized

    def _resize_tensor(self, tensor, new_h: int, new_w: int):
        """调整张量大小"""
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            resized = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            return resized.squeeze(0).permute(1, 2, 0)
        return tensor

    def prepare_video_for_inference(self, video_data: Dict) -> Optional[Dict]:
        """准备视频数据用于推理"""
        if not video_data or "frames" not in video_data:
            return None

        try:
            return video_data
        except Exception as e:
            print(f"准备视频数据时出错: {e}")
            return None


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
    
    def _filter_thinking_content(self, text):
        """
        清理模型输出中的思考标记
        
        Args:
            text: 原始生成文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return text
        
        # 优先处理Qwen3.5/3.6原生思考格式 - 结束标记后为实际回答
        thinking_end_markers = [
            '<|end_of_thinking|>',
            '<|end_of_solution|>',
            '<|finish_reason|>',
        ]
        
        for marker in thinking_end_markers:
            if marker in text:
                parts = text.split(marker)
                if len(parts) >= 2:
                    actual_response = parts[-1].strip()
                    actual_response = re.sub(r'^\s*\n+', '', actual_response)
                    if actual_response:
                        return self._clean_residual_markers(actual_response)
        
        # 清理残留标记
        return self._clean_residual_markers(text)
    
    def _clean_residual_markers(self, text):
        """
        清理文本中残留的思考标记
        
        Args:
            text: 待清理的文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return text
        
        markers_to_remove = [
            '<thinking>', '</thinking>',
            '<think>', '</think>',
            '<analysis>', '</analysis>',
            '<|end_of_thinking|>',
            '<|end_of_solution|>',
            '<|finish_reason|>',
        ]
        
        result = text
        for marker in markers_to_remove:
            result = result.replace(marker, '')
        
        result = re.sub(r'\n\s*\n', '\n', result)
        result = re.sub(r'^\s+|\s+$', '', result)
        
        return result
    
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
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保所有GPU操作完成
                torch.cuda.empty_cache()  # 清空GPU缓存
                gc.collect()  # 强制垃圾回收
                print("【内存管理】清理了 GPU 缓存并执行垃圾回收")
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
        "elderly_male": 9, "dialect_female": 10
    }
    
    # 初始化预设提示词字典
    preset_prompts = {}

    # 添加基础预设
    preset_prompts["Empty - Nothing"] = ""

    # 添加分类预设提示词（顺序与 prompt_enhancer_preset_zh.py / prompt_enhancer_preset_en.py 保持一致）
    preset_prompts["[Reverse] Tags"] = "IMAGE_REVERSE_TAGS"
    preset_prompts["[Reverse] Describe"] = "IMAGE_REVERSE_DESCRIBE"
    preset_prompts["[Normal] Expand"] = "PROMPT_EXPANDER"
    preset_prompts["[Anime] Expand Tags"] = "ILLUSTRIOUS"
    preset_prompts["[Anime] Prompt Expand"] = "ANIME_PROMPT"
    preset_prompts["[Anime] Thick Paint Role"] = "THICKPAINT_ROLE"
    preset_prompts["[Portrait] Asian Female"] = "REALISTIC_FEMALE"
    preset_prompts["[Portrait] Asian Male"] = "REALISTIC_MALE"
    preset_prompts["[Portrait] Western Female"] = "WESTERN_FEMALE"
    preset_prompts["[Portrait] Western Male"] = "WESTERN_MALE"
    preset_prompts["[Portrait] Influencer"] = "INFLUENCER_PORTRAIT"
    preset_prompts["[Portrait] Male Portrait"] = "MALE_PORTRAIT"
    preset_prompts["[Portrait] Young Boy"] = "YOUNG_BOY_PORTRAIT"
    preset_prompts["[Portrait] Middle Elderly Female"] = "MIDDLE_ELDERLY_FEMALE_PORTRAIT"
    preset_prompts["[Portrait] Middle Elderly Male"] = "MIDDLE_ELDERLY_MALE_PORTRAIT"
    preset_prompts["[Design] Art Illustration"] = "ART_ILLUSTRATION"
    preset_prompts["[Design] Poster Design"] = "POSTER_DESIGN"
    preset_prompts["[Design] Scene Design"] = "SCENE_DESIGN"
    preset_prompts["[Design] Interior Design"] = "INTERIOR_DESIGN"
    preset_prompts["[Design] Architecture Rendering"] = "ARCHITECTURE_RENDERING"
    preset_prompts["[Design] Ecommerce Product"] = "ECOMMERCE"
    preset_prompts["[Design] Food Photography"] = "FOOD_PHOTOGRAPHY"
    preset_prompts["[Edit] Combined"] = "EDIT_COMBINED"
    preset_prompts["[Text to Video] Universal"] = "UNIVERSAL_VIDEO"
    preset_prompts["[Image to Video] I2V"] = "CONTINUING_I2V"
    preset_prompts["[Image to Video] FLF2V"] = "CONTINUING_FLF2V"
    preset_prompts["[Image to Video] Multi Storyboard"] = "CONTINUING_MULTI_STORYBOARD"
    preset_prompts["[Video Analysis] Frame Sequence"] = "VIDEO_FRAME_SEQUENCE"
    preset_prompts["[Video Analysis] Reverse Prompt"] = "VIDEO_TO_PROMPT"
    preset_prompts["[Video Analysis] Scene Breakdown"] = "VIDEO_SCENE_BREAKDOWN"
    preset_prompts["[Video Analysis] Subtitle"] = "VIDEO_SUBTITLE_FORMAT"
    preset_prompts["[Audio] Multi-Speaker Dialogue"] = "MULTI_SPEAKER_DIALOGUE"
    preset_prompts["[Music] Lyrics Creation"] = "LYRICS_CREATION"
    preset_prompts["[Design] Ideogram-4"] = "IDEOGRAM4"

    preset_tags = list(preset_prompts.keys())
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ========== 模型配置 ==========
                "llama_model": ("LLAMACPPMODEL", {"tooltip": "加载的VL模型，用于图像理解和文本生成"}),
                
                # ========== 推理模式 ==========
                "inference_mode": ([
                        "[基础] 文本生成 (Text Generation)",
                        "[基础] 图像理解 (Image Understanding)",
                        "[基础] 批量图像理解 (Batch Image Understanding)",
                        "[基础] 音频转文本 (Audio to Text)",
                        "[基础] 文本转音频 (Text to Audio)",
                        "[高级] 视频理解 (Video Understanding)"
                    ], {
                        "default": "[基础] 文本生成 (Text Generation)",
                        "tooltip": "选择推理模式：\n• [基础] 文本生成：使用语言模型生成文本内容\n• [基础] 图像理解：处理单张图像内容并生成描述\n• [基础] 批量图像理解：一次性处理多张图片，减少推理调用次数\n• [基础] 音频转文本：使用ASR模型将音频转换为文本\n• [基础] 文本转音频：生成文本后使用TTS模型转换为语音\n• [高级] 视频理解：从视频中提取帧并进行分析"
                    }),
                
                # ========== 提示词配置 ==========
                "preset_prompt": (s.preset_tags, {"default": s.preset_tags[1], "tooltip": "选择预设提示词模板：\n• Empty - Nothing：无预设，完全自定义\n• [Reverse] Tags：反推XL标签格式提示词\n• [Reverse] Describe：通用图片反推提示词\n• [Normal] Expand：通用提示词文本优化\n• [Anime] Expand Tags：二次元角色风格文本优化\n• [Anime] Prompt Expand：二次元内容文本优化\n• [Anime] Thick Paint Role：角色厚涂CG文本优化\n• [Portrait] Asian Female：真实亚洲女性人像文本优化\n• [Portrait] Asian Male：真实亚洲男性人像文本优化\n• [Portrait] Western Female：真实欧美女性人像文本优化\n• [Portrait] Western Male：真实欧美男性人像文本优化\n• [Portrait] Influencer：超写实女性人像文本优化\n• [Portrait] Male Portrait：超写实男性人像文本优化\n• [Portrait] Young Boy：儿童人像文本优化\n• [Portrait] Middle Elderly Female：中老年女性人像文本优化\n• [Portrait] Middle Elderly Male：中老年男性人像文本优化\n• [Design] Art Illustration：艺术插画文本优化\n• [Design] Poster Design：海报设计文本优化\n• [Design] Scene Design：场景设计文本优化\n• [Design] Interior Design：室内设计文本优化\n• [Design] Architecture Rendering：建筑外观与园林渲染\n• [Design] Ecommerce Product：电商产品文本优化\n• [Design] Food Photography：美食摄影文本优化\n• [Edit] Combined：图像编辑文本优化\n• [Text to Video] Universal：扩写视频文本内容\n• [Image to Video] I2V：根据图片扩写视频文本内容\n• [Image to Video] FLF2V：根据首尾帧图片扩写视频文本内容\n• [Image to Video] Multi Storyboard:根据图片与文本扩写视频文本内容\n• [Video Analysis] Frame Sequence：帧分析视频内容\n• [Video Analysis] Reverse Prompt：通用视频反推提示词\n• [Video Analysis] Scene Breakdown：反推各分镜场景内容\n• [Video Analysis] Subtitle：结合字幕与视频反推情绪化文本\n• [Audio] Multi-Speaker Dialogue：多人对话情绪化优化\n• [Music] Lyrics Creation：歌词创作文本优化\n• [Design] Ideogram-4：json结构化提示词"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "你是一位优秀的AI提示词处理专家。", "tooltip": "系统提示词，定义AI助手的角色和行为，可包含预设模板占位符#和自定义内容"}),
                "text_input": ("STRING", {"default": "", "multiline": True, "tooltip": "用户输入文本，作为对话的用户消息内容"}),
                
                # ========== 语言设置 ==========
                "prompt_language": (["中文", "English"], {"default": "中文", "tooltip": "预设提示词的语言"}),
                "response_language": (["中文", "English"], {"default": "中文", "tooltip": "AI回复的语言"}),
                
                # ========== 输出格式设置 ==========
                "output_format": (["natural", "structured"], {
                    "default": "natural",
                    "tooltip": "输出格式控制：\n• natural：以自然段落格式输出纯文本内容\n• structured：输出结构化文本内容"
                }),
                "enable_constraints": ("BOOLEAN", {"default": False, "tooltip": "启用预设模板中的正向约束：\n• 正向约束：会添加到提示词最前面，引导模型生成更符合要求的内容"}),
                "enable_negative_prompts": ("BOOLEAN", {"default": False, "tooltip": "启用预设模板中的负向提示词：\n• 负向提示词：会追加到提示词末尾，避免模型生成不想要的内容"}),
                
                # ========== 视频处理参数 ==========
                "video_max_frames": ("INT", {"default": 16, "min": 2, "max": 1024, "step": 1, 
                                              "tooltip": "视频模式：最大提取帧数"}),
                "video_sampling": (["auto", "manual"], 
                                  {"default": "auto", 
                                   "tooltip": "视频帧采样方式：\n• auto：自动均衡提取视频帧\n• manual：自定义要提取的帧"}),
                "video_manual_indices": ("STRING", {"default": "", 
                                                     "placeholder": "例如: 0,10,20 或 0-10", 
                                                     "tooltip": "手动模式下的帧索引，仅在手动采样时生效"}),
                
                # ========== 图像处理参数 ==========
                "image_max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "step": 64,
                                           "tooltip": "图像处理的最大边长（像素），较大的值需要更多显存"}),
                
                # ========== 批量输出选项 ==========
                "batch_combination": (["separate", "combined"], {
                    "default": "separate",
                    "tooltip": "批量模式的结果处理方式：\n• separate：每张图片单独输出结果\n• combined：所有结果合并为一个输出"
                }),

                
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
                "queue_handler": ("*", {"tooltip": "队列处理器"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "AUDIO", "STRING")
    RETURN_NAMES = ("output", "output_list", "state_uid", "audio", "example_output")
    OUTPUT_IS_LIST = (False, True, False, False, False)
    FUNCTION = "process"
    CATEGORY = "omni-llm"
    
    def __init__(self):
        self.engine = None
        self.model_info = None
        self.video_processor = VideoProcessor()
    
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
        
        # 简单的模型信息 - 默认为VL模型
        model_info = {
            "key": "default",
            "type": "vl",
            "subtype": "default",
            "supports_audio": False,
            "supports_vision": True,
            "file_formats": [".gguf"]
        }
        
        # 缓存结果
        self._model_type_cache[cache_key] = model_info
        
        return model_info
    
    def get_preset_text_by_language(self, preset_key, language, output_format="JSON格式", input_mode="text"):
        """
        根据语言和输出格式获取预设提示词文本

        优先使用新的 output_format_suffix 字段，保持向后兼容性

        Args:
            preset_key: 预设键名
            language: 语言（"中文" 或 "English"）
            output_format: 输出格式（"structured" 或 "natural"）
            input_mode: 输入模式（"text"、"images"、"video"）

        Returns:
            str: 格式化后的预设提示词文本
        """
        if language == "中文":
            preset_map = {
                "IMAGE_REVERSE_TAGS": IMAGE_REVERSE_TAGS_ZH,
                "IMAGE_REVERSE_DESCRIBE": IMAGE_REVERSE_DESCRIBE_ZH,
                "PROMPT_EXPANDER": PROMPT_EXPANDER_ZH,
                "ILLUSTRIOUS": ILLUSTRIOUS_ZH,
                "ANIME_PROMPT": ANIME_PROMPT_ZH,
                "THICKPAINT_ROLE": THICKPAINT_ROLE_ZH,                
                "REALISTIC_FEMALE": REALISTIC_FEMALE_ZH,
                "REALISTIC_MALE": REALISTIC_MALE_ZH,
                "WESTERN_FEMALE": WESTERN_FEMALE_ZH,
                "WESTERN_MALE": WESTERN_MALE_ZH,
                "HYPER_REALISTIC_FEMALE": HYPER_REALISTIC_FEMALE_ZH,
                "HYPER_REALISTIC_MALE": HYPER_REALISTIC_MALE_ZH,
                "YOUNG_BOY_PORTRAIT": YOUNG_BOY_PORTRAIT_ZH,
                "MIDDLE_ELDERLY_FEMALE_PORTRAIT": MIDDLE_ELDERLY_FEMALE_PORTRAIT_ZH,
                "MIDDLE_ELDERLY_MALE_PORTRAIT": MIDDLE_ELDERLY_MALE_PORTRAIT_ZH,
                "ART_ILLUSTRATION": ART_ILLUSTRATION_ZH,
                "POSTER_DESIGN": POSTER_DESIGN_ZH,
                "SCENE_DESIGN": SCENE_DESIGN_ZH,
                "INTERIOR_DESIGN": INTERIOR_DESIGN_ZH,
                "ARCHITECTURE_RENDERING": ARCHITECTURE_RENDERING_ZH,
                "ECOMMERCE": ECOMMERCE_ZH,
                "FOOD_PHOTOGRAPHY": FOOD_PHOTOGRAPHY_ZH,
                "EDIT_COMBINED": EDIT_COMBINED_ZH,
                "UNIVERSAL_VIDEO": UNIVERSAL_VIDEO_ZH,
                "CONTINUING_I2V": CONTINUING_I2V_ZH,
                "CONTINUING_FLF2V": CONTINUING_FLF2V_ZH,
                "CONTINUING_MULTI_STORYBOARD": CONTINUING_MULTI_STORYBOARD_ZH,
                "VIDEO_FRAME_SEQUENCE": VIDEO_FRAME_SEQUENCE_ZH,
                "VIDEO_TO_PROMPT": VIDEO_TO_PROMPT_ZH,
                "VIDEO_SCENE_BREAKDOWN": VIDEO_SCENE_BREAKDOWN_ZH,
                "VIDEO_SUBTITLE_FORMAT": VIDEO_SUBTITLE_FORMAT_ZH,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_ZH,
                "LYRICS_CREATION": LYRICS_CREATION_ZH,
                "IDEOGRAM4": IDEOGRAM4_ZH,
            }
        else:
            preset_map = {
                "IMAGE_REVERSE_TAGS": IMAGE_REVERSE_TAGS_EN,
                "IMAGE_REVERSE_DESCRIBE": IMAGE_REVERSE_DESCRIBE_EN,
                "PROMPT_EXPANDER": PROMPT_EXPANDER_EN,
                "ILLUSTRIOUS": ILLUSTRIOUS_EN,
                "ANIME_PROMPT": ANIME_PROMPT_EN,
                "THICKPAINT_ROLE": THICKPAINT_ROLE_EN,
                "REALISTIC_FEMALE": REALISTIC_FEMALE_EN,
                "REALISTIC_MALE": REALISTIC_MALE_EN,
                "WESTERN_FEMALE": WESTERN_FEMALE_EN,
                "WESTERN_MALE": WESTERN_MALE_EN,
                "HYPER_REALISTIC_FEMALE": HYPER_REALISTIC_FEMALE_EN,
                "HYPER_REALISTIC_MALE": HYPER_REALISTIC_MALE_EN,
                "YOUNG_BOY_PORTRAIT": YOUNG_BOY_PORTRAIT_EN,
                "MIDDLE_ELDERLY_FEMALE_PORTRAIT": MIDDLE_ELDERLY_FEMALE_PORTRAIT_EN,
                "MIDDLE_ELDERLY_MALE_PORTRAIT": MIDDLE_ELDERLY_MALE_PORTRAIT_EN,
                "ART_ILLUSTRATION": ART_ILLUSTRATION_EN,
                "POSTER_DESIGN": POSTER_DESIGN_EN,
                "SCENE_DESIGN": SCENE_DESIGN_EN,
                "INTERIOR_DESIGN": INTERIOR_DESIGN_EN,
                "ARCHITECTURE_RENDERING": ARCHITECTURE_RENDERING_EN,
                "ECOMMERCE": ECOMMERCE_EN,
                "FOOD_PHOTOGRAPHY": FOOD_PHOTOGRAPHY_EN,
                "EDIT_COMBINED": EDIT_COMBINED_EN,
                "UNIVERSAL_VIDEO": UNIVERSAL_VIDEO_EN,
                "CONTINUING_I2V": CONTINUING_I2V_EN,
                "CONTINUING_FLF2V": CONTINUING_FLF2V_EN,
                "CONTINUING_MULTI_STORYBOARD": CONTINUING_MULTI_STORYBOARD_EN,
                "VIDEO_FRAME_SEQUENCE": VIDEO_FRAME_SEQUENCE_EN,
                "VIDEO_TO_PROMPT": VIDEO_TO_PROMPT_EN,
                "VIDEO_SCENE_BREAKDOWN": VIDEO_SCENE_BREAKDOWN_EN,
                "VIDEO_SUBTITLE_FORMAT": VIDEO_SUBTITLE_FORMAT_EN,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_EN,
                "LYRICS_CREATION": LYRICS_CREATION_EN,
                "IDEOGRAM4": IDEOGRAM4_EN,
            }

        preset = preset_map.get(preset_key, None)
        if preset is None:
            return preset_key

        # 获取基础模板，优先使用不同格式的专用模板
        if output_format == "natural" and "input_template_natural" in preset:
            base_template = preset.get("input_template_natural")
        elif output_format == "structured" and "input_template_structured" in preset:
            base_template = preset.get("input_template_structured")
        else:
            base_template = preset.get("input_template", preset_key)

        # 替换模板中的 {mode} 占位符为实际的输入模式（video/image/text）
        base_template = base_template.replace("{mode}", input_mode)

        # 构建完整的提示词：基础模板 + 约束条件 + 任务要求 + 输出格式后缀
        
        # 添加约束条件
        constraints = preset.get("constraints", {})
        if constraints:
            constraints_text = "\n\n**【约束条件】**\n"
            for key, value in constraints.items():
                if isinstance(value, str):
                    constraints_text += f"- {key}：{value}\n"
                else:
                    constraints_text += f"- {key}：{str(value)}\n"
            base_template += constraints_text
        
        # 添加任务要求
        task_requirements = preset.get("task_requirements", [])
        if task_requirements:
            requirements_text = "\n\n**【任务要求】**\n"
            for i, requirement in enumerate(task_requirements, 1):
                requirements_text += f"{i}. {requirement}\n"
            base_template += requirements_text
        
        # 添加四要素组合法（歌词创作等预设专用）
        four_element_method = preset.get("four_element_method", {})
        if four_element_method:
            method_text = "\n\n**【四要素组合法】**\n"
            for key, value in four_element_method.items():
                if isinstance(value, list):
                    method_text += f"- {key}：{', '.join(str(v) for v in value)}\n"
                elif isinstance(value, dict):
                    method_text += f"- {key}：\n"
                    for k2, v2 in value.items():
                        method_text += f"  - {k2}：{v2}\n"
                else:
                    method_text += f"- {key}：{value}\n"
            base_template += method_text
        
        # 添加情绪示例（对话等预设专用）
        emotion_examples = preset.get("emotion_examples", {})
        if emotion_examples:
            emotion_text = "\n\n**【情绪示例】**\n"
            for emotion, example in emotion_examples.items():
                emotion_text += f"- {emotion}：{example}\n"
            base_template += emotion_text
        
        # 添加音色映射（多人对话预设专用）
        voice_mapping = preset.get("voice_mapping", {})
        if voice_mapping:
            voice_text = "\n\n**【音色映射】**\n"
            for voice, info in voice_mapping.items():
                if isinstance(info, dict):
                    voice_text += f"- {voice}：speaker_id={info.get('speaker_id', '?')}, {info.get('description', '')}\n"
                else:
                    voice_text += f"- {voice}：{info}\n"
            base_template += voice_text
        
        # 添加通用原则（编辑提示等预设专用）
        general_principles = preset.get("general_principles", [])
        if general_principles:
            principles_text = "\n\n**【通用原则】**\n"
            for i, principle in enumerate(general_principles, 1):
                principles_text += f"{i}. {principle}\n"
            base_template += principles_text
        
        # 添加任务规则（编辑提示等预设专用）
        task_rules = preset.get("task_rules", {})
        if task_rules:
            rules_text = "\n\n**【任务规则】**\n"
            for key, value in task_rules.items():
                rules_text += f"- {key}：{value}\n"
            base_template += rules_text

        # 优先使用新的 output_format_suffix 字段
        suffix_map = preset.get("output_format_suffix", {})
        if output_format in suffix_map:
            return base_template + suffix_map[output_format]

        # 向后兼容：检查旧字段
        if output_format == "文本格式" and "input_template_text" in preset:
            return preset.get("input_template_text", base_template)

        return base_template
    
    def get_preset_examples(self, preset_key, language, output_format="natural", custom_prompt=""):
        """
        获取预设模板的示例内容
        
        Args:
            preset_key: 预设键名
            language: 语言（"中文" 或 "English"）
            output_format: 输出格式（"natural" 或 "structured"）
            custom_prompt: 用户输入的自定义提示词，用于匹配示例类别
            
        Returns:
            str: 格式化后的示例内容，随机显示一条；如果匹配到相同类型，显示对应示例
        """
        import random
        
        if language == "中文":
            preset_map = {
                "IMAGE_REVERSE_TAGS": IMAGE_REVERSE_TAGS_ZH,
                "IMAGE_REVERSE_DESCRIBE": IMAGE_REVERSE_DESCRIBE_ZH,
                "PROMPT_EXPANDER": PROMPT_EXPANDER_ZH,
                "ILLUSTRIOUS": ILLUSTRIOUS_ZH,
                "ANIME_PROMPT": ANIME_PROMPT_ZH,
                "THICKPAINT_ROLE": THICKPAINT_ROLE_ZH,
                "REALISTIC_FEMALE": REALISTIC_FEMALE_ZH,
                "REALISTIC_MALE": REALISTIC_MALE_ZH,
                "WESTERN_FEMALE": WESTERN_FEMALE_ZH,
                "WESTERN_MALE": WESTERN_MALE_ZH,
                "HYPER_REALISTIC_FEMALE": HYPER_REALISTIC_FEMALE_ZH,
                "HYPER_REALISTIC_MALE": HYPER_REALISTIC_MALE_ZH,
                "YOUNG_BOY_PORTRAIT": YOUNG_BOY_PORTRAIT_ZH,
                "MIDDLE_ELDERLY_FEMALE_PORTRAIT": MIDDLE_ELDERLY_FEMALE_PORTRAIT_ZH,
                "MIDDLE_ELDERLY_MALE_PORTRAIT": MIDDLE_ELDERLY_MALE_PORTRAIT_ZH,
                "ART_ILLUSTRATION": ART_ILLUSTRATION_ZH,
                "POSTER_DESIGN": POSTER_DESIGN_ZH,
                "SCENE_DESIGN": SCENE_DESIGN_ZH,
                "INTERIOR_DESIGN": INTERIOR_DESIGN_ZH,
                "ARCHITECTURE_RENDERING": ARCHITECTURE_RENDERING_ZH,
                "ECOMMERCE": ECOMMERCE_ZH,
                "FOOD_PHOTOGRAPHY": FOOD_PHOTOGRAPHY_ZH,
                "EDIT_COMBINED": EDIT_COMBINED_ZH,
                "UNIVERSAL_VIDEO": UNIVERSAL_VIDEO_ZH,
                "CONTINUING_I2V": CONTINUING_I2V_ZH,
                "CONTINUING_FLF2V": CONTINUING_FLF2V_ZH,
                "CONTINUING_MULTI_STORYBOARD": CONTINUING_MULTI_STORYBOARD_ZH,
                "VIDEO_FRAME_SEQUENCE": VIDEO_FRAME_SEQUENCE_ZH,
                "VIDEO_TO_PROMPT": VIDEO_TO_PROMPT_ZH,
                "VIDEO_SCENE_BREAKDOWN": VIDEO_SCENE_BREAKDOWN_ZH,
                "VIDEO_SUBTITLE_FORMAT": VIDEO_SUBTITLE_FORMAT_ZH,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_ZH,
                "LYRICS_CREATION": LYRICS_CREATION_ZH,
                "IDEOGRAM4": IDEOGRAM4_ZH,
            }
        else:
            preset_map = {
                "IMAGE_REVERSE_TAGS": IMAGE_REVERSE_TAGS_EN,
                "IMAGE_REVERSE_DESCRIBE": IMAGE_REVERSE_DESCRIBE_EN,
                "PROMPT_EXPANDER": PROMPT_EXPANDER_EN,
                "ILLUSTRIOUS": ILLUSTRIOUS_EN,
                "ANIME_PROMPT": ANIME_PROMPT_EN,
                "THICKPAINT_ROLE": THICKPAINT_ROLE_EN,
                "REALISTIC_FEMALE": REALISTIC_FEMALE_EN,
                "REALISTIC_MALE": REALISTIC_MALE_EN,
                "WESTERN_FEMALE": WESTERN_FEMALE_EN,
                "WESTERN_MALE": WESTERN_MALE_EN,
                "HYPER_REALISTIC_FEMALE": HYPER_REALISTIC_FEMALE_EN,
                "HYPER_REALISTIC_MALE": HYPER_REALISTIC_MALE_EN,
                "YOUNG_BOY_PORTRAIT": YOUNG_BOY_PORTRAIT_EN,
                "MIDDLE_ELDERLY_FEMALE_PORTRAIT": MIDDLE_ELDERLY_FEMALE_PORTRAIT_EN,
                "MIDDLE_ELDERLY_MALE_PORTRAIT": MIDDLE_ELDERLY_MALE_PORTRAIT_EN,
                "ART_ILLUSTRATION": ART_ILLUSTRATION_EN,
                "POSTER_DESIGN": POSTER_DESIGN_EN,
                "SCENE_DESIGN": SCENE_DESIGN_EN,
                "INTERIOR_DESIGN": INTERIOR_DESIGN_EN,
                "ARCHITECTURE_RENDERING": ARCHITECTURE_RENDERING_EN,
                "ECOMMERCE": ECOMMERCE_EN,
                "FOOD_PHOTOGRAPHY": FOOD_PHOTOGRAPHY_EN,
                "EDIT_COMBINED": EDIT_COMBINED_EN,
                "UNIVERSAL_VIDEO": UNIVERSAL_VIDEO_EN,
                "CONTINUING_I2V": CONTINUING_I2V_EN,
                "CONTINUING_FLF2V": CONTINUING_FLF2V_EN,
                "CONTINUING_MULTI_STORYBOARD": CONTINUING_MULTI_STORYBOARD_EN,
                "VIDEO_FRAME_SEQUENCE": VIDEO_FRAME_SEQUENCE_EN,
                "VIDEO_TO_PROMPT": VIDEO_TO_PROMPT_EN,
                "VIDEO_SCENE_BREAKDOWN": VIDEO_SCENE_BREAKDOWN_EN,
                "VIDEO_SUBTITLE_FORMAT": VIDEO_SUBTITLE_FORMAT_EN,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_EN,
                "LYRICS_CREATION": LYRICS_CREATION_EN,
                "IDEOGRAM4": IDEOGRAM4_EN,
            }
        
        preset = preset_map.get(preset_key, None)
        if preset is None:
            return ""
        
        examples = preset.get("examples", [])
        if not examples:
            return ""
        
        filtered_examples = []
        for example in examples:
            if isinstance(example, str):
                content = example
                category = ""
            elif isinstance(example, dict):
                if output_format == "natural" and "natural" in example:
                    content = example.get("natural", "")
                elif output_format == "structured" and "structured" in example:
                    content = example.get("structured", "")
                else:
                    content = example.get("natural", example.get("structured", ""))
                category = example.get("category", "")
            else:
                content = ""
                category = ""
            
            if content:
                filtered_examples.append({"content": content, "category": category})
        
        if not filtered_examples:
            return ""
        
        if custom_prompt:
            custom_prompt_lower = custom_prompt.lower()
            for example in filtered_examples:
                if example["category"] and example["category"].lower() in custom_prompt_lower:
                    return f"【示例】\n{example['content']}"
        
        random_example = random.choice(filtered_examples)
        return f"【示例】\n{random_example['content']}"
    
    def get_preset_constraints(self, preset_key, language):
        """
        获取预设模板中的正向约束和负向提示词
        
        Args:
            preset_key: 预设键名
            language: 语言（"中文" 或 "English"）
            
        Returns:
            tuple: (positive_constraints, negative_prompts)
        """
        if language == "中文":
            preset_map = {
                "PROMPT_EXPANDER": PROMPT_EXPANDER_ZH,
                "ILLUSTRIOUS": ILLUSTRIOUS_ZH,
                "ANIME_PROMPT": ANIME_PROMPT_ZH,
                "THICKPAINT_ROLE": THICKPAINT_ROLE_ZH,
                "REALISTIC_FEMALE": REALISTIC_FEMALE_ZH,
                "REALISTIC_MALE": REALISTIC_MALE_ZH,
                "WESTERN_FEMALE": WESTERN_FEMALE_ZH,
                "WESTERN_MALE": WESTERN_MALE_ZH,
                "HYPER_REALISTIC_FEMALE": HYPER_REALISTIC_FEMALE_ZH,
                "HYPER_REALISTIC_MALE": HYPER_REALISTIC_MALE_ZH,
                "YOUNG_BOY_PORTRAIT": YOUNG_BOY_PORTRAIT_ZH,
                "MIDDLE_ELDERLY_FEMALE_PORTRAIT": MIDDLE_ELDERLY_FEMALE_PORTRAIT_ZH,
                "MIDDLE_ELDERLY_MALE_PORTRAIT": MIDDLE_ELDERLY_MALE_PORTRAIT_ZH,
                "ART_ILLUSTRATION": ART_ILLUSTRATION_ZH,
                "POSTER_DESIGN": POSTER_DESIGN_ZH,
                "SCENE_DESIGN": SCENE_DESIGN_ZH,
                "INTERIOR_DESIGN": INTERIOR_DESIGN_ZH,
                "ARCHITECTURE_RENDERING": ARCHITECTURE_RENDERING_ZH,
                "ECOMMERCE": ECOMMERCE_ZH,
                "FOOD_PHOTOGRAPHY": FOOD_PHOTOGRAPHY_ZH,
                "UNIVERSAL_VIDEO": UNIVERSAL_VIDEO_ZH,
                "CONTINUING_I2V": CONTINUING_I2V_ZH,
                "CONTINUING_FLF2V": CONTINUING_FLF2V_ZH,
                "CONTINUING_MULTI_STORYBOARD": CONTINUING_MULTI_STORYBOARD_ZH,
                "IDEOGRAM4": IDEOGRAM4_ZH,
            }
        else:
            preset_map = {
                "PROMPT_EXPANDER": PROMPT_EXPANDER_EN,
                "ILLUSTRIOUS": ILLUSTRIOUS_EN,
                "ANIME_PROMPT": ANIME_PROMPT_EN,
                "THICKPAINT_ROLE": THICKPAINT_ROLE_EN,
                "REALISTIC_FEMALE": REALISTIC_FEMALE_EN,
                "REALISTIC_MALE": REALISTIC_MALE_EN,
                "WESTERN_FEMALE": WESTERN_FEMALE_EN,
                "WESTERN_MALE": WESTERN_MALE_EN,
                "HYPER_REALISTIC_FEMALE": HYPER_REALISTIC_FEMALE_EN,
                "HYPER_REALISTIC_MALE": HYPER_REALISTIC_MALE_EN,
                "YOUNG_BOY_PORTRAIT": YOUNG_BOY_PORTRAIT_EN,
                "MIDDLE_ELDERLY_FEMALE_PORTRAIT": MIDDLE_ELDERLY_FEMALE_PORTRAIT_EN,
                "MIDDLE_ELDERLY_MALE_PORTRAIT": MIDDLE_ELDERLY_MALE_PORTRAIT_EN,
                "ART_ILLUSTRATION": ART_ILLUSTRATION_EN,
                "POSTER_DESIGN": POSTER_DESIGN_EN,
                "SCENE_DESIGN": SCENE_DESIGN_EN,
                "INTERIOR_DESIGN": INTERIOR_DESIGN_EN,
                "ARCHITECTURE_RENDERING": ARCHITECTURE_RENDERING_EN,
                "ECOMMERCE": ECOMMERCE_EN,
                "FOOD_PHOTOGRAPHY": FOOD_PHOTOGRAPHY_EN,
                "UNIVERSAL_VIDEO": UNIVERSAL_VIDEO_EN,
                "CONTINUING_I2V": CONTINUING_I2V_EN,
                "CONTINUING_FLF2V": CONTINUING_FLF2V_EN,
                "CONTINUING_MULTI_STORYBOARD": CONTINUING_MULTI_STORYBOARD_EN,
                "IDEOGRAM4": IDEOGRAM4_EN,
            }
        
        preset = preset_map.get(preset_key, None)
        if preset is None:
            return "", ""
        
        positive_constraints = preset.get("positive_constraints", "")
        negative_prompts = preset.get("negative_prompts", "")
        
        return positive_constraints, negative_prompts
    
    def _process_batch_inference(self, llama_model, images, system_prompt, final_prompt,
                                  image_max_size, batch_combination, gen_params,
                                  engine, model_info, mode):
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
            
            # 记录总开始时间
            total_start_time = time.time()
            
            # 预处理：批量编码所有图片
            print(f"【批量推理】预处理图片编码...")
            image_contents = []
            
            # 批量转换为numpy数组，减少循环中的CPU-GPU数据传输
            if hasattr(images, 'cpu'):
                imgs_np = images.cpu().numpy()
            else:
                imgs_np = images
            
            # 批量处理图片
            for i in range(num_images):
                mm.throw_exception_if_processing_interrupted()
                
                img = imgs_np[i]
                img_np = scale_image(img, image_max_size)
                img_base64 = image2base64(img_np)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
                if (i + 1) % 5 == 0 or i + 1 == num_images:
                    print(f"【批量推理】编码图片 {i+1}/{num_images}")
            
            # 批量推理
            print(f"【批量推理】开始批量推理...")
            
            if batch_combination == "separate":
                # 优化：使用批处理减少推理调用次数
                # 对于数量较多的图片，分批次处理
                batch_size = 4  # 每批处理4张图片
                for batch_start in range(0, num_images, batch_size):
                    batch_end = min(batch_start + batch_size, num_images)
                    batch_images = image_contents[batch_start:batch_end]
                    
                    print(f"【批量推理】处理批次 {batch_start//batch_size + 1}/{(num_images + batch_size - 1)//batch_size}")
                    
                    # 记录批次开始时间
                    batch_start_time = time.time()
                    
                    # 批量构建消息
                    batch_messages = []
                    for i, img_content in enumerate(batch_images):
                        messages = []
                        if system_prompt and system_prompt.strip():
                            messages.append({"role": "system", "content": system_prompt})
                        
                        content = [{"type": "text", "text": final_prompt}]
                        content.append(img_content)
                        
                        messages.append({"role": "user", "content": content})
                        batch_messages.append(messages)
                    
                    # 批量执行推理
                    for i, messages in enumerate(batch_messages):
                        img_idx = batch_start + i
                        mm.throw_exception_if_processing_interrupted()
                        print(f"【批量推理】推理图片 {img_idx+1}/{num_images}")
                        
                        output = engine.create_chat_completion(llama_model.llm, messages, gen_params)
                        
                        if output and 'choices' in output:
                            content = output['choices'][0]['message']['content']
                            # 处理content可能是列表的情况（多模态模型）
                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        text_parts.append(item.get('text', ''))
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                text = ''.join(text_parts).lstrip().removeprefix(": ")
                            elif isinstance(content, str):
                                text = content.lstrip().removeprefix(": ")
                            else:
                                text = str(content).lstrip().removeprefix(": ")
                            batch_results.append(text)
                            print(f"【批量推理】图片 {img_idx+1} 完成: {text[:50]}...")
                        else:
                            batch_results.append("")
                            print(f"【批量推理】图片 {img_idx+1} 失败")
                    
                    # 记录批次结束时间
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    print(f"【批量推理】批次完成，耗时: {batch_time:.2f} 秒")
            else:
                # 合并输出模式：所有图片一次性处理
                # 记录推理开始时间
                start_time = time.time()
                
                messages = []
                if system_prompt and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                
                content = [{"type": "text", "text": final_prompt}]
                content.extend(image_contents)
                
                messages.append({"role": "user", "content": content})
                
                output = engine.create_chat_completion(llama_model.llm, messages, gen_params)
                
                # 记录推理结束时间
                end_time = time.time()
                inference_time = end_time - start_time
                
                if output and 'choices' in output:
                    content = output['choices'][0]['message']['content']
                    # 处理content可能是列表的情况（多模态模型）
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        combined_text = ''.join(text_parts).lstrip().removeprefix(": ")
                    elif isinstance(content, str):
                        combined_text = content.lstrip().removeprefix(": ")
                    else:
                        combined_text = str(content).lstrip().removeprefix(": ")
                    # 尝试分割结果（假设模型按顺序返回结果）
                    # 可以根据输出格式进行分割
                    lines = combined_text.split('\n')
                    if len(lines) >= num_images:
                        batch_results = [line.strip() for line in lines[:num_images] if line.strip()]
                    else:
                        # 如果无法分割，将整个结果作为第一个输出
                        batch_results = [combined_text]
                    
                    print(f"【批量推理】合并模式完成: {combined_text[:50]}... (耗时: {inference_time:.2f} 秒)")
                else:
                    batch_results = [""]
                    print(f"【批量推理】合并模式失败 (耗时: {inference_time:.2f} 秒)")
            
            # 记录总结束时间
            total_end_time = time.time()
            total_inference_time = total_end_time - total_start_time
            print(f"【批量推理】全部完成，总耗时: {total_inference_time:.2f} 秒")
            
            return batch_results
            
        except Exception as e:
            # 记录错误时的时间
            end_time = time.time()
            inference_time = end_time - total_start_time if 'total_start_time' in locals() else 0
            print(f"【批量推理错误】{str(e)} (耗时: {inference_time:.2f} 秒)")
            import traceback
            traceback.print_exc()
            return []
    

    

    
    def _run_inference(self, llama_model, messages, gen_params):
        """执行推理的内部方法，用于异步处理"""
        generated_text = ""
        audio_output = None
        
        # 生成缓存键
        model_path = getattr(llama_model, 'model_path', str(llama_model))
        
        # 如果没有选择模型，直接返回空结果
        if not model_path or model_path.lower() == "none":
            print("【推理】未选择模型，跳过推理")
            return generated_text, audio_output
        cache_content = {
            "model_path": model_path,
            "messages": messages,
            "gen_params": {k: v for k, v in gen_params.items() if k != "seed"}  # 排除seed
        }
        import json
        cache_key = hash(json.dumps(cache_content, sort_keys=True, default=str))
        
        # 检查缓存
        cached_result = self.get_from_cache("_inference_cache", cache_key)
        if cached_result:
            return cached_result
        
        retry_count = 0
        max_retries = 3
        success = False
        
        # 检查是否是MTP模型
        is_mtp_model = False
        if model_path:
            is_mtp_model = "mtp" in model_path.lower() or "multitoken" in model_path.lower()
        
        while retry_count < max_retries and not success:
            try:
                # 记录推理开始时间
                start_time = time.time()
                
                # 标准文本推理（所有模型类型）
                output = self.engine.create_chat_completion(llama_model.llm, messages, gen_params)
                if output and 'choices' in output:
                    content = output['choices'][0]['message']['content']
                    # 处理content可能是列表的情况（多模态模型）
                    if isinstance(content, list):
                        # 从列表中提取文本内容
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        generated_text = ''.join(text_parts).lstrip().removeprefix(": ")
                    elif isinstance(content, str):
                        generated_text = content.lstrip().removeprefix(": ")
                    else:
                        generated_text = str(content).lstrip().removeprefix(": ")
                
                # 记录推理结束时间
                end_time = time.time()
                inference_time = end_time - start_time
                print(f"【推理】推理完成，耗时: {inference_time:.2f} 秒")

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
                    
                    # MTP模型特殊处理：如果推测解码导致空结果，尝试调整参数重试
                    if is_mtp_model and retry_count == 1:
                        print(f"【MTP优化】MTP模型生成空结果，尝试禁用推测解码重试...")
                        # 临时调整参数，移除可能导致问题的MTP相关设置
                        gen_params = gen_params.copy()
                        # 增加temperature以鼓励生成
                        gen_params["temperature"] = min(gen_params.get("temperature", 0.7) + 0.2, 1.0)
                        print(f"【MTP优化】调整temperature到 {gen_params['temperature']}")

            except Exception as e:
                # 记录推理结束时间（即使出错）
                end_time = time.time()
                inference_time = end_time - start_time
                print(f"【推理】推理失败，耗时: {inference_time:.2f} 秒")
                
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
        
        # 过滤Qwen3.5的思考内容（在缓存前过滤）
        if generated_text and generated_text != "推理失败":
            generated_text = self._filter_thinking_content(generated_text)
        
        # 将结果添加到缓存
        result = (generated_text, audio_output)
        self.add_to_cache("_inference_cache", cache_key, result)
        
        return generated_text, audio_output
    
    def process(self, llama_model, inference_mode, preset_prompt, system_prompt, text_input,
                prompt_language, response_language, output_format, enable_constraints,
                enable_negative_prompts,
                video_max_frames, video_sampling, video_manual_indices, image_max_size, batch_combination,
                seed, force_offload,
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
                "[高级] 视频理解 (Video Understanding)": "video"
            }
            mode = mode_map.get(inference_mode, "text")
            
            # 检查输入
            has_images = images is not None and (hasattr(images, 'numel') and images.numel() > 0)
            has_video = self.video_processor._check_video_input(video)
            has_audio = audio is not None
            
            # 处理自定义提示词
            custom_prompt = text_input
            
            # 检查是否启用ASR和TTS
            enable_asr = mode in ["audio", "text_to_audio"] or preset_prompt in ["[Audio] Audio to Text"]
            # 启用TTS的条件：text_to_audio模式，或者使用了TTS模型，或者连接了TTS节点
            enable_tts = mode in ["text_to_audio"] or preset_prompt in ["[Audio] Text to Audio"] or (tts_model is not None)
            
            # 设置默认值
            audio_output_mode = "TTS音色美化输出" if enable_tts else "关闭音频输出"
            tts_speed = 1.0
            speaker_type = "default"
            
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
            preset_text = self.get_preset_text_by_language(preset_key, preset_prompts_language, output_format, mode)
            
            # 获取预设模板的示例内容
            example_output = self.get_preset_examples(preset_key, preset_prompts_language, output_format, custom_prompt)
            
            # 构建最终提示词
            if preset_prompt == "Empty - Nothing":
                final_prompt = custom_prompt.strip() if custom_prompt.strip() else system_prompt.strip()
            else:
                final_prompt = preset_text
                # 替换占位符
                if custom_prompt.strip():
                    if "下面是要优化的 Prompt：" in preset_text or "Below is the Prompt to optimize:" in preset_text:
                        final_prompt = preset_text + custom_prompt.strip()
                    else:
                        final_prompt = final_prompt.replace("#", custom_prompt.strip()).replace("@", "video" if has_video else "image")
                        # 清理条件语句
                        final_prompt = final_prompt.replace("如果提供了自定义内容，请以此为基础：", "")
                else:
                    # 没有自定义内容时，移除 # 占位符并替换 @ 占位符
                    final_prompt = final_prompt.replace("#", "").replace("@", "video" if has_video else "image")
                    # 清理条件语句和多余的空白
                    final_prompt = final_prompt.replace("如果提供了自定义内容，请以此为基础：", "")
                    final_prompt = final_prompt.replace("和提供的自定义内容", "")
                    final_prompt = final_prompt.replace("  ", " ").strip()
            
            # 添加示例内容作为参考，引导模型生成更详细的输出
            if example_output:
                final_prompt += f"\n\n**【示例】**\n{example_output}"
            
            # 添加语言指示
            if output_language == "中文":
                final_prompt += "\n\n请用中文回答。"
            elif output_language == "English":
                final_prompt += "\n\nPlease answer in English."
            
            # MiMo-VL模型在文本生成模式下需要特殊指令，避免输出思考过程
            try:
                from common import LLAMA_CPP_STORAGE
                if LLAMA_CPP_STORAGE.current_config:
                    chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                    is_mimo_vl = chat_handler in ["MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]
                    if is_mimo_vl and mode == "text":
                        if output_language == "中文":
                            final_prompt += "\n\n**【重要】**请直接输出最终结果，不要输出分析过程、思考步骤或解释说明。"
                        else:
                            final_prompt += "\n\n**【Important】**Please output only the final result directly, without analysis process, thinking steps, or explanations."
            except Exception as e:
                pass
            
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
                    return (generated_text, [generated_text], seed, None, example_output)
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
                                # 从TTS模型配置中获取speaker_id
                                speaker_id = getattr(tts_model, 'config', {}).get('speaker_id', 0)
                                print(f"【TTS】使用配置中的speaker_id: {speaker_id}")
                                audio_output = tts_model.synthesize(
                                    text=generated_text,
                                    speed=tts_speed,
                                    speaker_id=speaker_id
                                )
                                print(f"【TTS】使用TTS模型合成成功")
                            except Exception as e:
                                print(f"【TTS】TTS模型合成失败: {str(e)}")
                                audio_output = None
                        # 如果TTS模型失败或未提供，audio_output保持为None
                        if audio_output is None:
                            print(f"【TTS】TTS模型未提供或合成失败")
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
                    
                    return (generated_text, [generated_text], seed, audio_output, example_output)
                else:
                    # 其他模式但没有LLM模型，返回空结果
                    print(f"【无模型模式】模式: {mode}，未选择LLM模型，返回空结果")
                    return ("", [""], seed, None, example_output)

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
            
            # 创建推理引擎（使用缓存）
            engine_key = f"{self.model_info['key']}_{self.model_info['type']}_{self.model_info['subtype']}"
            if engine_key not in self._model_cache:
                self._model_cache[engine_key] = InferenceEngineFactory.create_engine(self.model_info)
                print(f"【模型缓存】创建并缓存推理引擎: {engine_key}")
            self.engine = self._model_cache[engine_key]
            
            # 确定性能级别
            perf_level = "balanced"
            if has_video:
                perf_level = self.video_processor.get_video_perf_level()
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
                # Qwen3系列模型特殊内存优化和禁用思考模式
            try:
                from common import LLAMA_CPP_STORAGE
                if LLAMA_CPP_STORAGE.current_config:
                    chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                    is_qwen3_series = chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3-VL"]
                    is_mimo_vl = chat_handler in ["MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]
                    has_visual = video_input or has_images
                    
                    if is_qwen3_series or (is_mimo_vl and has_visual):
                        # 视频模式需要更激进的参数调整以避免KV缓存耗尽
                        if video_input and has_video:
                            # 视频模式下大幅降低批处理大小以避免KV缓存耗尽
                            gen_params["n_batch"] = min(gen_params.get("n_batch", 512), 128)
                            print(f"【Qwen3-VL视频优化】视频模式：降低n_batch到128以避免KV缓存耗尽")
                        else:
                            # 图像模式使用适中的批处理大小
                            gen_params["n_batch"] = min(gen_params.get("n_batch", 512), 256)
                        # 减少最大token数
                        gen_params["max_tokens"] = min(gen_params.get("max_tokens", 1024), 768)
                        # 降低top_p以减少内存使用
                        gen_params["top_p"] = max(gen_params.get("top_p", 0.9), 0.8)
                        # 禁用思考模式以避免混乱输出
                        gen_params["enable_thinking"] = False
                        # 添加思考结束标记作为停止序列
                        stop_list = gen_params.get("stop", [])
                        if isinstance(stop_list, str):
                            stop_list = [stop_list]
                        stop_list.extend(["<|end_of_thinking|>", "<|end_of_solution|>", "</think>"])
                        gen_params["stop"] = stop_list
                        print(f"【Qwen3优化】调整参数: n_batch={gen_params['n_batch']}, max_tokens={gen_params['max_tokens']}, top_p={gen_params['top_p']}, enable_thinking=False")
            except Exception as e:
                print(f"【Qwen3优化】内存参数调整时出错（忽略）: {e}")
                self._perf_params_cache[cache_key] = gen_params.copy()
            
            # 应用用户自定义参数
            if parameters:
                gen_params.update({k: v for k, v in parameters.items() if k != "state_uid"})
            
            gen_params["seed"] = seed
            
            # ========== 视频处理逻辑 ==========
            video_frames = []
            if has_video and mode == "video":
                print(f"【视频处理】开始处理视频，帧数限制: {video_max_frames}, 采样方式: {video_sampling}")
                video_data = self.video_processor.process_video_input(
                    video=video,
                    max_frames=video_max_frames,
                    sampling_method=video_sampling,
                    manual_indices=video_manual_indices
                )
                
                if video_data and "frames" in video_data:
                    video_frames = video_data["frames"]
                    print(f"【视频处理】成功提取 {len(video_frames)} 帧用于推理")
                else:
                    print(f"【视频处理】视频提取失败或无可用帧")
                    
            # 统一处理所有模型
            content = []
            if final_prompt:
                content.append({"type": "text", "text": final_prompt})
            
            # 获取当前ChatHandler信息用于优化
            current_chat_handler = ""
            try:
                from common import LLAMA_CPP_STORAGE
                if LLAMA_CPP_STORAGE.current_config:
                    current_chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
            except Exception as e:
                pass
            
            # 先添加视频帧
            if len(video_frames) > 0:
                # Qwen3-VL视频模式：在推理前清理KV缓存以避免内存不足
                try:
                    from common import LLAMA_CPP_STORAGE
                    if LLAMA_CPP_STORAGE.llm and current_chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
                        # 清理KV缓存以腾出空间给视频帧
                        if hasattr(LLAMA_CPP_STORAGE.llm, '_ctx') and hasattr(LLAMA_CPP_STORAGE.llm._ctx, 'memory_clear'):
                            LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                            print(f"【Qwen3-VL视频优化】已清理KV缓存，为视频帧腾出空间")
                        if hasattr(LLAMA_CPP_STORAGE.llm, 'is_hybrid') and LLAMA_CPP_STORAGE.llm.is_hybrid:
                            if hasattr(LLAMA_CPP_STORAGE.llm, '_hybrid_cache_mgr') and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                                LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                except Exception as e:
                    print(f"【Qwen3-VL视频优化】KV缓存清理时出错（忽略）: {e}")

                # Qwen3系列模型特殊优化：限制帧数和降低图像大小
                qwen3_image_size = image_max_size
                qwen3_frame_limit = len(video_frames)
                
                if current_chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
                    # 限制帧数以避免内存不足（每帧大约需要256-512 tokens）
                    qwen3_frame_limit = min(len(video_frames), 8)
                    # 降低图像大小以减少内存使用
                    qwen3_image_size = min(image_max_size, 256)
                    if qwen3_image_size < 128:
                        qwen3_image_size = 128
                    if len(video_frames) > qwen3_frame_limit:
                        print(f"【Qwen3优化】限制视频帧数从 {len(video_frames)} 到 {qwen3_frame_limit} 以避免内存不足")
                    print(f"【Qwen3优化】调整视频帧大小为: {qwen3_image_size}")
                
                # 截取限制后的帧数
                video_frames_to_use = video_frames[:qwen3_frame_limit]
                
                for i, frame in enumerate(video_frames_to_use):
                    if hasattr(frame, 'cpu'):
                        img = frame.cpu().numpy()
                    else:
                        img = frame
                    img_np = scale_image(img, qwen3_image_size)
                    img_base64 = image2base64(img_np)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                    del img, img_np, img_base64
                    import gc
                    gc.collect()
            
            # 再添加图像
            if images is not None and mode in ["images"]:
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
                
                # Qwen3系列模型特殊图像大小优化
                qwen3_image_size = image_max_size
                if current_chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
                    qwen3_image_size = min(image_max_size, 256)
                    if qwen3_image_size < 128:
                        qwen3_image_size = 128
                    print(f"【Qwen3优化】调整图像大小为: {qwen3_image_size}")
                
                for i in range(images.shape[0]):
                    if hasattr(images[i], 'cpu'):
                        img = images[i].cpu().numpy()
                    else:
                        img = images[i]
                    img_np = scale_image(img, qwen3_image_size)
                    img_base64 = image2base64(img_np)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                    del img, img_np, img_base64
                    import gc
                    gc.collect()
            
            # 回退机制：当ChatHandler为None且没有图像/视频内容时，将列表格式转换为字符串格式
            # 这是因为某些模型（如MTP模型）不支持列表格式的content
            # 特殊处理：Qwen3.5/3.6启用mmproj时，根据推理模式决定是否使用纯文本格式
            # - text模式：强制使用纯文本格式，避免模型误判为图片反推
            # - images/batch_images/video模式：正常使用多模态功能
            has_visual_content = len(video_frames) > 0 or (images is not None and mode in ["images"])
            chat_handler_available = False
            enable_mmproj = False
            is_qwen3_model = False
            try:
                from common import LLAMA_CPP_STORAGE
                chat_handler_available = LLAMA_CPP_STORAGE.chat_handler is not None
                if LLAMA_CPP_STORAGE.current_config:
                    enable_mmproj = LLAMA_CPP_STORAGE.current_config.get("enable_mmproj", False)
                    current_chat_handler_name = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                    is_qwen3_model = current_chat_handler_name in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]
            except Exception as e:
                pass
            
            # 判断是否需要强制使用纯文本格式：
            # 1. ChatHandler不可用且无视觉内容（原有逻辑）
            # 2. Qwen3.5/3.6启用mmproj且为文本生成模式（根据推理模式判定）
            is_text_mode = mode == "text"
            force_text_mode = (not chat_handler_available and not has_visual_content) or \
                              (is_qwen3_model and enable_mmproj and is_text_mode)
            
            if force_text_mode and isinstance(content, list):
                # 将列表格式转换为纯文本字符串
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = ''.join(text_parts).strip()
                if is_qwen3_model and enable_mmproj and is_text_mode:
                    print(f"【Qwen3.5/3.6优化】文本生成模式，强制使用纯文本格式")
                else:
                    print(f"【回退模式】ChatHandler为None，已将content从列表格式转换为字符串格式")
            
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
                        engine=self.engine,
                        model_info=self.model_info,
                        mode=mode
                    )
                    
                    if batch_results:
                        if batch_combination == "combined":
                            generated_text = " | ".join(batch_results)
                            output_list = batch_results
                        else:
                            generated_text = batch_results[0] if batch_results else ""
                            output_list = batch_results
                        
                        # 过滤Qwen3.5的思考内容
                        generated_text = self._filter_thinking_content(generated_text)
                        output_list = [self._filter_thinking_content(item) for item in output_list]
                        
                        print(f"【批量推理】完成，生成 {len(batch_results)} 个结果")
                        
                        # 跳转到TTS处理部分
                        if audio_output_mode == "TTS音色美化输出" and enable_tts and audio_output is None and generated_text:
                            pass  # 继续到TTS处理
                        else:
                            # Qwen3系列模型特殊内存清理（防止多余内容）
                            try:
                                from common import LLAMA_CPP_STORAGE
                                if LLAMA_CPP_STORAGE.current_config and LLAMA_CPP_STORAGE.llm is not None:
                                    chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                                    if chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
                                        if hasattr(LLAMA_CPP_STORAGE.llm, 'n_tokens'):
                                            LLAMA_CPP_STORAGE.llm.n_tokens = 0
                                        if hasattr(LLAMA_CPP_STORAGE.llm, '_ctx') and hasattr(LLAMA_CPP_STORAGE.llm._ctx, 'memory_clear'):
                                            LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                                        if hasattr(LLAMA_CPP_STORAGE.llm, 'is_hybrid') and LLAMA_CPP_STORAGE.llm.is_hybrid:
                                            if hasattr(LLAMA_CPP_STORAGE.llm, '_hybrid_cache_mgr') and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                                                LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                                        print("【Qwen3优化】已清理模型内存，防止多余内容")
                            except Exception as e:
                                print(f"【Qwen3优化】内存清理时出错（忽略）: {e}")

                            # 添加预设模板中的正向约束和负向提示词到输出文本
                            positive_constraints, negative_prompts = "", ""
                            if (enable_constraints or enable_negative_prompts) and preset_key:
                                positive_constraints, negative_prompts = self.get_preset_constraints(preset_key, preset_prompts_language)
                            
                            if enable_constraints and positive_constraints:
                                generated_text = f"【正向约束】{positive_constraints}\n\n{generated_text}"
                            
                            if enable_negative_prompts and negative_prompts:
                                generated_text += f"\n\n【负向提示词】{negative_prompts}"
                            
                            if enable_constraints or enable_negative_prompts:
                                output_list = []
                                for item in output_list:
                                    new_item = item
                                    if enable_constraints and positive_constraints:
                                        new_item = f"【正向约束】{positive_constraints}\n\n{new_item}"
                                    if enable_negative_prompts and negative_prompts:
                                        new_item += f"\n\n【负向提示词】{negative_prompts}"
                                    output_list.append(new_item)
                            
                            # 直接返回结果
                            _uid = parameters.get("state_uid", None) if parameters else None
                            uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
                            return (generated_text, output_list, int(uid), None, example_output)
            
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
                        gen_params=gen_params
                    )
                    generated_text, audio_output = future.result()
            
            # 处理TTS语音合成（独立TTS模型）- 仅在TTS音色美化输出模式下启用
            if enable_tts and audio_output is None and generated_text:
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
                
                # 如果TTS模型失败或未提供，audio_output保持为None
                if audio_output is None:
                    print(f"【TTS】TTS模型未提供或合成失败")
            
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

            # Qwen3系列模型特殊内存清理（防止多余内容）
            try:
                from common import LLAMA_CPP_STORAGE
                if LLAMA_CPP_STORAGE.current_config and LLAMA_CPP_STORAGE.llm is not None:
                    chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                    if chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
                        if hasattr(LLAMA_CPP_STORAGE.llm, 'n_tokens'):
                            LLAMA_CPP_STORAGE.llm.n_tokens = 0
                        if hasattr(LLAMA_CPP_STORAGE.llm, '_ctx') and hasattr(LLAMA_CPP_STORAGE.llm._ctx, 'memory_clear'):
                            LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                        if hasattr(LLAMA_CPP_STORAGE.llm, 'is_hybrid') and LLAMA_CPP_STORAGE.llm.is_hybrid:
                            if hasattr(LLAMA_CPP_STORAGE.llm, '_hybrid_cache_mgr') and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                                LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()
                        print("【Qwen3优化】已清理模型内存，防止多余内容")
            except Exception as e:
                print(f"【Qwen3优化】内存清理时出错（忽略）: {e}")

            # 强制卸载
            if force_offload:
                mm.soft_empty_cache()
                # 清理所有缓存
                self.clear_all_caches()

            # 过滤思考内容
            generated_text = self._filter_thinking_content(generated_text)

            # 添加预设模板中的正向约束和负向提示词到输出文本
            positive_constraints, negative_prompts = "", ""
            if (enable_constraints or enable_negative_prompts) and preset_key:
                positive_constraints, negative_prompts = self.get_preset_constraints(preset_key, preset_prompts_language)
            
            if enable_constraints and positive_constraints:
                generated_text = f"【正向约束】{positive_constraints}\n\n{generated_text}"
            
            if enable_negative_prompts and negative_prompts:
                generated_text += f"\n\n【负向提示词】{negative_prompts}"

            # 处理UID
            _uid = parameters.get("state_uid", None) if parameters else None
            uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid

            return (generated_text, [generated_text], int(uid), audio_output, example_output)
        
        except Exception as e:
            error_message = ErrorHandler.handle_error(e, context={"mode": mode, "model_type": self.model_info.get("type", "unknown")})
            print(f"【处理错误】{str(e)}")
            return (error_message, [error_message], seed, None, example_output)
    
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