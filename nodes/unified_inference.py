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
#图像反推标签
from support.image_reverse_tags import (
    ImageReverseTags,
)
#图像反推通用
from support.image_reverse_describe import (
    ImageReverseDescribe,
)
#二次元角色
from support.illustrious import (
    Illustrious,
)
#动漫
from support.anime_dream_visual import (
    AnimeDreamVisual,
)
#厚涂CG角色
from support.thickpaint_role import (
    ThickPaintRole,
)
#通用提示词扩写
from support.prompt_expander import (
    PromptExpander,
)
#真实亚洲女性人像
from support.realistic_female import (
    RealisticFemale,
)
#真实亚洲男性人像
from support.realistic_male import (
    RealisticMale,
)
#西方女性人像
from support.western_female import (
    WesternFemale,
)
#西方男性人像
from support.western_male import (
    WesternMale,
)
#超写实女性人像
from support.hyper_realistic_female import (
    HyperRealisticFemale,
)
#超写实男性人像
from support.hyper_realistic_male import (
    HyperRealisticMale,
)
#中老年人女性人像
from support.middle_elderly_female import (
    MiddleElderlyFemale,
)
#中老年人男性人像
from support.middle_elderly_male import (
    MiddleElderlyMale,
)
#儿童人像
from support.young_boy_portrait import (
    YoungBoyPortrait,
)
#美食摄影
from support.food_photography import (
    FoodPhotography,
)
#室内设计
from support.interior_design import (
    InteriorDesign,
)
#场景设计
from support.scene_design import (
    SceneDesign,
)
#室外建筑
from support.architecture_rendering import (
    ArchitectureRendering,
)
#艺术插画
from support.art_illustration import (
    ArtIllustration,
)
#海报设计
from support.poster_design import (
    PosterDesign,
)
#电商产品
from support.ecommerce import (
    Ecommerce,
)
#文生视频
from support.continuing_t2v import (
    ContinuingT2V,
)
#首帧延续图生视频
from support.continuing_i2v import (
    ContinuingI2V,
)
#首尾帧图生视频
from support.continuing_fl2v import (
    ContinuingFL2V,
)
#多关键帧序列图生视频
from support.continuing_multi_storyboard import (
    ContinuingMultiStoryboard,
)
#视频帧序列
from support.video_frame_sequence import (
    VideoFrameSequence,
)
#视频反推
from support.video_to_prompt import (
    VideoToPrompt,
)
#视频分镜拆解
from support.video_scene_breskdown import (
    VideoSceneBreakdown,
)
#视频情绪字幕分析器与TTS合成指导
from support.video_subtutle_format import (
    VideoSubtitleFormat,
)
#多人对话
from support.multi_speaker_dialogue import (
    MultiSpeakerDialogue,
)
#歌曲创作
from support.song_creation import (
    SongCreation,
)

# 图像生成模型选项（通用类型）：Auto 为默认通用项，其余模型从 HyperRealisticFemale 生成器类动态读取保持同步（不含 SDXL）
IMAGE_MODEL_OPTIONS = ["Auto"] + [m for m in HyperRealisticFemale().model_formula_library.keys() if m != "SDXL"]

# 视频生成模型选项
VIDEO_MODEL_OPTIONS = ["Auto"] + list(ContinuingT2V().video_model_formula_library.keys())

# 音频生成模型选项（TTS语音合成 + 音乐生成，Auto 为默认通用项）
AUDIO_MODEL_OPTIONS = ["Auto", "IndexTTS-2.5", "VoxCPM2", "Ace-Step1.5", "MiniMax-Music3"]

# 音频模型分类：TTS语音合成 与 音乐生成（用于按预设校验所选模型是否合法）
AUDIO_TTS_MODELS = {"IndexTTS-2.5", "VoxCPM2"}
AUDIO_MUSIC_MODELS = {"Ace-Step1.5", "MiniMax-Music3"}

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


class omni_llm_unified_inference:
    """
    统一推理节点 - 支持VL模型
    自动检测模型类型并使用对应的推理方式
    支持ASR+VL组合处理音频
    """
    
    # 缓存变量
    _language_cache = {}
    _perf_params_cache = {}
    _model_type_cache = {}
    _model_cache = {}  # 模型缓存，避免重复加载
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
        caches = ["_language_cache", "_perf_params_cache", "_model_type_cache", "_model_cache", "_inference_cache"]
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
    
    # 初始化预设提示词字典
    preset_prompts = {}

    # 添加基础预设
    preset_prompts["Empty - Nothing"] = ""

    # 添加分类预设提示词（顺序与 prompt_enhancer_preset_zh.py / prompt_enhancer_preset_en.py 保持一致）
    preset_prompts["[Reverse] Tags"] = "IMAGE_REVERSE_TAGS"
    preset_prompts["[Reverse] Describe"] = "IMAGE_REVERSE_DESCRIBE"
    preset_prompts["[Normal] Expand"] = "PROMPT_EXPANDER"
    preset_prompts["[Anime] Illustrious"] = "ILLUSTRIOUS"
    preset_prompts["[Anime] Anime Prompt Expand"] = "ANIME_PROMPT"
    preset_prompts["[Anime] Thick Paint Role"] = "THICKPAINT_ROLE"
    preset_prompts["[Portrait] Asian Female"] = "REALISTIC_FEMALE"
    preset_prompts["[Portrait] Asian Male"] = "REALISTIC_MALE"
    preset_prompts["[Portrait] Western Female"] = "WESTERN_FEMALE"
    preset_prompts["[Portrait] Western Male"] = "WESTERN_MALE"
    preset_prompts["[Portrait] Hyper Realistic Female"] = "HYPER_REALISTIC_FEMALE"
    preset_prompts["[Portrait] Hyper Realistic Male"] = "HYPER_REALISTIC_MALE"
    preset_prompts["[Portrait] Young Boy"] = "YOUNG_BOY_PORTRAIT"
    preset_prompts["[Portrait] Middle Elderly Female"] = "MIDDLE_ELDERLY_FEMALE"
    preset_prompts["[Portrait] Middle Elderly Male"] = "MIDDLE_ELDERLY_MALE"
    preset_prompts["[Design] Art Illustration"] = "ART_ILLUSTRATION"
    preset_prompts["[Design] Poster Design"] = "POSTER_DESIGN"
    preset_prompts["[Design] Scene Design"] = "SCENE_DESIGN"
    preset_prompts["[Design] Interior Design"] = "INTERIOR_DESIGN"
    preset_prompts["[Design] Architecture Rendering"] = "ARCHITECTURE_RENDERING"
    preset_prompts["[Design] Ecommerce Product"] = "ECOMMERCE"
    preset_prompts["[Design] Food Photography"] = "FOOD_PHOTOGRAPHY"
    preset_prompts["[Text to Video] T2V"] = "CONTINUING_T2V"
    preset_prompts["[Image to Video] I2V"] = "CONTINUING_I2V"
    preset_prompts["[Image to Video] FL2V"] = "CONTINUING_FL2V"
    preset_prompts["[Image to Video] Multi"] = "CONTINUING_MULTI_STORYBOARD"
    preset_prompts["[Video Reverse] Frame Sequence"] = "VIDEO_FRAME_SEQUENCE"
    preset_prompts["[Video Reverse] Describe"] = "VIDEO_TO_PROMPT"
    preset_prompts["[Video Reverse] Scene Breakdown"] = "VIDEO_SCENE_BREAKDOWN"
    preset_prompts["[Video Reverse] Subtitle"] = "VIDEO_SUBTITLE_FORMAT"
    preset_prompts["[Audio] Multi-Speaker Dialogue"] = "MULTI_SPEAKER_DIALOGUE"
    preset_prompts["[Music] Song Creation"] = "SONG_CREATION"

    preset_tags = list(preset_prompts.keys())
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ========== 推理模式 ==========
                "inference_mode": ([
                        "[基础] 文本生成 (Text Generation)",
                        "[基础] 图像理解 (Image Understanding)",
                        "[基础] 批量图像理解 (Batch Image Understanding)",
                        "[基础] 音频转文本 (Audio to Text)",
                        "[高级] 视频理解 (Video Understanding)"
                    ], {
                        "default": "[基础] 文本生成 (Text Generation)",
                        "tooltip": "选择推理模式：\n• [基础] 文本生成：使用语言模型生成文本内容\n• [基础] 图像理解：处理单张图像内容并生成描述\n• [基础] 批量图像理解：一次性处理多张图片，减少推理调用次数\n• [基础] 音频转文本：使用ASR模型将音频转换为文本\n• [高级] 视频理解：从视频中提取帧并进行分析"
                    }),
                
                # ========== 提示词配置 ==========
                "preset_prompt": (s.preset_tags, {"default": s.preset_tags[1], "tooltip": "选择预设提示词模板：\n• Empty - Nothing：无预设，完全自定义\n• [Reverse] Tags：图片反推SDXL模型提示词内容\n• [Reverse] Describe：图片反推自然语言提示词内容\n• [Normal] Expand：通用提示词文本优化扩写\n• [Anime] Illustrious：二次元角色专用优化扩写\n• [Anime] Anime Prompt Expand：二次元内容优化扩写\n• [Anime] Thick Paint Role：角色厚涂CG人物内容优化扩写\n• [Portrait] Asian Female：真实亚洲女性人像内容优化扩写\n• [Portrait] Asian Male：真实亚洲男性人像内容优化扩写\n• [Portrait] Western Female：真实欧美女性人像内容优化扩写\n• [Portrait] Western Male：真实欧美男性人像内容优化扩写\n• [Portrait] Hyper Realistic Female：超写实女性人像内容优化扩写\n• [Portrait] Hyper Realistic Male：超写实男性人像内容优化扩写\n• [Portrait] Young Boy：儿童人像内容优化扩写\n• [Portrait] Middle Elderly Female：中老年女性人像内容优化扩写\n• [Portrait] Middle Elderly Male：中老年男性人像内容优化扩写\n• [Design] Art Illustration：艺术插画内容优化扩写\n• [Design] Poster Design：海报设计文本优化\n• [Design] Scene Design：场景设计内容优化扩写\n• [Design] Interior Design：室内装修设计内容优化扩写\n• [Design] Architecture Rendering：建筑外观与园林渲染内容优化扩写\n• [Design] Ecommerce Product：电商产品内容优化扩写\n• [Design] Food Photography：美食摄影内容优化扩写\n• [Text to Video] T2V：根据关键词内容优化扩写\n• [Image to Video] I2V：根据图片内容优化扩写\n• [Image to Video] FL2V：根据首尾帧图片内容优化扩写\n• [Image to Video] Multi Storyboard:根据图片与文本内容优化扩写\n• [Video Reverse] Frame Sequence：按帧序列反推视频内容\n• [Video Reverse] Describe：通用视频提示词反推\n• [Video Reverse] Scene Breakdown：按分镜场景反推视频内容\n• [Video Reverse] Subtitle：结合字幕与视频反推情绪化文本\n• [Audio] Multi-Speaker Dialogue：多人对话情绪化优化调整\n• [Music] Song Creation：歌词创作内容优化扩写"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "提示词书写结构：主体+细节/特征+环境/场景+构图/视角+风格/媒介+画质/光影", "tooltip": "系统提示词，定义AI助手的角色和行为，可包含预设模板占位符#和自定义内容"}),
                "text_input": ("STRING", {"default": "", "multiline": True, "tooltip": "用户输入文本，作为对话的用户消息内容"}),
                
                # ========== 语言设置 ==========
                "prompt_language": (["中文", "English"], {"default": "中文", "tooltip": "预设模板的语言"}),
                "response_language": (["中文", "English"], {"default": "中文", "tooltip": "AI回复的语言"}),
                
                # ========== 输出格式设置 ==========
                "output_format": (["natural", "structured"], {
                    "default": "natural",
                    "tooltip": "输出格式控制：\n• natural：以自然段落格式输出纯文本内容，推荐日常使用\n• structured：输出结构化文本内容，适用于强理解模型（仅支持文生图预设模板）"
                }),
                "enable_constraints": ("BOOLEAN", {"default": False, "tooltip": "启用预设模板中的正向约束：\n• 正向约束：会添加到提示词最前面，引导模型生成更符合要求的内容，适用于SDXL/Base模型"}),
                "enable_negative_prompts": ("BOOLEAN", {"default": False, "tooltip": "启用预设模板中的负向提示词：\n• 负向提示词：会追加到提示词末尾，避免模型生成不想要的内容，部分模型不需要负向提示词"}),
                "image_model": (IMAGE_MODEL_OPTIONS, {
                    "default": "Auto",
                    "tooltip": "图像生成模型选择（通用类型，影响提示词构建风格）：\n• Auto：通用类型，自动适配默认模型\n• Flux1：擅长写实人像/场景/空间逻辑\n• Flux2_klein：擅长快速出图/多参考图编辑\n• Z_image：擅长写实人像/风光/静物\n• Krea2：擅长摄影质感/自然语言写实/电影感\n• Qwen_Image2512：擅长长图文/海报/密集排版\n• Boogu：擅长极简产品图/电商/室内渲染\n• ERNIE_Image：擅长文字渲染/社媒图文/商业海报\n• HiDream-O1-Image：擅长高清写实人像/时尚编辑级\n• Mage_Flow：擅长指令编辑/任意比例构图/双语文字渲染\n• LongCat_Image：擅长复杂场景/多角色/长文本描述理解\n• GLM_Image：擅长创意图解/科学科普/文字渲染"
                }),
                "video_model": (VIDEO_MODEL_OPTIONS, {
                    "default": "Auto",
                    "tooltip": "视频生成模型选择（仅视频类预设模板生效）：\n• Auto：自动适配默认模型\n• Wan2.2：原生单段最大5秒，仅支持单镜头\n• LTX2.5：原生单段最长20秒，支持音画同步\n• MiniMax-H3：时长4-15秒整数，支持音画同步"
                }),
                "audio_model": (AUDIO_MODEL_OPTIONS, {
                    "default": "Auto",
                    "tooltip": "音频生成模型选择（仅音频类预设模板生效）：\n• Auto：自动适配默认模型\n\n────────── TTS 语音合成 ──────────\n• IndexTTS-2.5：精细调情绪\n• VoxCPM2：日常短视频人物对话\n\n────────── 音乐生成 ──────────\n• Ace-Step1.5：控制维度丰富\n• MiniMax-Music3：人声演唱自然度高"
                }),
                
                # ========== 视频处理参数 ==========
                "video_max_frames": ("INT", {"default": 16, "min": 2, "max": 1024, "step": 1, 
                                              "tooltip": "视频模式：最大提取帧数"}),
                "video_sampling": (["Auto", "Manual"], 
                                  {"default": "Auto", 
                                   "tooltip": "视频帧采样方式：\n• Auto：自动均衡提取视频帧\n• Manual：自定义要提取的帧"}),
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
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "强制卸载模型释放显存"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "llama_model": ("LLAMACPPMODEL", {"tooltip": "加载的VL模型，用于图像理解和文本生成（API模式可不连接）"}),
                "api_config": ("OMNI_LLM_API_CONFIG", {"tooltip": "API 配置（用于 API 推理模式）"}),
                "parameters": ("LLAMACPPARAMS", {"tooltip": "额外的生成参数配置"}),
                "images": ("IMAGE", {"tooltip": "图像输入（用于图像理解模式）"}),
                "video": ("VIDEO", {"tooltip": "视频输入（用于视频理解模式）"}),
                "audio": ("AUDIO", {"tooltip": "音频输入（用于ASR识别）"}),
                "asr_model": ("ASRMODEL", {"tooltip": "ASR模型输入（用于语音识别）"}),
                "queue_handler": ("*", {"tooltip": "队列处理器"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "omni-llm"
    
    def __init__(self):
        self.engine = None
        self.model_info = None
        self.video_processor = VideoProcessor()
    
    def _build_preset_map(self):
        """构建预设键名到 (生成器实例, template_id) 的映射（类实例内置中英双语，无需语言拆分）"""
        return {
            "IMAGE_REVERSE_TAGS": (ImageReverseTags(), "image_reverse_tags"),
            "IMAGE_REVERSE_DESCRIBE": (ImageReverseDescribe(), "image_reverse_describe"),
            "PROMPT_EXPANDER": (PromptExpander(), "prompt_expander"),
            "ILLUSTRIOUS": (Illustrious(), "illustrious"),
            "ANIME_PROMPT": (AnimeDreamVisual(), "anime_dream_visual"),
            "THICKPAINT_ROLE": (ThickPaintRole(), "thickpaint_role"),
            "REALISTIC_FEMALE": (RealisticFemale(), "realistic_female"),
            "REALISTIC_MALE": (RealisticMale(), "realistic_male"),
            "WESTERN_FEMALE": (WesternFemale(), "western_female"),
            "WESTERN_MALE": (WesternMale(), "western_male"),
            "HYPER_REALISTIC_FEMALE": (HyperRealisticFemale(), "hyper_realistic_female"),
            "HYPER_REALISTIC_MALE": (HyperRealisticMale(), "hyper_realistic_male"),
            "YOUNG_BOY_PORTRAIT": (YoungBoyPortrait(), "young_boy_portrait"),
            "MIDDLE_ELDERLY_FEMALE": (MiddleElderlyFemale(), "middle_elderly_female"),
            "MIDDLE_ELDERLY_MALE": (MiddleElderlyMale(), "middle_elderly_male"),
            "ART_ILLUSTRATION": (ArtIllustration(), "art_illustration"),
            "POSTER_DESIGN": (PosterDesign(), "poster_design"),
            "SCENE_DESIGN": (SceneDesign(), "scene_design"),
            "INTERIOR_DESIGN": (InteriorDesign(), "interior_design"),
            "ARCHITECTURE_RENDERING": (ArchitectureRendering(), "architecture_rendering"),
            "ECOMMERCE": (Ecommerce(), "ecommerce"),
            "FOOD_PHOTOGRAPHY": (FoodPhotography(), "food_photography"),
            "CONTINUING_T2V": (ContinuingT2V(), "continuing_t2v"),
            "CONTINUING_I2V": (ContinuingI2V(), "continuing_i2v"),
            "CONTINUING_FL2V": (ContinuingFL2V(), "continuing_fl2v"),
            "CONTINUING_MULTI_STORYBOARD": (ContinuingMultiStoryboard(), "continuing_multi_storyboard"),
            "VIDEO_FRAME_SEQUENCE": (VideoFrameSequence(), "video_frame_sequence"),
            "VIDEO_TO_PROMPT": (VideoToPrompt(), "video_to_prompt"),
            "VIDEO_SCENE_BREAKDOWN": (VideoSceneBreakdown(), "video_storyboard_breakdown"),
            "VIDEO_SUBTITLE_FORMAT": (VideoSubtitleFormat(), "video_subtitle_format"),
            "MULTI_SPEAKER_DIALOGUE": (MultiSpeakerDialogue(), "multi_speaker_dialogue"),
            "SONG_CREATION": (SongCreation(), "song_creation"),
        }
    
    def _get_generator_entry(self, preset_key):
        """获取预设对应的 (生成器实例, template_id)，不存在返回 (None, None)"""
        return self._build_preset_map().get(preset_key, (None, None))
    
    def _is_generator_preset(self, preset_key):
        """判断预设是否为类生成器格式（新格式模板）"""
        generator, template_id = self._get_generator_entry(preset_key)
        return generator is not None and template_id is not None
    
    def _is_reverse_preset(self, preset_key):
        """判断是否为反推/无模型类预设（不支持模型切换）"""
        return preset_key in (
            "IMAGE_REVERSE_TAGS", "IMAGE_REVERSE_DESCRIBE", "VIDEO_TO_PROMPT",
            "VIDEO_FRAME_SEQUENCE", "VIDEO_SCENE_BREAKDOWN", "VIDEO_SUBTITLE_FORMAT",
        )
    
    def _is_audio_preset(self, preset_key):
        """判断是否为音频类预设（TTS语音合成/音乐生成，使用 audio_model 切换）"""
        return preset_key in ("MULTI_SPEAKER_DIALOGUE", "SONG_CREATION")
    
    def _resolve_audio_model(self, preset_key, audio_model):
        """按预设类型校验音频模型合法性：
        MULTI_SPEAKER_DIALOGUE 仅接受 TTS 模型；SONG_CREATION 仅接受音乐模型；
        非法选项（含 Auto 之外的错误模型）一律回退为 Auto
        """
        if preset_key == "MULTI_SPEAKER_DIALOGUE":
            return audio_model if audio_model in AUDIO_TTS_MODELS else "Auto"
        if preset_key == "SONG_CREATION":
            return audio_model if audio_model in AUDIO_MUSIC_MODELS else "Auto"
        return "Auto"
    
    def _resolve_effective_model(self, preset_key, image_model="Auto", video_model="Auto", audio_model="Auto"):
        """按预设模板内可用的模型接口解析实际生效模型，忽略其他类型的模型选择：
        - 反推/无模型预设 → Auto
        - 音频类预设 → 按 TTS/音乐分类校验 audio_model，非法回退 Auto
        - 视频类预设 → video_model 必须在接口库内，否则回退库内默认模型（库为空回退 Auto）
        - 图像类预设 → image_model 必须在接口库内，否则回退库内默认模型（库为空回退 Auto）
        保证图像/视频/音频三个选项同时切换时，只按预设自身的模型接口输出且不产生错误
        """
        if not preset_key or self._is_reverse_preset(preset_key):
            return "Auto"
        if self._is_audio_preset(preset_key):
            return self._resolve_audio_model(preset_key, audio_model)
        generator, _ = self._get_generator_entry(preset_key)
        if generator is None:
            return "Auto"
        # 视频类预设：仅使用 video_model 接口
        if self._is_video_preset(preset_key):
            lib = generator.video_model_formula_library
            return video_model if video_model in lib else (next(iter(lib)) if lib else "Auto")
        # 图像类预设：仅使用 image_model 接口
        if self._is_image_model_preset(preset_key):
            lib = generator.model_formula_library
            return image_model if image_model in lib else (next(iter(lib)) if lib else "Auto")
        return "Auto"
    
    def _is_video_preset(self, preset_key):
        """判断是否为视频类预设（使用 video_model 切换）"""
        generator, _ = self._get_generator_entry(preset_key)
        return generator is not None and hasattr(generator, 'video_model_formula_library')
    
    def _is_image_model_preset(self, preset_key):
        """判断是否为图像模型类预设（使用 image_model 切换）"""
        generator, _ = self._get_generator_entry(preset_key)
        return generator is not None and hasattr(generator, 'model_formula_library')
    
    def _get_generator_negative_prompt(self, preset_key, language, model="Auto"):
        """
        获取类生成器格式的负向提示词
        反推/无模型类预设返回空字符串
        """
        generator, template_id = self._get_generator_entry(preset_key)
        if generator is None:
            return ""
        lang_code = "zh" if language == "中文" else "en"
        
        # 视频类预设
        if hasattr(generator, 'video_model_formula_library'):
            if model not in generator.video_model_formula_library:
                model = next(iter(generator.video_model_formula_library)) if generator.video_model_formula_library else "Auto"
            result = generator.build_prompt(
                user_input="",
                preset_name=template_id,
                video_model=model,
                output_language=lang_code,
                enable_global_preconstraint=False,
                enable_negative_prompt=True,
                output_format="both"
            )
            return result.get("negative_prompt", "")
        
        # 图像类预设
        if hasattr(generator, 'model_formula_library'):
            if model not in generator.model_formula_library:
                model = next(iter(generator.model_formula_library)) if generator.model_formula_library else "Auto"
            result = generator.build_prompt(
                user_input="",
                preset_name=template_id,
                downstream_model=model,
                output_language=lang_code,
                enable_global_preconstraint=False,
                enable_negative_prompt=True,
                output_format="both"
            )
            return result.get("negative_prompt", "")
        
        # 反推/无模型预设：尝试提取负向提示词
        try:
            result = generator.build_prompt(
                preset_name=template_id,
                user_keywords="",
                output_language=lang_code,
                output_format="both"
            )
            return result.get("negative_prompt", "")
        except Exception:
            try:
                result = generator.build_prompt(
                    user_input="",
                    preset_name=template_id,
                    output_language=lang_code,
                    enable_global_preconstraint=False,
                    enable_negative_prompt=True,
                    output_format="both"
                )
                return result.get("negative_prompt", "")
            except Exception:
                return ""
    
    def _get_generator_positive_prompt(self, preset_key, language, model="Auto"):
        """
        获取类生成器格式的正向约束
        反推/无模型类预设返回空字符串
        """
        generator, template_id = self._get_generator_entry(preset_key)
        if generator is None:
            return ""
        lang_code = "zh" if language == "中文" else "en"
        
        # 视频类预设
        if hasattr(generator, 'video_model_formula_library'):
            if model not in generator.video_model_formula_library:
                model = next(iter(generator.video_model_formula_library)) if generator.video_model_formula_library else "Auto"
            result = generator.build_prompt(
                user_input="",
                preset_name=template_id,
                video_model=model,
                output_language=lang_code,
                enable_global_preconstraint=True,
                enable_negative_prompt=False,
                output_format="both"
            )
            return f"【下游模型】{model}\n{result.get('positive_constraint', '')}"
        
        # 图像类预设
        if hasattr(generator, 'model_formula_library'):
            if model not in generator.model_formula_library:
                model = next(iter(generator.model_formula_library)) if generator.model_formula_library else "Auto"
            result = generator.build_prompt(
                user_input="",
                preset_name=template_id,
                downstream_model=model,
                output_language=lang_code,
                enable_global_preconstraint=True,
                enable_negative_prompt=False,
                output_format="both"
            )
            return f"【下游模型】{model}\n{result.get('positive_constraint', '')}"
        
        # 音频类预设：使用 audio_model，输出附带下游模型标识
        if self._is_audio_preset(preset_key):
            model = self._resolve_audio_model(preset_key, model)
            try:
                result = generator.build_prompt(
                    preset_name=template_id,
                    user_keywords="",
                    output_language=lang_code,
                    output_format="both"
                )
                positive = result.get("positive_constraint", "")
                if model != "Auto":
                    return f"【下游模型】{model}\n{positive}"
                return positive
            except Exception:
                return ""
        
        # 反推/无模型预设：支持正向约束
        try:
            result = generator.build_prompt(
                preset_name=template_id,
                user_keywords="",
                output_language=lang_code,
                output_format="both"
            )
            return result.get("positive_constraint", "")
        except Exception:
            try:
                result = generator.build_prompt(
                    user_input="",
                    preset_name=template_id,
                    output_language=lang_code,
                    enable_global_preconstraint=True,
                    enable_negative_prompt=False,
                    output_format="both"
                )
                return result.get("positive_constraint", "")
            except Exception:
                return ""
    
    def _get_generator_model_marker(self, preset_key, model="Auto"):
        """
        获取类生成器格式的下游模型标识（【下游模型】xxx）
        仅类生成器预设且解析到具体模型时返回标识，反推/无模型预设或 Auto 返回空字符串
        该标识与正向约束开关解耦，未启用正向约束时也可单独输出
        """
        if not self._is_generator_preset(preset_key):
            return ""
        if model in ("Auto", ""):
            return ""
        return f"【下游模型】{model}"
    
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
    
    def get_preset_text_by_language(self, preset_key, language, output_format="natural", input_mode="text",
                                    custom_prompt="", model="Auto",
                                    enable_constraints=True, enable_negative_prompts=True):
        """
        根据语言和输出格式获取预设提示词文本

        使用类生成器格式（build_prompt）动态构建提示词，内置中英双语支持。
        反推类生成器（无模型参数）单独适配。

        Args:
            preset_key: 预设键名
            language: 语言（"中文" 或 "English"）
            output_format: 输出格式（"structured" 或 "natural"）
            input_mode: 输入模式（"text"、"images"、"video"）
            custom_prompt: 用户输入文本（内嵌到提示词）
            model: 模型名称（图像模型或视频模型）
            enable_constraints: 是否启用正向约束
            enable_negative_prompts: 是否启用负向提示词

        Returns:
            str: 格式化后的预设提示词文本
        """
        generator, template_id = self._get_generator_entry(preset_key)
        if generator is None:
            return preset_key

        lang_code = "zh" if language == "中文" else "en"

        # 无 structured 输出接口的预设（视频/音频类等）一律强制 natural 输出
        if "structured" not in (getattr(generator, "format_guide", None) or {}):
            output_format = "natural"

        # 视频类预设：使用 video_model
        if hasattr(generator, 'video_model_formula_library'):
            if model not in generator.video_model_formula_library:
                model = next(iter(generator.video_model_formula_library)) if generator.video_model_formula_library else "Auto"
            result = generator.build_prompt(
                user_input=custom_prompt,
                preset_name=template_id,
                video_model=model,
                output_language=lang_code,
                enable_global_preconstraint=enable_constraints,
                enable_negative_prompt=enable_negative_prompts,
                output_format=output_format
            )
            print(f"【类生成器-视频】{preset_key} → 视频模型={model}，语言={lang_code}，格式={output_format}")
            return result["llm_input_prompt"]

        # 图像类预设：使用 downstream_model
        if hasattr(generator, 'model_formula_library'):
            if model not in generator.model_formula_library:
                model = next(iter(generator.model_formula_library)) if generator.model_formula_library else "Auto"
            result = generator.build_prompt(
                user_input=custom_prompt,
                preset_name=template_id,
                downstream_model=model,
                output_language=lang_code,
                enable_global_preconstraint=enable_constraints,
                enable_negative_prompt=enable_negative_prompts,
                output_format=output_format
            )
            print(f"【类生成器-图像】{preset_key} → 下游模型={model}，语言={lang_code}，格式={output_format}")
            # 前置输出实际生效的下游模型名称，与内容组织公式保持一致，便于用户判定模型选择是否正确
            return f"【目标模型适配：{model}】\n{result['llm_input_prompt']}"

        # 音频类预设：使用 audio_model（生成器无模型库，提示词末尾追加目标模型指示）
        if self._is_audio_preset(preset_key):
            model = self._resolve_audio_model(preset_key, model)
            try:
                result = generator.build_prompt(
                    preset_name=template_id,
                    user_keywords=custom_prompt,
                    output_language=lang_code,
                    output_format=output_format
                )
                prompt = result["llm_input_prompt"]
                if model and model != "Auto":
                    prompt += f"\n\n【目标音频模型】{model}"
                print(f"【类生成器-音频】{preset_key} → 音频模型={model}，语言={lang_code}，格式={output_format}")
                return prompt
            except Exception as e:
                print(f"【类生成器-音频】{preset_key} 调用 build_prompt 失败: {e}")

        # 反推/无模型预设：使用 preset_name + user_keywords 签名
        try:
            result = generator.build_prompt(
                preset_name=template_id,
                user_keywords=custom_prompt,
                output_language=lang_code,
                output_format=output_format
            )
            print(f"【类生成器-反推】{preset_key} → 语言={lang_code}，格式={output_format}")
            return result["llm_input_prompt"]
        except Exception:
            # 尝试带 user_input 的签名（AnimeDreamVisual, Illustrious 等）
            try:
                result = generator.build_prompt(
                    user_input=custom_prompt,
                    preset_name=template_id,
                    output_language=lang_code,
                    enable_global_preconstraint=enable_constraints,
                    enable_negative_prompt=enable_negative_prompts,
                    output_format=output_format
                )
                print(f"【类生成器-通用】{preset_key} → 语言={lang_code}，格式={output_format}")
                return result["llm_input_prompt"]
            except Exception as e:
                print(f"【类生成器错误】{preset_key} 调用 build_prompt 失败: {e}")
                return preset_key
    
    def get_preset_examples(self, preset_key, language, output_format="natural", custom_prompt=""):
        """
        获取预设模板的示例内容（新类生成器格式无示例字段，返回空字符串）
        """
        return ""
    
    def get_preset_constraints(self, preset_key, language):
        """
        获取预设模板中的正向约束和负向提示词
        （新类生成器格式已通过 _get_generator_positive/negative_prompt 处理，此方法作为兜底返回空）
        
        Returns:
            tuple: (positive_constraints, negative_prompts)
        """
        return "", ""
    
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
    @staticmethod
    def _call_api_chat_completion(api_base_url, api_key, api_model, messages, params):
        """调用外部 API（OpenAI 兼容格式）"""
        import requests as _requests

        if not api_key:
            raise RuntimeError("API 密钥为空，请在 API 配置节点中填写 api_key")
        if not api_base_url:
            raise RuntimeError("API 地址为空，请在 API 配置节点中填写 api_base")
        if not api_model:
            raise RuntimeError("模型名称为空，请在 API 配置节点中填写 model_name")

        base = api_base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        elif "/v1/" in base:
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": api_model,
            "messages": messages,
            "max_tokens": params.get("max_tokens", 1024),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "stream": False,
        }
        if params.get("seed", -1) >= 0:
            payload["seed"] = params["seed"]

        timeout = int(params.get("timeout", 300))

        def _do_request():
            try:
                return _requests.post(url, headers=headers, json=payload, timeout=timeout)
            except _requests.ProxyError:
                # 系统代理不可用（如 Clash 未启动）时绕过代理直连重试（国内 API 通常无需代理）
                try:
                    return _requests.post(url, headers=headers, json=payload, timeout=timeout,
                                          proxies={"http": None, "https": None})
                except _requests.ConnectionError as e:
                    raise RuntimeError(f"网络连接失败，无法访问 {api_base_url}（已尝试直连和系统代理）。请检查网络、VPN/代理、防火墙设置。") from e
            except _requests.ConnectionError as e:
                raise RuntimeError(f"网络连接失败，无法访问 {api_base_url}。请检查网络、VPN/代理、防火墙设置。") from e
            except _requests.Timeout as e:
                raise RuntimeError(f"API 请求超时（{timeout}秒）。请检查网络或稍后重试。") from e
            except _requests.RequestException as e:
                raise RuntimeError(f"API 请求异常：{e}") from e

        # 推理模型（如 deepseek-flash）的思考 token 也计入 max_tokens，复杂预设提示词可能导致
        # 思考过程占满配额、正文 content 为空（finish_reason=length），此时自动翻倍配额重试
        max_tokens_cap = 8192
        while True:
            resp = _do_request()

            if resp.status_code != 200:
                error_msg = resp.text[:500]
                status_hints = {
                    400: f"请求参数错误：{error_msg}",
                    401: "API 密钥无效或已过期，请检查 api_key",
                    402: "API 余额不足，请充值后重试",
                    403: "没有 API 访问权限",
                    404: f"接口不存在：{url}",
                    429: "请求过于频繁，请稍后重试",
                    500: "API 服务端异常，请稍后重试",
                    502: "API 服务暂时不可用，请稍后重试",
                    503: "API 服务繁忙，请稍后重试",
                }
                hint = status_hints.get(resp.status_code, f"HTTP {resp.status_code}")
                raise RuntimeError(f"API 错误 {resp.status_code}：{hint}")

            try:
                result = resp.json()
            except Exception:
                raise RuntimeError(f"API 响应解析失败：{resp.text[:300]}")

            choices = result.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"API 返回格式异常，无 choices 字段：{str(result)[:300]}")

            first_choice = choices[0]
            content = first_choice.get("message", {}).get("content")
            if content is None:
                content = first_choice.get("text", "")

            current_max_tokens = payload["max_tokens"]
            if (not content) and first_choice.get("finish_reason") == "length" \
                    and current_max_tokens < max_tokens_cap:
                payload["max_tokens"] = min(current_max_tokens * 2, max_tokens_cap)
                print(f"【API推理】思考过程占满 max_tokens={current_max_tokens} 导致正文为空，"
                      f"自动提升至 {payload['max_tokens']} 重试...")
                continue

            return content or ""

    def _run_inference(self, llama_model, messages, gen_params):
        """执行推理的内部方法，用于异步处理"""
        generated_text = ""
        
        # 生成缓存键
        model_path = getattr(llama_model, 'model_path', str(llama_model))
        
        # 如果没有选择模型，直接返回空结果
        if not model_path or model_path.lower() == "none":
            print("【推理】未选择模型，跳过推理")
            return generated_text
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
        result = generated_text
        self.add_to_cache("_inference_cache", cache_key, result)
        
        return generated_text
    
    def process(self, inference_mode, preset_prompt, system_prompt, text_input,
                prompt_language, response_language, output_format, enable_constraints,
                enable_negative_prompts,
                video_max_frames, video_sampling, video_manual_indices, image_max_size, batch_combination,
                force_offload,
                api_config=None, parameters=None, images=None, video=None, audio=None,
                asr_model=None, queue_handler=None, unique_id=None,
                image_model="Auto", video_model="Auto", audio_model="Auto",
                llama_model=None):
        """处理推理请求"""
        try:
            # 检测模型类型（无模型时使用"none"类型，走API推理路径）
            if llama_model is None:
                self.model_info = {
                    "key": "none",
                    "type": "none",
                    "subtype": "none",
                    "supports_audio": False,
                    "supports_vision": False,
                    "file_formats": []
                }
            else:
                self.model_info = self.detect_model_type(llama_model)
            
            # 处理推理模式
            mode_map = {
                "[基础] 文本生成 (Text Generation)": "text",
                "[基础] 图像理解 (Image Understanding)": "images",
                "[基础] 批量图像理解 (Batch Image Understanding)": "batch_images",
                "[基础] 音频转文本 (Audio to Text)": "audio",
                "[高级] 视频理解 (Video Understanding)": "video"
            }
            mode = mode_map.get(inference_mode, "text")
            
            # 检查输入
            has_images = images is not None and (hasattr(images, 'numel') and images.numel() > 0)
            has_video = self.video_processor._check_video_input(video)
            has_audio = audio is not None
            
            # 处理自定义提示词
            custom_prompt = text_input
            
            # 检查是否启用ASR
            enable_asr = mode in ["audio"]
            
            # 检查是否有ASR模型
            has_asr_model = asr_model is not None
            
            # 检查是否有音频输入
            has_audio_input = has_audio
            
            # 语言设置
            preset_prompts_language = prompt_language
            output_language = response_language
            
            # 获取预设提示词
            preset_key = self.preset_prompts.get(preset_prompt, "")
            
            # 模型切换逻辑：按预设模板可用的模型接口解析实际模型（图像/视频/音频可同时切换，互不干扰）
            effective_model = self._resolve_effective_model(preset_key, image_model, video_model, audio_model)
            
            print(f"【模型切换】{preset_prompt} → 实际模型={effective_model}（image_model={image_model}，video_model={video_model}，audio_model={audio_model}）")
            
            # 获取预设提示词文本
            preset_text = self.get_preset_text_by_language(
                preset_key, preset_prompts_language, output_format, mode,
                custom_prompt=custom_prompt,
                model=effective_model,
                enable_constraints=enable_constraints,
                enable_negative_prompts=enable_negative_prompts
            )
            # 新格式模板的正向约束
            generator_positive_prompt = self._get_generator_positive_prompt(
                preset_key, output_language, effective_model
            )
            # 新格式模板的负向提示词
            generator_negative_prompt = self._get_generator_negative_prompt(
                preset_key, output_language, effective_model
            )
            
            # 构建最终提示词
            if preset_prompt == "Empty - Nothing":
                final_prompt = custom_prompt.strip() if custom_prompt.strip() else system_prompt.strip()
            elif self._is_generator_preset(preset_key):
                # 类生成器格式：用户输入已内嵌在生成器输出的提示词中，直接使用
                final_prompt = preset_text
            else:
                # 兜底：直接使用 preset_text
                final_prompt = preset_text
            if output_language == "中文":
                final_prompt += "\n\n【输出语言硬性要求】最终输出的全部内容（包括所有标签、字段、描述文本）必须使用中文撰写，禁止输出任何英文或其他语言内容。"
            elif output_language == "English":
                final_prompt += "\n\n[Output Language Hard Requirement] All final output content (including all tags, fields and description text) must be written in English only. Do not output any Chinese or other language content."
            
            # MiMo-VL模型在文本生成模式下需要特殊指令，避免输出思考过程
            try:
                from common import LLAMA_CPP_STORAGE
                if LLAMA_CPP_STORAGE.current_config:
                    chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                    is_mimo_vl = "mimo" in chat_handler.lower()
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
            
            # 处理无模型情况：优先 API 推理，其次 ASR，最后返回空
            if self.model_info["type"] == "none":
                if mode == "audio" and asr_text:
                    print("【无模型模式】音频转文本模式，仅返回ASR识别结果")
                    return (generated_text, [generated_text], 0)

                # 尝试 API 推理
                api_base_url = ""
                api_key = ""
                api_model = ""
                api_timeout = 300
                if isinstance(api_config, dict):
                    api_base_url = api_config.get("base_url", "")
                    api_key = api_config.get("api_key", "")
                    api_model = api_config.get("model", "")
                    api_timeout = int(api_config.get("timeout", 300))

                if api_base_url and api_key and api_model:
                    print(f"【API推理】未加载本地模型，使用 API 模式（model={api_model}）")
                    api_messages = [{"role": "system", "content": system_prompt.strip()}]
                    api_messages.append({"role": "user", "content": final_prompt})
                    api_params = {
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "timeout": api_timeout,
                    }
                    if parameters:
                        api_params.update({k: v for k, v in parameters.items()
                                           if k in ("max_tokens", "temperature", "top_p", "seed", "timeout")})
                    try:
                        generated_text = self._call_api_chat_completion(
                            api_base_url, api_key, api_model, api_messages, api_params
                        )
                        generated_text = self._filter_thinking_content(generated_text)
                        _uid = parameters.get("state_uid", None) if parameters else None
                        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
                        return (generated_text, [generated_text], int(uid))
                    except Exception as e:
                        print(f"【API推理错误】{e}")
                        return (str(e), [str(e)], 0)
                else:
                    print(f"【无模型模式】模式: {mode}，未选择LLM模型且无API配置，返回空结果")
                    return ("", [""], 0)

            # 非音频转文本模式清空预置文本
            if mode != "audio":
                generated_text = ""

            # 本地模型优先：当本地模型已加载时，忽略 API 配置
            if isinstance(api_config, dict) and api_config.get("base_url") and api_config.get("api_key"):
                print("【模式选择】本地模型已加载，忽略 API 配置，使用本地模型推理")

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
                    is_qwen3_series = chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking", "Qwen3-VL"]
                    is_mimo_vl = "mimo" in chat_handler.lower()
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
                        # 思考模式由ChatHandler初始化时控制（Qwen3-VL: force_reasoning；MiMo-VL模板无思考机制）
                        # 注意：create_chat_completion不接受enable_thinking参数，此处不再设置
                        # 添加思考结束标记作为停止序列
                        stop_list = gen_params.get("stop", [])
                        if isinstance(stop_list, str):
                            stop_list = [stop_list]
                        stop_list.extend(["<|end_of_thinking|>", "<|end_of_solution|>", "</think>"])
                        gen_params["stop"] = stop_list
                        print(f"【Qwen3优化】调整参数: n_batch={gen_params['n_batch']}, max_tokens={gen_params['max_tokens']}, top_p={gen_params['top_p']}")
            except Exception as e:
                print(f"【Qwen3优化】内存参数调整时出错（忽略）: {e}")
                self._perf_params_cache[cache_key] = gen_params.copy()
            
            # 应用用户自定义参数
            if parameters:
                gen_params.update({k: v for k, v in parameters.items() if k != "state_uid"})
            
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
                    if LLAMA_CPP_STORAGE.llm and current_chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
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
                
                if current_chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
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
                if current_chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
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
                    is_qwen3_model = current_chat_handler_name in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking"]
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
                    print(f"【Qwen3.5/3.6/3.8优化】文本生成模式，强制使用纯文本格式")
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
                        
                        # Qwen3系列模型特殊内存清理（防止多余内容）
                        try:
                            from common import LLAMA_CPP_STORAGE
                            if LLAMA_CPP_STORAGE.current_config and LLAMA_CPP_STORAGE.llm is not None:
                                chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                                if chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
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
                        if generator_negative_prompt:
                            # 新格式模板：正向约束与负向提示词均由生成器返回，正向约束附带下游模型标识
                            positive_constraints = generator_positive_prompt
                            negative_prompts = generator_negative_prompt
                        elif (enable_constraints or enable_negative_prompts) and preset_key:
                            positive_constraints, negative_prompts = self.get_preset_constraints(preset_key, output_language)
                        
                        # 下游模型标识：类生成器预设始终输出，不依赖正向约束开关
                        model_marker = self._get_generator_model_marker(preset_key, effective_model)
                        
                        if enable_constraints and positive_constraints:
                            generated_text = f"【正向约束】{positive_constraints}\n\n{generated_text}"
                        elif model_marker:
                            generated_text = f"{model_marker}\n\n{generated_text}"
                        
                        if enable_negative_prompts and negative_prompts:
                            generated_text += f"\n\n【负向提示词】{negative_prompts}"
                        
                        if enable_constraints or enable_negative_prompts or model_marker:
                            new_output_list = []
                            for item in output_list:
                                new_item = item
                                if enable_constraints and positive_constraints:
                                    new_item = f"【正向约束】{positive_constraints}\n\n{new_item}"
                                elif model_marker:
                                    new_item = f"{model_marker}\n\n{new_item}"
                                if enable_negative_prompts and negative_prompts:
                                    new_item += f"\n\n【负向提示词】{negative_prompts}"
                                new_output_list.append(new_item)
                            output_list = new_output_list
                        
                        # 直接返回结果
                        _uid = parameters.get("state_uid", None) if parameters else None
                        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
                        return (generated_text, output_list, int(uid))
            
            # 执行推理（异步）
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            if not generated_text:
                # 使用线程池执行推理
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._run_inference,
                        llama_model=llama_model,
                        messages=messages,
                        gen_params=gen_params
                    )
                    generated_text = future.result()

            # Qwen3系列模型特殊内存清理（防止多余内容）
            try:
                from common import LLAMA_CPP_STORAGE
                if LLAMA_CPP_STORAGE.current_config and LLAMA_CPP_STORAGE.llm is not None:
                    chat_handler = LLAMA_CPP_STORAGE.current_config.get("chat_handler", "")
                    if chat_handler in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking", "Qwen3.8", "Qwen3.8-Thinking", "Qwen3-VL", "MiMo-VL-7B-RL", "MiMo-VL-7B-RL-2508"]:
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
            if generator_negative_prompt:
                # 新格式模板：正向约束与负向提示词均由生成器返回，正向约束附带下游模型标识
                positive_constraints = generator_positive_prompt
                negative_prompts = generator_negative_prompt
            elif (enable_constraints or enable_negative_prompts) and preset_key:
                positive_constraints, negative_prompts = self.get_preset_constraints(preset_key, output_language)
            
            # 下游模型标识：类生成器预设始终输出，不依赖正向约束开关
            model_marker = self._get_generator_model_marker(preset_key, effective_model)
            
            if enable_constraints and positive_constraints:
                generated_text = f"【正向约束】{positive_constraints}\n\n{generated_text}"
            elif model_marker:
                generated_text = f"{model_marker}\n\n{generated_text}"
            
            if enable_negative_prompts and negative_prompts:
                generated_text += f"\n\n【负向提示词】{negative_prompts}"

            # 处理UID
            _uid = parameters.get("state_uid", None) if parameters else None
            uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid

            return (generated_text, [generated_text], int(uid))
        
        except Exception as e:
            error_message = ErrorHandler.handle_error(e, context={"mode": mode, "model_type": self.model_info.get("type", "unknown")})
            print(f"【处理错误】{str(e)}")
            return (error_message, [error_message], locals().get("gen_params", {}).get("seed", 0))
    
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
    "omni_llm_unified_inference": omni_llm_unified_inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_unified_inference": "Omni LLM Unified Inference",
}