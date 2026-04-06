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

# 导入预设提示词库
from support.prompt_enhancer_preset_zh import (
    WAN_T2V_ZH,
    WAN_I2V_ZH,
    WAN_I2V_EMPTY_ZH,
    WAN_FLF2V_ZH,
    QWEN_IMAGE_LAYERED_ZH,
    QWEN_IMAGE_EDIT_COMBINED_ZH,
    QWEN_IMAGE_2512_ZH,
    ZIMAGE_TURBO_ZH,
    FLUX2_KLEIN_ZH,
    LTX2_ZH,
    VIDEO_TO_PROMPT_ZH,
    VIDEO_DETAILED_SCENE_BREAKDOWN_ZH,
    VIDEO_SUBTITLE_FORMAT_ZH,
    OCR_ENHANCED_ZH,
    ULTRA_HD_IMAGE_REVERSE_ZH,
    NORMAL_DESCRIBE_ZH,
    PROMPT_STYLE_TAGS_ZH,
    PROMPT_STYLE_SIMPLE_ZH,
    PROMPT_STYLE_DETAILED_ZH,
    PROMPT_STYLE_COMPREHENSIVE_ZH,
    CREATIVE_DETAILED_ANALYSIS_ZH,
    CREATIVE_SUMMARIZE_VIDEO_ZH,
    CREATIVE_SHORT_STORY_ZH,
    CREATIVE_REFINE_EXPAND_PROMPT_ZH,
    VISION_BOUNDING_BOX_ZH,
    AUDIO_SUBTITLE_CONVERT_ZH,
    VIDEO_TO_AUDIO_SUBTITLE_ZH,
    TEXT_TO_AUDIO_ZH,
    AUDIO_ANALYSIS_ZH,
    MULTI_SPEAKER_DIALOGUE_ZH,
    AUDIO_TO_TEXT_ZH,
    LYRICS_AND_AUDIO_MERGE_ZH,
)

from support.prompt_enhancer_preset_en import (
    WAN_T2V_EN,
    WAN_I2V_EN,
    WAN_I2V_EMPTY_EN,
    WAN_FLF2V_EN,
    FLUX2_KLEIN_EN,
    LTX2_EN,
    QWEN_IMAGE_LAYERED_EN,
    QWEN_IMAGE_EDIT_COMBINED_EN,
    QWEN_IMAGE_2512_EN,
    ZIMAGE_TURBO_EN,
    VIDEO_TO_PROMPT_EN,
    VIDEO_DETAILED_SCENE_BREAKDOWN_EN,
    VIDEO_SUBTITLE_FORMAT_EN,
    ULTRA_HD_IMAGE_REVERSE_EN,
    NORMAL_DESCRIBE_EN,
    PROMPT_STYLE_TAGS_EN,
    PROMPT_STYLE_SIMPLE_EN,
    PROMPT_STYLE_DETAILED_EN,
    PROMPT_STYLE_COMPREHENSIVE_EN,
    CREATIVE_DETAILED_ANALYSIS_EN,
    CREATIVE_SUMMARIZE_VIDEO_EN,
    CREATIVE_SHORT_STORY_EN,
    CREATIVE_REFINE_EXPAND_PROMPT_EN,
    VISION_BOUNDING_BOX_EN,
    AUDIO_SUBTITLE_CONVERT_EN,
    VIDEO_TO_AUDIO_SUBTITLE_EN,
    TEXT_TO_AUDIO_EN,
    AUDIO_ANALYSIS_EN,
    MULTI_SPEAKER_DIALOGUE_EN,
    AUDIO_TO_TEXT_EN,
    LYRICS_AND_AUDIO_MERGE_EN,
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
    
    # 添加分类预设提示词
    preset_prompts["[Normal] Describe"] = "NORMAL_DESCRIBE"
    preset_prompts["[Text to Image] ZIMAGE - Turbo"] = "ZIMAGE_TURBO"
    preset_prompts["[Text to Image] FLUX2 - Klein"] = "FLUX2_KLEIN"
    preset_prompts["[Text to Image] Qwen - Image 2512"] = "QWEN_IMAGE_2512"
    preset_prompts["[Image Edit] Qwen - Image Edit Combined"] = "QWEN_IMAGE_EDIT_COMBINED"
    preset_prompts["[Image Edit] Qwen - Image Layered"] = "QWEN_IMAGE_LAYERED"
    preset_prompts["[Text to Video] LTX-2"] = "LTX2"
    preset_prompts["[Text to Video] WAN - Text to Video"] = "WAN_T2V"
    preset_prompts["[Image to Video] WAN - Image to Video"] = "WAN_I2V"
    preset_prompts["[Image to Video] WAN - Image to Video Empty"] = "WAN_I2V_EMPTY"
    preset_prompts["[Image to Video] WAN - FLF to Video"] = "WAN_FLF2V"
    preset_prompts["[Prompt Style] Tags"] = "PROMPT_STYLE_TAGS"
    preset_prompts["[Prompt Style] Simple"] = "PROMPT_STYLE_SIMPLE"
    preset_prompts["[Prompt Style] Detailed"] = "PROMPT_STYLE_DETAILED"
    preset_prompts["[Prompt Style] Comprehensive Expansion"] = "PROMPT_STYLE_COMPREHENSIVE"
    preset_prompts["[Creative] Refine & Expand Prompt"] = "CREATIVE_REFINE_EXPAND_PROMPT"
    preset_prompts["[Creative] Detailed Analysis"] = "CREATIVE_DETAILED_ANALYSIS"
    preset_prompts["[Creative] Summarize Video"] = "CREATIVE_SUMMARIZE_VIDEO"
    preset_prompts["[Creative] Short Story"] = "CREATIVE_SHORT_STORY"
    preset_prompts["[Video Analysis] Video - Reverse Prompt"] = "VIDEO_TO_PROMPT"
    preset_prompts["[Video Analysis] Video - Detailed Scene Breakdown"] = "VIDEO_DETAILED_SCENE_BREAKDOWN"
    preset_prompts["[Video Analysis] Video - Subtitle Format"] = "VIDEO_SUBTITLE_FORMAT"
    preset_prompts["[Vision] Bounding Box"] = "VISION_BOUNDING_BOX"
    preset_prompts["[OCR] Enhanced OCR"] = "OCR_ENHANCED"
    preset_prompts["[HighRes] Ultra HD Image Reverse"] = "ULTRA_HD_IMAGE_REVERSE"
    preset_prompts["[Audio] Audio ↔ Subtitle Convert"] = "AUDIO_SUBTITLE_CONVERT"
    preset_prompts["[Audio] Video to Audio & Subtitle"] = "VIDEO_TO_AUDIO_SUBTITLE"
    preset_prompts["[Audio] Text to Audio"] = "TEXT_TO_AUDIO"
    preset_prompts["[Audio] Audio Analysis"] = "AUDIO_ANALYSIS"
    preset_prompts["[Audio] Multi-Person Dialogue"] = "MULTI_SPEAKER_DIALOGUE"
    preset_prompts["[Audio] Audio to Text"] = "AUDIO_TO_TEXT"
    preset_prompts["[Music] Lyrics & Audio Merge"] = "LYRICS_AND_AUDIO_MERGE"
    
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
                        "[基础] 音频转文本 (Audio to Text)",
                        "[基础] 文本转音频 (Text to Audio)",
                        "[高级] 全模态整合 (Multimodal Integration)",
                        "[高级] 视频理解 (Video Understanding)"
                    ], {
                        "default": "[基础] 文本生成 (Text Generation)", 
                        "tooltip": "选择推理模式：\n• [基础] 文本生成：使用语言模型生成文本内容\n• [基础] 图像理解：处理图像内容并生成描述\n• [基础] 音频转文本：使用ASR模型将音频转换为文本\n• [基础] 文本转音频：生成文本后使用TTS模型转换为语音\n• [高级] 全模态整合：同时处理图像、音频和文本（Omni模型专用）\n• [高级] 视频理解：从视频中提取帧并进行分析"
                    }),
                
                # ========== 提示词配置 ==========
                "preset_prompt": (s.preset_tags, {"default": s.preset_tags[1], "tooltip": "选择预设提示词模板"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "你是一位优秀的多模态助手。", "tooltip": "系统提示词，定义AI助手的角色和行为，可包含预设模板占位符#和自定义内容"}),
                "text_input": ("STRING", {"default": "", "multiline": True, "tooltip": "用户输入文本，作为对话的用户消息内容"}),
                
                # ========== 语言设置 ==========
                "prompt_language": (["中文", "English"], {"default": "中文", "tooltip": "预设提示词的语言"}),
                "response_language": (["自动检测", "中文", "English"], {"default": "自动检测", "tooltip": "AI回复的语言"}),
                
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
                
                # ========== TTS语音参数 ==========
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
                "save_states": ("BOOLEAN", {"default": False, "tooltip": "保存模型状态以便恢复"}),
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
    
    def get_preset_text_by_language(self, preset_key, language):
        """根据语言获取预设提示词文本"""
        if language == "中文":
            preset_map = {
                "NORMAL_DESCRIBE": NORMAL_DESCRIBE_ZH,
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
                "OCR_ENHANCED": OCR_ENHANCED_ZH,
                "ULTRA_HD_IMAGE_REVERSE": ULTRA_HD_IMAGE_REVERSE_ZH,
                "PROMPT_STYLE_TAGS": PROMPT_STYLE_TAGS_ZH,
                "PROMPT_STYLE_SIMPLE": PROMPT_STYLE_SIMPLE_ZH,
                "PROMPT_STYLE_DETAILED": PROMPT_STYLE_DETAILED_ZH,
                "PROMPT_STYLE_COMPREHENSIVE": PROMPT_STYLE_COMPREHENSIVE_ZH,
                "CREATIVE_DETAILED_ANALYSIS": CREATIVE_DETAILED_ANALYSIS_ZH,
                "CREATIVE_SUMMARIZE_VIDEO": CREATIVE_SUMMARIZE_VIDEO_ZH,
                "CREATIVE_SHORT_STORY": CREATIVE_SHORT_STORY_ZH,
                "CREATIVE_REFINE_EXPAND_PROMPT": CREATIVE_REFINE_EXPAND_PROMPT_ZH,
                "VISION_BOUNDING_BOX": VISION_BOUNDING_BOX_ZH,
                "AUDIO_SUBTITLE_CONVERT": AUDIO_SUBTITLE_CONVERT_ZH,
                "VIDEO_TO_AUDIO_SUBTITLE": VIDEO_TO_AUDIO_SUBTITLE_ZH,
                "TEXT_TO_AUDIO": TEXT_TO_AUDIO_ZH,
                "AUDIO_ANALYSIS": AUDIO_ANALYSIS_ZH,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_ZH,
                "AUDIO_TO_TEXT": AUDIO_TO_TEXT_ZH,
                "LYRICS_AND_AUDIO_MERGE": LYRICS_AND_AUDIO_MERGE_ZH,
            }
        else:
            preset_map = {
                "NORMAL_DESCRIBE": NORMAL_DESCRIBE_EN,
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
                "ULTRA_HD_IMAGE_REVERSE": ULTRA_HD_IMAGE_REVERSE_EN,
                "PROMPT_STYLE_TAGS": PROMPT_STYLE_TAGS_EN,
                "PROMPT_STYLE_SIMPLE": PROMPT_STYLE_SIMPLE_EN,
                "PROMPT_STYLE_DETAILED": PROMPT_STYLE_DETAILED_EN,
                "PROMPT_STYLE_COMPREHENSIVE": PROMPT_STYLE_COMPREHENSIVE_EN,
                "CREATIVE_DETAILED_ANALYSIS": CREATIVE_DETAILED_ANALYSIS_EN,
                "CREATIVE_SUMMARIZE_VIDEO": CREATIVE_SUMMARIZE_VIDEO_EN,
                "CREATIVE_SHORT_STORY": CREATIVE_SHORT_STORY_EN,
                "CREATIVE_REFINE_EXPAND_PROMPT": CREATIVE_REFINE_EXPAND_PROMPT_EN,
                "VISION_BOUNDING_BOX": VISION_BOUNDING_BOX_EN,
                "AUDIO_SUBTITLE_CONVERT": AUDIO_SUBTITLE_CONVERT_EN,
                "VIDEO_TO_AUDIO_SUBTITLE": VIDEO_TO_AUDIO_SUBTITLE_EN,
                "TEXT_TO_AUDIO": TEXT_TO_AUDIO_EN,
                "AUDIO_ANALYSIS": AUDIO_ANALYSIS_EN,
                "MULTI_SPEAKER_DIALOGUE": MULTI_SPEAKER_DIALOGUE_EN,
                "AUDIO_TO_TEXT": AUDIO_TO_TEXT_EN,
                "LYRICS_AND_AUDIO_MERGE": LYRICS_AND_AUDIO_MERGE_EN,
            }
        return preset_map.get(preset_key, preset_key)
    
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
                            parameters, unique_id, seed, force_offload):
        """处理分段模型的推理请求"""
        try:
            print("【分段模型】使用分段模型推理方式")
            
            # 获取模型和处理器
            model = llama_model.model
            processor = llama_model.processor
            tokenizer = llama_model.tokenizer
            
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
                if len(images.shape) == 3:
                    images = images.unsqueeze(0)
                
                # CPU模式下降低图像分辨率以提高速度
                target_size = 128 if model.device.type == 'cpu' else None
                
                for i in range(images.shape[0]):
                    if hasattr(images[i], 'cpu'):
                        img = images[i].cpu().numpy()
                    else:
                        img = images[i]
                    img_np = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    
                    # CPU模式下调整图像大小
                    if target_size and (pil_img.size[0] > target_size or pil_img.size[1] > target_size):
                        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        if i == 0:
                            print(f"【CPU模式优化】图像已调整大小至 {target_size}x{target_size}")
                    
                    pil_images.append(pil_img)
                    conversation[-1]["content"].append({"type": "image", "image": pil_img})
            
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
            
            # 生成配置
            generate_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "use_cache": True,
                "return_audio": audio_output_mode != "TTS音色美化输出",
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            
            # CPU模式特殊优化：降低生成参数以提高速度
            if model.device.type == 'cpu':
                generate_config["max_new_tokens"] = 128
                generate_config["temperature"] = 0.3
                generate_config["do_sample"] = False
                generate_config["top_p"] = 1.0
                print(f"【CPU模式优化】已调整生成参数: max_new_tokens={generate_config['max_new_tokens']}, temperature={generate_config['temperature']}, do_sample={generate_config['do_sample']}")
            
            # 视频处理特殊优化：调整参数以减少内存占用
            if has_video:
                # 降低max_new_tokens以避免内存不足
                generate_config["max_new_tokens"] = min(generate_config["max_new_tokens"], 1024)
                # 降低temperature以获得更稳定的输出
                generate_config["temperature"] = max(generate_config["temperature"] - 0.1, 0.5)
                print(f"【视频优化】视频处理模式，调整max_new_tokens={generate_config['max_new_tokens']}, temperature={generate_config['temperature']}")
            
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
                
                # 记录推理开始时间
                start_time = time.time()
                
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
                    if audio_tensor.dim() == 3:
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
                    print(f"【分段模型】音频生成成功")
            
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
    
    def process(self, llama_model, inference_mode, preset_prompt, system_prompt, text_input, 
                prompt_language, response_language, video_max_frames, 
                video_sampling, video_manual_indices, image_max_size, audio_output_mode,
                omni_speaker, tts_speed, seed, force_offload, save_states, 
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
            # 兼容 ASR 输出（字典）和原始音频（tensor）
            has_audio = False
            if audio is not None:
                if isinstance(audio, dict):
                    has_audio = "waveform" in audio or "text" in audio
                elif hasattr(audio, 'numel'):
                    has_audio = audio.numel() > 0
            
            # 处理自定义提示词
            custom_prompt = text_input
            
            # 检查是否启用ASR和TTS
            enable_asr = mode in ["audio", "text_to_audio"] or preset_prompt in ["[Audio] Audio to Text"]
            enable_tts = mode in ["text_to_audio"] or preset_prompt in ["[Audio] Text to Audio"]
            
            # 检查是否有ASR和TTS模型
            has_asr_model = asr_model is not None
            has_tts_model = tts_model is not None
            
            # 检查是否有音频输入（兼容 ASR 输出和原始音频）
            has_audio_input = False
            if audio is not None:
                if isinstance(audio, dict):
                    has_audio_input = "waveform" in audio or "text" in audio
                elif hasattr(audio, 'numel'):
                    has_audio_input = audio.numel() > 0
            
            # 语言设置
            preset_prompts_language = prompt_language
            output_language = response_language
            
            # 获取预设提示词
            preset_key = self.preset_prompts.get(preset_prompt, "")
            preset_text = self.get_preset_text_by_language(preset_key, preset_prompts_language)
            
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
                    parameters, unique_id, seed, force_offload
                )
            else:
                # 创建推理引擎
                self.engine = InferenceEngineFactory.create_engine(self.model_info)
                
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
                
                # 执行推理
                audio_output = None
                if not generated_text:
                    generated_text = ""

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
                                        print(f"【提示】Omni无音频输出，回退至文本生成")
                                else:
                                    # 非Omni模型或不支持原生音频输出，使用标准推理
                                    output = self.engine.create_chat_completion(llama_model.llm, messages, gen_params)
                                    if output and 'choices' in output:
                                        generated_text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                                    print(f"【提示】当前模型不支持Omni原生音频输出，已回退到文本输出")
                            else:
                                # 标准文本推理（所有模型类型）
                                output = self.engine.create_chat_completion(llama_model.llm, messages, gen_params)
                                if output and 'choices' in output:
                                    generated_text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")

                            if generated_text.strip():
                                success = True
                                print(f"【推理完成】生成文本长度: {len(generated_text)}")
                            else:
                                retry_count += 1
                                print(f"【提示】推理结果为空，重试 {retry_count}/{max_retries}...")

                        except Exception as e:
                            retry_count += 1
                            error_msg = str(e)
                            
                            # 分析错误类型，提供针对性建议
                            if "out of memory" in error_msg.lower() or "oom" in error_msg.lower() or "failed to find a memory slot" in error_msg.lower():
                                print(f"【显存错误】推理失败：{error_msg}")
                                print(f"【智能建议】")
                                print(f"  1. 减少视频帧数（当前：{video_max_frames}）")
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
            
            # 处理TTS语音合成（独立TTS模型）- 仅在TTS音色美化输出模式下启用
            if audio_output_mode == "TTS音色美化输出" and enable_tts and audio_output is None and generated_text:
                print(f"【TTS】开始语音合成...")
                
                # 尝试使用TTS模型
                if tts_model is not None and hasattr(tts_model, 'synthesize'):
                    try:
                        # TTS模型的音色和情感由TTS节点控制
                        audio_output = tts_model.synthesize(
                            text=generated_text,
                            speed=tts_speed
                        )
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
            
            # 处理UID
            _uid = parameters.get("state_uid", None) if parameters else None
            uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
            
            return (generated_text, [generated_text], int(uid), audio_output)
            
        except Exception as e:
            print(f"【处理错误】{str(e)}")
            return (f"处理失败: {str(e)}", [f"处理失败: {str(e)}"], seed, None)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "llama_cpp_unified_inference": llama_cpp_unified_inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_unified_inference": "Llama CPP Unified Inference (VL)",    
}
