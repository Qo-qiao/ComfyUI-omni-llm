# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm API Inference Node - API推理节点
优化版本：添加视频处理、图像预处理、错误处理增强
解决阿里云6MB请求体限制问题，支持图片/视频反推
"""
import os
import json
import requests
import torch
import sys
import tempfile
import numpy as np
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp_api_config import api_config_manager

# 导入预设提示词库
try:
    from support.prompt_enhancer_preset_zh import (
        WAN_T2V_ZH, WAN_I2V_ZH, WAN_I2V_EMPTY_ZH, WAN_FLF2V_ZH,
        QWEN_IMAGE_LAYERED_ZH, QWEN_IMAGE_EDIT_COMBINED_ZH, QWEN_IMAGE_2512_ZH,
        ZIMAGE_TURBO_ZH, FLUX2_KLEIN_ZH, LTX2_ZH,
        VIDEO_TO_PROMPT_ZH, VIDEO_DETAILED_SCENE_BREAKDOWN_ZH, VIDEO_SUBTITLE_FORMAT_ZH,
        OCR_ENHANCED_ZH, ULTRA_HD_IMAGE_REVERSE_ZH, NORMAL_DESCRIBE_ZH,
        NORMAL_DESCRIBE_TAGS_ZH, PROMPT_EXPANDER_ZH,
        VISION_BOUNDING_BOX_ZH, AUDIO_SUBTITLE_CONVERT_ZH, VIDEO_TO_AUDIO_SUBTITLE_ZH,
        AUDIO_ANALYSIS_ZH, MULTI_SPEAKER_DIALOGUE_ZH, LYRICS_CREATION_ZH,
    )

    from support.prompt_enhancer_preset_en import (
        WAN_T2V_EN, WAN_I2V_EN, WAN_I2V_EMPTY_EN, WAN_FLF2V_EN,
        FLUX2_KLEIN_EN, LTX2_EN, QWEN_IMAGE_LAYERED_EN, QWEN_IMAGE_EDIT_COMBINED_EN,
        QWEN_IMAGE_2512_EN, ZIMAGE_TURBO_EN,
        VIDEO_TO_PROMPT_EN, VIDEO_DETAILED_SCENE_BREAKDOWN_EN, VIDEO_SUBTITLE_FORMAT_EN,
        ULTRA_HD_IMAGE_REVERSE_EN, NORMAL_DESCRIBE_EN,
        NORMAL_DESCRIBE_TAGS_EN, PROMPT_EXPANDER_EN,
        VISION_BOUNDING_BOX_EN, AUDIO_SUBTITLE_CONVERT_EN, VIDEO_TO_AUDIO_SUBTITLE_EN,
        AUDIO_ANALYSIS_EN, MULTI_SPEAKER_DIALOGUE_EN, LYRICS_CREATION_EN,
    )
except ImportError:
    # 提示词缺失时的兜底
    WAN_T2V_ZH = WAN_T2V_EN = ""
    WAN_I2V_ZH = WAN_I2V_EN = ""
    WAN_I2V_EMPTY_ZH = WAN_I2V_EMPTY_EN = ""
    WAN_FLF2V_ZH = WAN_FLF2V_EN = ""
    QWEN_IMAGE_LAYERED_ZH = QWEN_IMAGE_LAYERED_EN = ""
    QWEN_IMAGE_EDIT_COMBINED_ZH = QWEN_IMAGE_EDIT_COMBINED_EN = ""
    QWEN_IMAGE_2512_ZH = QWEN_IMAGE_2512_EN = ""
    ZIMAGE_TURBO_ZH = ZIMAGE_TURBO_EN = ""
    FLUX2_KLEIN_ZH = FLUX2_KLEIN_EN = ""
    LTX2_ZH = LTX2_EN = ""
    VIDEO_TO_PROMPT_ZH = VIDEO_TO_PROMPT_EN = ""
    VIDEO_DETAILED_SCENE_BREAKDOWN_ZH = VIDEO_DETAILED_SCENE_BREAKDOWN_EN = ""
    VIDEO_SUBTITLE_FORMAT_ZH = VIDEO_SUBTITLE_FORMAT_EN = ""
    OCR_ENHANCED_ZH = OCR_ENHANCED_EN = ""
    ULTRA_HD_IMAGE_REVERSE_ZH = ULTRA_HD_IMAGE_REVERSE_EN = ""
    NORMAL_DESCRIBE_ZH = NORMAL_DESCRIBE_EN = ""
    PROMPT_STYLE_TAGS_ZH = PROMPT_STYLE_TAGS_EN = ""
    PROMPT_STYLE_SIMPLE_ZH = PROMPT_STYLE_SIMPLE_EN = ""
    PROMPT_STYLE_DETAILED_ZH = PROMPT_STYLE_DETAILED_EN = ""
    VISION_BOUNDING_BOX_ZH = VISION_BOUNDING_BOX_EN = ""
    AUDIO_SUBTITLE_CONVERT_ZH = AUDIO_SUBTITLE_CONVERT_EN = ""
    VIDEO_TO_AUDIO_SUBTITLE_ZH = VIDEO_TO_AUDIO_SUBTITLE_EN = ""
    AUDIO_ANALYSIS_ZH = AUDIO_ANALYSIS_EN = ""
    MULTI_SPEAKER_DIALOGUE_ZH = MULTI_SPEAKER_DIALOGUE_EN = ""
    LYRICS_CREATION_ZH = LYRICS_CREATION_EN = ""
    NORMAL_DESCRIBE_TAGS_ZH = NORMAL_DESCRIBE_TAGS_EN = ""
    PROMPT_EXPANDER_ZH = PROMPT_EXPANDER_EN = ""


class VideoProcessor:
    """视频处理类 - 提取帧、压缩、编码"""

    def __init__(self):
        self.cv2 = None
        self.imageio = None
        self._try_import_libs()

    def _try_import_libs(self):
        """尝试导入视频处理库"""
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            pass

        try:
            import imageio
            self.imageio = imageio
        except ImportError:
            pass

    def is_available(self):
        """检查是否有可用的视频处理库"""
        return self.cv2 is not None or self.imageio is not None

    def parse_frame_indices(self, indices_str, total_frames):
        """解析帧索引字符串，支持格式：0,10,20 或 0-10 或 0-10,2"""
        if not indices_str or not indices_str.strip():
            return None

        indices = []
        parts = indices_str.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                # 范围格式: start-end 或 start-end,step
                range_parts = part.split('-')
                start = int(range_parts[0].strip())

                if ',' in range_parts[1]:
                    # 有步长: start-end,step
                    end_step = range_parts[1].split(',')
                    end = int(end_step[0].strip())
                    step = int(end_step[1].strip())
                else:
                    end = int(range_parts[1].strip())
                    step = 1

                indices.extend(range(start, min(end + 1, total_frames), step))
            else:
                indices.append(int(part))

        # 去重并排序
        indices = sorted(list(set(indices)))
        # 过滤越界
        indices = [i for i in indices if 0 <= i < total_frames]

        return indices if indices else None

    def extract_frames(self, video_path, max_frames=16, sampling="自动均匀采样", manual_indices="", max_size=1024):
        """
        从视频中提取帧

        Args:
            video_path: 视频文件路径
            max_frames: 最大帧数
            sampling: 采样方式 ("自动均匀采样" 或 "手动指定帧索引")
            manual_indices: 手动帧索引字符串
            max_size: 图像最大边长

        Returns:
            帧列表（PIL Image列表）或None
        """
        if not self.is_available():
            print("【视频处理】错误：未安装视频处理库，请安装 opencv-python: pip install opencv-python")
            return None

        try:
            frames = []

            if self.cv2:
                # 使用OpenCV
                cap = self.cv2.VideoCapture(video_path)
                total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(self.cv2.CAP_PROP_FPS)

                print(f"【视频处理】总帧数: {total_frames}, FPS: {fps:.2f}")

                if sampling == "手动指定帧索引" and manual_indices:
                    frame_indices = self.parse_frame_indices(manual_indices, total_frames)
                    if frame_indices:
                        print(f"【视频处理】手动采样帧索引: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
                    else:
                        print("【视频处理】手动索引解析失败，使用自动采样")
                        frame_indices = None
                else:
                    frame_indices = None

                if frame_indices is None:
                    # 自动均匀采样
                    if total_frames <= max_frames:
                        frame_indices = list(range(total_frames))
                    else:
                        step = total_frames // max_frames
                        frame_indices = [i * step for i in range(max_frames)]
                    print(f"【视频处理】自动均匀采样 {len(frame_indices)} 帧")

                # 限制帧数
                frame_indices = frame_indices[:max_frames]

                for idx in frame_indices:
                    cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # BGR to RGB
                        frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        # 调整尺寸
                        pil_image = api_config_manager.resize_image(pil_image, max_size)
                        frames.append(pil_image)

                cap.release()

            elif self.imageio:
                # 使用imageio作为备选
                reader = self.imageio.get_reader(video_path)
                meta = reader.get_meta_data()
                total_frames = meta.get('nframes', 0)
                fps = meta.get('fps', 30)

                print(f"【视频处理】总帧数: {total_frames}, FPS: {fps:.2f}")

                if sampling == "手动指定帧索引" and manual_indices:
                    frame_indices = self.parse_frame_indices(manual_indices, total_frames)
                else:
                    if total_frames <= max_frames:
                        frame_indices = list(range(total_frames))
                    else:
                        step = total_frames // max_frames
                        frame_indices = [i * step for i in range(max_frames)]

                frame_indices = frame_indices[:max_frames]

                for idx in frame_indices:
                    try:
                        frame = reader.get_data(idx)
                        pil_image = Image.fromarray(frame)
                        pil_image = api_config_manager.resize_image(pil_image, max_size)
                        frames.append(pil_image)
                    except:
                        continue

                reader.close()

            print(f"【视频处理】成功提取 {len(frames)} 帧")
            return frames

        except Exception as e:
            print(f"【视频处理错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None


class llama_cpp_api_inference:
    """API多媒体推理节点 - 支持图像/视频/音频输入，文本/音频输出"""

    video_processor = VideoProcessor()

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

    def get_preset_text_by_language(self, preset_key, language):
        """根据语言获取预设提示词文本"""
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
        return preset_map.get(preset_key, preset_key)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ========== API配置 ==========
                "api_config": ("API_CONFIG", {"tooltip": "API配置"}),

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
                "preset_prompt": (s.preset_tags, {"default": s.preset_tags[1] if len(s.preset_tags) > 1 else s.preset_tags[0], "tooltip": "选择预设提示词模板：\n• Empty - Nothing：无预设，完全自定义\n• [Normal] Tags：反推标签格式的描述\n• [Normal] Describe：反推详细描述文本\n• [Normal] Expand：扩展和丰富提示词\n• [Portrait] ZIMAGE - Turbo：人像生成优化\n• [General] FLUX2 - Klein：通用图像生成\n• [Poster] Qwen - Image 2512：海报风格图像\n• [Image Edit] Qwen - Image Edit Combined：图像编辑模板\n• [Image Edit] Qwen - Image Layered：分层图像编辑\n• [Text to Video] LTX-2：文本到视频生成\n• [Text to Video] WAN - Text to Video：WAN模型文本生视频\n• [Image to Video] WAN - Image to Video：图生视频\n• [Image to Video] WAN - Image to Video Empty：图像生视频（无提示词）\n• [Image to Video] WAN - FLF to Video：首尾帧生视频\n• [Video Analysis] Video - Reverse Prompt：视频反推提示词\n• [Video Analysis] Video - Detailed Scene Breakdown：视频场景详细分析\n• [Video Analysis] Video - Subtitle Format：视频字幕格式\n• [Audio] Audio ↔ Subtitle Convert：音频与字幕互转\n• [Audio] Video to Audio & Subtitle：视频转音频和字幕\n• [Audio] Audio Analysis：音频内容分析\n• [Audio] Multi-Person Dialogue：多人对话处理\n• [Music] Lyrics & Audio Merge：歌词与音频合并\n• [OCR] Enhanced OCR：增强型文字识别\n• [HighRes] Ultra HD Image Reverse：超高清图像反推\n• [Vision] Bounding Box：视觉目标检测框"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are an excellent multimodal assistant.", "tooltip": "系统提示词，定义AI助手的角色和行为，可包含预设模板占位符#和自定义内容"}),
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
                "image_max_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64, 
                                           "tooltip": "图像处理的最大边长（像素）。建议值：1024（平衡质量和大小），512（小文件），2048（高质量但可能超限制）。阿里云建议不超过1024。"}),

                # ========== TTS语音参数 ==========
                "tts_voice": ([
                        "default",
                        "female", 
                        "male",
                        "loli",
                        "shota", 
                        "mature_female",
                        "mature_male"
                    ], {
                        "default": "default", 
                        "tooltip": "TTS语音合成音色选择"
                    }),
                "tts_emotion": ([
                        "default",
                        "happy",
                        "sad", 
                        "angry",
                        "calm",
                        "excited",
                        "gentle"
                    ], {
                        "default": "default", 
                        "tooltip": "TTS语音合成情感风格"
                    }),
                "tts_speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, 
                                        "display": "slider"}),

                # ========== 生成参数 ==========
                "seed": ("INT", {"default": 101, "min": 0, "max": 0xffffffffffffffff, "step": 1, "tooltip": "随机种子，用于复现结果"}),
                
                # ========== 输出格式 ==========
                "output_format": (["文本", "JSON"], {"default": "文本", "tooltip": "选择输出格式：\n• 文本：返回纯文本结果\n• JSON：返回包含详细信息的JSON格式"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "图像输入（用于图像理解模式）"}),
                "video": ("VIDEO", {"tooltip": "视频输入（用于视频理解模式）。支持格式：MP4, AVI, MOV等。需要安装opencv-python。"}),
                "audio": ("AUDIO", {"tooltip": "音频输入（用于音频理解模式）"}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "inference"
    CATEGORY = "llama-cpp-vlm"

    def inference(self, api_config, inference_mode, preset_prompt, system_prompt, text_input,
                  prompt_language, response_language, video_max_frames, video_sampling, video_manual_indices,
                  image_max_size, tts_voice, tts_emotion, tts_speed, seed, output_format,
                  images=None, video=None, audio=None):
        try:
            # 配置验证
            if not api_config or not isinstance(api_config, dict):
                if output_format == "JSON":
                    return (json.dumps({"error": "API配置无效"}, ensure_ascii=False), None)
                else:
                    return ("错误：API配置无效", None)

            api_provider = api_config.get("api_provider", "OpenAI")
            api_base = api_config.get("api_base", "").strip()
            api_key = api_config.get("api_key", "").strip()

            if not api_base:
                if output_format == "JSON":
                    return (json.dumps({"error": "API基础URL未配置"}, ensure_ascii=False), None)
                else:
                    return ("错误：API基础URL未配置", None)

            # 对于需要API密钥的提供商进行验证
            if api_provider in ["OpenAI", "Anthropic", "Moonshot AI", "Qwen", "llms-py", "Kimi", "GLM", "MiniMax"] and not api_key:
                if output_format == "JSON":
                    return (json.dumps({"error": "API密钥未配置"}, ensure_ascii=False), None)
                else:
                    return ("错误：API密钥未配置", None)

            # 检查视频处理库
            if "视频理解" in inference_mode and video is not None:
                if not self.video_processor.is_available():
                    if output_format == "JSON":
                        return (json.dumps({"error": "视频处理需要opencv-python。请运行：pip install opencv-python"}, ensure_ascii=False), None)
                    else:
                        return ("错误：视频处理需要opencv-python。请运行：pip install opencv-python", None)

            # 确定输入语言
            if prompt_language == "中文":
                input_language = "zh"
            else:
                input_language = "en"

            # 构建系统提示词
            if input_language == "zh":
                default_system = "你是一个优秀的多模态助手。"
            else:
                default_system = "You are an excellent multimodal assistant."

            final_system_prompt = system_prompt if system_prompt.strip() else default_system

            # 根据推理模式调整系统提示词
            if "视频理解" in inference_mode:
                if input_language == "zh":
                    final_system_prompt = "请分析视频内容，语言简洁明了。" + final_system_prompt
                else:
                    final_system_prompt = "Please analyze the video content clearly and concisely. " + final_system_prompt
            elif "图像理解" in inference_mode or "全模态整合" in inference_mode:
                if input_language == "zh":
                    final_system_prompt = "请分析图片内容，语言简洁明了。" + final_system_prompt
                else:
                    final_system_prompt = "Please analyze the image content clearly and concisely. " + final_system_prompt
            elif "音频转文本" in inference_mode or "文本转音频" in inference_mode:
                if input_language == "zh":
                    final_system_prompt = "请分析音频内容，语言简洁明了。" + final_system_prompt
                else:
                    final_system_prompt = "Please analyze the audio content clearly and concisely. " + final_system_prompt

            messages = []
            if final_system_prompt.strip():
                messages.append({"role": "system", "content": final_system_prompt})

            # 构建用户内容
            user_content = []

            # 处理音频输入（ASR）
            asr_text = ""
            if audio is not None and ("音频转文本" in inference_mode or "全模态整合" in inference_mode):
                try:
                    # 将音频编码为base64
                    audio_base64 = api_config_manager.encode_audio(audio)
                    if audio_base64:
                        # 使用API调用进行语音识别
                        asr_messages = [
                            {"role": "system", "content": "你是一个语音识别助手，请将音频内容转录为文本。"},
                            {"role": "user", "content": [
                                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                            ]}
                        ]
                        asr_result = api_config_manager.call_api(api_config, asr_messages)
                        if asr_result and 'choices' in asr_result:
                            asr_text = asr_result['choices'][0]['message']['content'].strip()
                            if asr_text:
                                if input_language == "zh":
                                    user_content.append({"type": "text", "text": f"[语音识别结果]：{asr_text}"})
                                else:
                                    user_content.append({"type": "text", "text": f"[ASR Result]：{asr_text}"})
                                print(f"【ASR】识别成功：{len(asr_text)}字符")
                            else:
                                print("【ASR】识别失败或无结果")
                        else:
                            print("【ASR】API返回格式错误")
                    else:
                        print("【ASR】音频编码失败")
                except Exception as e:
                    print(f"【ASR错误】语音识别失败: {str(e)}")

            # 处理视频输入 - 提取帧并作为图像处理
            video_frames = []
            if "视频理解" in inference_mode and video is not None:
                try:
                    # ComfyUI VIDEO格式通常是文件路径或tensor
                    video_path = None
                    if isinstance(video, str) and os.path.exists(video):
                        video_path = video
                    elif isinstance(video, torch.Tensor):
                        # 如果是tensor，暂不支持直接处理，提示用户
                        print("【视频处理】暂不支持直接tensor输入，请使用视频文件路径")

                    if video_path:
                        print(f"【视频处理】正在处理视频: {video_path}")
                        video_frames = self.video_processor.extract_frames(
                            video_path, 
                            max_frames=video_max_frames,
                            sampling=video_sampling,
                            manual_indices=video_manual_indices,
                            max_size=image_max_size
                        )
                        if video_frames:
                            print(f"【视频处理】成功提取 {len(video_frames)} 帧，将用于分析")
                        else:
                            print("【视频处理】视频帧提取失败")
                    else:
                        print("【视频处理】无效的视频输入")
                except Exception as e:
                    print(f"【视频处理错误】{str(e)}")

            # 处理图像输入（包括视频帧）
            has_images = False
            all_images = []

            # 添加视频帧到图像列表
            if video_frames:
                all_images.extend(video_frames)
                has_images = True

            # 添加常规图像输入
            if images is not None:
                try:
                    if isinstance(images, torch.Tensor):
                        if images.dim() == 4:
                            # 批次图像，限制数量避免请求过大
                            max_images = min(images.shape[0], 8)
                            for i in range(max_images):
                                img_np = images[i].cpu().numpy()
                                # 处理通道顺序
                                if img_np.shape[0] in [1, 3, 4] and img_np.ndim == 3:
                                    img_np = np.transpose(img_np, (1, 2, 0))
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                all_images.append(Image.fromarray(img_np))
                            print(f"【图像】添加了 {max_images} 张输入图像")
                            has_images = True
                        else:
                            img_np = images.cpu().numpy()
                            if img_np.shape[0] in [1, 3, 4] and img_np.ndim == 3:
                                img_np = np.transpose(img_np, (1, 2, 0))
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            all_images.append(Image.fromarray(img_np))
                            print("【图像】添加了 1 张输入图像")
                            has_images = True
                except Exception as e:
                    print(f"【图像处理错误】{str(e)}")

            # 编码所有图像（使用image_max_size参数）
            encoded_images = []
            if all_images and ("图像理解" in inference_mode or "全模态整合" in inference_mode or "视频理解" in inference_mode):
                print(f"【图像编码】正在编码 {len(all_images)} 张图像，最大边长: {image_max_size}")
                for idx, img in enumerate(all_images):
                    try:
                        img_base64 = api_config_manager.encode_image(img, max_size=image_max_size)
                        if img_base64:
                            encoded_images.append(img_base64)
                    except Exception as e:
                        print(f"【图像编码】第{idx+1}张图像编码失败: {str(e)}")

                # 添加到用户内容
                for img_base64 in encoded_images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })

                print(f"【图像编码】成功编码 {len(encoded_images)} 张图像")

            # 处理音频输入（用于全模态整合）
            if "全模态整合" in inference_mode and audio is not None and not asr_text:
                try:
                    # 如果没有进行ASR，将音频作为多模态输入
                    audio_base64 = api_config_manager.encode_audio(audio)
                    if audio_base64:
                        user_content.append({
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
                        })
                        print("【音频】添加为多模态输入")
                except Exception as e:
                    print(f"【音频处理错误】{str(e)}")

            # 获取预设提示词
            preset_key = self.preset_prompts.get(preset_prompt, "")
            preset_text = self.get_preset_text_by_language(preset_key, prompt_language)

            # 构建最终提示词
            custom_prompt = text_input
            if preset_prompt == "Empty - Nothing":
                final_prompt = custom_prompt.strip() if custom_prompt.strip() else final_system_prompt.strip()
            else:
                final_prompt = preset_text
                if custom_prompt.strip():
                    if "下面是要优化的 Prompt：" in preset_text or "Below is the Prompt to optimize:" in preset_text:
                        final_prompt = preset_text + custom_prompt.strip()
                    else:
                        final_prompt = preset_text.replace("#", custom_prompt.strip()).replace("@", "video" if video_frames else "image")

            # 添加语言指示
            if response_language == "中文":
                final_prompt += "\n\n请用中文回答。"
            elif response_language == "English":
                final_prompt += "\n\nPlease answer in English."

            # 将ASR结果合并到提示词中
            if asr_text:
                final_prompt = f"[音频内容: {asr_text}]\n{final_prompt}"

            # 添加用户输入文本
            user_content.append({"type": "text", "text": final_prompt})

            # 如果没有内容，添加默认文本
            if not user_content:
                default_text = "请描述此内容" if input_language == "zh" else "Please describe this content"
                user_content.append({"type": "text", "text": default_text})

            messages.append({"role": "user", "content": user_content})

            # 发送API请求
            print(f"【API调用】使用 {api_provider} 提供商，模型: {api_config.get('model_id', 'default')}")

            # 使用api_config_manager的call_api方法（带重试）
            result = api_config_manager.call_api(
                api_config, 
                messages, 
                max_tokens=api_config.get("max_tokens", 1024), 
                temperature=api_config.get("temperature", 0.7),
                retries=2
            )

            if not result:
                if output_format == "JSON":
                    return (json.dumps({"error": "API调用失败，请检查日志了解详情"}, ensure_ascii=False), None)
                else:
                    return ("错误：API调用失败，请检查日志了解详情", None)

            # 解析响应结果
            generated_text = ""
            if api_provider == "Ollama":
                generated_text = result.get('message', {}).get('content', '')
            elif 'choices' in result:
                generated_text = result['choices'][0]['message']['content']
            else:
                print("【API响应】未知的响应格式")
                if output_format == "JSON":
                    return (json.dumps({"error": "API返回格式错误"}, ensure_ascii=False), None)
                else:
                    return ("错误：API返回格式错误", None)

            # 验证生成结果
            if not generated_text or not isinstance(generated_text, str):
                if output_format == "JSON":
                    return (json.dumps({"error": "API返回结果无效"}, ensure_ascii=False), None)
                else:
                    return ("错误：API返回结果无效", None)

            generated_text = generated_text.strip()
            if not generated_text:
                if output_format == "JSON":
                    return (json.dumps({"error": "API返回空结果"}, ensure_ascii=False), None)
                else:
                    return ("错误：API返回空结果", None)

            print(f"【API响应】生成了 {len(generated_text)} 字符的文本")

            # 语言检测和输出调整
            if response_language != "自动检测":
                detected_lang = api_config_manager.detect_language(generated_text)
                target_lang = "中文" if response_language == "中文" else "English"
                if detected_lang != target_lang:
                    print(f"【提示】检测到输出语言与目标语言不一致，当前为{detected_lang}，目标为{target_lang}")

            # TTS处理 - 解析音色选项
            audio_output = None
            if generated_text and ("文本转音频" in inference_mode or "全模态整合" in inference_mode):
                # 从音色选项中提取实际的voice名称
                voice = "alloy"  # 默认voice
                if "female" in tts_voice or "女声" in tts_voice:
                    voice = "nova"
                elif "male" in tts_voice or "男声" in tts_voice:
                    voice = "onyx"
                elif "loli" in tts_voice or "萝莉" in tts_voice:
                    voice = "shimmer"
                elif "shota" in tts_voice or "正太音" in tts_voice:
                    voice = "echo"
                elif "mature_female" in tts_voice or "御姐" in tts_voice:
                    voice = "fable"
                elif "mature_male" in tts_voice or "大叔" in tts_voice:
                    voice = "alloy"

                try:
                    audio_output = api_config_manager.call_tts_api(api_config, generated_text, voice, tts_speed)
                    if audio_output:
                        print("【TTS】语音合成成功")
                    else:
                        print("【TTS】语音合成失败")
                except Exception as e:
                    print(f"【TTS错误】语音合成失败: {str(e)}")

            # 处理输出格式
            if output_format == "JSON":
                json_output = {
                    "text": generated_text,
                    "audio_generated": audio_output is not None,
                    "inference_mode": inference_mode,
                    "prompt_language": prompt_language,
                    "response_language": response_language,
                    "model": api_config.get("model_id", "default"),
                    "provider": api_config.get("api_provider", "OpenAI"),
                    "token_count": len(generated_text),
                    "timestamp": "2026-04-17"
                }
                return (json.dumps(json_output, ensure_ascii=False, indent=2), audio_output)
            else:
                return (generated_text, audio_output)

        except requests.exceptions.Timeout as e:
            error_msg = f"错误：API请求超时：{str(e)}"
            print(f"【超时错误】{error_msg}")
            if output_format == "JSON":
                return (json.dumps({"error": error_msg}, ensure_ascii=False), None)
            else:
                return (error_msg, None)
        except requests.exceptions.RequestException as e:
            error_msg = f"错误：API请求失败：{str(e)}"
            print(f"【网络错误】{error_msg}")
            if output_format == "JSON":
                return (json.dumps({"error": error_msg}, ensure_ascii=False), None)
            else:
                return (error_msg, None)
        except Exception as e:
            error_msg = f"错误：API调用失败：{str(e)}"
            print(f"【错误】{error_msg}")
            import traceback
            traceback.print_exc()
            if output_format == "JSON":
                return (json.dumps({"error": error_msg}, ensure_ascii=False), None)
            else:
                return (error_msg, None)


NODE_CLASS_MAPPINGS = {
    "llama_cpp_api_inference": llama_cpp_api_inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_api_inference": "Llama CPP API Inference (VL/Omni)",
}