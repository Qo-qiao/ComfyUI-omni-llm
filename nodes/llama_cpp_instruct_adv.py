# @亲卿于情 修改版本
# -*- coding: utf-8 -*-
"""
Llama-cpp Image Inference Node
"""
import numpy as np
from ..common import (
    HARDWARE_INFO, any_type, image2base64, scale_image,
    mm
)

# 导入预设提示词库
from ..support.prompt_enhancer_preset_zh import (
    PRESET_PROMPTS_ZH,
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
    BILINGUAL_PROMPT_GENERATE_ZH,
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
)

from ..support.prompt_enhancer_preset_en import (
    PRESET_PROMPTS_EN,
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
    BILINGUAL_PROMPT_GENERATE_EN,
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
)

class llama_cpp_instruct_adv:
    # 缓存变量
    _language_cache = {}
    _perf_params_cache = {}
    
    # 初始化预设提示词字典
    preset_prompts = {}
    
    # 添加基础预设（保留Empty选项）
    preset_prompts["Empty - Nothing"] = ""
    
    # 添加分类预设提示词
    # 通用描述 (Normal)
    preset_prompts["[Normal] Describe"] = "NORMAL_DESCRIBE"
    
    # 文生图 (Text to Image)
    preset_prompts["[Text to Image] ZIMAGE - Turbo"] = "ZIMAGE_TURBO"
    preset_prompts["[Text to Image] FLUX2 - Klein"] = "FLUX2_KLEIN"
    preset_prompts["[Text to Image] Qwen - Image 2512"] = "QWEN_IMAGE_2512"
    
    # 图编辑 (Image Edit)
    preset_prompts["[Image Edit] Qwen - Image Edit Combined"] = "QWEN_IMAGE_EDIT_COMBINED"
    preset_prompts["[Image Edit] Qwen - Image Layered"] = "QWEN_IMAGE_LAYERED"  
        
    # 文生视频 (Text to Video)
    preset_prompts["[Text to Video] LTX-2"] = "LTX2"
    preset_prompts["[Text to Video] WAN - Text to Video"] = "WAN_T2V"   
    
    # 图生视频 (Image to Video)
    preset_prompts["[Image to Video] WAN - Image to Video"] = "WAN_I2V"
    preset_prompts["[Image to Video] WAN - Image to Video Empty"] = "WAN_I2V_EMPTY"
    preset_prompts["[Image to Video] WAN - FLF to Video"] = "WAN_FLF2V"
     
    # 提示词风格 (Prompt Style)
    preset_prompts["[Prompt Style] Tags"] = "PROMPT_STYLE_TAGS"
    preset_prompts["[Prompt Style] Simple"] = "PROMPT_STYLE_SIMPLE"
    preset_prompts["[Prompt Style] Detailed"] = "PROMPT_STYLE_DETAILED"
    preset_prompts["[Prompt Style] Comprehensive Expansion"] = "PROMPT_STYLE_COMPREHENSIVE"
    
    # 创意功能 (Creative)
    preset_prompts["[Creative] Refine & Expand Prompt"] = "PROMPT_STYLE_REFINE"
    preset_prompts["[Creative] Detailed Analysis"] = "CREATIVE_DETAILED_ANALYSIS"
    preset_prompts["[Creative] Summarize Video"] = "CREATIVE_SUMMARIZE_VIDEO"
    preset_prompts["[Creative] Short Story"] = "CREATIVE_SHORT_STORY"
    
    # 视频分析 (Video Analysis)
    preset_prompts["[Video Analysis] Video - Reverse Prompt"] = "VIDEO_TO_PROMPT"
    preset_prompts["[Video Analysis] Video - Detailed Scene Breakdown"] = "VIDEO_DETAILED_SCENE_BREAKDOWN"
    preset_prompts["[Video Analysis] Video - Subtitle Format"] = "VIDEO_SUBTITLE_FORMAT"
     
    # 视觉任务 (Vision)
    preset_prompts["[Vision] Bounding Box"] = "VISION_BOUNDING_BOX"
    # OCR任务
    preset_prompts["[OCR] Enhanced OCR"] = "OCR_ENHANCED"
    # 多语言任务 (Multilingual)
    preset_prompts["[Multilingual] Bilingual Prompt Generate"] = "BILINGUAL_PROMPT_GENERATE"
    # 高清任务 (HighRes)
    preset_prompts["[HighRes] Ultra HD Image Reverse"] = "ULTRA_HD_IMAGE_REVERSE"


    

    
    preset_tags = list(preset_prompts.keys())
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (s.preset_tags, {"default": s.preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are an excellent image description assistant."}),
                "inference_mode": (["text", "one by one", "images", "video"], {"default": "images"}),
                "preset_prompts_language": (["中文", "English"], {"default": "中文"}),
                "output_language": (["Auto", "中文", "English"], {"default": "Auto"}),
                "max_frames": ("INT", {"default": 16, "min": 2, "max": 1024, "step": 1}),
                "max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "step": 64}),
                "sampling_mode": (["Auto (Uniform)", "Manual (Indices)"], {"default": "Auto (Uniform)", "tooltip": "抽帧模式: 自动均匀采样或手动指定索引"}),
                "manual_indices": ("STRING", {"default": "", "placeholder": "输入索引 (例如: 0,10,20) 或范围 (例如: 0-10)", "tooltip": "手动模式下的帧索引，仅在sampling_mode为Manual时生效"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {"default": False}),
                "save_states": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "images": ("IMAGE",),
                "video": ("VIDEO",),
                "queue_handler": (any_type,),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def sanitize_messages(self, messages):
        clean_messages = messages.copy()
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages
    
    def detect_language(self, text):
        """Detect if text is Chinese or English (with caching)"""
        if not text or not isinstance(text, str):
            return "en"
        
        # 检查缓存
        if text in self._language_cache:
            return self._language_cache[text]
        
        # Count Chinese characters
        chinese_chars = 0
        total_chars = len(text)
        
        for char in text:
            # Check if character is Chinese
            if '\u4e00' <= char <= '\u9fff':
                chinese_chars += 1
        
        # If more than 30% are Chinese characters, consider it Chinese
        result = "zh" if total_chars > 0 and (chinese_chars / total_chars) > 0.3 else "en"
        
        # 缓存结果
        self._language_cache[text] = result
        
        # 限制缓存大小，避免内存泄漏
        if len(self._language_cache) > 1000:
            self._language_cache.clear()
        
        return result
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, preset_prompts_language, output_language, max_frames, max_size, sampling_mode, manual_indices, seed, force_offload, save_states, unique_id, parameters=None, images=None, video=None, queue_handler=None):
        if not llama_model.llm:
            raise RuntimeError("【错误】模型未加载或已卸载，请先加载LLM模型")
        
        video_input = inference_mode == "video"
        text_input = inference_mode == "text"
        
        # 处理视频输入
        if video is not None:
            print(f"【视频处理】开始处理视频输入")
            try:
                # 加载视频为tensor
                video_tensor = self._load_video_as_tensor(video)
                print(f"【视频处理】视频加载成功，帧数量: {video_tensor.shape[0]}")
                
                # 提取帧
                if sampling_mode == "Manual (Indices)":
                    frames_tensor = self._extract_frames_and_tensor(video_tensor, target_indices_str=manual_indices)
                else:  # Auto (Uniform)
                    frames_tensor = self._extract_frames_and_tensor(video_tensor, target_count=max_frames)
                print(f"【视频处理】帧提取成功，提取帧数量: {frames_tensor.shape[0]}")
                
                # 使用视频帧覆盖images变量
                images = frames_tensor
                video_input = True  # 强制使用视频模式
            except Exception as e:
                print(f"【视频处理错误】视频处理失败: {str(e)}")
                raise RuntimeError(f"视频处理失败: {str(e)}")
        
        if parameters is None:
            # 根据硬件性能和任务类型自动调整参数（使用缓存）
            perf_level = HARDWARE_INFO.get("perf_level", "low")
            
            # 生成缓存键
            cache_key = f"{perf_level}_{video_input}_{text_input}"
            
            # 检查缓存
            if cache_key in self._perf_params_cache:
                parameters = self._perf_params_cache[cache_key]
            else:
                # 基础参数设置
                base_params = {
                    "min_p": 0.05,
                    "typical_p": 1.0,
                    "repeat_penalty": 1.0,
                    "frequency_penalty": 0.0,
                    "mirostat_mode": 0,
                    "mirostat_eta": 0.1,
                    "mirostat_tau": 5.0,
                    "state_uid": -1
                }
                
                # 根据硬件性能调整核心参数
                if perf_level == "high":  # 24GB+
                    base_params.update({
                        "max_tokens": 2048,
                        "top_k": 40,
                        "top_p": 0.92,
                        "temperature": 0.85
                    })
                elif perf_level == "mid_high":  # 16GB
                    base_params.update({
                        "max_tokens": 1536,
                        "top_k": 35,
                        "top_p": 0.9,
                        "temperature": 0.8
                    })
                elif perf_level == "mid":  # 12GB
                    base_params.update({
                        "max_tokens": 1024,
                        "top_k": 30,
                        "top_p": 0.88,
                        "temperature": 0.75
                    })
                elif perf_level == "mid_low":  # 8GB
                    base_params.update({
                        "max_tokens": 768,
                        "top_k": 25,
                        "top_p": 0.85,
                        "temperature": 0.7
                    })
                else:  # <8GB
                    base_params.update({
                        "max_tokens": 512,
                        "top_k": 20,
                        "top_p": 0.8,
                        "temperature": 0.6
                    })
                
                # 根据任务类型进一步调整
                if video_input:
                    # 视频处理需要更多tokens
                    base_params["max_tokens"] = min(base_params["max_tokens"] * 2, 2048)
                elif text_input:
                    # 纯文本可以使用更大的temperature
                    base_params["temperature"] = min(base_params["temperature"] + 0.1, 1.0)
                
                parameters = base_params
                
                # 缓存结果
                self._perf_params_cache[cache_key] = parameters
                
                # 限制缓存大小，避免内存泄漏
                if len(self._perf_params_cache) > 100:
                    self._perf_params_cache.clear()
        
        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
        last_sys_prompt = llama_model.sys_prompts.get(f"{uid}", None)
        
        # Determine language based on preset_prompts_language selection
        if preset_prompts_language == "中文":
            input_language = "zh"
        else:  # English
            input_language = "en"
        
        # Set system prompt based on detected language
        if input_language == "zh":
            # Default Chinese system prompt
            default_system_prompt = "你是一个优秀的文本生成助手。"
        else:
            # Default English system prompt
            default_system_prompt = "You are an excellent text generation assistant."
        
        # Use user-provided system prompt if available, otherwise use language-specific default
        final_system_prompt = system_prompt if system_prompt.strip() else default_system_prompt
        
        # Adjust system prompt for different input types
        if text_input:
            system_prompts = final_system_prompt
        elif video_input:
            if input_language == "zh":
                system_prompts = "请分析视频内容，语言简洁明了。" + final_system_prompt
            else:
                system_prompts = "Please analyze the video content clearly and concisely." + final_system_prompt
        else:
            if input_language == "zh":
                system_prompts = "请分析图片内容，语言简洁明了。" + final_system_prompt
            else:
                system_prompts = "Please analyze the image content clearly and concisely." + final_system_prompt
        
        if last_sys_prompt != system_prompts:
            messages = []
            llama_model.clean_state()
            llama_model.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            messages = llama_model.messages.get(f"{uid}", []) if save_states else []
        
        out1 = ""
        out2 = []
        user_content = []
        preset_text = self.preset_prompts.get(preset_prompt, "")
        
        # 处理双语言预设提示词
        if preset_text == "QWEN_IMAGE_LAYERED":
            # Qwen Image Layered 双语言预设
            preset_text = QWEN_IMAGE_LAYERED_ZH if input_language == "zh" else QWEN_IMAGE_LAYERED_EN
        elif preset_text == "ZIMAGE_TURBO":
            # ZIMAGE Turbo 双语言预设
            preset_text = ZIMAGE_TURBO_ZH if input_language == "zh" else ZIMAGE_TURBO_EN
        elif preset_text == "FLUX2_KLEIN":
            # FLUX Klein 双语言预设
            preset_text = FLUX2_KLEIN_ZH if input_language == "zh" else FLUX2_KLEIN_EN
        elif preset_text == "LTX2":
            # LTX-2 双语言预设
            preset_text = LTX2_ZH if input_language == "zh" else LTX2_EN
        elif preset_text == "QWEN_IMAGE_EDIT_COMBINED":
            # Qwen Image Edit Combined 双语言预设
            preset_text = QWEN_IMAGE_EDIT_COMBINED_ZH if input_language == "zh" else QWEN_IMAGE_EDIT_COMBINED_EN
        elif preset_text == "QWEN_IMAGE_2512":
            # 根据输入语言选择对应的预设
            preset_text = QWEN_IMAGE_2512_ZH if input_language == "zh" else QWEN_IMAGE_2512_EN
        elif preset_text == "WAN_T2V":
            # WAN Text to Video 双语言预设
            preset_text = WAN_T2V_ZH if input_language == "zh" else WAN_T2V_EN
        elif preset_text == "WAN_I2V":
            # WAN Image to Video 双语言预设
            preset_text = WAN_I2V_ZH if input_language == "zh" else WAN_I2V_EN
        elif preset_text == "WAN_I2V_EMPTY":
            # WAN Image to Video Empty 双语言预设
            preset_text = WAN_I2V_EMPTY_ZH if input_language == "zh" else WAN_I2V_EMPTY_EN
        elif preset_text == "WAN_FLF2V":
            # WAN FLF to Video 双语言预设
            preset_text = WAN_FLF2V_ZH if input_language == "zh" else WAN_FLF2V_EN
        elif preset_text == "VIDEO_TO_PROMPT":
            # Video Reverse Prompt 双语言预设
            preset_text = VIDEO_TO_PROMPT_ZH if input_language == "zh" else VIDEO_TO_PROMPT_EN
        elif preset_text == "PROMPT_STYLE_TAGS":
            # Prompt Style - Tags 双语言预设
            preset_text = PROMPT_STYLE_TAGS_ZH if input_language == "zh" else PROMPT_STYLE_TAGS_EN
        elif preset_text == "PROMPT_STYLE_SIMPLE":
            # Prompt Style - Simple 双语言预设
            preset_text = PROMPT_STYLE_SIMPLE_ZH if input_language == "zh" else PROMPT_STYLE_SIMPLE_EN
        elif preset_text == "PROMPT_STYLE_DETAILED":
            # Prompt Style - Detailed 双语言预设
            preset_text = PROMPT_STYLE_DETAILED_ZH if input_language == "zh" else PROMPT_STYLE_DETAILED_EN
        elif preset_text == "PROMPT_STYLE_COMPREHENSIVE":
            # Prompt Style - Comprehensive Expansion 双语言预设
            preset_text = PROMPT_STYLE_COMPREHENSIVE_ZH if input_language == "zh" else PROMPT_STYLE_COMPREHENSIVE_EN
        elif preset_text == "PROMPT_STYLE_REFINE":
            # Creative - Refine & Expand Prompt 双语言预设
            preset_text = CREATIVE_REFINE_EXPAND_PROMPT_ZH if input_language == "zh" else CREATIVE_REFINE_EXPAND_PROMPT_EN
        elif preset_text == "CREATIVE_DETAILED_ANALYSIS":
            # Creative - Detailed Analysis 双语言预设
            preset_text = CREATIVE_DETAILED_ANALYSIS_ZH if input_language == "zh" else CREATIVE_DETAILED_ANALYSIS_EN
        elif preset_text == "CREATIVE_SUMMARIZE_VIDEO":
            # Creative - Summarize Video 双语言预设
            preset_text = CREATIVE_SUMMARIZE_VIDEO_ZH if input_language == "zh" else CREATIVE_SUMMARIZE_VIDEO_EN
        elif preset_text == "CREATIVE_SHORT_STORY":
            # Creative - Short Story 双语言预设
            preset_text = CREATIVE_SHORT_STORY_ZH if input_language == "zh" else CREATIVE_SHORT_STORY_EN
        elif preset_text == "VISION_BOUNDING_BOX":
            # Vision - Bounding Box 双语言预设
            preset_text = VISION_BOUNDING_BOX_ZH if input_language == "zh" else VISION_BOUNDING_BOX_EN
        elif preset_text == "NORMAL_DESCRIBE":
            # Normal - Describe 双语言预设
            preset_text = NORMAL_DESCRIBE_ZH if input_language == "zh" else NORMAL_DESCRIBE_EN
        elif preset_text == "VIDEO_DETAILED_SCENE_BREAKDOWN":
            # Video - Detailed Scene Breakdown 双语言预设
            preset_text = VIDEO_DETAILED_SCENE_BREAKDOWN_ZH if input_language == "zh" else VIDEO_DETAILED_SCENE_BREAKDOWN_EN
        elif preset_text == "VIDEO_SUBTITLE_FORMAT":
            # Video - Subtitle Format 双语言预设
            preset_text = VIDEO_SUBTITLE_FORMAT_ZH if input_language == "zh" else VIDEO_SUBTITLE_FORMAT_EN
        elif preset_text == "OCR_ENHANCED":
            # OCR - Enhanced 双语言预设
            preset_text = OCR_ENHANCED_ZH if input_language == "zh" else ""
        elif preset_text == "BILINGUAL_PROMPT_GENERATE":
            # Bilingual Prompt Generate 双语言预设
            preset_text = BILINGUAL_PROMPT_GENERATE_ZH if input_language == "zh" else BILINGUAL_PROMPT_GENERATE_EN
        elif preset_text == "ULTRA_HD_IMAGE_REVERSE":
            # Ultra HD Image Reverse 双语言预设
            preset_text = ULTRA_HD_IMAGE_REVERSE_ZH if input_language == "zh" else ULTRA_HD_IMAGE_REVERSE_EN

        
        # 处理输出语言设置
        def get_output_language_instruction():
            # 根据output_language设置生成语言指令
            output_lang = output_language
            if output_lang == "Auto":
                # 自动模式不需要额外指令
                return ""
            elif output_lang == "中文":
                return "\n\n请用中文回答。"
            else:  # English
                return "\n\nPlease answer in English."
        
        # 处理提示词逻辑
        if preset_prompt == "Empty - Nothing":
            # 当使用Empty - Nothing预设时：
            # 1. 如果有custom_prompt，直接使用它作为用户输入
            # 2. system_prompt已经作为系统指令在messages列表开头设置
            # 这样模型会根据system_prompt的指令处理custom_prompt的内容
            if custom_prompt.strip():
                final_prompt = custom_prompt.strip() + get_output_language_instruction()
            else:
                # 如果没有custom_prompt，就使用system_prompt作为输入
                final_prompt = (system_prompts.strip() if system_prompts.strip() else custom_prompt.strip()) + get_output_language_instruction()
        else:
            # 处理其他预设
            final_prompt = preset_text
            if custom_prompt.strip():
                if "*" in preset_prompt:
                    final_prompt = custom_prompt.strip() + get_output_language_instruction()
                else:
                    # 检查是否是专属预设模板（末尾有"下面是要优化的 Prompt："）
                    if "下面是要优化的 Prompt：" in preset_text:
                        final_prompt = preset_text + custom_prompt.strip() + get_output_language_instruction()
                    elif "Below is the Prompt to optimize:" in preset_text:
                        final_prompt = preset_text + custom_prompt.strip() + get_output_language_instruction()
                    else:
                        final_prompt = preset_text.replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image") + get_output_language_instruction()
        
        user_content.append({"type": "text", "text": final_prompt})
        
        if not text_input and images is not None:
            # 与nodes.py完全一致的检查方式
            if not hasattr(llama_model, "chat_handler") or not llama_model.chat_handler:
                raise ValueError("【错误】处理图片需要启用MMProj并选择对应的ChatHandler（如Qwen3-VL）")
            
            frames = images
            if video_input:
                indices = np.linspace(0, len(images) - 1, min(max_frames, len(images)), dtype=int)
                frames = [images[i] for i in indices]
            
            preprocessed_images = []
            print(f"【图片处理】开始预处理{len(frames)}张图片，尺寸限制：{max_size}px")
            for image in frames:
                try:
                    if len(frames) > 1:
                        img_np = scale_image(image, max_size)
                    else:
                        img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                    preprocessed_images.append(image2base64(img_np))
                except Exception as e:
                    print(f"【提示】图片预处理失败，跳过该图片：{e}")
                    preprocessed_images.append("")
            
            if inference_mode == "one by one":
                tmp_list = []
                image_content = {"type": "image_url", "image_url": {"url": ""}}
                user_content.append(image_content)
                messages.append({"role": "user", "content": user_content})
                
                for i, img_base64 in enumerate(preprocessed_images):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    if not img_base64:
                        continue
                    
                    for item in user_content:
                        if item.get("type") == "image_url":
                            item["image_url"]["url"] = f"data:image/jpeg;base64,{img_base64}"
                            break
                    
                    infer_params = {
                        "max_tokens": _parameters.get("max_tokens", 1024),
                        "temperature": _parameters.get("temperature", 0.6),
                        "top_k": _parameters.get("top_k", 20),
                        "top_p": _parameters.get("top_p", 0.85),
                        "min_p": _parameters.get("min_p", 0.05),
                        "typical_p": _parameters.get("typical_p", 1.0),
                        "repeat_penalty": _parameters.get("repeat_penalty", 1.0),
                        "frequency_penalty": _parameters.get("frequency_penalty", 0.0),
                        "mirostat_mode": _parameters.get("mirostat_mode", 0),
                        "mirostat_eta": _parameters.get("mirostat_eta", 0.1),
                        "mirostat_tau": _parameters.get("mirostat_tau", 5.0),
                        "seed": seed,
                        "stream": False,
                        "stop": ["</s>"]
                    }
                    
                    retry_count = 0
                    max_retries = 2
                    success = False
                    
                    while retry_count < max_retries and not success:
                        try:
                            output = llama_model.llm.create_chat_completion(messages=messages, **infer_params)
                            if output and 'choices' in output and len(output['choices']) > 0:
                                text = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                                if text.strip():
                                    out2.append(text)
                                    tmp_list.append(f"====== 图片 {i+1} ======\n{text}")
                                    success = True
                                else:
                                    retry_count += 1
                                    print(f"【提示】图片{i+1}推理结果为空，重试 {retry_count}/{max_retries}...")
                            else:
                                retry_count += 1
                                print(f"【提示】图片{i+1}推理无结果，重试 {retry_count}/{max_retries}...")
                        except Exception as e:
                            retry_count += 1
                            error_msg = str(e)
                            
                            # 分析错误类型，提供针对性建议
                            if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                                print(f"【显存错误】图片{i+1}推理失败：{error_msg}")
                                print(f"【智能建议】")
                                print(f"  1. 减少max_tokens值（当前：{infer_params['max_tokens']}）")
                                print(f"  2. 降低n_gpu_layers值")
                                print(f"  3. 减少n_ctx值")
                                print(f"  4. 使用更小的模型")
                                
                                # 尝试降低参数后重试
                                if retry_count < max_retries:
                                    # 降低max_tokens重试
                                    infer_params['max_tokens'] = max(128, infer_params['max_tokens'] // 2)
                                    print(f"【自动调整】已将max_tokens降低到{infer_params['max_tokens']}，重新尝试推理...")
                            elif "assertion failed" in error_msg.lower() or "ggml_assert" in error_msg.lower():
                                print(f"【硬件错误】图片{i+1}推理失败：{error_msg}")
                                print(f"【智能建议】")
                                print(f"  1. 降低GPU层数或切换到CPU模式")
                                print(f"  2. 减少上下文长度")
                                print(f"  3. 检查显卡驱动是否最新")
                            else:
                                print(f"【提示】图片{i+1}推理失败，重试 {retry_count}/{max_retries}：{e}")
                            
                    if not success:
                        print(f"【提示】图片{i+1}多次推理失败，跳过该图片")
                        out2.append("推理失败")
                
                out1 = "\n\n".join(tmp_list)
            else:
                for img_base64 in preprocessed_images:
                    if img_base64:
                        image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        user_content.append(image_content)
                
                messages.append({"role": "user", "content": user_content})
                infer_params = {
                    "max_tokens": _parameters.get("max_tokens", 1024),
                    "temperature": _parameters.get("temperature", 0.6),
                    "top_k": _parameters.get("top_k", 20),
                    "top_p": _parameters.get("top_p", 0.85),
                    "repeat_penalty": _parameters.get("repeat_penalty", 1.0),
                    "seed": seed,
                    "stream": False,
                    "stop": ["</s>"]
                }
                
                retry_count = 0
                max_retries = 2
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        output = llama_model.llm.create_chat_completion(messages=messages, **infer_params)
                        if output and 'choices' in output and len(output['choices']) > 0:
                            out1 = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                            if out1.strip():
                                out2 = [out1]
                                success = True
                            else:
                                retry_count += 1
                                print(f"【提示】批量图片推理结果为空，重试 {retry_count}/{max_retries}...")
                        else:
                            retry_count += 1
                            print(f"【提示】批量图片推理无结果，重试 {retry_count}/{max_retries}...")
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        
                        # 分析错误类型，提供针对性建议
                        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                            print(f"【显存错误】批量图片推理失败：{error_msg}")
                            print(f"【智能建议】")
                            print(f"  1. 减少max_tokens值（当前：{infer_params['max_tokens']}）")
                            print(f"  2. 降低n_gpu_layers值")
                            print(f"  3. 减少n_ctx值")
                            print(f"  4. 使用更小的模型")
                            
                            # 尝试降低参数后重试
                            if retry_count < max_retries:
                                # 降低max_tokens重试
                                infer_params['max_tokens'] = max(128, infer_params['max_tokens'] // 2)
                                print(f"【自动调整】已将max_tokens降低到{infer_params['max_tokens']}，重新尝试推理...")
                        elif "assertion failed" in error_msg.lower() or "ggml_assert" in error_msg.lower():
                            print(f"【硬件错误】批量图片推理失败：{error_msg}")
                            print(f"【智能建议】")
                            print(f"  1. 降低GPU层数或切换到CPU模式")
                            print(f"  2. 减少上下文长度")
                            print(f"  3. 检查显卡驱动是否最新")
                        else:
                            print(f"【提示】批量图片推理失败，重试 {retry_count}/{max_retries}：{e}")
                        
                if not success:
                    print(f"【错误】批量图片推理多次失败")
                    out1 = "推理失败"
                    out2 = [out1]
        else:
            # For pure text generation, if chat_handler is None, use simple string format instead of list
            from ..common import LLAMA_CPP_STORAGE
            if LLAMA_CPP_STORAGE.chat_handler is None and len(user_content) == 1 and user_content[0].get("type") == "text":
                messages.append({"role": "user", "content": user_content[0]["text"]})
            else:
                messages.append({"role": "user", "content": user_content})
            infer_params = {
                "max_tokens": _parameters.get("max_tokens", 1024),
                "temperature": _parameters.get("temperature", 0.8),
                "top_k": _parameters.get("top_k", 30),
                "top_p": _parameters.get("top_p", 0.9),
                "min_p": _parameters.get("min_p", 0.05),
                "typical_p": _parameters.get("typical_p", 1.0),
                "repeat_penalty": _parameters.get("repeat_penalty", 1.0),
                "frequency_penalty": _parameters.get("frequency_penalty", 0.0),
                "mirostat_mode": _parameters.get("mirostat_mode", 0),
                "mirostat_eta": _parameters.get("mirostat_eta", 0.1),
                "mirostat_tau": _parameters.get("mirostat_tau", 5.0),
                "seed": seed,
                "stream": False,
                "stop": ["</s>"]
            }
            
            retry_count = 0
            max_retries = 2
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    output = llama_model.llm.create_chat_completion(messages=messages, **infer_params)
                    if output and 'choices' in output and len(output['choices']) > 0:
                        out1 = output['choices'][0]['message']['content'].lstrip().removeprefix(": ")
                        if out1.strip():
                            out2 = [out1]
                            success = True
                        else:
                            retry_count += 1
                            print(f"【提示】纯文本推理结果为空，重试 {retry_count}/{max_retries}...")
                    else:
                        retry_count += 1
                        print(f"【提示】纯文本推理无结果，重试 {retry_count}/{max_retries}...")
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    # 分析错误类型，提供针对性建议
                    if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                        print(f"【显存错误】纯文本推理失败：{error_msg}")
                        print(f"【智能建议】")
                        print(f"  1. 减少max_tokens值（当前：{infer_params['max_tokens']}）")
                        print(f"  2. 降低n_gpu_layers值")
                        print(f"  3. 减少n_ctx值")
                        print(f"  4. 使用更小的模型")
                        
                        # 尝试降低参数后重试
                        if retry_count < max_retries:
                            # 降低max_tokens重试
                            infer_params['max_tokens'] = max(128, infer_params['max_tokens'] // 2)
                            print(f"【自动调整】已将max_tokens降低到{infer_params['max_tokens']}，重新尝试推理...")
                    elif "assertion failed" in error_msg.lower() or "ggml_assert" in error_msg.lower():
                        print(f"【硬件错误】纯文本推理失败：{error_msg}")
                        print(f"【智能建议】")
                        print(f"  1. 降低GPU层数或切换到CPU模式")
                        print(f"  2. 减少上下文长度")
                        print(f"  3. 检查显卡驱动是否最新")
                    else:
                        print(f"【提示】纯文本推理失败，重试 {retry_count}/{max_retries}：{e}")
                        
            if not success:
                print(f"【错误】纯文本推理多次失败")
                out1 = "推理失败"
                out2 = [out1]
        
        # 保存会话状态
        if save_states and llama_model.llm:
            llama_model.messages[f"{uid}"] = messages
        
        # 处理输出语言设置
        def detect_output_language():
            # 检测输出语言
            if output_language == "Auto":
                # 首先检查custom_prompt的语言
                if custom_prompt.strip():
                    return self.detect_language(custom_prompt)
                # 如果custom_prompt为空，检查system_prompt的语言
                elif system_prompt.strip():
                    return self.detect_language(system_prompt)
                # 如果都为空，根据system_prompts_language决定
                else:
                    return input_language
            elif output_language == "中文":
                return "zh"
            else:  # English
                return "en"
        
        # 强制卸载（如果需要）
        if force_offload and llama_model.llm:
            llama_model.clean()
        
        return (out1, out2, int(uid))
    
    def _uniform_sample(self, l, n):
        """
        从列表中均匀采样 n 个元素
        算法: 将列表分成 n 个等长区间,从每个区间的中心位置取样
        """
        if n >= len(l):
            return l
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        # 确保索引不越界
        idxs = [min(i, len(l) - 1) for i in idxs]
        return [l[i] for i in idxs]
    
    def _parse_frame_indices(self, indices_str, total_frames):
        """解析手动输入的帧索引字符串"""
        indices = set()
        if not indices_str:
            return []
            
        parts = indices_str.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                if '-' in part:
                    # Range: 0-10
                    start_str, end_str = part.split('-')
                    start = int(start_str)
                    end = int(end_str)
                    # Handle negative indices
                    if start < 0: start += total_frames
                    if end < 0: end += total_frames
                    
                    start = max(0, min(start, total_frames - 1))
                    end = max(0, min(end, total_frames - 1))
                    
                    if start <= end:
                        indices.update(range(start, end + 1))
                else:
                    # Single index
                    idx = int(part)
                    if idx < 0: idx += total_frames
                    idx = max(0, min(idx, total_frames - 1))
                    indices.add(idx)
            except ValueError:
                print(f"【提示】忽略无效的帧索引格式: {part}")
                
        return sorted(list(indices))
    
    def _extract_frames_and_tensor(self, tensor, target_count=None, target_indices_str=None):
        """从tensor中提取指定数量或指定索引的帧"""
        total_frames = tensor.shape[0]
        
        # 生成所有帧的索引列表
        all_indices = list(range(total_frames))
        
        selected_indices = []
        if target_indices_str is not None:
            # 手动模式
            selected_indices = self._parse_frame_indices(target_indices_str, total_frames)
            if not selected_indices:
                print(f"【提示】手动帧索引无效或为空,回退到自动采样")
                selected_indices = self._uniform_sample(all_indices, 8) # Default fallback
        elif target_count is not None:
            # 自动模式
            selected_indices = self._uniform_sample(all_indices, target_count)
        else:
            # 默认全量
            selected_indices = all_indices
        
        # 提取选中的帧 [N, H, W, C]
        selected_tensor = tensor[selected_indices]
        
        return selected_tensor
    
    def _load_video_as_tensor(self, video):
        """
        从视频输入加载为tensor
        支持VideoFromFile对象、字典格式和直接的tensor
        """
        import os
        import tempfile
        import numpy as np
        import torch
        
        # 处理VideoFromFile对象(需要先保存再加载)
        if hasattr(video, 'save_to') and callable(getattr(video, 'save_to')):
            try:
                fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
                os.close(fd)
                video.save_to(temp_video_path)
                print(f"【视频处理】VideoFromFile saved to temp")
                
                # 尝试使用imageio读取视频
                try:
                    import imageio
                    # 读取视频
                    reader = imageio.get_reader(temp_video_path, 'ffmpeg')
                    
                    frames = []
                    for frame in reader:
                        frame_float = frame.astype(np.float32) / 255.0
                        frames.append(frame_float)
                    reader.close()
                    
                    if not frames:
                        raise RuntimeError("视频文件中没有帧")
                    
                    frames_array = np.stack(frames, axis=0)
                    tensor = torch.from_numpy(frames_array)
                    
                    # 清理临时文件
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                    
                    return tensor
                except Exception as e:
                    # 清理临时文件
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                    raise RuntimeError(f"视频文件读取失败: {str(e)}")
            except Exception as e:
                if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                raise RuntimeError(f"VideoFromFile processing failed: {str(e)}")
        
        # 处理字典格式
        elif isinstance(video, dict):
            input_tensor = video.get('frames') or video.get('video')
            if input_tensor is None:
                for v in video.values():
                    if isinstance(v, torch.Tensor):
                        input_tensor = v
                        break
            if input_tensor is not None:
                return input_tensor
            else:
                raise ValueError(f"Failed to extract tensor from VIDEO input")
        
        # 处理tensor
        elif isinstance(video, torch.Tensor):
            return video
        
        # 尝试索引访问
        elif hasattr(video, '__getitem__'):
            try:
                first_item = video[0]
                if isinstance(first_item, torch.Tensor):
                    return first_item
                elif isinstance(first_item, dict):
                    input_tensor = first_item.get('frames') or first_item.get('video')
                    if input_tensor is not None:
                        return input_tensor
            except Exception:
                pass
        
        raise ValueError(f"Failed to extract tensor from VIDEO input")