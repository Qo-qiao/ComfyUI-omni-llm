# -*- coding: utf-8 -*-
"""
Llama-cpp Image Inference Node
"""
import numpy as np
from .common import (
    HARDWARE_INFO, any_type, image2base64, scale_image,
    mm
)

class llama_cpp_instruct_adv:
    preset_prompts = {
        "Empty - Nothing": "",
        "Normal - Describe": "IMPORTANT: Respond in the same language as the input. Describe this @ in detail. If custom content is provided, use it as the basis: #",
        "Prompt Style - Tags": "IMPORTANT: Respond in the same language as the input. Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based on the visual information in the @ and any custom content provided. If custom content is provided, use it as the basis: #\n\nLimit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
        "Prompt Style - Simple": "IMPORTANT: Respond in the same language as the input. Your task is to expand the following user-provided prompt into detailed, vivid, and contextually rich text, enhancing its clarity and expressiveness for AI generation (image/video/story, etc.), without altering the original intent or core keywords. Steps: 1. Extract core elements (subject, setting, action, emotional tone); 2. Supplement specific sensory and visual details. Requirements: Focus on key information, ensure logical coherence, avoid redundancy, keep the expanded text within 300 words, use specific descriptive words extensively, and provide clear guidance for AI. If custom content is provided, use it as the basis: #",
        "Prompt Style - Detailed": "IMPORTANT: Respond in the same language as the input. Your task is to expand the following user-provided prompt into detailed, vivid, and contextually rich text, enhancing its clarity and expressiveness for AI generation tasks (text-to-image/video/story, etc.), strictly retaining the original intent and core keywords. Process: 1. Accurately identify core elements, including subject, setting, action (if any), emotional tone, and key themes; 2. Supplement details targeted: for the subject, add appearance, characteristics, and contextual relevance; for the setting, add environment, sensory cues, and temporal context; for the action, add process and interaction; for the emotional tone, strengthen it through appropriate descriptive language. Requirements: Ensure coherent content and clear logic, no redundant or irrelevant content, keep the expanded text within 600 words, prioritize specific and clear descriptive words, and provide precise and executable visual and contextual guidance for AI. If custom content is provided, use it as the basis: #",
        "Prompt Style - Comprehensive Expansion": "Your task is to expand the following user-provided prompt into detailed, vivid, and contextually rich text, enhancing its clarity and expressiveness for AI generation tasks such as text-to-image, text-to-video, text-to-story, etc. IMPORTANT: Respond in the same language as the user-provided prompt. The user-provided prompt is: #\n\nFirst, identify the core elements of the original prompt, including the central subject, scene setting, actions (if any), emotional tone (implied or explicit), and key themes. Then, without changing the original intent and core keywords, add specific, sensory-rich, and visually/contextually appropriate descriptive details to each element.\n\nFor the central subject, add specific attributes such as appearance features (e.g., physical details, clothing textures, accessories, posture), traits (e.g., mannerisms, movement rhythm, emotional state), and contextual relevance (e.g., the reason the subject is in this scene, the purpose of their actions). For the scene setting, elaborate on environmental details (e.g., spatial layout, surrounding objects, natural/urban elements), sensory clues (e.g., light quality, color palette, sound hints, surface textures), and temporal context (e.g., time of day, season, weather conditions).\n\nIf the original prompt includes actions, expand on the action process (e.g., action sequence, intensity, interaction between the subject and environment/other subjects). For emotional tone and themes, strengthen expression through descriptive language that evokes corresponding feelings (e.g., using warm light and soft colors to create a comfortable atmosphere, using sharp shadows and cool tones to convey tension).\n\nEnsure the expanded prompt is coherent and natural, with clear logic between expanded details. Avoid redundant descriptions and irrelevant additions. Keep the expanded prompt under 1000 words. Prioritize specific, clear descriptive words over vague, abstract expressions to provide the AI with clear and executable visual/contextual guidance.",
        "Creative - Detailed Analysis": "IMPORTANT: Respond in the same language as the input. Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections. If custom content is provided, use it as the basis: #",
        "Creative - Summarize Video": "IMPORTANT: Respond in the same language as the input. Summarize the key events and narrative points in this video. If custom content is provided, use it as the basis: #",
        "Creative - Short Story": "IMPORTANT: Respond in the same language as the input. Write a short, imaginative story inspired by this @ or video. If custom content is provided, use it as the basis: #",
        "Creative - Refine & Expand Prompt": "IMPORTANT: Respond in the same language as the input. Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.\n\nUser prompt: #",
        "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
    }
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
                "max_frames": ("INT", {"default": 16, "min": 2, "max": 1024, "step": 1}),
                "max_size": ("INT", {"default": 256, "min": 128, "max": 16384, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {"default": False}),
                "save_states": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "images": ("IMAGE",),
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
        """Detect if the text is Chinese or English"""
        if not text or not isinstance(text, str):
            return "en"
        
        # Count Chinese characters
        chinese_chars = 0
        total_chars = len(text)
        
        for char in text:
            # Check if character is Chinese
            if '\u4e00' <= char <= '\u9fff':
                chinese_chars += 1
        
        # If more than 30% are Chinese characters, consider it Chinese
        if total_chars > 0 and (chinese_chars / total_chars) > 0.3:
            return "zh"
        else:
            return "en"
    
    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, images=None, queue_handler=None):
        if not llama_model.llm:
            raise RuntimeError("【错误】模型未加载或已卸载，请先加载LLM模型")
        
        if parameters is None:
            parameters = {
                "max_tokens": 1024 if HARDWARE_INFO["is_low_perf"] else 2048,
                "top_k": 20 if HARDWARE_INFO["is_low_perf"] else 30,
                "top_p": 0.85 if HARDWARE_INFO["is_low_perf"] else 0.9,
                "min_p": 0.05,
                "typical_p": 1.0,
                "temperature": 0.6 if HARDWARE_INFO["is_low_perf"] else 0.8,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 1.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0,
                "state_uid": -1
            }
        
        _uid = parameters.get("state_uid", None)
        _parameters = parameters.copy()
        _parameters.pop("state_uid", None)
        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid
        last_sys_prompt = llama_model.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        text_input = inference_mode == "text"
        
        # Detect language based on custom_prompt
        input_language = self.detect_language(custom_prompt)
        
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
        
        # 处理提示词逻辑
        if preset_prompt == "Empty - Nothing":
            # 当使用Empty - Nothing预设时：
            # 1. 如果有custom_prompt，直接使用它作为用户输入
            # 2. system_prompt已经作为系统指令在messages列表开头设置
            # 这样模型会根据system_prompt的指令处理custom_prompt的内容
            if custom_prompt.strip():
                final_prompt = custom_prompt.strip()
            else:
                # 如果没有custom_prompt，就使用system_prompt作为输入
                final_prompt = system_prompts.strip() if system_prompts.strip() else custom_prompt.strip()
        else:
            # 处理其他预设
            final_prompt = preset_text
            if custom_prompt.strip():
                if "*" in preset_prompt:
                    final_prompt = custom_prompt.strip()
                else:
                    final_prompt = preset_text.replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
        
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
                        "presence_penalty": _parameters.get("presence_penalty", 1.0),
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
            from .common import LLAMA_CPP_STORAGE
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
                "presence_penalty": _parameters.get("presence_penalty", 1.0),
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
        
        # 强制卸载（如果需要）
        if force_offload and llama_model.llm:
            llama_model.clean()
        
        return (out1, out2, int(uid))
