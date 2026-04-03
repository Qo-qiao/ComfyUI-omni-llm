# -*- coding: utf-8 -*-
"""
Omni系列推理引擎模块

提供Omni系列模型（Qwen3.5, Qwen2.5-Omni, MiniCPM-O, DreamOmni2等）的推理引擎实现。
支持音频和视觉多模态输入，包括文本生成、音频生成、图像理解等功能。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import torch
import re
import base64
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_engine import BaseInferenceEngine, scale_image, image2base64


class QwenOmniInferenceEngine(BaseInferenceEngine):
    """Qwen Omni系列推理引擎 (Qwen3.5, Qwen2.5-Omni)"""
    
    def __init__(self, model_info: Dict):
        super().__init__(model_info)
        self.audio_start_token = "<|audio_start|>"
        self.audio_end_token = "<|audio_end|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.audio_pad_token = "<|audio_pad|>"
        self.image_pad_token = "<|image_pad|>"
    
    def build_omni_messages(self, system_prompt: str, text: str, 
                            images: torch.Tensor = None, audio: Dict = None,
                            max_size: int = 512) -> List[Dict]:
        """构建Qwen Omni格式的消息"""
        content = []
        
        # 添加文本
        if text:
            content.append({"type": "text", "text": text})
        
        # 添加图像
        if images is not None and self.supports_vision:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()
                img_np = scale_image(img, max_size)
                img_base64 = image2base64(img_np)
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
        
        # 添加音频
        if audio is not None:
            if self.supports_audio:
                audio_content = self.process_audio_to_content(audio, self.model_subtype)
                if audio_content:
                    content.append(audio_content)
                else:
                    print("【Omni 音频】音频处理函数未返回可用内容，已忽略音频输入。")
            else:
                print("【Omni 音频】模型不支持直接原生音频输入，已忽略音频输入。")
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": content})
        
        return messages
    
    def generate_with_audio_output(self, llm, messages: List[Dict], params: Dict,
                                   voice_type: str = "female_cn", emotion: str = "default") -> Tuple[str, Optional[Dict]]:
        """生成文本和音频输出 - Qwen2.5-Omni优化版"""
        try:
            print(f"【Qwen-Omni音频生成】音色: {voice_type}, 情感: {emotion}")
            
            # 添加音频生成指令
            audio_instruction = f"\n请同时生成音频，使用{voice_type}音色，{emotion}情感。"
            if isinstance(messages[-1]["content"], list):
                for item in messages[-1]["content"]:
                    if item.get("type") == "text":
                        item["text"] += audio_instruction
                        break
            else:
                messages[-1]["content"] += audio_instruction
            
            # 调用API
            output = self.create_chat_completion(llm, messages, params)
            
            if output and 'choices' in output and len(output['choices']) > 0:
                content = output['choices'][0]['message']['content']
                print(f"【Qwen-Omni音频生成】原始内容长度: {len(content)}")
                
                # 尝试解析音频数据
                audio_output = None
                if '<|audio|' in content or 'data:audio' in content:
                    audio_output = self._extract_audio_from_response(content)
                    if audio_output:
                        print(f"【Qwen-Omni音频生成】音频提取成功")
                        # 清理文本中的音频标记
                        content = re.sub(r'<\|audio.*?\|>', '', content)
                        content = re.sub(r'data:audio/[^\s]+\s*', '', content)
                        content = content.strip()
                    else:
                        print(f"【Qwen-Omni音频生成】音频提取失败")
                else:
                    print(f"【Qwen-Omni音频生成】未检测到音频标记")
                
                return content.lstrip().removeprefix(": "), audio_output
            
            print(f"【Qwen-Omni音频生成】API返回异常")
            return "生成失败", None
            
        except Exception as e:
            print(f"【Qwen Omni生成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return f"生成失败: {str(e)}", None
    
    def _extract_audio_from_response(self, content: str) -> Optional[Dict]:
        """从响应中提取音频数据"""
        try:
            # 查找base64编码的音频
            audio_pattern = r'data:audio/[^;]+;base64,([A-Za-z0-9+/=]+)'
            match = re.search(audio_pattern, content)
            
            if match:
                audio_base64 = match.group(1)
                audio_bytes = base64.b64decode(audio_base64)
                
                # 转换为numpy数组
                import numpy as np
                waveform = np.frombuffer(audio_bytes, dtype=np.int16)
                waveform = waveform.astype(np.float32) / 32768.0
                
                # 转换为tensor
                audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
                
                return {
                    "waveform": audio_tensor,
                    "sample_rate": 24000  # 匹配参考文档的采样率
                }
            
            return None
            
        except Exception as e:
            print(f"【音频提取错误】{str(e)}")
            return None


class MiniCPMOInferenceEngine(BaseInferenceEngine):
    """MiniCPM-O系列推理引擎"""
    
    def __init__(self, model_info: Dict):
        super().__init__(model_info)
        self.audio_start_token = ""
        self.audio_end_token = " Audio"
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
    
    def build_omni_messages(self, system_prompt: str, text: str,
                            images: torch.Tensor = None, audio: Dict = None,
                            max_size: int = 512) -> List[Dict]:
        """构建MiniCPM-O格式的消息"""
        content_parts = []
        
        # 构建文本内容
        full_text = text if text else ""
        
        # 添加图像标记
        if images is not None and self.supports_vision:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            image_contents = []
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()
                img_np = scale_image(img, max_size)
                img_base64 = image2base64(img_np)
                
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
            
            # MiniCPM-O使用特殊格式
            content_parts = [{"type": "text", "text": full_text}]
            content_parts.extend(image_contents)
        else:
            content_parts = [{"type": "text", "text": full_text}]
        
        # 添加音频
        if audio is not None and self.supports_audio:
            audio_content = self.process_audio_to_content(audio, self.model_subtype)
            if audio_content:
                content_parts.append(audio_content)
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": content_parts})
        
        return messages
    
    def generate_with_audio_output(self, llm, messages: List[Dict], params: Dict,
                                   voice_type: str = "female_cn", emotion: str = "default") -> Tuple[str, Optional[Dict]]:
        """生成文本和音频输出 - MiniCPM-o-4.5优化版"""
        try:
            print(f"【MiniCPM-O音频生成】音色: {voice_type}, 情感: {emotion}")
            
            # 添加音频生成指令
            audio_instruction = f"\n请同时生成音频，使用{voice_type}音色，{emotion}情感。"
            if isinstance(messages[-1]["content"], list):
                for item in messages[-1]["content"]:
                    if item.get("type") == "text":
                        item["text"] += audio_instruction
                        break
            else:
                messages[-1]["content"] += audio_instruction
            
            # 调用API
            output = self.create_chat_completion(llm, messages, params)
            
            if output and 'choices' in output and len(output['choices']) > 0:
                content = output['choices'][0]['message']['content']
                print(f"【MiniCPM-O音频生成】原始内容长度: {len(content)}")
                
                # 尝试解析音频数据
                audio_output = None
                if '<audio>' in content or 'data:audio' in content:
                    audio_output = self._extract_audio_from_response(content)
                    if audio_output:
                        print(f"【MiniCPM-O音频生成】音频提取成功")
                        # 清理文本中的音频标记
                        content = content.replace('<audio>', '').replace('</audio>', '')
                        content = content.replace(' Audio', '')
                        content = re.sub(r'data:audio/[^\s]+\s*', '', content)
                        content = content.strip()
                    else:
                        print(f"【MiniCPM-O音频生成】音频提取失败")
                else:
                    print(f"【MiniCPM-O音频生成】未检测到音频标记")
                
                return content.lstrip().removeprefix(": "), audio_output
            
            print(f"【MiniCPM-O音频生成】API返回异常")
            return "生成失败", None
            
        except Exception as e:
            print(f"【MiniCPM-O生成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return f"生成失败: {str(e)}", None
    
    def _extract_audio_from_response(self, content: str) -> Optional[Dict]:
        """从响应中提取音频数据 - MiniCPM-o-4.5优化版"""
        try:
            # 查找base64编码的音频
            audio_pattern = r'data:audio/[^;]+;base64,([A-Za-z0-9+/=]+)'
            match = re.search(audio_pattern, content)
            
            if match:
                audio_base64 = match.group(1)
                audio_bytes = base64.b64decode(audio_base64)
                
                # 转换为numpy数组
                import numpy as np
                waveform = np.frombuffer(audio_bytes, dtype=np.int16)
                waveform = waveform.astype(np.float32) / 32768.0
                
                # 转换为tensor
                audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
                
                return {
                    "waveform": audio_tensor,
                    "sample_rate": 24000  # 匹配MiniCPM-o-4.5的采样率
                }
            
            return None
            
        except Exception as e:
            print(f"【音频提取错误】{str(e)}")
            return None



