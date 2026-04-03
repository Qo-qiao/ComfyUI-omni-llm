# -*- coding: utf-8 -*-
"""
视觉语言推理引擎模块

提供视觉语言模型（VLM）的推理引擎实现，支持图像理解和视觉推理功能。
主要针对仅支持视觉输入的模型，不支持音频输入。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import torch
from typing import Dict, List, Optional
from .base_engine import BaseInferenceEngine, scale_image, image2base64


class GLM4VInferenceEngine(BaseInferenceEngine):
    """GLM-4V系列推理引擎 (仅视觉，不支持音频)"""
    
    def __init__(self, model_info: Dict):
        super().__init__(model_info)
        self.supports_audio = False
        self.image_start_token = "<|begin_of_image|>"
        self.image_end_token = "<|end_of_image|>"
    
    def build_vision_messages(self, system_prompt: str, text: str,
                              images: torch.Tensor = None, max_size: int = 512) -> List[Dict]:
        """构建GLM-4V格式的消息"""
        content_parts = []
        
        # 添加文本
        if text:
            content_parts.append({"type": "text", "text": text})
        
        # 添加图像
        if images is not None:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            
            for i in range(images.shape[0]):
                img = images[i].cpu().numpy()
                img_np = scale_image(img, max_size)
                img_base64 = image2base64(img_np)
                
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": content_parts})
        
        return messages
