# -*- coding: utf-8 -*-
"""
模型类型检测器模块

自动识别模型架构和类型，根据模型名称或配置文件推断模型的特性。
支持多种模型系列（Qwen、MiniCPM、GLM、DreamOmni等）的智能检测。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

from typing import Dict


class ModelTypeDetector:
    """模型类型检测器 - 自动识别模型架构"""
    
    MODEL_SIGNATURES = {
        "qwen3.5": {
            "keywords": ["qwen3.5", "qwen35", "qwen-3.5", "qwen_35"],
            "type": "omni",
            "subtype": "qwen35",
            # llama-cpp 当前 Qwen35ChatHandler 不支持 audio_url 项，跳过原生音频输入
            "supports_audio": False,
            "supports_vision": True,
            "audio_special_tokens": ["<|audio|>", "<|audio_start|>", "<|audio_end|>"],
            "vision_special_tokens": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"],
            "file_formats": [".gguf"],
        },
        "qwen2.5-omni": {
            "keywords": ["qwen2.5-omni", "qwen25-omni", "qwen-2.5-omni", "qwen2.5omni", "qwen25omni", "qwen2.5-omni-3b", "qwen2.5omni3b"],
            "type": "omni",
            "subtype": "qwen25_omni",
            # llama-cpp 当前 Qwen25VLChatHandler 不支持 audio_url 项，跳过原生音频输入
            "supports_audio": False,
            "supports_vision": True,
            "audio_special_tokens": ["<|audio|>", "<|audio_start|>", "<|audio_end|>"],
            "vision_special_tokens": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"],
            "file_formats": [".gguf"],
        },
        "dreamomni2": {
            "keywords": ["dreamomni2", "dream-omni2", "dream_omni2"],
            "type": "omni",
            "subtype": "dreamomni2",
            "supports_audio": True,
            "supports_vision": True,
            "audio_special_tokens": ["[AUDIO]", "[/AUDIO]"],
            "vision_special_tokens": ["[IMG]", "[/IMG]"],
            "file_formats": [".gguf"],
        },
        "glm-4.6v": {
            "keywords": ["glm-4.6v", "glm4.6v", "glm_4.6v"],
            "type": "vl",
            "subtype": "glm4v",
            "supports_audio": False,
            "supports_vision": True,
            "vision_special_tokens": ["<|begin_of_image|>", "<|end_of_image|>"],
        },
        "glm-4v": {
            "keywords": ["glm-4v", "glm4v", "glm_4v"],
            "type": "vl",
            "subtype": "glm4v",
            "supports_audio": False,
            "supports_vision": True,
            "vision_special_tokens": ["<|begin_of_image|>", "<|end_of_image|>"],
        },
        "default_vl": {
            "keywords": [],
            "type": "vl",
            "subtype": "default",
            "supports_audio": False,
            "supports_vision": True,
            "vision_special_tokens": [],
        }
    }
    
    @classmethod
    def detect_model_type(cls, model_path: str, model_config: Dict = None) -> Dict:
        """
        检测模型类型
        
        Args:
            model_path: 模型路径或名称
            model_config: 模型配置信息
            
        Returns:
            Dict: 模型类型信息
        """
        model_path_lower = model_path.lower()
        
        for model_key, signature in cls.MODEL_SIGNATURES.items():
            if model_key == "default_vl":
                continue
                
            for keyword in signature["keywords"]:
                if keyword in model_path_lower:
                    return {
                        "key": model_key,
                        **signature
                    }
        
        # 检查模型配置中的特殊标记
        if model_config:
            chat_format = model_config.get("chat_format", "").lower()
            chat_handler = model_config.get("chat_handler", "").lower()
            model_name = model_config.get("model", "").lower()
            
            for model_key, signature in cls.MODEL_SIGNATURES.items():
                if model_key == "default_vl":
                    continue
                for keyword in signature["keywords"]:
                    # 检查chat_format
                    if keyword in chat_format:
                        return {
                            "key": model_key,
                            **signature
                        }
                    # 检查chat_handler
                    if keyword in chat_handler:
                        return {
                            "key": model_key,
                            **signature
                        }
                    # 检查model名称
                    if keyword in model_name:
                        return {
                            "key": model_key,
                            **signature
                        }
        
        # 默认返回VL模型
        return {
            "key": "default_vl",
            **cls.MODEL_SIGNATURES["default_vl"]
        }