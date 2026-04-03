# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm 角色配置节点

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RoleConfig:
    """
    角色配置节点
    用于配置角色的参数，包括名称、音色、语速、音高、音量等
    支持本地TTS模型和API TTS模型
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # 本地TTS模型的音色选项
        local_voices = ["Vivian - 明亮略带锐气的年轻女声 (中文)", "Serena - 温暖柔和的年轻女声 (中文)", "Uncle_Fu - 音色低沉醇厚的成熟男声 (中文)", "Dylan - 清晰自然的北京青年男声 (中文北京方言)"]
        
        # API TTS模型的音色选项（OpenAI兼容）
        api_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        return {
            "required": {
                "role_name": ("STRING", {"default": "角色1", "tooltip": "角色名称"}),
                "config_type": (["Local model", "API model"], {"default": "Local model", "tooltip": "配置类型"}),
            },
            "optional": {
                # 本地模型参数
                "local_voice": (local_voices, {"default": "Vivian - 明亮略带锐气的年轻女声 (中文)", "tooltip": "本地模型音色"}),
                "local_emotion": (["default", "happy", "sad", "angry", "surprised", "calm", "excited", "gentle"], {"default": "default", "tooltip": "本地模型情绪"}),
                
                # API模型参数
                "api_voice": (api_voices, {"default": "alloy", "tooltip": "API模型音色"}),
                
                # 通用参数
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider"}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "display": "slider"}),
                "volume": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "display": "slider"}),
                
                # API配置（用于多API模式）
                "api_config_name": ("STRING", {"default": "", "tooltip": "API配置名称（多API模式使用）"}),
                "use_custom_api": ("BOOLEAN", {"default": False, "tooltip": "是否使用自定义API配置"}),
                "api_provider": (["OpenAI", "Ollama", "llms-py"], {"default": "OpenAI", "tooltip": "API提供商"}),
                "api_base": ("STRING", {"default": "https://api.openai.com/v1", "tooltip": "API地址"}),
                "api_key": ("STRING", {"default": "", "tooltip": "API密钥"}),
            }
        }
    
    RETURN_TYPES = ("ROLE_CONFIG",)
    RETURN_NAMES = ("role_config",)
    FUNCTION = "create_role_config"
    CATEGORY = "llama-cpp-vlm"
    
    def create_role_config(self, role_name, config_type,
                         local_voice="Vivian - 明亮略带锐气的年轻女声 (中文)", local_emotion="默认",
                         api_voice="alloy",
                         speed=1.0, pitch=0.0, volume=1.0,
                         api_config_name="", use_custom_api=False,
                         api_provider="OpenAI", api_base="https://api.openai.com/v1", api_key=""):
        """
        创建角色配置
        
        Args:
            role_name: 角色名称
            config_type: 配置类型（本地模型/API模型）
            local_voice: 本地模型音色
            local_emotion: 本地模型情绪
            api_voice: API模型音色
            speed: 语速
            pitch: 音高偏移
            volume: 音量
            api_config_name: API配置名称（多API模式使用）
            use_custom_api: 是否使用自定义API配置
            api_provider: API提供商
            api_base: API地址
            api_key: API密钥
            
        Returns:
            角色配置字典
        """
        role_config = {
            "role_name": role_name,
            "config_type": config_type,
            "speed": speed,
            "pitch": pitch,
            "volume": volume
        }
        
        # 根据配置类型添加特定参数
        if config_type == "本地模型":
            role_config["voice"] = local_voice
            role_config["emotion"] = local_emotion
            role_config["model_type"] = "local"
        else:  # API模型
            role_config["voice"] = api_voice
            role_config["model_type"] = "api"
            
            # 添加API配置信息
            if api_config_name:
                role_config["api_config_name"] = api_config_name
            
            if use_custom_api:
                role_config["api_config"] = {
                    "api_provider": api_provider,
                    "api_base": api_base,
                    "api_key": api_key
                }
        
        return (role_config,)


NODE_CLASS_MAPPINGS = {
    "RoleConfig": RoleConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RoleConfig": "角色配置",
}
