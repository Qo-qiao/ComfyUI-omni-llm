# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm API Config Nodes - 配置管理相关节点
优化版本：添加图像压缩、视频处理、错误处理增强
解决阿里云6MB请求体限制问题
"""
import os
import sys
import json
import requests
import base64
import io
import re
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union
from PIL import Image
from urllib.parse import urlparse

# 尝试导入OpenAI SDK（可选）
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from support.api_provider_presets import PRESET_MODELS, DEFAULT_PROVIDER_CONFIGS

class APIConfigManager:
    _instance = None
    current_provider = "OpenAI"
    active_config = DEFAULT_PROVIDER_CONFIGS["OpenAI"].copy()

    # API限制配置（字节）- 阿里云限制6MB，设置5MB为安全阈值
    MAX_REQUEST_SIZE = 5 * 1024 * 1024  # 5MB安全限制
    MAX_IMAGE_SIZE = 2048  # 最大图像边长

    @classmethod
    def validate_api_config(cls, api_config: Dict[str, Any]) -> Union[bool, str]:
        """验证API配置的有效性"""
        if not api_config:
            return "API配置为空"
        
        api_provider = api_config.get("api_provider", "")
        api_base = api_config.get("api_base", "").strip()
        api_key = api_config.get("api_key", "").strip()
        model_id = api_config.get("model_id", "").strip()
        
        # 验证API基础URL
        if not api_base:
            return "API基础URL未配置"
        
        try:
            result = urlparse(api_base)
            if result.scheme not in ["http", "https"]:
                return "URL格式无效，必须以http或https开头"
            if not result.netloc:
                return "URL缺少主机名"
        except ValueError as e:
            return f"URL格式无效: {str(e)}"
        
        # 验证模型ID
        if not model_id:
            return "模型ID未配置"
        
        # 验证API密钥（针对需要密钥的提供商）
        if api_provider in ["OpenAI", "Anthropic", "Moonshot AI", "Qwen", "Kimi", "GLM", "MiniMax"]:
            if not api_key:
                return f"{api_provider} 需要配置API密钥"
        
        return True
    JPEG_QUALITY = 85  # JPEG质量

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIConfigManager, cls).__new__(cls)
            cls._instance.api_configs = DEFAULT_PROVIDER_CONFIGS.copy()
            cls._instance.api_models = list(cls._instance.api_configs.keys())
            cls._instance.switch_provider(cls.current_provider)
        return cls._instance

    def switch_provider(self, provider_name):
        if provider_name not in self.api_configs:
            self.current_provider = "自定义"
            self.active_config = DEFAULT_PROVIDER_CONFIGS["自定义"].copy()
            return False
        target_config = self.api_configs[provider_name]
        self.current_provider = provider_name
        self.active_config = target_config.copy()
        return True

    def get_active_params(self):
        return self.active_config.copy()

    def update_active_config(self, update_dict):
        self.active_config.update(update_dict)
        self.api_configs[self.current_provider] = self.active_config.copy()

    def get_provider_list(self):
        return list(self.api_configs.keys())

    def get_supported_models(self, provider_name=None):
        provider = provider_name or self.current_provider
        return PRESET_MODELS.get(provider, ["自定义输入"])

    def resize_image(self, pil_image, max_size=1024):
        """调整图像尺寸，保持长宽比"""
        width, height = pil_image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return pil_image

    def compress_image_to_limit(self, pil_image, max_bytes=5*1024*1024, initial_quality=85):
        """
        压缩图像到指定大小限制内
        策略：先尝试JPEG质量调节，如果仍过大则缩小尺寸
        """
        buffer = io.BytesIO()

        # 转换为RGB（移除alpha通道）
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            if pil_image.mode in ('RGBA', 'LA'):
                background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                pil_image = background
            else:
                pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        quality = initial_quality
        min_quality = 30

        # 尝试降低质量
        while quality >= min_quality:
            buffer.seek(0)
            buffer.truncate()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            size = buffer.tell()

            if size <= max_bytes:
                break
            quality -= 10

        # 如果质量降到最低还是太大，缩小尺寸
        if buffer.tell() > max_bytes:
            width, height = pil_image.size
            scale = 0.8
            while buffer.tell() > max_bytes and scale > 0.3:
                new_size = (int(width * scale), int(height * scale))
                resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                buffer.seek(0)
                buffer.truncate()
                resized.save(buffer, format='JPEG', quality=min_quality, optimize=True)
                scale -= 0.1

        buffer.seek(0)
        return buffer

    def encode_image(self, image_tensor: Union[torch.Tensor, Image.Image, np.ndarray], 
                    max_size: int = 1024, 
                    max_bytes: int = 5*1024*1024) -> Optional[str]:
        """
        编码图像为base64，自动压缩以适应API限制

        Args:
            image_tensor: 输入图像tensor或PIL Image或numpy数组
            max_size: 最大边长（像素）
            max_bytes: 最大字节数（默认5MB）

        Returns:
            base64编码的图像字符串，失败时返回None
        """
        try:
            if isinstance(image_tensor, torch.Tensor):
                image_np = image_tensor.cpu().numpy()
                if image_np.ndim == 4:
                    image_np = image_np[0]
                # 处理通道顺序 (H, W, C) 或 (C, H, W)
                if image_np.shape[0] in [1, 3, 4] and image_np.ndim == 3:
                    image_np = np.transpose(image_np, (1, 2, 0))
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            elif isinstance(image_tensor, Image.Image):
                pil_image = image_tensor
            elif isinstance(image_tensor, np.ndarray):
                if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3, 4]:
                    image_tensor = np.transpose(image_tensor, (1, 2, 0))
                if image_tensor.max() <= 1.0:
                    image_tensor = (image_tensor * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_tensor)
            else:
                print(f"【图像编码】不支持的输入类型: {type(image_tensor)}")
                return None

            # 调整尺寸
            pil_image = self.resize_image(pil_image, max_size)

            # 压缩到限制内
            buffer = self.compress_image_to_limit(pil_image, max_bytes)

            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            actual_size = len(base64_str) * 3 // 4  # 估算实际字节数
            print(f"【图像编码】尺寸: {pil_image.size}, 预估大小: {actual_size/1024:.1f}KB")

            return base64_str

        except Exception as e:
            print(f"【图像编码错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def encode_audio(self, audio_data: Union[Dict[str, Any], torch.Tensor, np.ndarray], 
                    max_bytes: int = 5*1024*1024) -> Optional[str]:
        """
        编码音频为base64，自动压缩以适应API限制

        Args:
            audio_data: 音频数据（ComfyUI AUDIO格式或tensor或numpy数组）
            max_bytes: 最大字节数

        Returns:
            base64编码的音频字符串，失败时返回None
        """
        try:
            if audio_data is None:
                return None

            # ComfyUI音频格式通常是字典，包含 waveform 和 sample_rate
            if isinstance(audio_data, dict):
                waveform = audio_data.get('waveform')
                sample_rate = audio_data.get('sample_rate', 16000)
            else:
                waveform = audio_data
                sample_rate = 16000

            if waveform is None:
                print("【音频编码】音频数据为空")
                return None

            # 转换为numpy数组
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()

            # 确保是1D或2D数组
            if waveform.ndim > 2:
                waveform = waveform.squeeze()

            # 转换为16位PCM WAV格式
            import wave
            buffer = io.BytesIO()

            # 归一化到16位范围
            if waveform.dtype != np.int16:
                if waveform.max() <= 1.0:
                    waveform = (waveform * 32767).astype(np.int16)
                else:
                    waveform = waveform.astype(np.int16)

            # 如果是单声道，添加维度
            if waveform.ndim == 1:
                waveform = waveform.reshape(-1, 1)

            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(waveform.shape[1] if waveform.ndim > 1 else 1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(waveform.tobytes())

            wav_bytes = buffer.getvalue()

            # 检查大小并压缩（重采样）如果需要
            if len(wav_bytes) > max_bytes:
                print(f"【音频编码】音频过大({len(wav_bytes)/1024:.1f}KB)，进行降采样...")
                try:
                    from scipy import signal
                    target_rate = 8000
                    num_samples = int(len(waveform) * target_rate / sample_rate)
                    resampled = signal.resample(waveform, num_samples)

                    buffer = io.BytesIO()
                    with wave.open(buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(target_rate)
                        wav_file.writeframes(resampled.astype(np.int16).tobytes())
                    wav_bytes = buffer.getvalue()
                    print(f"【音频编码】降采样后大小: {len(wav_bytes)/1024:.1f}KB")
                except ImportError:
                    print("【音频编码】scipy未安装，无法降采样")

            return base64.b64encode(wav_bytes).decode('utf-8')

        except Exception as e:
            print(f"【音频编码错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def detect_language(self, text):
        """检测文本语言（简单版本）"""
        if not text:
            return "未知"

        # 检查中文字符比例
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len(text.strip())

        if total_chars == 0:
            return "未知"

        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3:
            return "中文"
        else:
            return "English"

    def call_api(self, api_config: Dict[str, Any], 
                messages: List[Dict[str, Any]], 
                max_tokens: int = 1024, 
                temperature: float = 0.7, 
                retries: int = 2) -> Optional[Dict[str, Any]]:
        """
        调用API，支持自动重试

        Args:
            api_config: API配置字典
            messages: 消息列表
            max_tokens: 最大token数
            temperature: 温度参数
            retries: 失败重试次数

        Returns:
            API响应JSON或None
        """
        # 验证API配置
        validation_result = self.validate_api_config(api_config)
        if validation_result is not True:
            print(f"【API配置验证失败】{validation_result}")
            return None
        
        for attempt in range(retries + 1):
            try:
                api_provider = api_config.get("api_provider", "OpenAI")
                api_base = api_config.get("api_base", "").strip()
                api_key = api_config.get("api_key", "").strip()
                model_id = api_config.get("model_id", "gpt-3.5-turbo")
                
                # 如果API密钥为空，尝试从环境变量获取
                if not api_key and api_provider in ["Qwen", "Qwen-SG"]:
                    env_key = os.getenv("DASHSCOPE_API_KEY")
                    if env_key:
                        api_key = env_key
                        print("【API配置】从环境变量DASHSCOPE_API_KEY获取API密钥")

                url = f"{api_base}/chat/completions"
                headers = {"Content-Type": "application/json"}

                # 设置认证头
                if api_provider in ["OpenAI", "Anthropic", "Moonshot AI", "Qwen", "llms-py", "Kimi", "GLM", "MiniMax"]:
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"

                payload = {
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                }

                # 图像URL格式转换 - 适配不同API提供商
                if api_provider in ["Qwen", "OpenAI", "Anthropic", "Moonshot AI", "Kimi", "GLM", "MiniMax"]:
                    fixed_messages = []
                    for msg in messages:
                        if isinstance(msg.get("content"), list):
                            new_content = []
                            for item in msg["content"]:
                                if item.get("type") == "image_url":
                                    img_url = item.get("image_url", {}).get("url", "")
                                    if img_url.startswith("data:image/"):
                                        # 根据API提供商选择合适的格式
                                        if api_provider == "Qwen":
                                            # Qwen API使用base64://格式
                                            b64 = img_url.split(",")[-1]
                                            new_content.append({
                                                "type": "image_url",
                                                "image_url": {"url": f"base64://{b64}"}
                                            })
                                        else:
                                            # 其他API使用标准data:image格式
                                            new_content.append(item)
                                    else:
                                        new_content.append(item)
                                else:
                                    new_content.append(item)
                            fixed_messages.append({"role": msg["role"], "content": new_content})
                        else:
                            fixed_messages.append(msg)
                    payload["messages"] = fixed_messages

                # 估算请求大小
                request_size = len(json.dumps(payload).encode('utf-8'))
                print(f"【API请求】提供商: {api_provider}, 模型: {model_id}, 请求大小: {request_size/1024:.1f}KB")

                if request_size > self.MAX_REQUEST_SIZE:
                    print(f"【警告】请求体过大({request_size/1024:.1f}KB > {self.MAX_REQUEST_SIZE/1024:.1f}KB)，可能超过API限制")
                    print("【建议】请减小image_max_size参数（建议1024或更小），或减少输入图像数量")

                response = requests.post(url, headers=headers, json=payload, timeout=120)
                print(f"【API响应】状态码: {response.status_code}")

                # 处理特定错误
                if response.status_code == 400:
                    error_detail = response.text
                    print(f"【API 400错误】{error_detail}")
                    if "max bytes" in error_detail.lower() or "exceeded limit" in error_detail.lower():
                        print("【建议】请求体超过API限制，请：")
                        print("  1. 减小 image_max_size 参数（当前建议值：1024）")
                        print("  2. 减少输入图像数量")
                        print("  3. 使用更高压缩率（代码已自动优化）")
                    return None
                elif response.status_code == 401:
                    print("【API 401错误】API密钥无效或已过期")
                    return None
                elif response.status_code == 403:
                    print("【API 403错误】没有访问权限")
                    return None
                elif response.status_code == 429:
                    print("【API 429错误】请求过于频繁，请稍后重试")
                    if attempt < retries:
                        import time
                        backoff_time = 2 ** attempt  # 指数退避
                        print(f"【重试】等待 {backoff_time} 秒后重试...")
                        time.sleep(backoff_time)
                        continue
                    return None
                elif response.status_code >= 500:
                    print(f"【API服务器错误】状态码: {response.status_code}, 响应: {response.text}")
                    if attempt < retries:
                        import time
                        time.sleep(1)  # 短暂等待后重试
                        continue
                    return None

                # 尝试解析响应
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(f"【响应解析错误】无法解析JSON响应: {response.text}")
                    return None

            except requests.exceptions.Timeout as e:
                print(f"【API超时】第{attempt+1}次尝试超时: {str(e)}")
                if attempt < retries:
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                print(f"【网络连接错误】第{attempt+1}次尝试失败: {str(e)}")
                if attempt < retries:
                    continue
                return None
            except requests.exceptions.RequestException as e:
                print(f"【API请求错误】第{attempt+1}次尝试失败: {str(e)}")
                if attempt < retries:
                    continue
                return None
            except Exception as e:
                print(f"【API调用错误】{str(e)}")
                import traceback
                traceback.print_exc()
                return None

        return None

    def call_tts_api(self, api_config, text, voice="alloy", speed=1.0):
        """
        调用TTS API生成语音

        Args:
            api_config: API配置
            text: 要合成的文本
            voice: 音色
            speed: 语速

        Returns:
            音频数据字典或None
        """
        try:
            api_provider = api_config.get("api_provider", "OpenAI")
            api_base = api_config.get("api_base", "").strip()
            api_key = api_config.get("api_key", "").strip()

            if not api_base or not api_key:
                print("【TTS错误】API配置不完整")
                return None

            # OpenAI TTS格式
            url = f"{api_base}/audio/speech"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "speed": speed
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            # 返回音频数据
            audio_bytes = response.content
            waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            return {
                "waveform": waveform,
                "sample_rate": 24000
            }

        except Exception as e:
            print(f"【TTS调用错误】{str(e)}")
            return None

api_config_manager = APIConfigManager()

DEFAULT_API_CONFIGS = {k: {"api_base": v["api_base"], "max_tokens": v["max_tokens"], "temperature": v["temperature"]} for k, v in DEFAULT_PROVIDER_CONFIGS.items()}

class llama_cpp_api_model_config:
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        api_providers = list(PRESET_MODELS.keys())
        return {
            "required": {
                "model_name": ("STRING", {"default": "", "refresh_on_change": True}),
                "api_provider": (api_providers, {"default": api_providers[0] if api_providers else "", "refresh_on_change": True}),
                "api_base": ("STRING", {"default": "", "refresh_on_change": True}),
                "api_key": ("STRING", {"default": "", "refresh_on_change": True}),
                "max_tokens": ("INT", {"default": "", "min": 0, "max": 8192, "refresh_on_change": True}),
                "temperature": ("FLOAT", {"default": "", "min": 0, "max": 2, "refresh_on_change": True}),
            },
            "optional": {
                "preview_info": ("STRING", {"default": "", "multiline": True, "readonly": True, "refresh_on_change": True})
            }
        }
    RETURN_TYPES = ("API_CONFIG", "STRING")
    RETURN_NAMES = ("api_config", "preview")
    FUNCTION = "get_config"
    CATEGORY = "llama-cpp-vlm"

    def get_config(self, model_name="Qwen", api_provider="Qwen", api_base="", api_key="", max_tokens=1024, temperature=0.7, preview_info=""):
        cfg = DEFAULT_PROVIDER_CONFIGS.get(api_provider, DEFAULT_PROVIDER_CONFIGS["自定义"])

        api_base = (api_base or "").strip()
        api_key = (api_key or "").strip()
        model_name = (model_name or "").strip()

        if api_provider != "自定义":
            # provider 默认值 + 已修改值
            model_id = model_name or cfg["model_id"]
            api_base = api_base or cfg["api_base"]
            api_key = api_key or cfg.get("api_key", "")
            max_tokens = max_tokens if max_tokens not in (None, "") else cfg["max_tokens"]
            temperature = temperature if temperature not in (None, "") else cfg["temperature"]
        else:
            model_id = model_name or cfg["model_id"]
            api_base = api_base or cfg["api_base"]
            api_key = api_key or cfg.get("api_key", "")
            max_tokens = max_tokens if max_tokens not in (None, "") else cfg["max_tokens"]
            temperature = temperature if temperature not in (None, "") else cfg["temperature"]

        config = {
            "model_id": model_id,
            "api_provider": api_provider,
            "api_base": api_base,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        show_key = api_key[:4] + "***" + api_key[-4:] if len(api_key) > 8 else api_key
        preview = f"""===== API 配置 =====
模型：{model_id}
厂商：{api_provider}
地址：{api_base}
密钥：{show_key}
最大Token：{max_tokens}
温度：{temperature}
===================="""
        return (config, preview)

    def preview(self, model_name="Qwen", api_provider="Qwen", api_base="", api_key="", max_tokens=1024, temperature=0.7, preview_info=""):
        _, preview_text = self.get_config(model_name, api_provider, api_base, api_key, max_tokens, temperature, preview_info)
        return preview_text

NODE_CLASS_MAPPINGS = {"llama_cpp_api_model_config": llama_cpp_api_model_config}
NODE_DISPLAY_NAME_MAPPINGS = {"llama_cpp_api_model_config": "API配置管理器"}