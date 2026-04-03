# -*- coding: utf-8 -*-
"""
基础推理引擎模块

定义推理引擎的基类和通用功能，所有具体的推理引擎都继承自此基类。
提供图像处理、音频处理、消息构建等通用方法。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import scale_image, image2base64


def convert_audio_to_wav_bytes(audio: Dict) -> Optional[bytes]:
    """将音频转换为WAV字节流 - 优化版本"""
    try:
        import io
        import wave
        import numpy as np
        
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)
        
        if waveform is None:
            return None
        
        # 确保波形是CPU上的numpy数组
        if isinstance(waveform, torch.Tensor):
            if waveform.is_cuda:
                # 使用异步移除以提高性能
                waveform = waveform.cpu()
            # 直接转换为numpy数组，避免额外的内存复制
            waveform = waveform.numpy()
        
        # 确保波形是一维的
        if len(waveform.shape) > 1:
            # 对于多通道音频，取第一个通道
            waveform = waveform.squeeze()
        
        # 归一化到16位PCM范围，使用向量化操作提高速度
        waveform = waveform * 32767
        waveform = waveform.astype(np.int16)  # 直接使用np.int16提高性能
        
        # 创建WAV文件字节流
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(sample_rate)
            wf.writeframes(waveform.tobytes())
        
        buffer.seek(0)
        return buffer.read()
        
    except Exception as e:
        print(f"【音频转换错误】{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def stream_audio_processing(audio: Dict, chunk_size: int = 1024) -> Optional[bytes]:
    """流式音频处理 - 适用于大型音频文件"""
    try:
        import io
        import wave
        import numpy as np
        
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)
        
        if waveform is None:
            return None
        
        # 确保波形是CPU上的numpy数组
        if isinstance(waveform, torch.Tensor):
            if waveform.is_cuda:
                waveform = waveform.cpu()
            waveform = waveform.numpy()
        
        # 确保波形是一维的
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # 创建WAV文件字节流
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位
            wf.setframerate(sample_rate)
            
            # 流式处理音频数据
            total_samples = len(waveform)
            for i in range(0, total_samples, chunk_size):
                chunk = waveform[i:i+chunk_size]
                # 归一化到16位PCM范围
                chunk = chunk * 32767
                chunk = chunk.astype(np.int16)
                wf.writeframes(chunk.tobytes())
        
        buffer.seek(0)
        return buffer.read()
        
    except Exception as e:
        print(f"【流式音频处理错误】{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def convert_audio_to_format(audio: Dict, format: str = "wav") -> Optional[bytes]:
    """将音频转换为指定格式的字节流"""
    try:
        import io
        import numpy as np
        import soundfile as sf
        
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 16000)
        
        if waveform is None:
            return None
        
        # 确保波形是CPU上的numpy数组
        if isinstance(waveform, torch.Tensor):
            if waveform.is_cuda:
                waveform = waveform.cpu()
            waveform = waveform.numpy()
        
        # 确保波形是一维的
        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()
        
        # 创建字节流
        buffer = io.BytesIO()
        
        # 根据格式保存音频
        if format.lower() == "wav":
            import wave
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16位
                wf.setframerate(sample_rate)
                # 归一化到16位PCM范围
                waveform = waveform * 32767
                waveform = waveform.astype(np.int16)
                wf.writeframes(waveform.tobytes())
        else:
            # 使用soundfile库保存其他格式
            sf.write(buffer, waveform, sample_rate, format=format.lower())
        
        buffer.seek(0)
        return buffer.read()
        
    except Exception as e:
        print(f"【音频格式转换错误】{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_audio_data_uri(audio_bytes: bytes, format: str = "wav") -> Optional[str]:
    """创建音频数据URI"""
    try:
        import base64
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return f"data:audio/{format};base64,{audio_base64}"
        
    except Exception as e:
        print(f"【音频URI创建错误】{str(e)}")
        return None


class BaseInferenceEngine:
    """基础推理引擎 - 封装通用推理逻辑"""
    
    def __init__(self, model_info: Dict):
        self.model_info = model_info
        self.model_type = model_info.get("type", "vl")
        self.model_subtype = model_info.get("subtype", "default")
        self.supports_audio = model_info.get("supports_audio", False)
        self.supports_vision = model_info.get("supports_vision", True)
        
        # 添加图像处理缓存
        self._image_cache = {}
        self._cache_size_limit = 50
        
        # 添加音频处理缓存
        self._audio_cache = {}
        self._audio_cache_size_limit = 20
        
        # 添加内存管理
        self._memory_threshold = 0.85  # 内存使用阈值（85%）
        self._enable_memory_monitoring = True
        
    def get_generation_params(self, perf_level: str, video_input: bool = False, text_input: bool = False) -> Dict:
        """获取生成参数"""
        # 根据模型类型设置基础参数
        if self.model_subtype in ["qwen35", "qwen25_omni"]:
            # Qwen系列模型的优化参数
            base_params = {
                "min_p": 0.03,  # Qwen模型对min_p更敏感
                "typical_p": 1.0,
                "repeat_penalty": 1.05,  # Qwen模型需要稍高的重复惩罚
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 4.0,  # Qwen模型使用稍低的tau
            }
        elif self.model_subtype == "minicpm_o":
            # MiniCPM-O模型的优化参数
            base_params = {
                "min_p": 0.04,
                "typical_p": 1.0,
                "repeat_penalty": 1.02,
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 4.5,
            }
        else:
            # 默认参数
            base_params = {
                "min_p": 0.05,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0,
            }
        
        # 根据硬件性能调整
        perf_configs = {
            "high": {"max_tokens": 2048, "top_k": 40, "top_p": 0.92, "temperature": 0.85},
            "mid_high": {"max_tokens": 1536, "top_k": 35, "top_p": 0.9, "temperature": 0.8},
            "mid": {"max_tokens": 1024, "top_k": 30, "top_p": 0.88, "temperature": 0.75},
            "mid_low": {"max_tokens": 768, "top_k": 25, "top_p": 0.85, "temperature": 0.7},
            "low": {"max_tokens": 512, "top_k": 20, "top_p": 0.8, "temperature": 0.6},
        }
        
        base_params.update(perf_configs.get(perf_level, perf_configs["low"]))
        
        # 根据模型类型调整参数
        if self.model_subtype in ["qwen35", "qwen25_omni"]:
            # Qwen模型的特定优化
            base_params["temperature"] = max(base_params["temperature"] - 0.05, 0.6)  # Qwen模型对温度更敏感
            base_params["top_p"] = min(base_params["top_p"] + 0.02, 0.95)  # Qwen模型喜欢稍高的top_p
        elif self.model_subtype == "minicpm_o":
            # MiniCPM-O模型的特定优化
            base_params["temperature"] = max(base_params["temperature"] - 0.03, 0.65)  # MiniCPM-O模型需要稍低的温度
            base_params["top_k"] = min(base_params["top_k"] + 5, 45)  # MiniCPM-O模型喜欢稍高的top_k
        
        # 根据任务类型调整
        if video_input:
            base_params["max_tokens"] = min(base_params["max_tokens"] * 2, 2048)
            base_params["temperature"] = max(base_params["temperature"] - 0.1, 0.5)  # 视频任务需要更稳定的输出
        elif text_input:
            base_params["temperature"] = min(base_params["temperature"] + 0.1, 1.0)
            
        print(f"【模型特定优化】为{self.model_subtype}模型使用优化参数")
        return base_params
    
    def build_messages(self, system_prompt: str, user_content: Union[str, List], 
                       history: List[Dict] = None) -> List[Dict]:
        """构建消息列表"""
        messages = []
        
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def process_images_to_content(self, images: torch.Tensor, max_size: int, 
                                  preset_prompt: str = "") -> List[Dict]:
        """
        将图像转换为消息内容格式（优化版本，添加缓存和并行处理）
        """
        content = []
        
        if preset_prompt:
            content.append({"type": "text", "text": preset_prompt})
        
        # 处理多张图片
        if images is not None:
            if len(images.shape) == 3:  # 单张图片 [H, W, C]
                images = images.unsqueeze(0)
            
            # 根据模型类型调整图像处理策略
            # 对于视觉编码器，优先保证图像质量和特征提取效果
            if self.model_subtype in ["qwen35", "qwen25_omni", "minicpm_o"]:
                # 对于高级模型，使用更优化的图像处理
                content.extend(self._process_images_with_encoder_optimization(images, max_size))
            else:
                # 使用并行处理多张图片
                if images.shape[0] > 3:  # 如果图片数量大于3，使用并行处理
                    content.extend(self._process_images_parallel(images, max_size))
                else:
                    # 图片数量较少，使用串行处理
                    for i in range(images.shape[0]):
                        # 生成缓存键
                        cache_key = self._generate_image_cache_key(images[i], max_size)
                        
                        # 检查缓存
                        if cache_key in self._image_cache:
                            print(f"【图像缓存】使用缓存结果，索引: {i}")
                            content.append(self._image_cache[cache_key])
                            continue
                        
                        # 处理图像
                        img = images[i].cpu().numpy()
                        img_np = scale_image(img, max_size)
                        img_base64 = image2base64(img_np)
                        
                        image_content = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                        
                        # 缓存结果
                        self._cache_image_result(cache_key, image_content)
                        
                        content.append(image_content)
        
        return content
    
    def _process_images_with_encoder_optimization(self, images: torch.Tensor, max_size: int) -> List[Dict]:
        """
        针对视觉编码器优化的图像处理
        考虑视觉编码器的特性，调整图像处理策略
        """
        content = []
        
        # 根据模型类型调整图像处理参数
        if self.model_subtype in ["qwen35", "qwen25_omni"]:
            # Qwen系列模型的视觉编码器优化
            # 使用适当的图像大小，保证特征提取效果
            optimized_max_size = min(max_size, 1024)  # Qwen模型支持较大的图像
            quality = 85  # 提高图像质量以获得更好的特征
        elif self.model_subtype == "minicpm_o":
            # MiniCPM-O模型的视觉编码器优化
            optimized_max_size = min(max_size, 768)  # MiniCPM-O模型的最佳图像大小
            quality = 80
        else:
            # 默认优化
            optimized_max_size = max_size
            quality = 75
        
        print(f"【视觉编码器优化】使用优化参数: 最大尺寸={optimized_max_size}, 质量={quality}")
        
        # 处理多张图片
        for i in range(images.shape[0]):
            # 生成缓存键
            cache_key = self._generate_image_cache_key(images[i], optimized_max_size)
            
            # 检查缓存
            if cache_key in self._image_cache:
                print(f"【图像缓存】使用缓存结果，索引: {i}")
                content.append(self._image_cache[cache_key])
                continue
            
            # 处理图像
            img = images[i].cpu().numpy()
            img_np = scale_image(img, optimized_max_size)
            
            # 针对视觉编码器优化的图像编码
            # 提高图像质量以获得更好的特征提取效果
            import io
            import base64
            from PIL import Image as PILImage
            
            pil_img = PILImage.fromarray(img_np)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            }
            
            # 缓存结果
            self._cache_image_result(cache_key, image_content)
            
            content.append(image_content)
        
        return content
    
    def _process_images_parallel(self, images: torch.Tensor, max_size: int) -> List[Dict]:
        """
        并行处理多张图片
        """
        def process_single_image(img_tensor, idx):
            cache_key = self._generate_image_cache_key(img_tensor, max_size)
            
            # 检查缓存
            if cache_key in self._image_cache:
                print(f"【图像缓存并行】使用缓存结果，索引: {idx}")
                return self._image_cache[cache_key]
            
            # 处理图像
            img = img_tensor.cpu().numpy()
            img_np = scale_image(img, max_size)
            img_base64 = image2base64(img_np)
            
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            }
            
            # 缓存结果
            self._cache_image_result(cache_key, image_content)
            
            return image_content
        
        # 使用线程池并行处理
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_single_image, images[i], i): i
                for i in range(images.shape[0])
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    print(f"【图像并行处理错误】索引 {idx}: {str(e)}")
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        return [result for idx, result in results]
    
    def _generate_image_cache_key(self, image_tensor: torch.Tensor, max_size: int) -> str:
        """
        生成图像缓存键
        """
        import hashlib
        
        try:
            # 使用图像数据的哈希作为缓存键
            img_bytes = image_tensor.cpu().numpy().tobytes()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            cache_key = f"{img_hash}_{max_size}"
            
            return cache_key
        except Exception as e:
            print(f"【图像缓存错误】生成缓存键失败: {str(e)}")
            return None
    
    def _cache_image_result(self, cache_key: str, result: Dict):
        """
        缓存图像处理结果
        """
        if cache_key is None:
            return
        
        try:
            # 检查内存使用情况（图像处理任务）
            if self._enable_memory_monitoring and self._check_memory_usage(task_type="images"):
                print("【内存管理】内存使用率过高，清理缓存")
                self._clear_cache()
            
            # 如果缓存已满，删除最旧的条目
            if len(self._image_cache) >= self._cache_size_limit:
                oldest_key = next(iter(self._image_cache))
                del self._image_cache[oldest_key]
            
            self._image_cache[cache_key] = result
            
        except Exception as e:
            print(f"【图像缓存错误】缓存结果失败: {str(e)}")
    
    def _check_memory_usage(self, task_type: str = "text") -> bool:
        """
        检查内存使用情况 - 增强版
        
        Args:
            task_type: 任务类型 (text, images, audio, video, multimodal)
        """
        try:
            # 根据模型类型和任务类型动态调整内存阈值
            if self.model_subtype in ["qwen35", "qwen25_omni"]:
                # Qwen模型需要更多内存
                adjusted_threshold = min(self._memory_threshold + 0.05, 0.9)
            elif self.model_subtype == "minicpm_o":
                # MiniCPM-O模型内存需求适中
                adjusted_threshold = self._memory_threshold
            else:
                # 默认阈值
                adjusted_threshold = self._memory_threshold
            
            # 根据任务类型调整阈值
            task_memory_demands = {
                "text": 0.0,  # 文本任务内存需求最低
                "images": 0.1,  # 图像处理需要更多内存
                "audio": 0.05,  # 音频处理内存需求适中
                "video": 0.15,  # 视频处理内存需求最高
                "multimodal": 0.12  # 多模态整合内存需求较高
            }
            
            task_demand = task_memory_demands.get(task_type, 0.0)
            final_threshold = max(adjusted_threshold - task_demand, 0.7)  # 确保阈值不低于70%
            
            memory_status = {}
            
            if torch.cuda.is_available():
                # 检查GPU内存
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated()
                    gpu_max_memory = torch.cuda.max_memory_allocated()
                    if gpu_max_memory > 0:
                        gpu_memory_used = gpu_memory_allocated / gpu_max_memory
                        memory_status["gpu"] = {
                            "used": gpu_memory_used,
                            "allocated": gpu_memory_allocated / 1024 / 1024 / 1024,  # GB
                            "max": gpu_max_memory / 1024 / 1024 / 1024  # GB
                        }
                        if gpu_memory_used > final_threshold:
                            print(f"【内存监控】GPU内存使用率: {gpu_memory_used:.2%}, 阈值: {final_threshold:.2%}, 任务: {task_type}")
                            print(f"【内存监控】GPU内存使用: {memory_status['gpu']['allocated']:.2f}GB / {memory_status['gpu']['max']:.2f}GB")
                            return True
                except Exception as gpu_error:
                    print(f"【内存监控】GPU内存检查失败: {str(gpu_error)}")
                    import traceback
                    traceback.print_exc()
            
            # 检查系统内存
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_percent = memory.percent / 100.0
                memory_status["system"] = {
                    "used": memory_percent,
                    "total": memory.total / 1024 / 1024 / 1024,  # GB
                    "available": memory.available / 1024 / 1024 / 1024  # GB
                }
                
                if memory_percent > final_threshold:
                    print(f"【内存监控】系统内存使用率: {memory_percent:.2%}, 阈值: {final_threshold:.2%}, 任务: {task_type}")
                    print(f"【内存监控】系统内存使用: {memory_status['system']['total'] - memory_status['system']['available']:.2f}GB / {memory_status['system']['total']:.2f}GB")
                    return True
            except ImportError:
                print("【内存监控】未安装psutil库，跳过系统内存监控")
            except Exception as system_error:
                print(f"【内存监控】系统内存检查失败: {str(system_error)}")
                import traceback
                traceback.print_exc()
            
            # 打印内存状态信息（调试用）
            if memory_status:
                print(f"【内存监控】当前状态: {memory_status}")
            
            return False
            
        except Exception as e:
            print(f"【内存监控错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_audio_cache_key(self, audio: Dict, model_subtype: str) -> str:
        """
        生成音频缓存键 - 优化版本
        """
        import hashlib
        
        try:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate", 16000)
            
            if waveform is None:
                return None
            
            # 优化：对于大型音频文件，使用采样哈希以减少计算时间
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.cpu().numpy()
            else:
                waveform_np = waveform
            
            # 对于大型音频文件，只取部分数据进行哈希计算
            if len(waveform_np) > 100000:
                # 采样计算哈希，减少计算时间
                step = len(waveform_np) // 1000
                sampled_data = waveform_np[::step]
                audio_bytes = sampled_data.tobytes()
            else:
                audio_bytes = waveform_np.tobytes()
            
            # 使用SHA-256获得更可靠的哈希值
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()
            cache_key = f"audio_{audio_hash}_{sample_rate}_{model_subtype}"
            
            return cache_key
        except Exception as e:
            print(f"【音频缓存错误】生成缓存键失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _cache_audio_result(self, cache_key: str, result: Dict):
        """
        缓存音频处理结果 - 智能缓存管理
        """
        if cache_key is None:
            return
        
        try:
            # 检查内存使用情况（音频处理任务）
            if self._enable_memory_monitoring and self._check_memory_usage(task_type="audio"):
                print("【内存管理】内存使用率过高，清理缓存")
                self._clear_cache()
            
            # 动态调整缓存大小基于可用内存
            import psutil
            import os
            
            # 获取当前进程的内存使用情况
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_used = memory_info.rss / 1024 / 1024  # 转换为MB
            
            # 基于可用内存动态调整缓存大小
            adjusted_cache_limit = self._audio_cache_size_limit
            if memory_used > 8000:  # 如果内存使用超过8GB
                adjusted_cache_limit = max(10, int(self._audio_cache_size_limit * 0.4))
            elif memory_used > 4000:  # 如果内存使用超过4GB
                adjusted_cache_limit = max(20, int(self._audio_cache_size_limit * 0.7))
            
            # 如果缓存已满，删除最旧的条目
            if len(self._audio_cache) >= adjusted_cache_limit:
                # 实现LRU缓存清理策略
                # 注意：这里简化实现，实际应该记录访问时间
                # 为了兼容性，我们仍然使用删除最早条目的策略
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]
                print(f"【音频缓存】清理缓存，当前缓存大小: {len(self._audio_cache)}, 调整后缓存限制: {adjusted_cache_limit}")
            
            # 缓存结果
            self._audio_cache[cache_key] = result
            print(f"【音频缓存】缓存结果，当前缓存大小: {len(self._audio_cache)}")
            
        except Exception as e:
            print(f"【音频缓存错误】缓存结果失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _clear_cache(self):
        """
        清理缓存
        """
        try:
            # 清理图像缓存
            image_cache_size = len(self._image_cache)
            self._image_cache.clear()
            print(f"【内存管理】已清理 {image_cache_size} 个图像缓存条目")
            
            # 清理音频缓存
            audio_cache_size = len(self._audio_cache)
            self._audio_cache.clear()
            print(f"【内存管理】已清理 {audio_cache_size} 个音频缓存条目")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("【内存管理】已清理GPU缓存")
            
        except Exception as e:
            print(f"【内存管理错误】{str(e)}")
    
    def cleanup(self):
        """
        清理资源
        """
        try:
            print("【资源清理】开始清理推理引擎资源")
            
            # 清理缓存
            self._clear_cache()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("【资源清理】已清理GPU内存")
            
            print("【资源清理】推理引擎资源清理完成")
            
        except Exception as e:
            print(f"【资源清理错误】{str(e)}")
    
    def process_audio_to_content(self, audio: Dict, model_subtype: str = "default", audio_format: str = "wav") -> Optional[Dict]:
        """将音频转换为消息内容格式 - 支持多种音频格式"""
        if audio is None:
            return None
        
        try:
            # 生成缓存键，包含音频格式
            cache_key = self._generate_audio_cache_key(audio, f"{model_subtype}_{audio_format}")
            
            # 检查缓存
            if cache_key and cache_key in self._audio_cache:
                print("【音频缓存】使用缓存结果")
                return self._audio_cache[cache_key]
            
            # 检查内存使用情况，决定使用哪种处理方式
            use_streaming = False
            waveform = audio.get("waveform")
            if waveform is not None:
                # 估算音频数据大小
                if isinstance(waveform, torch.Tensor):
                    num_samples = waveform.numel()
                else:
                    num_samples = len(waveform)
                
                # 如果音频样本数超过100万，使用流式处理
                if num_samples > 1000000:
                    print("【音频处理】使用流式处理大型音频文件")
                    use_streaming = True
            
            # 根据音频大小和格式选择处理方式
            if use_streaming:
                # 流式处理只支持WAV格式
                audio_bytes = stream_audio_processing(audio)
                current_format = "wav"
            else:
                # 使用新的格式转换函数
                audio_bytes = convert_audio_to_format(audio, audio_format)
                current_format = audio_format
            
            if audio_bytes is None:
                return None
            
            # 创建Data URI
            audio_uri = create_audio_data_uri(audio_bytes, current_format)
            if audio_uri is None:
                return None
            
            # 根据模型类型返回不同格式
            if model_subtype in ["qwen35", "qwen25_omni"]:
                # llama-cpp 中的 Qwen35/Qwen25VLChatHandler 在本环境下不支持 audio_url item，
                # 否则会抛出 "Unexpected item type in content."。
                print("【音频处理】Qwen Omni 模型当前不支持直接原生音频输入，已忽略音频内容。")
                return None
            elif model_subtype == "minicpm_o":
                audio_content = {
                    "type": "audio",
                    "audio": audio_uri
                }
            else:
                # 默认格式
                audio_content = {
                    "type": "audio_url",
                    "audio_url": {"url": audio_uri}
                }
            
            # 缓存结果
            if cache_key:
                self._cache_audio_result(cache_key, audio_content)
                
            return audio_content
                
        except Exception as e:
            print(f"【音频处理错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_chat_completion(self, llm, messages: List[Dict], params: Dict) -> Dict:
        """创建聊天完成请求"""
        try:
            # 为不同模型类型添加特定的优化策略
            if self.model_subtype in ["qwen35", "qwen25_omni"]:
                # Qwen模型的特殊处理
                # 确保使用正确的停止词
                stop_words = params.get("stop", ["</s>", "<|im_end|>"])
                
                # 构建参数，确保seed参数正确传递
                completion_params = {
                    "messages": messages,
                    "max_tokens": params.get("max_tokens", 512),
                    "temperature": params.get("temperature", 0.7),
                    "top_k": params.get("top_k", 40),
                    "top_p": params.get("top_p", 0.9),
                    "min_p": params.get("min_p", 0.03),
                    "repeat_penalty": params.get("repeat_penalty", 1.05),
                    "frequency_penalty": params.get("frequency_penalty", 0.0),
                    "stream": False,
                    "stop": stop_words
                }
                
                # 添加seed参数，兼容llama-cpp-python 0.3.32
                if "seed" in params:
                    completion_params["seed"] = params["seed"]
                
                print(f"【Qwen-Omni API】调用参数: max_tokens={completion_params['max_tokens']}, temperature={completion_params['temperature']}")
                output = llm.create_chat_completion(**completion_params)
                
            elif self.model_subtype in ["minicpm_o", "minicpm_o_26", "minicpm_o_45"]:
                # MiniCPM-O模型的特殊处理
                stop_words = params.get("stop", ["</s>", "[END]", "<|end|>"])
                
                # 构建参数，确保seed参数正确传递
                completion_params = {
                    "messages": messages,
                    "max_tokens": params.get("max_tokens", 512),
                    "temperature": params.get("temperature", 0.7),
                    "top_k": params.get("top_k", 40),
                    "top_p": params.get("top_p", 0.9),
                    "min_p": params.get("min_p", 0.04),
                    "repeat_penalty": params.get("repeat_penalty", 1.02),
                    "frequency_penalty": params.get("frequency_penalty", 0.0),
                    "stream": False,
                    "stop": stop_words
                }
                
                # 添加seed参数，兼容llama-cpp-python 0.3.32
                if "seed" in params:
                    completion_params["seed"] = params["seed"]
                
                # MiniCPM-O-4.5 特殊优化
                if self.model_subtype == "minicpm_o_45":
                    # 优化参数以获得更好的音频生成效果
                    completion_params["temperature"] = max(completion_params["temperature"], 0.6)
                    completion_params["top_p"] = min(completion_params["top_p"], 0.95)
                    completion_params["repeat_penalty"] = 1.03
                    print(f"【MiniCPM-O-4.5 API】优化参数: max_tokens={completion_params['max_tokens']}, temperature={completion_params['temperature']}")
                else:
                    print(f"【MiniCPM-O API】调用参数: max_tokens={completion_params['max_tokens']}, temperature={completion_params['temperature']}")
                
                output = llm.create_chat_completion(**completion_params)
                
            else:
                # 默认处理
                # 构建参数，确保seed参数正确传递
                completion_params = {
                    "messages": messages,
                    "max_tokens": params.get("max_tokens", 512),
                    "temperature": params.get("temperature", 0.7),
                    "top_k": params.get("top_k", 40),
                    "top_p": params.get("top_p", 0.9),
                    "min_p": params.get("min_p", 0.05),
                    "repeat_penalty": params.get("repeat_penalty", 1.0),
                    "frequency_penalty": params.get("frequency_penalty", 0.0),
                    "stream": False,
                    "stop": params.get("stop", ["</s>"])
                }
                
                # 添加seed参数，兼容llama-cpp-python 0.3.32
                if "seed" in params:
                    completion_params["seed"] = params["seed"]
                
                print(f"【默认API】调用参数: max_tokens={completion_params['max_tokens']}, temperature={completion_params['temperature']}")
                output = llm.create_chat_completion(**completion_params)
            
            # 验证输出格式
            if output and 'choices' in output and len(output['choices']) > 0:
                print(f"【API成功】返回内容长度: {len(output['choices'][0]['message']['content'])}")
            else:
                print(f"【API警告】输出格式异常: {output}")
                
            return output
            
        except Exception as e:
            print(f"【API调用错误】模型类型: {self.model_subtype}, 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
