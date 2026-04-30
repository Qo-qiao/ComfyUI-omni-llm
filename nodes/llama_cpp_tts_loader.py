# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm TTS模型加载器节点

支持以下功能：
- 主流TTS模型：Qwen3-TTS、KaniTTS
- 其他TTS模型：Bark、VITS、XTTS、Coqui、Supertonic等

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import json
import torch
import numpy as np
import sys
import requests
import shutil
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import HARDWARE_INFO, folder_paths


class BaseTTSEngine:
    """
    基础TTS引擎类
    所有TTS引擎的基类，定义了通用的接口和方法
    """
    
    def __init__(self, model_path, config, device="cpu"):
        """
        初始化TTS引擎
        
        Args:
            model_path: 模型路径
            config: 配置参数
            device: 运行设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.config = config or {}
        self.device = device
        self.model = None
        self.processor = None
        self.model_type = "base"
    
    def load_model(self):
        """
        加载模型
        子类需要实现此方法
        """
        raise NotImplementedError("Subclass must implement load_model method")
    
    def generate(self, text, **kwargs):
        """
        生成语音
        子类需要实现此方法
        
        Args:
            text: 要转换的文本
            **kwargs: 额外的生成参数
            
        Returns:
            生成的音频数据和采样率
        """
        raise NotImplementedError("Subclass must implement generate method")
    
    def get_speaker_ids(self):
        """
        获取可用的说话人ID列表
        子类需要实现此方法
        
        Returns:
            说话人ID列表
        """
        raise NotImplementedError("Subclass must implement get_speaker_ids method")
    
    def get_languages(self):
        """
        获取支持的语言列表
        子类需要实现此方法
        
        Returns:
            语言列表
        """
        raise NotImplementedError("Subclass must implement get_languages method")
    
    def get_emotions(self):
        """
        获取支持的情绪列表
        子类需要实现此方法
        
        Returns:
            情绪列表
        """
        raise NotImplementedError("Subclass must implement get_emotions method")
    
    def release(self):
        """
        释放模型资源
        子类需要实现此方法
        """
        raise NotImplementedError("Subclass must implement release method")


class QwenTTSEngine(BaseTTSEngine):
    """
    Qwen TTS引擎
    支持Qwen3-TTS模型的加载和推理
    """
    
    def __init__(self, model_path, config, device="cpu"):
        super().__init__(model_path, config, device)
        self.model_type = "qwen_tts"
        self.qwen_tts_variant = "CustomVoice"
        self.qwen_tts_sampling_rate = 24000
        self.user_selected_model_file = None
        self.load_model()
    
    def load_model(self):
        """
        加载Qwen3-TTS模型
        
        支持三种加载方式:
        1. qwen-tts 包方式 (推荐，如果已安装)
        2. 标准 transformers 方式
        3. ModelScope pipeline 方式 (CustomVoice/VoiceDesign 变体)
        """
        try:
            print(f"【Qwen3-TTS】正在加载模型...")
            print(f"【Qwen3-TTS】原始模型路径: {self.model_path}")

            # 检查模型路径是否存在
            if not os.path.exists(self.model_path):
                print(f"【Qwen3-TTS错误】模型路径不存在: {self.model_path}")
                self.model = None
                return

            # 保存原始路径
            original_path = self.model_path
            
            # 检查是否是文件路径
            is_model_file = os.path.isfile(original_path)
            
            # 保存用户选择的模型文件名
            user_selected_model_file = None
            
            # 如果是文件路径，获取文件夹路径和文件名
            if is_model_file:
                self.model_path = os.path.dirname(original_path)
                user_selected_model_file = os.path.basename(original_path)
                print(f"【Qwen3-TTS】检测到文件路径，已转换为文件夹路径: {self.model_path}")
                print(f"【Qwen3-TTS】原文件: {original_path}")
                print(f"【Qwen3-TTS】用户选择的模型文件: {user_selected_model_file}")
            else:
                # 直接使用原始路径作为文件夹路径
                self.model_path = original_path
                print(f"【Qwen3-TTS】使用文件夹路径: {self.model_path}")
            
            # 检查模型文件夹中的 .safetensors 文件
            safetensors_files = [f for f in os.listdir(self.model_path) if f.endswith('.safetensors')]
            print(f"【Qwen3-TTS】检测到 .safetensors 文件: {safetensors_files}")
            
            # 检查是否存在 model.safetensors 文件
            if 'model.safetensors' not in safetensors_files and safetensors_files:
                print(f"【Qwen3-TTS】未找到 model.safetensors 文件")
                if user_selected_model_file:
                    print(f"【Qwen3-TTS】用户选择了模型文件: {user_selected_model_file}")
                    # 保存用户选择的模型文件名，用于后续加载
                    self.user_selected_model_file = user_selected_model_file
                else:
                    print(f"【Qwen3-TTS】使用第一个找到的 .safetensors 文件")
                    self.user_selected_model_file = safetensors_files[0]
            else:
                self.user_selected_model_file = None

            # 检测模型变体
            path_lower = self.model_path.lower()
            original_path_lower = original_path.lower()
            combined_path = f"{path_lower} {original_path_lower}"
            
            is_customvoice = "customvoice" in combined_path or "custom_voice" in combined_path
            is_voicedesign = "voicedesign" in combined_path or "voice_design" in combined_path
            is_12hz = "12hz" in combined_path
            is_8bit = "8bit" in combined_path
            
            print(f"【Qwen3-TTS】变体检测: CustomVoice={is_customvoice}, VoiceDesign={is_voicedesign}, 12Hz={is_12hz}, 8bit={is_8bit}")

            # 设置采样率
            self.qwen_tts_sampling_rate = 12000 if is_12hz else 24000
            if is_customvoice:
                self.qwen_tts_variant = "CustomVoice"
            elif is_voicedesign:
                self.qwen_tts_variant = "VoiceDesign"
            else:
                self.qwen_tts_variant = "CustomVoice"
            
            print(f"【Qwen3-TTS】使用变体: {self.qwen_tts_variant}, 采样率: {self.qwen_tts_sampling_rate}Hz, 8bit量化: {is_8bit}")
            
            # 获取8bit量化配置
            # 注意：对于已经是8bit量化的模型，不强制使用8bit量化，避免配置冲突
            use_8bit_quant = self.config.get("use_8bit_quant", False) and not is_8bit
            print(f"【Qwen3-TTS】最终8bit量化设置: {use_8bit_quant}")

            # 方式1: 尝试使用 qwen-tts 包加载 (推荐方式)
            try:
                # 尝试导入 qwen-tts 包 - 使用正确的导入路径
                try:
                    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                    print(f"【Qwen3-TTS】qwen-tts 包导入成功")
                except ImportError as e:
                    print(f"【Qwen3-TTS】qwen-tts 包未安装或导入失败: {str(e)}")
                    print(f"【Qwen3-TTS】尝试其他方式...")
                    # 继续尝试其他方式
                else:
                    print(f"【Qwen3-TTS】尝试使用 qwen-tts 包加载...")
                    
                    # 使用 qwen-tts 的加载方式
                    torch_dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
                    
                    # 构建加载参数
                    qwen_kwargs = {
                        "device_map": "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu",
                        "dtype": torch_dtype,
                        "low_cpu_mem_usage": True,
                        "trust_remote_code": True
                    }

                    # 注意：对于 8bit 量化模型，qwen-tts 包可能需要特殊处理
                    # 建议禁用量化支持，让 qwen-tts 包自动处理
                    if is_8bit:
                        print(f"【Qwen3-TTS】检测到8bit量化模型，禁用量化参数让qwen-tts自动处理")
                        # 不添加 load_in_8bit 参数，让 qwen-tts 根据模型自动判断
                    elif use_8bit_quant:
                        # 仅在明确启用量化且模型本身不是8bit时才添加参数
                        print(f"【Qwen3-TTS】尝试使用8bit量化加载（非量化模型）")
                        qwen_kwargs["load_in_8bit"] = True
                    
                    try:
                        # 检查用户是否选择了特定的模型文件
                        load_path = self.model_path
                        if hasattr(self, 'user_selected_model_file') and self.user_selected_model_file:
                            print(f"【Qwen3-TTS】用户选择了模型文件: {self.user_selected_model_file}")
                            # 检查是否存在 model.safetensors 文件
                            model_file_path = os.path.join(self.model_path, "model.safetensors")
                            if not os.path.exists(model_file_path):
                                # 尝试修改 model.safetensors.index.json 文件
                                index_file_path = os.path.join(self.model_path, "model.safetensors.index.json")
                                if os.path.exists(index_file_path):
                                    try:
                                        import json
                                        with open(index_file_path, 'r', encoding='utf-8') as f:
                                            index_data = json.load(f)
                                        # 修改 weight_map 中的所有文件名
                                        for key in index_data.get('weight_map', {}):
                                            index_data['weight_map'][key] = self.user_selected_model_file
                                        # 保存修改后的 index 文件到临时目录
                                        temp_dir = tempfile.mkdtemp()
                                        temp_index_path = os.path.join(temp_dir, "model.safetensors.index.json")
                                        with open(temp_index_path, 'w', encoding='utf-8') as f:
                                            json.dump(index_data, f, indent=2, ensure_ascii=False)
                                        print(f"【Qwen3-TTS】已创建临时 index 文件: {temp_index_path}")
                                        # 复制用户选择的模型文件到临时目录
                                        temp_model_path = os.path.join(temp_dir, "model.safetensors")
                                        shutil.copy(os.path.join(self.model_path, self.user_selected_model_file), temp_model_path)
                                        print(f"【Qwen3-TTS】已复制模型文件到临时目录")
                                        # 复制其他必要的配置文件到临时目录
                                        for config_file in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt']:
                                            src_file = os.path.join(self.model_path, config_file)
                                            if os.path.exists(src_file):
                                                shutil.copy(src_file, os.path.join(temp_dir, config_file))
                                        # 复制 speech_tokenizer 目录
                                        speech_tokenizer_dir = os.path.join(self.model_path, 'speech_tokenizer')
                                        if os.path.exists(speech_tokenizer_dir) and os.path.isdir(speech_tokenizer_dir):
                                            temp_speech_tokenizer_dir = os.path.join(temp_dir, 'speech_tokenizer')
                                            shutil.copytree(speech_tokenizer_dir, temp_speech_tokenizer_dir)
                                            print(f"【Qwen3-TTS】已复制 speech_tokenizer 目录到临时目录")
                                        # 使用临时目录加载模型
                                        load_path = temp_dir
                                        print(f"【Qwen3-TTS】使用临时目录加载模型: {load_path}")
                                    except Exception as e:
                                        print(f"【Qwen3-TTS】处理用户选择的模型文件失败: {str(e)}")
                                        raise Exception(f"Failed to process user selected model file: {str(e)}")
                                else:
                                    print(f"【Qwen3-TTS】没有找到索引文件，但用户选择了模型文件，直接创建临时目录")
                                    try:
                                        temp_dir = tempfile.mkdtemp()
                                        # 复制用户选择的模型文件到临时目录并重命名为model.safetensors
                                        temp_model_path = os.path.join(temp_dir, "model.safetensors")
                                        shutil.copy(os.path.join(self.model_path, self.user_selected_model_file), temp_model_path)
                                        print(f"【Qwen3-TTS】已复制并重命名模型文件到临时目录")
                                        # 复制其他必要的配置文件到临时目录
                                        for config_file in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt']:
                                            src_file = os.path.join(self.model_path, config_file)
                                            if os.path.exists(src_file):
                                                shutil.copy(src_file, os.path.join(temp_dir, config_file))
                                        # 复制 speech_tokenizer 目录
                                        speech_tokenizer_dir = os.path.join(self.model_path, 'speech_tokenizer')
                                        if os.path.exists(speech_tokenizer_dir) and os.path.isdir(speech_tokenizer_dir):
                                            temp_speech_tokenizer_dir = os.path.join(temp_dir, 'speech_tokenizer')
                                            shutil.copytree(speech_tokenizer_dir, temp_speech_tokenizer_dir)
                                            print(f"【Qwen3-TTS】已复制 speech_tokenizer 目录到临时目录")
                                        # 使用临时目录加载模型
                                        load_path = temp_dir
                                        print(f"【Qwen3-TTS】使用临时目录加载模型: {load_path}")
                                    except Exception as e:
                                        print(f"【Qwen3-TTS】创建临时目录失败: {str(e)}")
                                        raise Exception(f"Failed to create temporary directory: {str(e)}")
                        
                        # 始终使用文件夹路径加载模型
                        self.model = Qwen3TTSModel.from_pretrained(
                            load_path,
                            **qwen_kwargs
                        )
                        self.processor = None  # qwen-tts 包内部处理
                        print(f"【Qwen3-TTS】qwen-tts 包加载成功")
                        return
                    except ValueError as load_error:
                        if "quant_method" in str(load_error):
                            print(f"【Qwen3-TTS】检测到量化配置缺少'quant_method'属性，尝试修复...")
                            # 检查是否存在量化配置文件
                            import json
                            
                            quant_config_path = os.path.join(self.model_path, "quantization_config.json")
                            config_path = os.path.join(self.model_path, "config.json")
                            fixed = False
                            
                            # 尝试修复 quantization_config.json
                            if os.path.exists(quant_config_path):
                                try:
                                    print(f"【Qwen3-TTS】发现quantization_config.json，尝试修复...")
                                    with open(quant_config_path, 'r', encoding='utf-8') as f:
                                        config = json.load(f)
                                    if 'quant_method' not in config:
                                        config['quant_method'] = 'bitsandbytes'
                                        with open(quant_config_path, 'w', encoding='utf-8') as f:
                                            json.dump(config, f, indent=2, ensure_ascii=False)
                                        print(f"【Qwen3-TTS】已修复quantization_config.json文件")
                                        fixed = True
                                except Exception as fix_err:
                                    print(f"【Qwen3-TTS】修复quantization_config.json失败: {fix_err}")
                            
                            # 尝试修复 config.json 中的 quantization_config
                            if not fixed and os.path.exists(config_path):
                                try:
                                    print(f"【Qwen3-TTS】尝试修复config.json中的quantization_config...")
                                    with open(config_path, 'r', encoding='utf-8') as f:
                                        config = json.load(f)
                                    if 'quantization_config' in config and isinstance(config['quantization_config'], dict):
                                        if 'quant_method' not in config['quantization_config']:
                                            config['quantization_config']['quant_method'] = 'bitsandbytes'
                                            with open(config_path, 'w', encoding='utf-8') as f:
                                                json.dump(config, f, indent=2, ensure_ascii=False)
                                            print(f"【Qwen3-TTS】已修复config.json中的quantization_config")
                                            fixed = True
                                except Exception as fix_err:
                                    print(f"【Qwen3-TTS】修复config.json失败: {fix_err}")
                            
                            # 如果修复了配置文件，尝试重新加载
                            if fixed:
                                try:
                                    # 始终使用文件夹路径加载模型
                                    self.model = Qwen3TTSModel.from_pretrained(
                                        self.model_path,
                                        **qwen_kwargs
                                    )
                                    self.processor = None
                                    print(f"【Qwen3-TTS】qwen-tts 包加载成功")
                                    return
                                except Exception as e:
                                    print(f"【Qwen3-TTS】修复后仍然失败: {str(e)}")
                            
                            # 如果修复失败或没有配置文件，尝试不使用任何量化参数
                            print(f"【Qwen3-TTS】尝试不使用量化参数加载...")
                            # 移除所有可能的量化相关参数
                            clean_kwargs = {k: v for k, v in qwen_kwargs.items() if not k.startswith('load_in_') and not k == 'quantization_config'}
                            try:
                                # 始终使用文件夹路径加载模型
                                self.model = Qwen3TTSModel.from_pretrained(
                                    self.model_path,
                                    **clean_kwargs
                                )
                                self.processor = None
                                print(f"【Qwen3-TTS】qwen-tts 包（非量化）加载成功")
                                return
                            except Exception as e:
                                print(f"【Qwen3-TTS】再次失败: {str(e)}")
                        else:
                            print(f"【Qwen3-TTS】qwen-tts 包加载失败: {str(load_error)}")
                            import traceback
                            traceback.print_exc()
                            # 如果是8bit量化导致的失败，尝试不使用8bit量化
                            if use_8bit_quant:
                                print(f"【Qwen3-TTS】8bit量化加载失败，尝试不使用8bit量化")
                                qwen_kwargs.pop("load_in_8bit", None)
                                try:
                                    # 始终使用文件夹路径加载模型
                                    self.model = Qwen3TTSModel.from_pretrained(
                                        self.model_path,
                                        **qwen_kwargs
                                    )
                                    self.processor = None
                                    print(f"【Qwen3-TTS】qwen-tts 包（非8bit）加载成功")
                                    return
                                except Exception as e:
                                    print(f"【Qwen3-TTS】再次失败: {str(e)}")
                        self.model = None
                        print(f"【Qwen3-TTS】尝试其他方式...")

            except Exception as qwen_error:
                if "cannot pickle" in str(qwen_error) and "dict_keys" in str(qwen_error):
                    print(f"【Qwen3-TTS】qwen-tts 包存在已知pickle错误（dict_keys不可序列化）")
                    print(f"【Qwen3-TTS提示】这是qwen-tts包的已知问题，建议使用transformers方式加载")
                else:
                    print(f"【Qwen3-TTS】qwen-tts 包处理失败: {str(qwen_error)}")
                self.model = None

            # 方式2: 尝试使用 transformers 加载 (标准方式)
            try:
                print(f"【Qwen3-TTS】尝试使用 transformers 加载...")
                
                # 动态导入
                from transformers import AutoModel, AutoProcessor
                print(f"【Qwen3-TTS】transformers 导入成功")
                
                # 检查用户是否选择了特定的模型文件，可能需要使用临时目录
                processor_load_path = self.model_path
                model_load_path = self.model_path
                
                # 检查是否存在 model.safetensors 文件
                model_file_path = os.path.join(self.model_path, "model.safetensors")
                if not os.path.exists(model_file_path):
                    # 检查用户是否选择了特定的模型文件
                    if hasattr(self, 'user_selected_model_file') and self.user_selected_model_file:
                        print(f"【Qwen3-TTS】用户选择了模型文件: {self.user_selected_model_file}")
                        # 尝试修改 model.safetensors.index.json 文件
                        index_file_path = os.path.join(self.model_path, "model.safetensors.index.json")
                        if os.path.exists(index_file_path):
                            try:
                                import json
                                with open(index_file_path, 'r', encoding='utf-8') as f:
                                    index_data = json.load(f)
                                # 修改 weight_map 中的所有文件名
                                for key in index_data.get('weight_map', {}):
                                    index_data['weight_map'][key] = self.user_selected_model_file
                                # 保存修改后的 index 文件到临时目录
                                temp_dir = tempfile.mkdtemp()
                                temp_index_path = os.path.join(temp_dir, "model.safetensors.index.json")
                                with open(temp_index_path, 'w', encoding='utf-8') as f:
                                    json.dump(index_data, f, indent=2, ensure_ascii=False)
                                print(f"【Qwen3-TTS】已创建临时 index 文件: {temp_index_path}")
                                # 复制用户选择的模型文件到临时目录
                                temp_model_path = os.path.join(temp_dir, "model.safetensors")
                                shutil.copy(os.path.join(self.model_path, self.user_selected_model_file), temp_model_path)
                                print(f"【Qwen3-TTS】已复制模型文件到临时目录")
                                # 复制其他必要的配置文件到临时目录
                                for config_file in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt']:
                                    src_file = os.path.join(self.model_path, config_file)
                                    if os.path.exists(src_file):
                                        shutil.copy(src_file, os.path.join(temp_dir, config_file))
                                # 复制 speech_tokenizer 目录
                                speech_tokenizer_dir = os.path.join(self.model_path, 'speech_tokenizer')
                                if os.path.exists(speech_tokenizer_dir) and os.path.isdir(speech_tokenizer_dir):
                                    temp_speech_tokenizer_dir = os.path.join(temp_dir, 'speech_tokenizer')
                                    shutil.copytree(speech_tokenizer_dir, temp_speech_tokenizer_dir)
                                    print(f"【Qwen3-TTS】已复制 speech_tokenizer 目录到临时目录")
                                # 使用临时目录加载处理器和模型
                                processor_load_path = temp_dir
                                model_load_path = temp_dir
                                print(f"【Qwen3-TTS】使用临时目录加载: {temp_dir}")
                            except Exception as e:
                                print(f"【Qwen3-TTS】处理用户选择的模型文件失败: {str(e)}")
                                raise Exception(f"Failed to process user selected model file: {str(e)}")
                        else:
                            print(f"【Qwen3-TTS】没有找到索引文件，但用户选择了模型文件，直接创建临时目录")
                            try:
                                temp_dir = tempfile.mkdtemp()
                                # 复制用户选择的模型文件到临时目录并重命名为model.safetensors
                                temp_model_path = os.path.join(temp_dir, "model.safetensors")
                                shutil.copy(os.path.join(self.model_path, self.user_selected_model_file), temp_model_path)
                                print(f"【Qwen3-TTS】已复制并重命名模型文件到临时目录")
                                # 复制其他必要的配置文件到临时目录
                                for config_file in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt']:
                                    src_file = os.path.join(self.model_path, config_file)
                                    if os.path.exists(src_file):
                                        shutil.copy(src_file, os.path.join(temp_dir, config_file))
                                # 复制 speech_tokenizer 目录
                                speech_tokenizer_dir = os.path.join(self.model_path, 'speech_tokenizer')
                                if os.path.exists(speech_tokenizer_dir) and os.path.isdir(speech_tokenizer_dir):
                                    temp_speech_tokenizer_dir = os.path.join(temp_dir, 'speech_tokenizer')
                                    shutil.copytree(speech_tokenizer_dir, temp_speech_tokenizer_dir)
                                    print(f"【Qwen3-TTS】已复制 speech_tokenizer 目录到临时目录")
                                # 使用临时目录加载处理器和模型
                                processor_load_path = temp_dir
                                model_load_path = temp_dir
                                print(f"【Qwen3-TTS】使用临时目录加载: {temp_dir}")
                            except Exception as e:
                                print(f"【Qwen3-TTS】创建临时目录失败: {str(e)}")
                                raise Exception(f"Failed to create temporary directory: {str(e)}")
                
                # 加载处理器 - 使用正确的加载路径
                self.processor = AutoProcessor.from_pretrained(
                    processor_load_path, 
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"【Qwen3-TTS】处理器加载成功")
            except OSError as oo:
                if "speech_tokenizer/config.json" in str(oo):
                    print("【Qwen3-TTS错误】缺少 speech_tokenizer/config.json，无法继续加载。请使用完整的模型文件夹或重新下载模型。")
                    self.model = None
                    return
                raise
                
                # 加载模型 - 使用 AutoModel 代替 AutoModelForTextToSpeech
                torch_dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
                print(f"【Qwen3-TTS】使用数据类型: {torch_dtype}")
                
                # 构建模型加载参数
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "local_files_only": True
                }

                # 对于已经量化的模型，不添加额外的量化配置
                # 检查是否存在量化配置文件来判断
                quant_config_path = os.path.join(self.model_path, "quantization_config.json")
                is_pre_quantized = os.path.exists(quant_config_path)

                # 对于8bit量化模型，不添加额外的量化配置
                if is_8bit:
                    print(f"【Qwen3-TTS】检测到8bit量化模型，不添加量化配置参数")
                elif is_pre_quantized:
                    print(f"【Qwen3-TTS】检测到预量化模型，不添加量化配置参数")
                    # 检查并修复量化配置文件
                    try:
                        import json
                        with open(quant_config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        if 'quant_method' not in config:
                            print(f"【Qwen3-TTS】检测到量化配置缺少'quant_method'属性，尝试修复...")
                            config['quant_method'] = 'bitsandbytes'  # 使用正确的量化方法
                            with open(quant_config_path, 'w', encoding='utf-8') as f:
                                json.dump(config, f, indent=2, ensure_ascii=False)
                            print(f"【Qwen3-TTS】已修复quantization_config.json文件")
                    except Exception as e:
                        print(f"【Qwen3-TTS】修复量化配置文件失败: {str(e)}")
                elif use_8bit_quant:
                    print(f"【Qwen3-TTS】启用8bit量化（非预量化模型）")
                    try:
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                            quant_method="bitsandbytes"  # 显式指定量化方法
                        )
                    except ImportError:
                        print(f"【Qwen3-TTS】未安装bitsandbytes，无法使用8bit量化")
                        use_8bit_quant = False
                
                try:
                    # 始终使用文件夹路径加载模型
                    self.model = AutoModel.from_pretrained(
                        model_load_path,
                        **model_kwargs
                    )
                except (ValueError, RuntimeError) as e:
                    if "quant_method" in str(e):
                        print(f"【Qwen3-TTS】检测到量化配置缺少'quant_method'属性，尝试修复...")
                        # 尝试修改量化配置
                        import json
                        
                        # 首先检查 quantization_config.json
                        quant_config_path = os.path.join(self.model_path, "quantization_config.json")
                        # 然后检查 config.json
                        config_path = os.path.join(self.model_path, "config.json")
                        
                        fixed = False
                        
                        # 尝试修复 quantization_config.json
                        if os.path.exists(quant_config_path):
                            try:
                                with open(quant_config_path, 'r', encoding='utf-8') as f:
                                    config = json.load(f)
                                if 'quant_method' not in config:
                                    config['quant_method'] = 'bitsandbytes'
                                    with open(quant_config_path, 'w', encoding='utf-8') as f:
                                        json.dump(config, f, indent=2, ensure_ascii=False)
                                    print(f"【Qwen3-TTS】已修复quantization_config.json文件")
                                    fixed = True
                            except Exception as fix_err:
                                print(f"【Qwen3-TTS】修复quantization_config.json失败: {fix_err}")
                        
                        # 尝试修复 config.json 中的 quantization_config
                        if not fixed and os.path.exists(config_path):
                            try:
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config = json.load(f)
                                if 'quantization_config' in config and isinstance(config['quantization_config'], dict):
                                    if 'quant_method' not in config['quantization_config']:
                                        config['quantization_config']['quant_method'] = 'bitsandbytes'
                                        with open(config_path, 'w', encoding='utf-8') as f:
                                            json.dump(config, f, indent=2, ensure_ascii=False)
                                        print(f"【Qwen3-TTS】已修复config.json中的quantization_config")
                                        fixed = True
                            except Exception as fix_err:
                                print(f"【Qwen3-TTS】修复config.json失败: {fix_err}")
                        
                        # 如果修复了配置文件，尝试重新加载
                        if fixed:
                            try:
                                # 始终使用文件夹路径加载模型
                                self.model = AutoModel.from_pretrained(
                                    model_load_path,
                                    **model_kwargs
                                )
                                print(f"【Qwen3-TTS】修复后重新加载成功")
                            except Exception as fix_error:
                                print(f"【Qwen3-TTS】修复后仍然失败: {str(fix_error)}")
                                # 如果修复失败，尝试不使用量化
                                print(f"【Qwen3-TTS】尝试不使用量化重新加载...")
                                if "quantization_config" in model_kwargs:
                                    del model_kwargs["quantization_config"]
                                # 清除 transformers 缓存的配置
                                from transformers import AutoConfig
                                AutoConfig.clear_cached_config(model_load_path)
                                try:
                                    self.model = AutoModel.from_pretrained(
                                        model_load_path,
                                        **model_kwargs
                                    )
                                    print(f"【Qwen3-TTS】不使用量化重新加载成功")
                                except Exception as e:
                                    print(f"【Qwen3-TTS】再次失败: {str(e)}")
                                    self.model = None
                    else:
                        print(f"【Qwen3-TTS】transformers 加载失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        self.model = None
            
            if self.model is not None:
                print(f"【Qwen3-TTS】模型加载成功")
            else:
                print(f"【Qwen3-TTS】所有加载方式都失败了")
                
        except Exception as e:
            print(f"【Qwen3-TTS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def generate(self, text, **kwargs):
        """
        生成语音
        
        Args:
            text: 要转换的文本
            **kwargs: 额外的生成参数
            
        Returns:
            生成的音频数据和采样率
        """
        try:
            print(f"【Qwen3-TTS】正在生成语音: {text[:50]}...")
            
            # 获取生成参数
            speaker_id = kwargs.get("speaker_id", 0)
            emotion = kwargs.get("emotion", "default")
            language = kwargs.get("language", "Chinese")
            speed = kwargs.get("speed", 1.0)
            
            # 检查模型是否加载成功
            if self.model is None:
                print(f"【Qwen3-TTS错误】模型未加载，无法生成语音")
                return None, None
            
            # 使用 qwen-tts 包的生成方式
            if hasattr(self.model, "generate"):
                # 构建生成参数
                generate_kwargs = {
                    "text": text,
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                    "language": language,
                    "speed": speed
                }
                
                # 调用生成方法
                output = self.model.generate(**generate_kwargs)
                
                # 处理输出
                if isinstance(output, dict) and "audio" in output:
                    audio = output["audio"]
                    sampling_rate = self.qwen_tts_sampling_rate
                    print(f"【Qwen3-TTS】语音生成成功，长度: {len(audio)} samples")
                    return audio, sampling_rate
                else:
                    print(f"【Qwen3-TTS错误】生成输出格式不正确")
                    return None, None
            
            # 使用 transformers 的生成方式
            elif self.processor is not None:
                # 准备输入
                inputs = self.processor(
                    text=text,
                    speaker_id=speaker_id,
                    emotion=emotion,
                    language=language,
                    return_tensors="pt"
                ).to(self.device)
                
                # 生成语音
                with torch.no_grad():
                    output = self.model.generate(**inputs, speed=speed)
                
                # 处理输出
                audio = output["audio"].cpu().numpy().squeeze()
                sampling_rate = self.qwen_tts_sampling_rate
                print(f"【Qwen3-TTS】语音生成成功，长度: {len(audio)} samples")
                return audio, sampling_rate
            
            else:
                print(f"【Qwen3-TTS错误】模型或处理器未正确加载")
                return None, None
                
        except Exception as e:
            print(f"【Qwen3-TTS错误】生成语音失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_speaker_ids(self):
        """
        获取可用的说话人ID列表
        
        Returns:
            说话人ID列表
        """
        # Qwen3-TTS CustomVoice 支持9种音色
        return list(range(9))
    
    def get_languages(self):
        """
        获取支持的语言列表
        
        Returns:
            语言列表
        """
        return ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
    
    def get_emotions(self):
        """
        获取支持的情绪列表
        
        Returns:
            情绪列表
        """
        return ["default", "happy", "sad", "angry", "surprised", "calm", "excited", "gentle"]
    
    def release(self):
        """
        释放模型资源
        """
        if self.model is not None:
            # 将模型移动到CPU以确保GPU内存释放
            try:
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
            except Exception as e:
                print(f"【Qwen3-TTS】将模型移至CPU失败: {e}")
            
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 同步GPU操作并清空缓存
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有GPU操作完成
            torch.cuda.empty_cache()  # 清空GPU缓存
            print("【Qwen3-TTS】模型资源已释放并清理GPU缓存")
        else:
            print("【Qwen3-TTS】模型资源已释放，CUDA不可用")


class TTSEngineFactory:
    """
    TTS引擎工厂类
    使用工厂模式创建不同类型的TTS引擎
    """
    
    @staticmethod
    def create_engine(model_path, config, device="cpu"):
        """
        根据模型类型创建TTS引擎
        
        Args:
            model_path: 模型路径
            config: 配置参数
            device: 设备（cuda或cpu）
            
        Returns:
            相应的TTS引擎实例
        """
        # 从配置中获取模型类型
        model_type = config.get("model_type", "Auto-detect")
        
        # 检测模型类型
        detected_type = TTSEngineFactory._detect_model_type(model_path, model_type)
        
        # 根据检测到的类型创建引擎
        if detected_type in ["qwen_tts", "Qwen3-TTS", "qwen3_tts"]:
            return QwenTTSEngine(model_path, config, device)
        elif detected_type in ["kani_tts", "KaniTTS"]:
            # 后续可以添加Kani TTS引擎
            pass
        elif detected_type in ["bark", "Bark"]:
            # 后续可以添加Bark TTS引擎
            pass
        elif detected_type in ["vits", "VITS"]:
            # 后续可以添加VITS TTS引擎
            pass
        elif detected_type in ["xtts", "XTTS"]:
            # 后续可以添加XTTS引擎
            pass
        
        # 默认返回None
        return None
    
    @staticmethod
    def _detect_model_type(model_path, manual_type):
        """
        检测模型类型
        
        Args:
            model_path: 模型路径
            manual_type: 手动指定的模型类型
            
        Returns:
            检测到的模型类型
        """
        import os
        import json
        
        if manual_type != "自动检测" and manual_type != "Auto-detect":
            # 处理手动指定的模型类型
            type_lower = manual_type.lower()
            if type_lower == "kanitts":
                return "kani_tts"
            elif type_lower == "qwen3-tts":
                return "qwen_tts"
            else:
                return type_lower.replace("-", "_")
        
        filename = os.path.basename(model_path).lower()
        dirname = os.path.basename(os.path.dirname(model_path)).lower()
        full_path_lower = model_path.lower()
        detection_text = f"{filename} {dirname} {full_path_lower}"
        
        # 尝试从config.json检测
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, 'config.json')
        else:
            config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                architectures = config_data.get('architectures', [])
                model_type_cfg = config_data.get('model_type', '')
                
                if "Qwen3TTSForConditionalGeneration" in architectures or model_type_cfg == "qwen3_tts":
                    return "qwen_tts"
                elif "Lfm2ForCausalLM" in architectures or model_type_cfg == "lfm2":
                    return "kani_tts"
            except Exception:
                pass
        
        # 基于文件名和路径检测
        if "voicedesign" in detection_text or "customvoice" in detection_text:
            return "qwen_tts"
        
        if "qwen" in detection_text and "tts" in detection_text:
            return "qwen_tts"
        elif "ming" in detection_text and "tts" in detection_text:
            # Ming Omni TTS 目前与 Qwen3-TTS 兼容
            return "qwen_tts"
        elif "omni" in detection_text and "tts" in detection_text:
            return "qwen_tts"
        elif "kani" in detection_text and "tts" in detection_text:
            return "kani_tts"
        elif "bark" in detection_text:
            return "bark"
        elif "tortoise" in detection_text:
            return "tortoise"
        elif "coqui" in detection_text or "your_tts" in detection_text:
            return "coqui"
        elif "xtts" in detection_text:
            return "xtts"
        elif "vits" in detection_text:
            return "vits"
        elif "glow" in detection_text:
            return "glow_tts"
        elif "supertonic" in detection_text:
            if "supertonic-2" in detection_text or "supertonic2" in detection_text:
                return "supertonic_2"
            return "supertonic"
        elif filename.endswith(".gguf"):
            return "gguf_tts"
        elif filename.endswith(".safetensors") or filename.endswith(".pt") or filename.endswith(".pth"):
            return "pytorch_tts"
        else:
            # 尝试基于文件扩展名检测
            ext = os.path.splitext(filename)[1].lower()
            if ext in [".safetensors", ".pt", ".pth", ".bin"]:
                return "pytorch_tts"
            elif ext == ".gguf":
                return "gguf_tts"
            else:
                return "unknown"


# ========== 模型文件检测和下载功能 ==========

def _check_network_connection():
    """检查网络连接"""
    try:
        response = requests.get("https://www.baidu.com", timeout=5)
        return response.status_code == 200
    except:
        try:
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except:
            return False

def _get_model_source():
    """根据网络连接选择模型源"""
    if _check_network_connection():
        # 国内网络使用魔搭社区
        return "modelscope"
    else:
        # 国外网络使用Hugging Face
        return "huggingface"

def _download_file(url, save_path):
    """下载文件"""
    try:
        print(f"【文件下载】开始下载: {url}")
        print(f"【文件下载】保存路径: {save_path}")
        
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 下载文件
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        
        print(f"【文件下载】下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"【文件下载错误】下载失败: {str(e)}")
        return False

def _download_model_files(model_name, model_path):
    """下载模型必备文件"""
    source = _get_model_source()
    print(f"【模型下载】使用源: {source}")
    
    # 定义模型文件映射
    model_files = {
        # Qwen3-TTS模型文件
        "qwen3_tts": {
            "modelscope": {
                "model.safetensors": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-TTS/snapshots/master/model.safetensors",
                "config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-TTS/snapshots/master/config.json",
                "tokenizer.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-TTS/snapshots/master/tokenizer.json",
                "preprocessor_config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-TTS/snapshots/master/preprocessor_config.json",
                "speech_tokenizer/config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-TTS/snapshots/master/speech_tokenizer/config.json"
            },
            "huggingface": {
                "model.safetensors": f"https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/model.safetensors",
                "config.json": f"https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/config.json",
                "tokenizer.json": f"https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/tokenizer.json",
                "preprocessor_config.json": f"https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/preprocessor_config.json",
                "speech_tokenizer/config.json": f"https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/speech_tokenizer/config.json"
            }
        },
        # KaniTTS模型文件
        "kani_tts": {
            "modelscope": {
                "model.safetensors": f"https://modelscope.cn/api/v1/models/Kani/KaniTTS/snapshots/master/model.safetensors",
                "config.json": f"https://modelscope.cn/api/v1/models/Kani/KaniTTS/snapshots/master/config.json"
            },
            "huggingface": {
                "model.safetensors": f"https://huggingface.co/Kani/KaniTTS/resolve/main/model.safetensors",
                "config.json": f"https://huggingface.co/Kani/KaniTTS/resolve/main/config.json"
            }
        }
    }
    
    model_type = None
    if "qwen3" in model_name.lower() and "tts" in model_name.lower():
        model_type = "qwen3_tts"
    elif "kani" in model_name.lower() and "tts" in model_name.lower():
        model_type = "kani_tts"
    
    if model_type and model_type in model_files:
        files_to_download = model_files[model_type][source]
        success = True
        
        for filename, url in files_to_download.items():
            save_path = os.path.join(model_path, filename)
            if not os.path.exists(save_path):
                if "/" in filename:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if not _download_file(url, save_path):
                    success = False
        
        return success
    else:
        print(f"【模型下载】未知模型类型: {model_name}")
        return False

def _check_tts_model_files(model_path, model_type):
    """检查TTS模型必备文件"""
    required_files = []
    
    if model_type == "qwen3_tts":
        required_files = [
            "model.safetensors",
            "config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "speech_tokenizer/config.json"
        ]
    elif model_type == "kani_tts":
        required_files = [
            "model.safetensors",
            "config.json"
        ]
    
    missing_files = []
    for filename in required_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    return missing_files


class llama_cpp_tts_loader:
    """
    TTS模型加载器
    支持所有TTS模型类型，通过模型类型选择器自动识别或手动指定
    """

    @classmethod
    def INPUT_TYPES(s):
        # 检查LLM文件夹是否存在
        try:
            has_llm_folder = "LLM" in folder_paths.folder_names_and_paths
        except Exception:
            has_llm_folder = False

        # 所有TTS模型 - 只加载文件夹，不加载文件
        tts_list = ["None"]
        model_set = set()
        model_set.add("None")
        
        if has_llm_folder:
            try:
                # 获取所有LLM文件夹路径
                llm_folders = folder_paths.get_folder_paths("LLM")
                use_root_prefix = len(llm_folders) > 1

                for folder in llm_folders:
                    try:
                        root_tag = os.path.basename(folder) if use_root_prefix else ""

                        # 递归扫描所有子目录和文件
                        for root, dirs, files in os.walk(folder):
                            # 计算相对于LLM文件夹的路径
                            rel_path = os.path.relpath(root, folder)
                            if rel_path == '.':
                                current_folder_name = ""
                            else:
                                current_folder_name = rel_path.replace(os.sep, '/')

                            # 获取目录名称和完整路径（用于关键词检测）
                            dir_name = os.path.basename(root).lower()
                            full_path_lower = root.lower()

                            # 检查是否是包含TTS关键词的目录
                            # TTS关键词列表
                            tts_keywords = [
                                "tts", "kani",
                                "bark", "vits", "xtts", "coqui", "supertonic",
                                "tortoise", "glow", "your_tts",
                                "voice", "voicedesign", "customvoice", "custom_voice"
                            ]

                            # 检查目录名或完整路径是否包含TTS关键词
                            has_tts_keyword = any(keyword in dir_name or keyword in full_path_lower for keyword in tts_keywords)

                            # 对于Qwen和Omni系列，必须有明确的TTS关键词
                            if "qwen" in dir_name or "qwen" in full_path_lower or "omni" in dir_name or "omni" in full_path_lower:
                                # Qwen-Omni不是TTS模型，跳过
                                if "omni" in dir_name or "omni" in full_path_lower:
                                    # 只有包含明确TTS关键词的Omni才是TTS模型
                                    has_tts_keyword = any(kw in dir_name or kw in full_path_lower for kw in ["tts", "voice", "customvoice", "voicedesign"])
                                else:
                                    # 其他Qwen模型也必须有TTS关键词
                                    has_tts_keyword = any(kw in dir_name or kw in full_path_lower for kw in ["tts", "voice", "customvoice", "voicedesign"])

                            is_tts_dir = has_tts_keyword

                            if is_tts_dir:
                                # 检查目录中是否包含模型文件
                                has_model_files = any(os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".pt", ".pth", ".bin"] for f in files)

                                # 只有当目录包含模型文件时才添加目录
                                if has_model_files:
                                    if current_folder_name:
                                        folder_entry = current_folder_name
                                    else:
                                        folder_entry = dir_name

                                    if root_tag:
                                        folder_entry = f"{root_tag}/{folder_entry}"

                                    # 目录类型显示带斜杠后缀，避免与文件重名冲突
                                    folder_entry = folder_entry.rstrip('/') + '/'

                                    if folder_entry not in model_set:
                                        model_set.add(folder_entry)
                                        tts_list.append(folder_entry)

                            # 检查是否是分段模型文件夹（需要有TTS关键词才添加）
                            sharded_files1 = [f for f in files if "-of-" in f.lower() and f.endswith(".safetensors")]
                            sharded_files2 = [f for f in files if f.startswith("model.safetensors-") and "-of-" in f.lower()]
                            if sharded_files1 or sharded_files2:
                                # 检查文件夹名称是否包含TTS关键词
                                folder_path_lower = root.lower()
                                folder_name_lower = dir_name.lower()

                                # TTS关键词列表
                                tts_keywords = [
                                    "tts", "kani", "bark", "vits", "xtts",
                                    "coqui", "supertonic", "tortoise", "glow", "your_tts",
                                    "voice", "voicedesign", "customvoice", "custom_voice"
                                ]

                                # 对于Qwen系列，必须有TTS相关的关键词
                                if "qwen" in folder_path_lower or "qwen" in folder_name_lower or "omni" in folder_path_lower or "omni" in folder_name_lower:
                                    # Qwen-Omni不是TTS模型，跳过
                                    if "omni" in folder_path_lower or "omni" in folder_name_lower:
                                        if not any(kw in folder_path_lower or kw in folder_name_lower for kw in ["tts", "voice", "customvoice", "voicedesign"]):
                                            continue
                                    # 其他Qwen模型也必须有TTS关键词
                                    elif not any(kw in folder_path_lower or kw in folder_name_lower for kw in ["tts", "voice", "customvoice", "voicedesign"]):
                                        continue
                                elif not any(kw in folder_path_lower or kw in folder_name_lower for kw in tts_keywords):
                                    # 非Qwen模型也必须有TTS关键词
                                    continue

                                # 通过了所有检查，添加为TTS模型
                                if current_folder_name:
                                    folder_entry = current_folder_name
                                else:
                                    folder_entry = dir_name

                                if root_tag:
                                    folder_entry = f"{root_tag}/{folder_entry}"

                                # 目录类型显示带斜杠后缀，避免与文件重名冲突
                                folder_entry = folder_entry.rstrip('/') + '/'

                                if folder_entry not in model_set:
                                    model_set.add(folder_entry)
                                    tts_list.append(folder_entry)

                            # 检查非TTS目录中是否包含模型文件（用于检测包含模型文件的文件夹）
                            has_model_files_in_dir = any(os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".pt", ".pth", ".bin"] for f in files)
                            if has_model_files_in_dir:
                                # 检查文件夹名称或路径中是否包含TTS关键词
                                folder_path_lower = root.lower()
                                folder_name_lower = dir_name.lower()

                                # TTS关键词列表
                                tts_keywords = [
                                    "tts", "kani",
                                    "bark", "vits", "xtts", "coqui", "supertonic",
                                    "tortoise", "glow", "your_tts",
                                    "voice", "voicedesign", "customvoice", "custom_voice"
                                ]

                                # 检查是否包含TTS关键词
                                has_tts_keyword = any(keyword in folder_path_lower or keyword in folder_name_lower for keyword in tts_keywords)

                                # 对于Qwen系列，必须有TTS相关的关键词
                                if "qwen" in folder_path_lower or "qwen" in folder_name_lower:
                                    # Qwen-Omni不是TTS模型，跳过
                                    if "omni" in folder_path_lower or "omni" in folder_name_lower:
                                        if not any(kw in folder_path_lower or kw in folder_name_lower for kw in ["tts", "voice", "customvoice", "voicedesign"]):
                                            continue
                                    # 其他Qwen模型也必须有TTS关键词
                                    elif not has_tts_keyword:
                                        continue
                                elif not has_tts_keyword:
                                    # 非Qwen模型也必须有TTS关键词
                                    continue
                                    
                                    if current_folder_name:
                                        folder_entry = current_folder_name
                                    else:
                                        folder_entry = dir_name

                                    if root_tag:
                                        folder_entry = f"{root_tag}/{folder_entry}"

                                    # 目录类型显示带斜杠后缀，避免与文件重名冲突
                                    folder_entry = folder_entry.rstrip('/') + '/'

                                    if folder_entry not in model_set:
                                        model_set.add(folder_entry)
                                        tts_list.append(folder_entry)
                                        print(f"【TTS模型检测】检测到包含TTS模型文件的目录: {folder_entry}")
                    except Exception as e:
                        print(f"【TTS模型检测】扫描文件夹 {folder} 失败: {e}")
            except Exception as e:
                print(f"【TTS模型检测】获取LLM文件列表失败: {e}")
                pass

        if len(tts_list) == 1:
            tts_list = ["None", "请将TTS模型放入models/LLM文件夹"]

        perf_level = HARDWARE_INFO.get("perf_level", "low")

        if perf_level in ["high", "mid_high", "mid", "mid_low"]:
            default_n_gpu_layers = -1
        else:
            default_n_gpu_layers = 20

        return {
            "required": {
                "tts_model": (tts_list, {"tooltip": "选择TTS模型文件\n• 主流：Qwen3-TTS、KaniTTS\n• 其他：Bark、VITS、XTTS、Coqui、Supertonic等"}),
                "n_gpu_layers": ("INT", {"default": default_n_gpu_layers, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载"}),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000, "step": 100, "tooltip": "音频采样率（Hz），Qwen3-TTS推荐24000"}),
                "model_type": (["Auto-detect", "Qwen3-TTS", "KaniTTS", "Bark", "VITS", "XTTS", "Coqui", "Supertonic", "Supertonic-2", "GGUF-TTS", "PyTorch-TTS"], {"default": "Auto-detect", "tooltip": "模型类型，通常选择自动检测即可"}),
            },
            "optional": {
                "voice": (["Vivian - 明亮略带锐气的年轻女声 (中文)", "Serena - 温暖柔和的年轻女声 (中文)", "Uncle_Fu - 音色低沉醇厚的成熟男声 (中文)", "Dylan - 清晰自然的北京青年男声 (中文北京方言)", "Eric - 活泼略带沙哑明亮感的成都男声 (中文四川方言)", "Ryan - 富有节奏感的动态男声 (英语)", "Aiden - 清晰中频的阳光美式男声 (英语)", "Ono_Anna - 轻快灵活的俏皮日语女声 (日语)", "Sohee - 富含情感的温暖韩语女声 (韩语)"], {"default": "Vivian - 明亮略带锐气的年轻女声 (中文)", "tooltip": "选择音色类型（Qwen3-TTS CustomVoice支持9种高级音色）"}),
                "emotion": (["default", "happy", "sad", "angry", "surprised", "calm", "excited", "gentle"], {"default": "default", "tooltip": "选择情绪风格（主流TTS模型支持）"}),
                "language": (["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {"default": "Chinese", "tooltip": "选择合成语言（Qwen3-TTS支持10种语言）"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "生成温度，值越高越随机"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "核采样参数，控制生成多样性"}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "tooltip": "Top-K采样参数"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1, "tooltip": "重复惩罚"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 100, "max": 4096, "step": 100, "tooltip": "最大生成token数"}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "是否使用KV缓存加速推理"}),
                "ref_audio_path": ("STRING", {"default": "", "tooltip": "音色克隆参考音路径（wav格式，3~60秒最佳，单声道48000Hz）"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "display": "slider"}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "display": "slider"}),
                "volume": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("TTSMODEL",)
    RETURN_NAMES = ("tts_model",)
    FUNCTION = "load_tts_model"
    CATEGORY = "llama-cpp-vlm"

    @classmethod
    def _resolve_tts_model_path(s, tts_model):
        key = tts_model.rstrip('/')
        if key == "None":
            return None

        # 尝试所有LLM目录下的直接路径
        for folder in folder_paths.get_folder_paths("LLM"):
            candidate = os.path.join(folder, key)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)

        # 尝试带根目录前缀模式
        if "/" in key:
            root_name, inner = key.split("/", 1)
            for folder in folder_paths.get_folder_paths("LLM"):
                if os.path.basename(folder) == root_name:
                    candidate = os.path.join(folder, inner)
                    if os.path.exists(candidate):
                        return os.path.normpath(candidate)

        # 兼容 folder_paths.get_full_path 的老路径
        full = folder_paths.get_full_path("LLM", key)
        if full and os.path.exists(full):
            return os.path.normpath(full)

        return None

    def __init__(self):
        self.loaded_model = None
        self.current_config = None

    VOICE_MAP = {
        "Vivian": 0,
        "Serena": 1,
        "Uncle_Fu": 2,
        "Dylan": 3,
        "Eric": 4,
        "Ryan": 5,
        "Aiden": 6,
        "Ono_Anna": 7,
        "Sohee": 8,
    }

    EMOTION_MAP = {
        "默认": "default",
        "开心": "happy",
        "悲伤": "sad",
        "愤怒": "angry",
        "惊讶": "surprised",
        "平静": "calm",
        "兴奋": "excited",
        "温柔": "gentle",
    }
    
    LANGUAGE_MAP = {
        "Chinese": "chinese",
        "English": "english",
        "Japanese": "japanese",
        "Korean": "korean",
        "German": "german",
        "French": "french",
        "Russian": "russian",
        "Portuguese": "portuguese",
        "Spanish": "spanish",
        "Italian": "italian",
    }

    @classmethod
    def IS_CHANGED(s, tts_model, n_gpu_layers, sample_rate, model_type,
                  voice="Vivian", emotion="default", language="Chinese", speed=1.0, temperature=0.7, top_p=0.9, top_k=50,
                  repetition_penalty=1.1, max_new_tokens=2048, use_cache=True,
                  ref_audio_path="", pitch=0.0, volume=1.0):
        # 提取纯音色名称（去掉描述部分）
        pure_voice = voice.split(' - ')[0] if ' - ' in voice else voice
        speaker_id = s.VOICE_MAP.get(pure_voice, 0)
        emotion_value = s.EMOTION_MAP.get(emotion, "default")

        resolved_path = s._resolve_tts_model_path(tts_model)

        # 使用有序字典确保键顺序一致
        from collections import OrderedDict
        config = OrderedDict([
            ("tts_model", tts_model),
            ("tts_model_path", resolved_path or ""),
            ("n_gpu_layers", n_gpu_layers),
            ("sample_rate", sample_rate),
            ("model_type", model_type),
            ("speaker_id", speaker_id),
            ("emotion", emotion_value),
            ("language", language),
            ("speed", speed),
            ("temperature", temperature),
            ("top_p", top_p),
            ("top_k", top_k),
            ("repetition_penalty", repetition_penalty),
            ("max_new_tokens", max_new_tokens),
            ("use_cache", use_cache),
            ("ref_audio_path", ref_audio_path),
            ("pitch", pitch),
            ("volume", volume),
        ])

        result = json.dumps(config, ensure_ascii=False)
        print(f"【IS_CHANGED】模型: {tts_model}, 解析路径: {resolved_path}, 配置哈希: {hash(result) % 10000}")
        return result

    def load_tts_model(self, tts_model, n_gpu_layers, sample_rate, model_type,
                      voice="Vivian", emotion="default", language="Chinese", speed=1.0, temperature=0.7, top_p=0.9, top_k=50,
                      repetition_penalty=1.1, max_new_tokens=2048, use_cache=True,
                      ref_audio_path="", pitch=0.0, volume=1.0):
        if tts_model == "None" or tts_model == "请将TTS模型放入models/LLM文件夹":
            print("【TTS统一加载器】未选择TTS模型")
            return (None,)

        resolved_model_path = self._resolve_tts_model_path(tts_model)
        if not resolved_model_path:
            raise RuntimeError(f"无法解析TTS模型路径: {tts_model}")

        # 提取纯音色名称（去掉描述部分）
        pure_voice = voice.split(' - ')[0] if ' - ' in voice else voice
        speaker_id = self.VOICE_MAP.get(pure_voice, 0)
        emotion_value = self.EMOTION_MAP.get(emotion, "default")

        try:
            print(f"【TTS统一加载器】正在加载TTS模型: {tts_model} (解析路径: {resolved_model_path})")

            # 使用有序字典确保键顺序一致
            from collections import OrderedDict
            config = OrderedDict([
                ("tts_model", tts_model),
                ("n_gpu_layers", n_gpu_layers),
                ("sample_rate", sample_rate),
                ("model_type", model_type),
                ("speaker_id", speaker_id),
                ("emotion", emotion_value),
                ("language", language),
                ("speed", speed),
                ("temperature", temperature),
                ("top_p", top_p),
                ("top_k", top_k),
                ("repetition_penalty", repetition_penalty),
                ("max_new_tokens", max_new_tokens),
                ("use_cache", use_cache),
                ("ref_audio_path", ref_audio_path),
                ("pitch", pitch),
                ("volume", volume),
            ])

            # 缓存检查 - 添加详细调试信息
            if self.loaded_model is not None and self.current_config is not None:
                print(f"【TTS统一加载器】缓存检查:")
                print(f"  当前模型: {self.current_config.get('tts_model', 'None')}")
                print(f"  请求模型: {config.get('tts_model', 'None')}")
                print(f"  配置相同: {self.current_config == config}")

                if self.current_config == config:
                    print("【TTS统一加载器】使用缓存的TTS模型")
                    return (self.loaded_model,)
                else:
                    print("【TTS统一加载器】配置已改变，重新加载模型")
                    # 打印配置差异
                    for key in config:
                        if key in self.current_config:
                            if self.current_config[key] != config[key]:
                                print(f"  差异 - {key}: {self.current_config[key]} -> {config[key]}")
                        else:
                            print(f"  新增 - {key}: {config[key]}")
            else:
                print(f"【TTS统一加载器】无缓存或缓存为空，加载新模型")

            # 检查LLM文件夹是否存在
            try:
                has_llm_folder = "LLM" in folder_paths.folder_names_and_paths
            except Exception:
                has_llm_folder = False

            if not has_llm_folder:
                raise RuntimeError(f"LLM文件夹不存在，请在models目录下创建LLM文件夹并放入TTS模型")

            # 使用解析后的绝对路径（现在只返回文件夹路径）
            model_path = resolved_model_path
            
            if model_path is None or not os.path.exists(model_path):
                raise RuntimeError(f"TTS模型文件夹不存在: {tts_model} (解析路径：{resolved_model_path})")
            
            # 确保是文件夹路径
            if not os.path.isdir(model_path):
                # 如果是文件路径，获取其所在目录
                model_path = os.path.dirname(model_path)
                print(f"【TTS统一加载器】检测到文件路径，自动使用所在目录: {model_path}")
            
            print(f"【TTS统一加载器】模型路径: {model_path}")
            print(f"【TTS统一加载器】运行模式: GPU, GPU层数: {n_gpu_layers}")
            print(f"【TTS统一加载器】采样率: {sample_rate}Hz, 模型类型: {model_type}")

            # 检测模型类型并检查必备文件
            model_dir = model_path
            
            # 根据模型名称判断类型
            detected_model_type = None
            if "qwen3" in tts_model.lower() and "tts" in tts_model.lower():
                detected_model_type = "qwen3_tts"
            elif "kani" in tts_model.lower() and "tts" in tts_model.lower():
                detected_model_type = "kani_tts"
            
            # 检查必备文件
            if detected_model_type:
                print(f"【TTS统一加载器】检测模型类型: {detected_model_type}")
                missing_files = _check_tts_model_files(model_dir, detected_model_type)
                
                if missing_files:
                    print(f"【TTS统一加载器】检测到缺失文件: {missing_files}")
                    print(f"【TTS统一加载器】尝试自动下载缺失文件...")
                    
                    # 尝试下载缺失文件
                    if _download_model_files(tts_model, model_dir):
                        print(f"【TTS统一加载器】缺失文件下载成功")
                    else:
                        print(f"【TTS统一加载器警告】部分文件下载失败，尝试继续加载")

            tts_wrapper = UnifiedTTSModelWrapper(
                model_path=model_path,
                config=config,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            self.loaded_model = tts_wrapper
            self.current_config = config

            # 检查模型是否成功加载
            if tts_wrapper.model is None:
                print("【TTS统一加载器错误】TTS模型包装器创建成功，但内部模型加载失败")
                print("【TTS统一加载器错误】请检查上面的错误日志了解详细原因")
                return (None,)

            print("【TTS统一加载器】TTS模型加载成功")
            return (tts_wrapper,)

        except Exception as e:
            print(f"【TTS统一加载器错误】加载TTS模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 重置缓存状态，避免失败的模型影响后续加载
            self.loaded_model = None
            self.current_config = None
            print("【TTS统一加载器】已重置缓存状态")
            return (None,)


# 全局模型缓存
model_cache = {}

class UnifiedTTSModelWrapper:
    """
    统一TTS模型包装器
    支持所有TTS模型类型
    """
    
    VOICE_MAP = {
        "Vivian": 0,
        "Serena": 1,
        "Uncle_Fu": 2,
        "Dylan": 3,
        "Eric": 4,
        "Ryan": 5,
        "Aiden": 6,
        "Ono_Anna": 7,
        "Sohee": 8,
    }

    LANGUAGE_MAP = {
        "Chinese": "chinese",
        "English": "english",
        "Japanese": "japanese",
        "Korean": "korean",
        "German": "german",
        "French": "french",
        "Russian": "russian",
        "Portuguese": "portuguese",
        "Spanish": "spanish",
        "Italian": "italian"
    }

    def __init__(self, model_path, config, device="cpu"):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.model = None
        self.processor = None
        self.model_type = "unknown"
        self.tts_engine = None

        # 生成缓存键
        cache_key = self._generate_cache_key(model_path, config, device)
        print(f"【TTS统一包装器】生成缓存键: {cache_key}")

        # 检查缓存
        global model_cache
        if cache_key in model_cache:
            print(f"【TTS统一包装器】从缓存加载模型")
            cached_instance = model_cache[cache_key]
            self.model = cached_instance.model
            self.processor = cached_instance.processor
            self.model_type = cached_instance.model_type
            self.tts_engine = cached_instance.tts_engine
            print(f"【TTS统一包装器】缓存加载成功，模型类型: {self.model_type}")
            return

        # 检测模型类型
        model_type_manual = config.get("model_type", "Auto-detect")
        self.model_type = TTSEngineFactory._detect_model_type(model_path, model_type_manual)
        print(f"【TTS统一包装器】检测到模型类型: {self.model_type}")

        # 根据模型类型直接加载
        if self.model_type in ["qwen_tts", "Qwen3-TTS", "qwen3_tts"]:
            self._load_qwen_tts_model()
        elif self.model_type in ["kani_tts", "KaniTTS"]:
            self._load_kani_tts_model()
        elif self.model_type in ["bark", "Bark"]:
            self._load_bark_model()
        elif self.model_type in ["vits", "VITS"]:
            self._load_vits_model()
        elif self.model_type in ["xtts", "XTTS"]:
            self._load_xtts_model()
        elif self.model_type in ["coqui", "Coqui"]:
            self._load_coqui_model()
        elif self.model_type in ["supertonic", "Supertonic"]:
            self._load_supertonic_model()
        elif self.model_type in ["supertonic_2", "Supertonic-2"]:
            self._load_supertonic_2_model()
        elif self.model_type in ["gguf_tts", "GGUF-TTS"]:
            self._load_gguf_model()
        elif self.model_type in ["pytorch_tts", "PyTorch-TTS"]:
            self._load_pytorch_model()
        else:
            print(f"【TTS统一包装器警告】未知模型类型: {self.model_type}")

        # 加载完成后检查模型状态
        if self.model is None:
            print(f"【TTS统一包装器错误】模型加载失败，self.model 为 None")
            print(f"【TTS统一包装器错误】模型类型: {self.model_type}")
            print(f"【TTS统一包装器错误】模型路径: {self.model_path}")
        else:
            print(f"【TTS统一包装器】模型加载完成，模型对象类型: {type(self.model).__name__}")
            # 缓存模型
            model_cache[cache_key] = self
            print(f"【TTS统一包装器】模型已缓存，当前缓存大小: {len(model_cache)}")

    def _generate_cache_key(self, model_path, config, device):
        """
        生成缓存键
        
        Args:
            model_path: 模型路径
            config: 配置参数
            device: 设备
            
        Returns:
            缓存键字符串
        """
        import json
        from collections import OrderedDict
        
        # 提取关键配置参数
        key_config = OrderedDict([
            ("model_path", os.path.normpath(model_path)),
            ("model_type", config.get("model_type", "Auto-detect")),
            ("device", device),
            ("n_gpu_layers", config.get("n_gpu_layers", 0)),
        ])
        
        return json.dumps(key_config, ensure_ascii=False)
    
    def generate(self, text, **kwargs):
        """
        生成语音

        Args:
            text: 要转换的文本
            **kwargs: 额外的生成参数

        Returns:
            生成的音频数据和采样率
        """
        return self.synthesize(
            text=text,
            speaker_id=kwargs.get("speaker_id", 0),
            speed=kwargs.get("speed", 1.0),
            emotion=kwargs.get("emotion", "default"),
            language=kwargs.get("language"),
            pitch=kwargs.get("pitch", 0.0),
            volume=kwargs.get("volume", 1.0),
            ref_audio_path=kwargs.get("ref_audio_path")
        )
    
    def get_speaker_ids(self):
        """
        获取可用的说话人ID列表

        Returns:
            说话人ID列表
        """
        return list(range(9))

    def get_languages(self):
        """
        获取支持的语言列表

        Returns:
            语言列表
        """
        return ["chinese", "english", "japanese", "korean", "german", "french", "russian", "portuguese", "spanish", "italian"]

    def get_emotions(self):
        """
        获取支持的情绪列表

        Returns:
            情绪列表
        """
        return ["default", "happy", "sad", "angry", "surprised", "calm", "excited", "gentle"]

    def release(self):
        """
        释放模型资源
        """
        if self.model is not None:
            try:
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
            except Exception as e:
                print(f"【TTS资源释放】将模型移至CPU失败: {e}")
        
        # 删除模型引用
        self.model = None
        self.processor = None
        self.tts_engine = None
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 同步GPU操作并清空缓存
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有GPU操作完成
            torch.cuda.empty_cache()  # 清空GPU缓存
            print("【TTS资源释放】已清理GPU缓存")
        else:
            print("【TTS资源释放】CUDA不可用，仅清理CPU内存")

    def _load_qwen_tts_model(self):
        try:
            import os
            import tempfile
            import shutil
            print(f"【Qwen3-TTS】正在加载模型...")
            print(f"【Qwen3-TTS】模型路径: {self.model_path}")

            if not os.path.exists(self.model_path):
                print(f"【Qwen3-TTS错误】模型路径不存在: {self.model_path}")
                self.model = None
                return

            original_path = self.model_path
            is_model_file = os.path.isfile(original_path)
            user_selected_model_file = None

            if is_model_file:
                self.model_path = os.path.dirname(original_path)
                user_selected_model_file = os.path.basename(original_path)
                print(f"【Qwen3-TTS】检测到文件路径，已转换为文件夹路径: {self.model_path}")

            safetensors_files = [f for f in os.listdir(self.model_path) if f.endswith('.safetensors')]
            print(f"【Qwen3-TTS】检测到 .safetensors 文件: {safetensors_files}")

            if 'model.safetensors' not in safetensors_files and safetensors_files:
                if user_selected_model_file:
                    self.user_selected_model_file = user_selected_model_file
                else:
                    self.user_selected_model_file = safetensors_files[0]
            else:
                self.user_selected_model_file = None

            path_lower = self.model_path.lower()
            original_path_lower = original_path.lower()
            combined_path = f"{path_lower} {original_path_lower}"

            is_customvoice = "customvoice" in combined_path or "custom_voice" in combined_path
            is_voicedesign = "voicedesign" in combined_path or "voice_design" in combined_path
            is_12hz = "12hz" in combined_path
            is_8bit = "8bit" in combined_path

            self.qwen_tts_sampling_rate = 12000 if is_12hz else 24000
            if is_customvoice:
                self.qwen_tts_variant = "CustomVoice"
            elif is_voicedesign:
                self.qwen_tts_variant = "VoiceDesign"
            else:
                self.qwen_tts_variant = "CustomVoice"

            print(f"【Qwen3-TTS】使用变体: {self.qwen_tts_variant}, 采样率: {self.qwen_tts_sampling_rate}Hz")

            try:
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
                print(f"【Qwen3-TTS】qwen-tts 包导入成功")

                torch_dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
                qwen_kwargs = {
                    "device_map": "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu",
                    "dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }

                load_path = self.model_path
                if hasattr(self, 'user_selected_model_file') and self.user_selected_model_file:
                    model_file_path = os.path.join(self.model_path, "model.safetensors")
                    if not os.path.exists(model_file_path):
                        index_file_path = os.path.join(self.model_path, "model.safetensors.index.json")
                        if os.path.exists(index_file_path):
                            with open(index_file_path, 'r', encoding='utf-8') as f:
                                index_data = json.load(f)
                            for key in index_data.get('weight_map', {}):
                                index_data['weight_map'][key] = self.user_selected_model_file
                            temp_dir = tempfile.mkdtemp()
                            temp_index_path = os.path.join(temp_dir, "model.safetensors.index.json")
                            with open(temp_index_path, 'w', encoding='utf-8') as f:
                                json.dump(index_data, f, indent=2, ensure_ascii=False)
                            shutil.copy(os.path.join(self.model_path, self.user_selected_model_file), os.path.join(temp_dir, "model.safetensors"))
                            for config_file in ['config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'vocab.json', 'merges.txt']:
                                src_file = os.path.join(self.model_path, config_file)
                                if os.path.exists(src_file):
                                    shutil.copy(src_file, os.path.join(temp_dir, config_file))
                            speech_tokenizer_dir = os.path.join(self.model_path, 'speech_tokenizer')
                            if os.path.exists(speech_tokenizer_dir) and os.path.isdir(speech_tokenizer_dir):
                                shutil.copytree(speech_tokenizer_dir, os.path.join(temp_dir, 'speech_tokenizer'))
                            load_path = temp_dir

                self.model = Qwen3TTSModel.from_pretrained(load_path, **qwen_kwargs)
                self.processor = None

                # 验证模型确实在GPU上
                if hasattr(self.model, 'device') or hasattr(self.model, 'hf_device_map'):
                    model_device = getattr(self.model, 'device', None)
                    hf_device_map = getattr(self.model, 'hf_device_map', None)
                    print(f"【Qwen3-TTS】模型设备: {model_device}, 设备映射: {hf_device_map}")

                    # 检查模型参数的实际设备
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'parameters'):
                        param_device = next(self.model.model.parameters()).device
                        print(f"【Qwen3-TTS】模型参数实际设备: {param_device}")

                print(f"【Qwen3-TTS】qwen-tts 包加载成功")

            except ImportError:
                print("【Qwen3-TTS】qwen-tts 包未安装，尝试使用 transformers...")
                from transformers import AutoModelForTextToSpeech, AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForTextToSpeech.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                print(f"【Qwen3-TTS】transformers 加载成功")

        except Exception as e:
            print(f"【Qwen3-TTS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _load_kani_tts_model(self):
        try:
            from transformers import AutoModelForTextToSpeech, AutoProcessor
            print(f"【KaniTTS】正在加载模型...")

            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForTextToSpeech.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                print(f"【KaniTTS】模型已加载到 GPU")
            else:
                print(f"【KaniTTS】模型已加载到 CPU")
            print(f"【KaniTTS】模型加载成功")

        except ImportError:
            print("【KaniTTS错误】未安装transformers库")
        except Exception as e:
            print(f"【KaniTTS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_bark_model(self):
        try:
            from bark import preload_models
            print(f"【Bark】正在加载模型...")

            preload_models(
                text_use_gpu=self.device == "cuda",
                fine_use_gpu=self.device == "cuda",
                coarse_use_gpu=self.device == "cuda",
                codec_use_gpu=self.device == "cuda"
            )
            self.model = "bark"
            print(f"【Bark】模型加载成功")

        except ImportError:
            print("【Bark错误】未安装bark库，请运行: pip install bark")
        except Exception as e:
            print(f"【Bark错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_vits_model(self):
        try:
            print(f"【VITS】正在加载模型...")
            if self.model_path.endswith(".pt") or self.model_path.endswith(".pth"):
                state_dict = torch.load(self.model_path, map_location="cpu")
                self.model = {"state_dict": state_dict}
                print(f"【VITS】模型加载成功")
            else:
                print(f"【VITS错误】不支持的模型格式")
        except Exception as e:
            print(f"【VITS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_xtts_model(self):
        try:
            from TTS.api import TTS
            print(f"【XTTS】正在加载模型...")

            self.model = TTS(model_path=self.model_path, config_path=os.path.join(os.path.dirname(self.model_path), "config.json"))
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                print(f"【XTTS】模型已加载到 GPU")
            else:
                print(f"【XTTS】模型已加载到 CPU")
            print(f"【XTTS】模型加载成功")

        except ImportError:
            print("【XTTS错误】未安装TTS库，请运行: pip install TTS")
        except Exception as e:
            print(f"【XTTS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_coqui_model(self):
        try:
            from TTS.api import TTS
            print(f"【Coqui】正在加载模型...")

            self.model = TTS(model_path=self.model_path, config_path=os.path.join(os.path.dirname(self.model_path), "config.json"))
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                print(f"【Coqui】模型已加载到 GPU")
            else:
                print(f"【Coqui】模型已加载到 CPU")
            print(f"【Coqui】模型加载成功")

        except ImportError:
            print("【Coqui错误】未安装TTS库")
        except Exception as e:
            print(f"【Coqui错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_supertonic_model(self):
        try:
            import onnxruntime as ort
            print(f"【Supertonic】正在加载模型...")

            providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.model = ort.InferenceSession(self.model_path, providers=providers)
            print(f"【Supertonic】模型加载成功")

        except ImportError:
            print("【Supertonic错误】未安装onnxruntime库")
        except Exception as e:
            print(f"【Supertonic错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_supertonic_2_model(self):
        try:
            import onnxruntime as ort
            print(f"【Supertonic-2】正在加载模型...")

            providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.model = ort.InferenceSession(self.model_path, providers=providers)
            print(f"【Supertonic-2】模型加载成功")

        except ImportError:
            print("【Supertonic-2错误】未安装onnxruntime库")
        except Exception as e:
            print(f"【Supertonic-2错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_gguf_model(self):
        try:
            from llama_cpp import Llama
            print(f"【GGUF TTS】正在加载模型...")

            n_gpu_layers = self.config.get("n_gpu_layers", 0)
            n_ctx = 8192
            n_audio_ctx = 8192

            self.model = Llama(
                model_path=self.model_path, 
                n_gpu_layers=n_gpu_layers, 
                n_ctx=n_ctx, 
                verbose=False,
                audio=True,
                n_audio_ctx=n_audio_ctx,
                tensor_split=None,
                offload_kqv=True
            )
            print(f"【GGUF TTS】模型加载成功，GPU层数: {n_gpu_layers}")

        except ImportError:
            print("【GGUF TTS错误】未安装llama-cpp-python库")
        except Exception as e:
            print(f"【GGUF TTS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _load_pytorch_model(self):
        try:
            print(f"【PyTorch TTS】正在加载模型...")

            if self.model_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location="cpu")

            self.model = {"state_dict": state_dict}
            print(f"【PyTorch TTS】模型加载成功")

        except Exception as e:
            print(f"【PyTorch TTS错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # ========== 语音合成方法 ==========

    def synthesize(self, text, speaker_id=None, speed=None, emotion=None, language=None, temperature=None, top_p=None,
                  top_k=None, repetition_penalty=None, max_new_tokens=None, use_cache=None,
                  ref_audio_path=None, pitch=0.0, volume=1.0):
        if not text:
            return None

        # 检查模型是否已正确加载
        if self.model is None:
            print(f"【TTS合成错误】模型未加载，self.model 为 None")
            print(f"【TTS合成错误】模型路径: {self.model_path}")
            print(f"【TTS合成错误】模型类型: {self.model_type}")
            print(f"【TTS合成错误】请检查模型是否正确安装并加载")
            return None

        # 使用传入参数或配置默认值
        # 优先使用传入的speaker_id，如果没有传入则使用配置中的值
        if speaker_id is not None:
            print(f"【TTS合成】使用传入的speaker_id: {speaker_id}")
        else:
            speaker_id = self.config.get("speaker_id", 0)
            print(f"【TTS合成】使用配置中的speaker_id: {speaker_id}")
        
        speed = speed if speed is not None else self.config.get("speed", 1.0)
        emotion = emotion if emotion is not None else self.config.get("emotion", "default")
        language = language if language is not None else self.config.get("language", "中文")
        temperature = temperature if temperature is not None else self.config.get("temperature", 0.7)
        top_p = top_p if top_p is not None else self.config.get("top_p", 0.9)
        top_k = top_k if top_k is not None else self.config.get("top_k", 50)
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.get("repetition_penalty", 1.1)
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.get("max_new_tokens", 2048)
        use_cache = use_cache if use_cache is not None else self.config.get("use_cache", True)
        pitch = pitch if pitch is not None else self.config.get("pitch", 0.0)
        volume = volume if volume is not None else self.config.get("volume", 1.0)
        sample_rate = self.config.get("sample_rate", 24000)

        self.generation_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "use_cache": use_cache,
            "emotion": emotion,
            "language": language,
            "pitch": pitch,
            "volume": volume,
        }

        try:
            # 根据模型类型调用对应的合成方法
            if self.model_type in ["qwen_tts", "Qwen3-TTS", "qwen3_tts"]:
                return self._synthesize_qwen_tts(text, speaker_id, speed, sample_rate, ref_audio_path, pitch, volume, emotion, language)
            elif self.model_type in ["kani_tts", "KaniTTS"]:
                return self._synthesize_kani_tts(text, speaker_id, speed, sample_rate)
            elif self.model_type in ["bark", "Bark"]:
                return self._synthesize_bark(text, speed, sample_rate)
            elif self.model_type in ["vits", "VITS"]:
                return self._synthesize_vits(text, speed, sample_rate)
            elif self.model_type in ["xtts", "XTTS"]:
                return self._synthesize_xtts(text, speed, sample_rate)
            elif self.model_type in ["coqui", "Coqui"]:
                return self._synthesize_coqui(text, speed, sample_rate)
            elif self.model_type in ["supertonic", "Supertonic"]:
                return self._synthesize_supertonic(text, speed, sample_rate)
            elif self.model_type in ["supertonic_2", "Supertonic-2"]:
                return self._synthesize_supertonic_2(text, speed, sample_rate)
            elif self.model_type in ["gguf_tts", "GGUF-TTS"]:
                return self._synthesize_gguf(text, speed, sample_rate)
            elif self.model_type in ["pytorch_tts", "PyTorch-TTS"]:
                return self._synthesize_pytorch_tts(text, speed, sample_rate)
            else:
                print(f"【TTS合成错误】不支持的模型类型: {self.model_type}")
                return None

        except Exception as e:
            print(f"【TTS合成错误】语音合成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_qwen_tts(self, text, speaker_id, speed, sample_rate, ref_audio_path, pitch, volume, emotion="default", language="Chinese"):
        """Qwen3-TTS 语音合成

        支持两种模式:
        1. transformers 方式 (processor + model.generate)
        2. ModelScope pipeline 方式 (直接调用)
        """
        import time
        start_time = time.time()

        try:
            print(f"【Qwen3-TTS合成】开始合成，文本长度: {len(text) if text else 0}")
            print(f"【Qwen3-TTS合成】模型状态: {self.model is not None}, 处理器状态: {self.processor is not None}")
            print(f"【Qwen3-TTS合成】模型类型: {type(self.model).__name__ if self.model else 'None'}")

            # 检查GPU可用性和显存使用
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"【Qwen3-TTS合成】GPU可用，显存已分配: {gpu_memory_allocated:.2f}GB, 显存已保留: {gpu_memory_reserved:.2f}GB")
            else:
                print(f"【Qwen3-TTS合成警告】CUDA不可用，模型可能在CPU上运行")

            if self.model is None:
                print(f"【Qwen3-TTS合成错误】self.model 为 None，模型未正确加载")
                print(f"【Qwen3-TTS合成错误】模型路径: {self.model_path}")
                print(f"【Qwen3-TTS合成错误】模型类型: {self.model_type}")
                raise RuntimeError("模型未加载")

            # 判断加载方式 - Qwen3TTSModel直接调用
            if self.processor is None and 'Qwen3TTSModel' in type(self.model).__name__:
                # 使用 qwen-tts 包方式
                print(f"【Qwen3-TTS 合成】使用 qwen-tts 包直接调用方式")

                try:
                    # 直接使用数字speaker_id，根据Qwen3-TTS模型的要求
                    language_code = self.LANGUAGE_MAP.get(language, "chinese")

                    print(f"【Qwen3-TTS 合成】参数: text='{text}', speaker_id={speaker_id}, language={language_code}, speed={speed}")

                    # 检查可用方法
                    has_generate_custom_voice = hasattr(self.model, 'generate_custom_voice')
                    has_generate_voice_design = hasattr(self.model, 'generate_voice_design')
                    has_generate = hasattr(self.model, 'generate')
                    has_tts = hasattr(self.model, 'tts')

                    print(f"【Qwen3-TTS 合成】可用方法: generate_custom_voice={has_generate_custom_voice}, generate_voice_design={has_generate_voice_design}, generate={has_generate}, tts={has_tts}")

                    with torch.no_grad():
                        audio = None

                        voice_id_to_name = {v: k for k, v in self.VOICE_MAP.items()}
                        speaker_name = voice_id_to_name.get(speaker_id, "Vivian")
                        print(f"【Qwen3-TTS 合成】speaker_id={speaker_id} -> speaker_name={speaker_name}")

                        # 方式1: generate_custom_voice (CustomVoice模型专用，最快)
                        if has_generate_custom_voice:
                            try:
                                print(f"【Qwen3-TTS 合成】使用generate_custom_voice方法...")
                                outputs = self.model.generate_custom_voice(
                                    text=text,
                                    language=language_code,
                                    speaker=speaker_name,
                                    speed=speed,
                                    pitch=pitch,
                                    volume=volume
                                )
                                if isinstance(outputs, dict):
                                    audio = outputs.get("audio") or outputs.get("waveform") or outputs.get("speech")
                                elif isinstance(outputs, tuple):
                                    audio = outputs[0]
                                    if len(outputs) > 1:
                                        sample_rate = outputs[1]
                                        print(f"【Qwen3-TTS 合成】从返回值获取采样率: {sample_rate}")
                                else:
                                    audio = outputs
                                if audio is not None:
                                    print(f"【Qwen3-TTS 合成】generate_custom_voice成功")
                            except Exception as e:
                                print(f"【Qwen3-TTS 合成】generate_custom_voice失败: {e}")

                        # 方式2: generate方法
                        if audio is None and has_generate:
                            try:
                                print(f"【Qwen3-TTS 合成】使用generate方法...")
                                outputs = self.model.generate(
                                    text=text,
                                    speaker_id=speaker_id,
                                    speed=speed,
                                    pitch=pitch,
                                    volume=volume
                                )
                                if isinstance(outputs, dict):
                                    audio = outputs.get("audio") or outputs.get("waveform") or outputs.get("speech")
                                else:
                                    audio = outputs
                                if audio is not None:
                                    print(f"【Qwen3-TTS 合成】generate方法成功")
                            except Exception as e:
                                print(f"【Qwen3-TTS 合成】generate方法失败: {e}")

                        # 方式3: tts方法
                        if audio is None and has_tts:
                            try:
                                print(f"【Qwen3-TTS 合成】使用tts方法...")
                                outputs = self.model.tts(
                                    text=text,
                                    speaker_id=speaker_id,
                                    speed=speed
                                )
                                if isinstance(outputs, dict):
                                    audio = outputs.get("audio") or outputs.get("waveform") or outputs.get("speech")
                                else:
                                    audio = outputs
                                if audio is not None:
                                    print(f"【Qwen3-TTS 合成】tts方法成功")
                            except Exception as e:
                                print(f"【Qwen3-TTS 合成】tts方法失败: {e}")

                        # 方式4: generate_voice_design (VoiceDesign模型专用，较慢)
                        if audio is None and has_generate_voice_design:
                            try:
                                print(f"【Qwen3-TTS 合成】使用generate_voice_design方法...")
                                simple_instruct = self._get_voice_design_instruct(emotion, speed, pitch, volume, speaker_id)
                                outputs = self.model.generate_voice_design(
                                    text=text,
                                    instruct=simple_instruct,
                                    language=language_code
                                )
                                if isinstance(outputs, dict):
                                    audio = outputs.get("audio") or outputs.get("waveform") or outputs.get("speech")
                                elif isinstance(outputs, tuple):
                                    audio = outputs[0]
                                else:
                                    audio = outputs
                                if audio is not None:
                                    print(f"【Qwen3-TTS 合成】generate_voice_design成功")
                            except Exception as e:
                                print(f"【Qwen3-TTS 合成】generate_voice_design失败: {e}")

                    if audio is None:
                        print("【Qwen3-TTS 合成错误】所有调用方式都失败")
                        return None

                    # 转换为 numpy
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    elif isinstance(audio, list):
                        audio = np.array(audio, dtype=np.float32)

                    print(f"【Qwen3-TTS 合成】qwen-tts 生成完成，音频形状: {audio.shape}")
                    
                    # 转换为 numpy
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    elif isinstance(audio, list):
                        audio = np.array(audio, dtype=np.float32)
                    
                    print(f"【Qwen3-TTS 合成】qwen-tts 生成完成，音频形状: {audio.shape}")
                    
                except Exception as qwen_error:
                    print(f"【Qwen3-TTS 合成错误】qwen-tts 生成失败: {str(qwen_error)}")
                    import traceback
                    traceback.print_exc()
                    return None

            # 方式2: transformers 方式
            elif self.processor is not None and hasattr(self.model, 'generate'):
                # 使用 transformers 方式
                print(f"【Qwen3-TTS合成】使用 transformers 方式")
                
                try:
                    # 准备输入
                    inputs = self.processor(
                        text=text, 
                        speaker_id=speaker_id,
                        return_tensors="pt"
                    )
                    
                    # 移动到设备
                    if self.device == "cuda" and torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        print(f"【Qwen3-TTS合成】输入已移动到 GPU")
                    
                    # 生成音频
                    print(f"【Qwen3-TTS合成】开始生成...")
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            speed=speed,
                            pitch=pitch,
                            volume=volume
                        )
                    
                    # 提取音频数据
                    if isinstance(outputs, dict):
                        audio = outputs.get("audio") or outputs.get("waveform") or outputs.get("speech")
                    else:
                        audio = outputs
                    
                    if audio is None:
                        print("【Qwen3-TTS合成错误】模型输出中没有音频数据")
                        return None
                    
                    # 转换为 numpy
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    
                    print(f"【Qwen3-TTS合成】生成完成，音频形状: {audio.shape}")
                    
                except Exception as gen_error:
                    print(f"【Qwen3-TTS合成错误】transformers 生成失败: {str(gen_error)}")
                    import traceback
                    traceback.print_exc()
                    return None
                    
            # 方式3: ModelScope pipeline 方式
            elif hasattr(self.model, '__call__') and self.processor is None:
                # 使用 ModelScope pipeline 方式
                print(f"【Qwen3-TTS合成】使用 ModelScope pipeline 方式")
                
                try:
                    # 构建参数
                    gen_params = {
                        "text": text,
                        "speed": speed,
                        "pitch": pitch,
                        "volume": volume,
                    }
                    
                    # 添加参考音频（如果是 CustomVoice）
                    if ref_audio_path and os.path.exists(ref_audio_path):
                        gen_params["ref_audio"] = ref_audio_path
                        print(f"【Qwen3-TTS合成】使用参考音频: {ref_audio_path}")
                    # 添加预设音色（如果是 VoiceDesign）
                    elif speaker_id is not None:
                        gen_params["speaker"] = speaker_id
                    
                    # 调用 pipeline
                    print(f"【Qwen3-TTS合成】调用 pipeline...")
                    result = self.model(gen_params)
                    
                    # 提取音频数据
                    if isinstance(result, dict):
                        audio_data = result.get("audio") or result.get("waveform") or result.get("speech")
                        sample_rate = result.get("sample_rate", sample_rate)
                    else:
                        audio_data = result
                    
                    if audio_data is None:
                        print("【Qwen3-TTS合成错误】pipeline 未返回音频数据")
                        return None
                    
                    # 转换为 numpy
                    if isinstance(audio_data, list):
                        audio = np.array(audio_data, dtype=np.float32)
                    elif isinstance(audio_data, torch.Tensor):
                        audio = audio_data.cpu().numpy().astype(np.float32)
                    else:
                        audio = np.array(audio_data, dtype=np.float32)

                    print(f"【Qwen3-TTS合成】pipeline 生成完成，音频形状: {audio.shape}")
                    
                    print(f"【Qwen3-TTS合成】pipeline 生成完成，音频形状: {audio.shape}")
                    
                except Exception as pipe_error:
                    print(f"【Qwen3-TTS合成错误】pipeline 调用失败: {str(pipe_error)}")
                    import traceback
                    traceback.print_exc()
                    return None
            else:
                print(f"【Qwen3-TTS合成错误】无法确定合成方式")
                print(f"【Qwen3-TTS合成错误】模型有 generate: {hasattr(self.model, 'generate')}, 有 processor: {self.processor is not None}")
                print(f"【Qwen3-TTS合成错误】模型类型: {type(self.model).__name__}")
                print(f"【Qwen3-TTS合成错误】模型方法: {[m for m in dir(self.model) if not m.startswith('_')]}")
                return None

            # 后处理音频
            audio = audio.squeeze()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 归一化
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            elapsed_time = time.time() - start_time
            print(f"【Qwen3-TTS合成】合成完成，采样率: {sample_rate}Hz, 耗时: {elapsed_time:.2f}秒")
            return {"waveform": audio, "sample_rate": sample_rate}

        except Exception as e:
            print(f"【Qwen3-TTS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_kani_tts(self, text, speaker_id, speed, sample_rate):
        try:
            if self.model is None or self.processor is None:
                raise RuntimeError("模型未加载")

            inputs = self.processor(text=text, speaker_id=speaker_id, return_tensors="pt")
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, speed=speed)

            audio = outputs.get("audio", outputs.get("waveform", outputs)) if isinstance(outputs, dict) else outputs
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            audio = audio.squeeze()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return {"waveform": audio, "sample_rate": sample_rate}

        except Exception as e:
            print(f"【KaniTTS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_bark(self, text, speed, sample_rate):
        try:
            from bark import generate_audio
            audio_array = generate_audio(text, history_prompt=None)

            audio = np.array(audio_array) if not isinstance(audio_array, np.ndarray) else audio_array
            audio = audio.squeeze()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return {"waveform": audio, "sample_rate": sample_rate}

        except Exception as e:
            print(f"【Bark合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_vits(self, text, speed, sample_rate):
        try:
            print(f"【VITS合成】VITS模型合成需要自定义实现")
            return None
        except Exception as e:
            print(f"【VITS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_xtts(self, text, speed, sample_rate):
        try:
            if self.model is None:
                raise RuntimeError("模型未加载")

            wav = self.model.tts(text=text, speed=speed)
            audio = np.array(wav) if not isinstance(wav, np.ndarray) else wav

            audio = audio.squeeze()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return {"waveform": audio, "sample_rate": sample_rate}

        except Exception as e:
            print(f"【XTTS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_coqui(self, text, speed, sample_rate):
        try:
            if self.model is None:
                raise RuntimeError("模型未加载")

            wav = self.model.tts(text=text, speed=speed)
            audio = np.array(wav) if not isinstance(wav, np.ndarray) else wav

            audio = audio.squeeze()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return {"waveform": audio, "sample_rate": sample_rate}

        except Exception as e:
            print(f"【Coqui合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_supertonic(self, text, speed, sample_rate):
        try:
            print(f"【Supertonic合成】Supertonic模型合成需要自定义实现")
            return None
        except Exception as e:
            print(f"【Supertonic合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_supertonic_2(self, text, speed, sample_rate):
        try:
            print(f"【Supertonic-2合成】Supertonic-2模型合成需要自定义实现")
            return None
        except Exception as e:
            print(f"【Supertonic-2合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_gguf(self, text, speed, sample_rate):
        try:
            if self.model is None:
                raise RuntimeError("模型未加载")

            print(f"【GGUF TTS合成】开始合成音频...")

            # 检查模型是否支持 create_audio 方法
            if not hasattr(self.model, 'create_audio'):
                print("【GGUF TTS合成错误】模型不支持音频合成功能")
                return None

            # 默认参数
            voice = "default"
            pitch = 1.0

            # 生成音频
            audio_result = self.model.create_audio(
                text=text,
                speed=speed,
                voice=voice
            )

            # 处理音频数据
            samples = audio_result.get("samples")
            if samples is None:
                print("【GGUF TTS合成错误】未获取到音频数据")
                return None

            audio = np.array(samples, dtype=np.float32)
            audio = audio.squeeze()

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            return {"waveform": audio, "sample_rate": 24000}  # GGUF TTS 固定采样率

        except Exception as e:
            print(f"【GGUF TTS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _synthesize_pytorch_tts(self, text, speed, sample_rate):
        try:
            print(f"【PyTorch TTS合成】进入通用回退处理，尝试从已加载模型获取音频")

            if self.model is None:
                print("【PyTorch TTS合成】模型未加载，无法生成")
                return None

            # 如果加载的模型是HuggingFace Transformers Text-to-Speech，可尝试直接调用 generate
            if hasattr(self.model, 'generate'):
                print("【PyTorch TTS合成】检测到generate方法，尝试调用")
                with torch.no_grad():
                    outputs = self.model.generate(text)

                audio = None
                if isinstance(outputs, dict):
                    audio = outputs.get('audio') or outputs.get('waveform') or outputs.get('speech')
                else:
                    audio = outputs

                if audio is None:
                    print("【PyTorch TTS合成】generate未返回音频字段")
                    return None

                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                elif isinstance(audio, list):
                    audio = np.array(audio, dtype=np.float32)

                audio = np.array(audio, dtype=np.float32)
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)

                max_val = np.max(np.abs(audio)) if audio.size else 0
                if max_val > 0:
                    audio = audio / max_val

                return {"waveform": audio, "sample_rate": sample_rate}

            print("【PyTorch TTS合成】当前模型缺少generate方法，无法自动合成，请手动指定模型类型或实现自定义合成逻辑")
            return None

        except Exception as e:
            print(f"【PyTorch TTS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_model_info(self):
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.device,
            "sample_rate": self.config.get("sample_rate", 24000),
        }

    def synthesize(self, text, speaker_id=0, speed=1.0, emotion="default", language=None, pitch=0.0, volume=1.0, ref_audio_path=None):
        """统一的TTS合成接口"""
        try:
            print(f"【TTS合成】开始合成，文本: {text[:50]}...")
            print(f"【TTS合成】参数: speaker_id={speaker_id}, speed={speed}, emotion={emotion}, language={language}, pitch={pitch}, volume={volume}")

            if self.model is None:
                print("【TTS合成错误】模型未加载")
                return None

            # 根据模型类型调用对应的合成方法
            if self.model_type in ["qwen_tts", "Qwen3-TTS", "qwen3_tts"]:
                tts_language = language if language is not None else self.config.get("language", "中文")
                return self._synthesize_qwen_tts(text, speaker_id, speed, self.config.get("sample_rate", 24000), ref_audio_path, pitch, volume, emotion, tts_language)
            elif self.model_type in ["kani_tts", "KaniTTS"]:
                return self._synthesize_kani_tts(text, speaker_id, speed, self.config.get("sample_rate", 24000), ref_audio_path, pitch, volume)
            elif self.model_type == "bark":
                return self._synthesize_bark(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "vits":
                return self._synthesize_vits(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "xtts":
                return self._synthesize_xtts(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "coqui":
                return self._synthesize_coqui(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "supertonic":
                return self._synthesize_supertonic(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "supertonic_2":
                return self._synthesize_supertonic_2(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "gguf_tts":
                return self._synthesize_gguf(text, self.config.get("sample_rate", 24000))
            elif self.model_type == "pytorch_tts":
                return self._synthesize_pytorch_tts(text, self.config.get("sample_rate", 24000))
            else:
                print(f"【TTS合成错误】不支持的模型类型: {self.model_type}")
                return None

        except Exception as e:
            print(f"【TTS合成错误】{str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _get_voice_design_instruct(self, emotion, speed, pitch, volume, speaker_id=None):
        """生成语音设计指令"""
        instruct_parts = []
        
        if speaker_id is not None:
            speaker_desc_map = {
                0: "明亮略带锐气的年轻女声",
                1: "温暖柔和的年轻女声",
                2: "音色低沉醇厚的成熟男声",
                3: "清晰自然的北京青年男声",
                4: "活泼略带沙哑明亮感的成都男声",
                5: "富有节奏感的动态男声",
                6: "清晰中频的阳光美式男声",
                7: "轻快灵活的俏皮日语女声",
                8: "富含情感的温暖韩语女声"
            }
            speaker_desc = speaker_desc_map.get(speaker_id, "")
            if speaker_desc:
                instruct_parts.append(f"音色: {speaker_desc}")
        
        if emotion and emotion != "default":
            emotion_map = {
                "default": "中性",
                "happy": "开心",
                "sad": "悲伤",
                "angry": "愤怒",
                "excited": "兴奋",
                "calm": "平静",
                "surprised": "惊讶",
                "gentle": "温柔"
            }
            emotion_desc = emotion_map.get(emotion, emotion)
            instruct_parts.append(f"情绪: {emotion_desc}")
        
        if speed != 1.0:
            if speed > 1.0:
                instruct_parts.append(f"语速: 快速")
            else:
                instruct_parts.append(f"语速: 慢速")
        
        if pitch != 0.0:
            if pitch > 0.0:
                instruct_parts.append(f"音调: 高")
            else:
                instruct_parts.append(f"音调: 低")
        
        if volume != 1.0:
            if volume > 1.0:
                instruct_parts.append(f"音量: 大")
            else:
                instruct_parts.append(f"音量: 小")
        
        if not instruct_parts:
            return "请自然地朗读文本"
        
        return "，".join(instruct_parts) + "。请自然地朗读文本。"
