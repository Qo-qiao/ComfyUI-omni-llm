# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Forced Aligner Model Loader Node

独立的强制对齐模型加载节点，支持 Qwen3-ForcedAligner 模型

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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import folder_paths

# 自动检测并安装缺失的依赖
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

def _ensure_qwen_asr_installed():
    """确保 qwen-asr 已安装"""
    try:
        from qwen_asr import Qwen3ForcedAlignerModel
        return True
    except ImportError:
        try:
            from qwen_asr import Qwen3ForcedAligner
            return True
        except ImportError:
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
        # Qwen3-ForcedAligner模型文件
        "qwen_forced_aligner": {
            "modelscope": {
                "model.safetensors": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ForcedAligner-0.6B/snapshots/master/model.safetensors",
                "config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ForcedAligner-0.6B/snapshots/master/config.json",
                "tokenizer_config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ForcedAligner-0.6B/snapshots/master/tokenizer_config.json",
                "preprocessor_config.json": f"https://modelscope.cn/api/v1/models/qwen/Qwen3-ForcedAligner-0.6B/snapshots/master/preprocessor_config.json"
            },
            "huggingface": {
                "model.safetensors": f"https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/resolve/main/model.safetensors",
                "config.json": f"https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/resolve/main/config.json",
                "tokenizer_config.json": f"https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/resolve/main/tokenizer_config.json",
                "preprocessor_config.json": f"https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/resolve/main/preprocessor_config.json"
            }
        }
    }
    
    # 根据模型名称判断模型类型
    model_type = None
    if "forcedaligner" in model_name.lower() or "forced_aligner" in model_name.lower():
        model_type = "qwen_forced_aligner"
    
    if model_type and model_type in model_files:
        files_to_download = model_files[model_type][source]
        success = True
        
        for filename, url in files_to_download.items():
            save_path = os.path.join(model_path, filename)
            if not os.path.exists(save_path):
                if not _download_file(url, save_path):
                    success = False
        
        return success
    else:
        print(f"【模型下载】未知模型类型: {model_name}")
        return False

def _check_forced_aligner_model_files(model_path):
    """检查强制对齐模型必备文件"""
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json"
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    return missing_files

class forced_aligner_loader:
    """强制对齐模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 检查LLM文件夹是否存在
        try:
            has_llm_folder = "LLM" in folder_paths.folder_names_and_paths
        except Exception:
            has_llm_folder = False

        # 所有强制对齐模型 - 只加载文件夹，不加载文件
        forced_aligner_list = ["None"]
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

                            # 获取目录名称
                            dir_name = os.path.basename(root).lower()

                            # 检查是否是包含ForcedAligner关键词的目录
                            is_aligner_dir = any(keyword in dir_name for keyword in ["forcedaligner", "forced_aligner"])

                            if is_aligner_dir:
                                # 检查目录中是否包含模型文件
                                has_model_files = any(os.path.splitext(f)[1].lower() in [".safetensors", ".bin", ".gguf", ".pt", ".pth"] for f in files)

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
                                        forced_aligner_list.append(folder_entry)
                                        print(f"【强制对齐模型检测】检测到强制对齐目录: {folder_entry}")

                            # 检查非ForcedAligner目录中是否包含模型文件（用于检测包含模型文件的文件夹）
                            has_model_files_in_dir = any(os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".pt", ".pth", ".bin"] for f in files)
                            if has_model_files_in_dir:
                                # 检查文件夹名称或路径中是否包含ForcedAligner关键词
                                folder_path_lower = root.lower()
                                folder_name_lower = dir_name.lower()
                                if any(keyword in folder_path_lower or keyword in folder_name_lower for keyword in ["forcedaligner", "forced_aligner"]):
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
                                        forced_aligner_list.append(folder_entry)
                                        print(f"【强制对齐模型检测】检测到包含强制对齐模型文件的目录: {folder_entry}")
                    except Exception as e:
                        print(f"【强制对齐模型检测】扫描文件夹 {folder} 失败: {e}")
            except Exception as e:
                print(f"【强制对齐模型检测】获取LLM文件列表失败: {e}")
                pass

        # 如果没有找到强制对齐模型，显示提示
        if len(forced_aligner_list) == 1:
            forced_aligner_list = ["None", "请将Qwen3-ForcedAligner-0.6B模型放入models/LLM文件夹"]

        return {
            "required": {
                "forced_aligner_model": (forced_aligner_list, {"tooltip": "选择强制对齐模型文件\n• Qwen3-ForcedAligner-0.6B：阿里通义千问强制对齐模型"}),
                "n_gpu_layers": ("INT", {"default": 0, "min": -1, "max": 1000, "step": 1, "tooltip": "加载到GPU的模型层数，-1=全部加载"}),
            },
            "optional": {
                "precision": (["float16", "bfloat16", "float32"], {"default": "float16", "tooltip": "模型精度选择"}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1, "tooltip": "批处理大小"}),
                "flash_attention": ("BOOLEAN", {"default": False, "tooltip": "启用FlashAttention2优化（需要安装flash-attn）"}),
            }
        }
    
    RETURN_TYPES = ("ALIGNERMODEL",)
    RETURN_NAMES = ("aligner_model",)
    FUNCTION = "load_forced_aligner_model"
    CATEGORY = "llama-cpp-vlm"
    
    @classmethod
    def _resolve_forced_aligner_model_path(s, forced_aligner_model):
        key = forced_aligner_model.rstrip('/')
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
    
    @classmethod
    def IS_CHANGED(s, forced_aligner_model, n_gpu_layers=0, precision="float16", batch_size=32, flash_attention=False):
        resolved_path = s._resolve_forced_aligner_model_path(forced_aligner_model)
        
        # 使用有序字典确保键顺序一致
        from collections import OrderedDict
        config = OrderedDict([
            ("forced_aligner_model", forced_aligner_model),
            ("forced_aligner_model_path", resolved_path or ""),
            ("n_gpu_layers", n_gpu_layers),
            ("precision", precision),
            ("batch_size", batch_size),
            ("flash_attention", flash_attention),
        ])
        
        result = json.dumps(config, ensure_ascii=False)
        print(f"【IS_CHANGED】模型: {forced_aligner_model}, 解析路径: {resolved_path}, 配置哈希: {hash(result) % 10000}")
        return result
    
    def __init__(self):
        self.loaded_model = None
        self.current_config = None
    
    def load_forced_aligner_model(self, forced_aligner_model, n_gpu_layers=0, precision="float16", batch_size=32, flash_attention=False):
        if forced_aligner_model == "None" or forced_aligner_model == "请将Qwen3-ForcedAligner-0.6B模型放入models/LLM文件夹":
            print("【强制对齐模型加载器】未选择强制对齐模型")
            return (None,)
        
        resolved_model_path = self._resolve_forced_aligner_model_path(forced_aligner_model)
        if not resolved_model_path:
            raise RuntimeError(f"无法解析强制对齐模型路径: {forced_aligner_model}")
        
        try:
            print(f"【强制对齐模型加载器】正在加载强制对齐模型: {forced_aligner_model} (解析路径: {resolved_model_path})")
            
            # 使用有序字典确保键顺序一致
            from collections import OrderedDict
            config = OrderedDict([
                ("forced_aligner_model", forced_aligner_model),
                ("n_gpu_layers", n_gpu_layers),
                ("precision", precision),
                ("batch_size", batch_size),
                ("flash_attention", flash_attention),
            ])
            
            # 缓存检查 - 添加详细调试信息
            if self.loaded_model is not None and self.current_config is not None:
                print(f"【强制对齐模型加载器】缓存检查:")
                print(f"  当前模型: {self.current_config.get('forced_aligner_model', 'None')}")
                print(f"  请求模型: {config.get('forced_aligner_model', 'None')}")
                print(f"  配置相同: {self.current_config == config}")
                
                if self.current_config == config:
                    print("【强制对齐模型加载器】使用缓存的强制对齐模型")
                    return (self.loaded_model,)
                else:
                    print("【强制对齐模型加载器】配置已改变，重新加载模型")
                    # 打印配置差异
                    for key in config:
                        if key in self.current_config:
                            if self.current_config[key] != config[key]:
                                print(f"  差异 - {key}: {self.current_config[key]} -> {config[key]}")
                        else:
                            print(f"  新增 - {key}: {config[key]}")
            else:
                print(f"【强制对齐模型加载器】无缓存或缓存为空，加载新模型")
            
            # 检查LLM文件夹是否存在
            try:
                has_llm_folder = "LLM" in folder_paths.folder_names_and_paths
            except Exception:
                has_llm_folder = False
            
            if not has_llm_folder:
                raise RuntimeError(f"LLM文件夹不存在，请在models目录下创建LLM文件夹并放入强制对齐模型")
            
            # 使用解析后的绝对路径
            model_path = resolved_model_path
            
            if model_path is None or not os.path.exists(model_path):
                raise RuntimeError(f"强制对齐模型文件夹不存在: {forced_aligner_model} (解析路径：{resolved_model_path})")
            
            # 确保是文件夹路径
            if not os.path.isdir(model_path):
                # 如果是文件路径，获取其所在目录
                model_path = os.path.dirname(model_path)
                print(f"【强制对齐模型加载器】检测到文件路径，自动使用所在目录: {model_path}")
            
            print(f"【强制对齐模型加载器】模型路径: {model_path}")
            print(f"【强制对齐模型加载器】运行模式: {'GPU' if torch.cuda.is_available() else 'CPU'}, GPU层数: {n_gpu_layers}")
            print(f"【强制对齐模型加载器】精度: {precision}")
            
            # 检查必备文件
            print(f"【强制对齐模型加载器】检查必备文件...")
            missing_files = _check_forced_aligner_model_files(model_path)
            
            if missing_files:
                print(f"【强制对齐模型加载器】检测到缺失文件: {missing_files}")
                print(f"【强制对齐模型加载器】尝试自动下载缺失文件...")
                
                # 尝试下载缺失文件
                if _download_model_files(forced_aligner_model, model_path):
                    print(f"【强制对齐模型加载器】缺失文件下载成功")
                else:
                    print(f"【强制对齐模型加载器警告】部分文件下载失败，尝试继续加载")
            
            # 检查qwen-asr库是否可用
            qwen_asr_available = _ensure_qwen_asr_installed()
            print(f"【强制对齐模型加载器】检查qwen-asr库状态: {'可用' if qwen_asr_available else '不可用'}")
            
            # 加载模型
            if not qwen_asr_available:
                print("【强制对齐模型加载器】qwen-asr库未安装，请运行: pip install -U qwen-asr")
                return (None,)
            
            try:
                print(f"【强制对齐模型加载器】成功导入qwen_asr模块")

                # 兼容不同版本类名
                try:
                    from qwen_asr import Qwen3ForcedAlignerModel as aligner_cls
                except ImportError:
                    from qwen_asr import Qwen3ForcedAligner as aligner_cls

                # 准备模型参数（简化，只传递必要参数）
                model_kwargs = {}
                
                print(f"【Qwen3-ForcedAligner】正在加载模型...")
                print(f"【Qwen3-ForcedAligner】模型路径: {model_path}")
                print(f"【Qwen3-ForcedAligner】运行模式: {'GPU' if torch.cuda.is_available() else 'CPU'}")
                print(f"【Qwen3-ForcedAligner】精度: {precision}")
                print(f"【Qwen3-ForcedAligner】模型参数: {model_kwargs}")
                
                # 加载模型（优先from_pretrained）
                if hasattr(aligner_cls, 'from_pretrained'):
                    model = aligner_cls.from_pretrained(model_path, **model_kwargs)
                else:
                    model = aligner_cls(model=model_path, **model_kwargs)
                print(f"【Qwen3-ForcedAligner】模型加载成功")
                
                # 将模型移到指定设备
                if hasattr(model, 'to'):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)
                    print(f"【Qwen3-ForcedAligner】模型已移至设备: {device}")
                
                # 创建模型包装器
                self.loaded_model = ForcedAlignerModelWrapper(model, config)
                self.current_config = config
                
                # 检查模型是否成功加载
                if self.loaded_model.model is None:
                    print("【强制对齐模型加载器错误】模型包装器创建成功，但内部模型加载失败")
                    return (None,)
                
                print(f"【强制对齐模型加载器】强制对齐模型加载成功！")
                print(f"【强制对齐模型加载器】📁 模型路径：{resolved_model_path}")
                print(f"【强制对齐模型加载器】⚙️  精度：{precision}")
                print(f"【强制对齐模型加载器】🖥️  设备模式：{'GPU' if torch.cuda.is_available() else 'CPU'}")
                
                return (self.loaded_model,)
            except ImportError as e:
                print(f"【强制对齐模型加载器】导入qwen-asr库失败: {str(e)}")
                print("【强制对齐模型加载器】请运行: pip install -U qwen-asr")
                return (None,)
            except Exception as e:
                print(f"【强制对齐模型加载器】加载Qwen3ForcedAlignerModel失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return (None,)
            
        except Exception as e:
            print(f"【强制对齐模型加载器错误】加载模型失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 重置缓存状态，避免失败的模型影响后续加载
            self.loaded_model = None
            self.current_config = None
            print("【强制对齐模型加载器】已重置缓存状态")
            return (None,)

class ForcedAlignerModelWrapper:
    """强制对齐模型包装器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model_type = "qwen_forced_aligner"
    
    def align(self, audio_data, text, sample_rate=16000, language="zh"):
        """执行强制对齐
        
        Args:
            audio_data: 音频数据（numpy数组或torch张量）
            text: 要对齐的文本
            sample_rate: 音频采样率
            language: 文本语言，默认为中文
            
        Returns:
            对齐结果，包含时间戳信息
        """
        try:
            print(f"【Qwen3-ForcedAligner】开始强制对齐，文本长度: {len(text) if text else 0}")
            
            # 将音频数据转换为numpy数组
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            elif isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            
            # 确保音频数据是一维的
            if isinstance(audio_data, np.ndarray):
                if audio_data.ndim > 1:
                    if audio_data.ndim == 2:
                        audio_data = audio_data.mean(axis=0)  # 转为单声道
                    elif audio_data.ndim == 3:
                        audio_data = audio_data.squeeze()
                        if audio_data.ndim == 2:
                            audio_data = audio_data.mean(axis=0)
                # 确保是float32类型
                if audio_data.dtype != np.float32:
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_data = audio_data.astype(np.float32)
            
            # qwen-asr库期望的格式是 (waveform, sample_rate) 元组
            # 或者文件路径字符串
            audio_input = (audio_data, sample_rate)
            
            # 执行对齐（Qwen3ForcedAligner需要language参数）
            result = self.model.align(
                audio=audio_input,
                text=text,
                language=language
            )
            
            print(f"【Qwen3-ForcedAligner】强制对齐完成")
            return result
            
        except Exception as e:
            print(f"【Qwen3-ForcedAligner错误】强制对齐失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# 节点映射
NODE_CLASS_MAPPINGS = {
    "forced_aligner_loader": forced_aligner_loader,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "forced_aligner_loader": "强制对齐模型加载器",
}