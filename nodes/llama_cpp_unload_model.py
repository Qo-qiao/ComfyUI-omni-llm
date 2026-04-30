# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Unload Model Node

模型卸载节点，用于清理LLM、TTS和ASR模型占用的资源，释放内存和显存

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import LLAMA_CPP_STORAGE

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": ("*",)}}
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"
    
    def process(self, any):
        # 清理LLM模型和相关资源
        LLAMA_CPP_STORAGE.clean()
        
        # 清理TTS模型
        try:
            from .llama_cpp_tts_loader import llama_cpp_tts_loader
            # 获取所有TTS加载器实例并清理
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, llama_cpp_tts_loader):
                    if hasattr(obj, 'loaded_model') and obj.loaded_model:
                        # 调用模型的clean方法（如果有）
                        if hasattr(obj.loaded_model, 'clean'):
                            obj.loaded_model.clean()
                        obj.loaded_model = None
                        print(f"【资源释放】已清理TTS模型")
        except Exception as e:
            print(f"【提示】清理TTS模型失败（忽略）：{e}")
        
        # 清理ASR模型
        try:
            from .llama_cpp_asr_loader import llama_cpp_asr_loader
            # 获取所有ASR加载器实例并清理
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, llama_cpp_asr_loader):
                    if hasattr(obj, 'loaded_model') and obj.loaded_model:
                        # 调用模型的clean方法（如果有）
                        if hasattr(obj.loaded_model, 'clean'):
                            obj.loaded_model.clean()
                        obj.loaded_model = None
                        print(f"【资源释放】已清理ASR模型")
        except Exception as e:
            print(f"【提示】清理ASR模型失败（忽略）：{e}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 同步GPU操作并清空缓存
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保所有GPU操作完成
            torch.cuda.empty_cache()  # 清空GPU缓存
            print("【资源释放】已清理GPU缓存")
        else:
            print("【资源释放】CUDA不可用，仅清理CPU内存")
        
        return (any,)
