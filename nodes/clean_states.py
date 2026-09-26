# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Clean States Node

模型清理节点，用于清理模型缓存和状态，释放内存和显存资源
支持LLM、ASR、TTS模型的卸载，以及卸载所有ComfyUI模型

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import LLAMA_CPP_STORAGE, clear_all_caches

class omni_llm_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": ("*",),
                "state_uid": ("INT", {"default": -1, "min": -1, "max": 999999, "step": 1}),
            },
            "optional": {
                "clean_llm": ("BOOLEAN", {"default": True, "tooltip": "清理LLM模型"}),
                "clean_asr": ("BOOLEAN", {"default": True, "tooltip": "清理ASR模型"}),
                "clean_tts": ("BOOLEAN", {"default": True, "tooltip": "清理TTS模型"}),
                "unload_all_comfyui_models": ("BOOLEAN", {"default": False, "tooltip": "卸载所有ComfyUI模型"}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "omni-llm"

    def process(self, any, state_uid, clean_llm=True, clean_asr=True, clean_tts=True, unload_all_comfyui_models=False):
        print("【资源释放】开始清理模型资源...")
        
        # 记录清理前的显存使用
        pre_allocated = 0
        pre_reserved = 0
        try:
            import torch
            if torch.cuda.is_available():
                pre_allocated = torch.cuda.memory_allocated() / 1024**3
                pre_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"【显存状态】清理前 - 已分配: {pre_allocated:.2f}GB, 已保留: {pre_reserved:.2f}GB")
        except Exception:
            pass
        
        # 清理会话状态
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        
        # 清理LLM模型
        if clean_llm:
            self._clean_llm_model()
        
        # 清理ASR模型
        if clean_asr:
            self._clean_asr_model()
        
        # 清理TTS模型
        if clean_tts:
            self._clean_tts_model()
        
        # 清理所有缓存
        clear_all_caches()
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理GPU显存（根据选项决定是否卸载所有ComfyUI模型）
        self._clean_gpu_memory(unload_all_comfyui_models)
        
        # 记录清理后的显存使用
        try:
            import torch
            if torch.cuda.is_available():
                post_allocated = torch.cuda.memory_allocated() / 1024**3
                post_reserved = torch.cuda.memory_reserved() / 1024**3
                freed = pre_allocated - post_allocated
                print(f"【显存状态】清理后 - 已分配: {post_allocated:.2f}GB, 已保留: {post_reserved:.2f}GB")
                print(f"【显存状态】释放显存: {max(0, freed):.2f}GB")
        except Exception:
            pass
        
        print("【资源释放】全面清理完成")
        return (any,)
    
    def _clean_llm_model(self):
        """清理LLM模型"""
        try:
            print("【LLM清理】开始清理LLM模型...")
            LLAMA_CPP_STORAGE.clean(all=True)
            print("【LLM清理】LLM模型已清理")
        except Exception as e:
            print(f"【LLM清理】清理失败: {e}")
    
    def _get_module_cache(self, module_name, cache_attr):
        """从sys.modules获取已加载模块的缓存字典"""
        # 尝试多个可能的模块名
        possible_names = [module_name, f"nodes.{module_name}"]
        for name in possible_names:
            if name in sys.modules:
                module = sys.modules[name]
                if hasattr(module, cache_attr):
                    return getattr(module, cache_attr)
        
        # 如果sys.modules中没有，尝试通过文件路径加载
        nodes_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(nodes_dir, f'{module_name}.py')
        if os.path.exists(file_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, cache_attr):
                return getattr(module, cache_attr)
        
        return None

    def _clean_asr_model(self):
        """清理ASR模型"""
        try:
            print("【ASR清理】开始清理ASR模型...")
            
            # 从sys.modules获取已加载的模块缓存
            asr_model_cache = self._get_module_cache("asr_loader", "asr_model_cache")
            if asr_model_cache is None:
                print("【ASR清理】找不到ASR模型缓存，跳过清理")
                return
            
            model_count = len(asr_model_cache)
            for key in list(asr_model_cache.keys()):
                asr_wrapper = asr_model_cache[key]
                if asr_wrapper is not None:
                    # 优先调用release方法
                    if hasattr(asr_wrapper, 'release'):
                        try:
                            asr_wrapper.release()
                            print(f"【ASR清理】已调用模型缓存[{key}].release()")
                        except Exception as e:
                            print(f"【ASR清理】调用release()失败: {e}")
                    else:
                        # 手动释放
                        if hasattr(asr_wrapper, 'model') and asr_wrapper.model is not None:
                            if hasattr(asr_wrapper.model, 'to'):
                                try:
                                    asr_wrapper.model = asr_wrapper.model.to('cpu')
                                except Exception:
                                    pass
                            asr_wrapper.model = None
                            asr_wrapper._audio_cache = {}
                del asr_model_cache[key]
            
            print(f"【ASR清理】ASR模型缓存已清空（共{model_count}个模型）")
            print("【ASR清理】ASR模型已清理")
        except Exception as e:
            print(f"【ASR清理】清理失败: {e}")

    def _clean_tts_model(self):
        """清理TTS模型"""
        try:
            print("【TTS清理】开始清理TTS模型...")

            tts_model_cache = self._get_module_cache("tts_loader", "tts_model_cache")
            if tts_model_cache is None:
                print("【TTS清理】找不到TTS模型缓存，跳过清理")
                return

            model_count = len(tts_model_cache)
            for key in list(tts_model_cache.keys()):
                tts_wrapper = tts_model_cache[key]
                if tts_wrapper is not None:
                    if hasattr(tts_wrapper, 'release'):
                        try:
                            tts_wrapper.release()
                            print(f"【TTS清理】已调用模型缓存[{key}].release()")
                        except Exception as e:
                            print(f"【TTS清理】调用release()失败: {e}")
                del tts_model_cache[key]

            print(f"【TTS清理】TTS模型缓存已清空（共{model_count}个模型）")
            print("【TTS清理】TTS模型已清理")
        except Exception as e:
            print(f"【TTS清理】清理失败: {e}")
    
    def _release_model(self, model):
        """释放模型资源"""
        try:
            # 尝试将模型移到CPU
            if hasattr(model, 'model') and model.model is not None:
                if hasattr(model.model, 'to'):
                    try:
                        model.model = model.model.to('cpu')
                    except Exception:
                        pass
            
            # 尝试各种卸载方法
            for method_name in ["close", "free", "unload", "clean", "release"]:
                if hasattr(model, method_name):
                    try:
                        getattr(model, method_name)()
                        break
                    except Exception:
                        pass
            
            # 删除模型引用
            if hasattr(model, 'model'):
                model.model = None
            if hasattr(model, 'processor'):
                model.processor = None
            
        except Exception as e:
            print(f"【资源释放】释放模型时出错: {e}")
    
    def _clean_gpu_memory(self, unload_all_comfyui_models=False):
        """清理GPU显存"""
        try:
            import torch
            import comfy.model_management as mm
            
            # 同步CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 根据选项决定是否卸载所有ComfyUI模型
            if unload_all_comfyui_models:
                print("【显存清理】卸载所有ComfyUI模型...")
                mm.unload_all_models()
                print("【显存清理】已调用mm.unload_all_models()")
            
            # 使用ComfyUI内置的模型管理功能
            mm.soft_empty_cache()
            print("【显存清理】已调用mm.soft_empty_cache()")
            
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("【显存清理】已调用torch.cuda.empty_cache()和ipc_collect()")
            
            print("【显存清理】GPU缓存清理完成")
            
        except Exception as e:
            print(f"【显存清理】清理GPU缓存时出错: {e}")

NODE_CLASS_MAPPINGS = {
    "omni_llm_clean_states": omni_llm_clean_states
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_clean_states": "Omni LLM Clean States"
}