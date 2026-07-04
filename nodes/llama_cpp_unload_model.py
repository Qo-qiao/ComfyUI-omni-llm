# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Unload Model Node

模型卸载节点，用于清理LLM、TTS、ASR和强制对齐模型占用的资源，释放内存和显存

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.model_management as mm
from common import LLAMA_CPP_STORAGE

class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": ("*",)}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "omni-llm"

    def process(self, any):
        print("【资源释放】开始全面清理模型资源...")
        
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
        
        # 1. 调用LLAMA_CPP_STORAGE的clean方法 - 这是核心清理
        from common import LLAMA_CPP_STORAGE
        LLAMA_CPP_STORAGE.clean(all=True)
        print("【资源释放】已调用LLAMA_CPP_STORAGE.clean(all=True)")
        
        # 2. 清理MODEL_CACHE和RESULT_CACHE
        try:
            from common import MODEL_CACHE, RESULT_CACHE
            # 清理MODEL_CACHE中的每个模型
            for key in list(MODEL_CACHE.keys()):
                model = MODEL_CACHE[key]
                # 尝试调用close/free/unload等方法
                for method_name in ["close", "free", "unload", "cleanup"]:
                    if hasattr(model, method_name):
                        try:
                            getattr(model, method_name)()
                            print(f"【缓存清理】已调用MODEL_CACHE[{key}].{method_name}()")
                        except Exception as e:
                            print(f"【缓存清理】调用MODEL_CACHE[{key}].{method_name}()失败: {e}")
                del MODEL_CACHE[key]
            print(f"【缓存清理】已清理MODEL_CACHE")
            # 清理RESULT_CACHE
            RESULT_CACHE.clear()
            print("【缓存清理】已清理RESULT_CACHE")
        except Exception as e:
            print(f"【缓存清理】清理缓存失败: {e}")
        
        # 3. 清理TTS/ASR/强制对齐模型
        self._clean_custom_models()
        
        # 4. 第一次强制垃圾回收
        print("【内存清理】执行第一次垃圾回收...")
        gc.collect()
        
        # 5. 清理GPU显存（按照官方方式）
        self._clean_gpu_memory()
        
        # 6. 第二次强制垃圾回收
        print("【内存清理】执行第二次垃圾回收...")
        gc.collect()
        
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

    def _clean_llama_cpp_model(self):
        """清理 llama.cpp 模型，使用 llama.cpp 官方卸载方式"""
        try:
            if LLAMA_CPP_STORAGE.llm is not None:
                llm = LLAMA_CPP_STORAGE.llm
                
                # 步骤1: 清理 Qwen3.5 特殊内存（如果是Qwen3.5模型）
                try:
                    if hasattr(llm, '_ctx') and hasattr(llm._ctx, 'memory_clear'):
                        llm._ctx.memory_clear(True)
                        print("【LLM清理】已调用 ctx.memory_clear(True)")
                    if hasattr(llm, 'is_hybrid') and llm.is_hybrid:
                        if hasattr(llm, '_hybrid_cache_mgr') and llm._hybrid_cache_mgr is not None:
                            llm._hybrid_cache_mgr.clear()
                            print("【LLM清理】已清理混合缓存管理器")
                    if hasattr(llm, 'n_tokens'):
                        llm.n_tokens = 0
                except Exception as e:
                    print(f"【LLM清理】Qwen3.5特殊清理失败（忽略）: {e}")
                
                # 步骤2: 尝试 llama.cpp 官方卸载方法
                # llama-cpp-python 的正确卸载顺序
                unload_methods = [
                    ('close', lambda x: x.close()),
                    ('free', lambda x: x.free()),
                    ('unload', lambda x: x.unload()),
                ]
                
                success = False
                for method_name, method_func in unload_methods:
                    if hasattr(llm, method_name):
                        try:
                            method_func(llm)
                            print(f"【LLM清理】已调用 {method_name}()")
                            success = True
                            break
                        except Exception as e:
                            print(f"【LLM清理】调用 {method_name}() 失败（尝试下一个）: {e}")
                
                # 步骤3: 如果都失败，尝试直接删除模型对象
                if not success:
                    print("【LLM清理】未找到合适的卸载方法，将通过del强制删除")
                
                # 步骤4: 强制设置为None
                LLAMA_CPP_STORAGE.llm = None
                print("【LLM清理】LLM模型引用已置空")
                
                # 步骤5: 立即进行一次垃圾回收
                gc.collect()
                
        except Exception as e:
            print(f"【LLM清理】清理失败: {e}")

    def _clean_storage_state(self):
        """清理存储状态"""
        try:
            if LLAMA_CPP_STORAGE.chat_handler is not None:
                LLAMA_CPP_STORAGE.chat_handler = None
                print("【LLM清理】ChatHandler已释放")

            # 清理所有存储项
            LLAMA_CPP_STORAGE.current_config = None
            LLAMA_CPP_STORAGE.model_path = None
            LLAMA_CPP_STORAGE.model_name = None
            LLAMA_CPP_STORAGE.messages.clear()
            LLAMA_CPP_STORAGE.sys_prompts.clear()
            print("【LLM清理】存储数据已清空")
            
        except Exception as e:
            print(f"【LLM清理】清理存储失败: {e}")

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

    def _clean_custom_models(self):
        """清理TTS、ASR和强制对齐模型"""
        
        # 清理TTS模型
        try:
            model_cache = self._get_module_cache("llama_cpp_tts_loader", "model_cache")
            if model_cache is None:
                print("【TTS清理】找不到TTS模型缓存，跳过清理")
            else:
                model_count = len(model_cache)
                for key in list(model_cache.keys()):
                    model_wrapper = model_cache[key]
                    if model_wrapper is not None:
                        # 调用release方法
                        if hasattr(model_wrapper, 'release'):
                            try:
                                model_wrapper.release()
                            except Exception:
                                pass
                        else:
                            # 手动释放
                            self._release_model_object(model_wrapper)
                    del model_cache[key]
                
                print(f"【TTS清理】TTS模型缓存已清空（共{model_count}个模型）")
                
        except Exception as e:
            print(f"【TTS清理】清理失败: {e}")

        # 清理ASR模型
        try:
            asr_model_cache = self._get_module_cache("llama_cpp_asr_loader", "asr_model_cache")
            if asr_model_cache is None:
                print("【ASR清理】找不到ASR模型缓存，跳过清理")
            else:
                model_count = len(asr_model_cache)
                for key in list(asr_model_cache.keys()):
                    asr_wrapper = asr_model_cache[key]
                    if asr_wrapper is not None:
                        if hasattr(asr_wrapper, 'release'):
                            try:
                                asr_wrapper.release()
                            except Exception:
                                pass
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
                
        except Exception as e:
            print(f"【ASR清理】清理失败: {e}")

        # 清理强制对齐模型
        try:
            aligner_model_cache = self._get_module_cache("forced_aligner_loader", "aligner_model_cache")
            if aligner_model_cache is None:
                print("【强制对齐清理】找不到强制对齐模型缓存，跳过清理")
            else:
                model_count = len(aligner_model_cache)
                for key in list(aligner_model_cache.keys()):
                    aligner_wrapper = aligner_model_cache[key]
                    if aligner_wrapper is not None:
                        if hasattr(aligner_wrapper, 'release'):
                            try:
                                aligner_wrapper.release()
                            except Exception:
                                pass
                        else:
                            # 手动释放
                            if hasattr(aligner_wrapper, 'model') and aligner_wrapper.model is not None:
                                if hasattr(aligner_wrapper.model, 'to'):
                                    try:
                                        aligner_wrapper.model = aligner_wrapper.model.to('cpu')
                                    except Exception:
                                        pass
                                del aligner_wrapper.model
                                aligner_wrapper.model = None
                    del aligner_model_cache[key]
                
                print(f"【强制对齐清理】强制对齐模型缓存已清空（共{model_count}个模型）")
                
        except Exception as e:
            print(f"【强制对齐清理】清理失败: {e}")
    
    def _release_model_object(self, model):
        """释放模型对象资源"""
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
            if hasattr(model, 'tts_engine'):
                model.tts_engine = None
            
        except Exception as e:
            print(f"【资源释放】释放模型时出错: {e}")

    def _force_garbage_collection(self):
        """强制垃圾回收（多次）"""
        # 连续3次垃圾回收确保彻底清理
        for i in range(3):
            collected = gc.collect()
            if collected > 0 or i == 0:
                print(f"【内存清理】第{i+1}次垃圾回收，回收了 {collected} 个对象")

    def _clean_gpu_memory(self):
        """清理GPU显存（按照ComfyUI-Easy-Use方式）"""
        try:
            import torch
            
            # 1. 首先进行垃圾回收
            gc.collect()
            print("【显存清理】已调用gc.collect()")
            
            # 2. 同步CUDA（如果可用）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                print("【显存清理】已调用torch.cuda.synchronize()")
            
            # 3. 使用ComfyUI内置的模型管理功能
            mm.unload_all_models()
            print("【显存清理】已调用mm.unload_all_models()")
            
            # 4. 软清空缓存（官方方式）
            mm.soft_empty_cache()
            print("【显存清理】已调用mm.soft_empty_cache()")
            
            # 5. 再次进行垃圾回收
            gc.collect()
            
            # 6. 硬清空缓存和IPC
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("【显存清理】已调用torch.cuda.empty_cache()和ipc_collect()")
            
            print("【显存清理】GPU缓存清理完成")
            
        except Exception as e:
            print(f"【显存清理】清理GPU缓存时出错: {e}")
