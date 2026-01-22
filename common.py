# -*- coding: utf-8 -*-
"""
ComfyUI-llama-cpp-vlmforQo 公共组件
"""
import os
import io
import gc
import json
import base64
import random
import torch
import inspect
import numpy as np
import psutil
import platform
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# -------------------------- 自动导入依赖（缺失会提示） --------------------------
try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import (
        Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
        NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
    )
except ImportError as e:
    print(f"【错误】缺少llama-cpp-python依赖，请运行：pip install llama-cpp-python")
    exit(1)

try:
    import folder_paths
    import comfy.model_management as mm
    import comfy.utils
except ImportError as e:
    print(f"【错误】未检测到ComfyUI环境，请将该文件放入ComfyUI/custom_nodes/ComfyUI-llama-cpp-vlmforQo/目录下")
    exit(1)

# -------------------------- 硬件检测（保留提速优化） --------------------------
def get_hardware_info():
    hardware_info = {
        "has_cuda": torch.cuda.is_available(),
        "gpu_name": "未知显卡",
        "gpu_vram_total": 0.0,
        "cpu_cores": os.cpu_count() or 4,
        "is_high_perf": False,
        "is_low_perf": True
    }
    
    if hardware_info["has_cuda"]:
        try:
            device_prop = torch.cuda.get_device_properties(0)
            hardware_info["gpu_name"] = device_prop.name
            hardware_info["gpu_vram_total"] = round(device_prop.total_memory / (1024 ** 3), 2)
            
            high_perf_flags = ["40", "3090", "3080", "a100", "a10", "rtx 6000", "titan"]
            mid_perf_flags = ["30", "20", "1660", "1650 super"]
            if any(flag.lower() in hardware_info["gpu_name"].lower() for flag in high_perf_flags):
                hardware_info["is_high_perf"] = True
                hardware_info["is_low_perf"] = False
            elif any(flag.lower() in hardware_info["gpu_name"].lower() for flag in mid_perf_flags):
                hardware_info["is_high_perf"] = False
                hardware_info["is_low_perf"] = False
            print(f"【硬件检测】显卡：{hardware_info['gpu_name']}，显存：{hardware_info['gpu_vram_total']}GB")
        except Exception as e:
            print(f"【提示】显卡信息检测失败，自动使用兼容模式：{e}")
    else:
        print(f"【硬件检测】未检测到NVIDIA CUDA显卡，自动使用CPU/通用显卡兼容模式")
    
    print(f"【硬件检测】CPU核心数：{hardware_info['cpu_cores']}")
    return hardware_info

HARDWARE_INFO = get_hardware_info()

# -------------------------- 设置进程高优先级 --------------------------
def set_high_priority():
    try:
        p = psutil.Process()
        if platform.system() == "Windows":
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(max(-10, p.nice() - 5))
        print(f"【优化】已设置进程高优先级，提速更明显")
    except Exception as e:
        print(f"【提示】进程优先级设置失败（不影响核心功能）：{e}")

set_high_priority()

# -------------------------- 初始化ChatHandler（支持所有模型） --------------------------
chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", 
                 "llama3-Vision-Alpha", "MiniCPM-v2.6", "MiniCPM-v4", "llama-joycaption"]

# 动态导入各类ChatHandler
Gemma3ChatHandler = None
try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
    print(f"【模型支持】成功兼容Gemma3模型")
except ImportError:
    print(f"【模型支持】Gemma3模型暂不兼容（忽略，不影响其他功能）")

Qwen25VLChatHandler = None
try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL"]
    print(f"【模型支持】成功兼容Qwen2.5-VL模型")
except ImportError:
    print(f"【模型支持】Qwen2.5-VL模型暂不兼容（忽略，不影响其他功能）")

Qwen3VLChatHandler = None
try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Chat", "Qwen3-VL-Instruct"]
    print(f"【模型支持】成功兼容Qwen3-VL模型")
except ImportError as e:
    print(f"【模型支持】Qwen3-VL模型导入失败：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    # 尝试查看llama_cpp.llama_chat_format模块中可用的ChatHandler
    try:
        import llama_cpp.llama_chat_format
        available_handlers = [attr for attr in dir(llama_cpp.llama_chat_format) if attr.endswith('ChatHandler')]
        print(f"【调试】llama_cpp.llama_chat_format中可用的ChatHandler：{available_handlers}")
    except Exception as e2:
        print(f"【调试】查看可用ChatHandler失败：{e2}")

GLM46VChatHandler = None
try:
    from llama_cpp.llama_chat_format import GLM46VChatHandler
    chat_handlers += ["GLM-4.6V"]
    print(f"【模型支持】成功兼容GLM-4.6V模型")
except ImportError:
    print(f"【模型支持】GLM-4.6V模型暂不兼容（忽略，不影响其他功能）")

GLM41VChatHandler = None
try:
    from llama_cpp.llama_chat_format import GLM41VChatHandler
    chat_handlers += ["GLM-4.1V-Thinking"]
    print(f"【模型支持】成功兼容GLM-4.1V-Thinking模型")
except ImportError:
    print(f"【模型支持】GLM-4.1V-Thinking模型暂不兼容（忽略，不影响其他功能）")

LFM2VLChatHandler = None
try:
    from llama_cpp.llama_chat_format import LFM2VLChatHandler
    chat_handlers += ["LFM2-VL"]
    print(f"【模型支持】成功兼容LFM2-VL模型")
except ImportError:
    print(f"【模型支持】LFM2-VL模型暂不兼容（忽略，不影响其他功能）")

# -------------------------- 通用工具类 --------------------------
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    jpeg_quality = 70 if HARDWARE_INFO["is_low_perf"] else 75
    optimize = True if HARDWARE_INFO["is_high_perf"] else False
    img.save(buffered, format="JPEG", quality=jpeg_quality, optimize=optimize)
    img_base64 = base64.b64encode(buffered.getbuffer()).decode('utf-8')
    return img_base64

def scale_image(image: torch.Tensor, max_size: int = 128):
    try:
        img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        w, h = img_pil.size
        scale = min(max_size / max(w, h), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resample_mode = Image.Resampling.LANCZOS if HARDWARE_INFO["cpu_cores"] >= 8 else Image.Resampling.BILINEAR
            img_pil = img_pil.resize((new_w, new_h), resample=resample_mode)
        img_np = np.array(img_pil)
        return img_np
    except Exception as e:
        print(f"【错误】图片缩放失败：{e}")
        return np.array(Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)))

def parse_json(json_str):
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except Exception as e:
        print(f"【错误】解析JSON失败：{e}")
        return {}

def qwen3bbox(image, json_data):
    bbox_list = []
    img_np = np.array(image)
    h, w, _ = img_np.shape
    
    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and "bbox" in item:
                    bbox = item["bbox"]
                    label = item.get("label", "object")
                    x1, y1, x2, y2 = map(int, bbox)
                    bbox_list.append({"bbox_2d": [x1, y1, x2, y2], "label": label})
    except Exception as e:
        print(f"【错误】解析Qwen3边界框失败：{e}")
    
    return bbox_list

def draw_bbox(image, json_data, mode):
    try:
        img_np = np.array(image)
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        
        if mode == "Qwen3-VL":
            bbox_list = qwen3bbox(img_pil, json_data)
        elif mode == "simple":
            bbox_list = parse_json(json_data)
            if not isinstance(bbox_list, list):
                return img_np
        else:
            bbox_list = parse_json(json_data)
            if not isinstance(bbox_list, list):
                return img_np
        
        for bbox_item in bbox_list:
            if isinstance(bbox_item, dict) and "bbox_2d" in bbox_item:
                bbox = bbox_item["bbox_2d"]
                label = bbox_item.get("label", "object")
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 20), label, fill="red", font=None)
    except Exception as e:
        print(f"【错误】绘制边界框失败：{e}")
    
    return np.array(img_pil)

# -------------------------- 模型存储类（单例模式） --------------------------
class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    messages = {}
    sys_prompts = {}
    
    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
            print(f"【会话管理】已清理所有会话状态")
        else:
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)
            print(f"【会话管理】已清理会话状态 (id={id})")
    
    @classmethod
    def clean(cls, all=False):
        try:
            if cls.llm is not None:
                cls.llm.close()
                print(f"【资源释放】成功关闭LLM模型")
        except Exception as e:
            print(f"【提示】关闭LLM模型失败（忽略，继续释放资源）：{e}")
        
        if cls.chat_handler is not None:
            release_methods = [("_exit_stack", lambda x: x.close()), ("close", lambda x: x()), ("cleanup", lambda x: x())]
            for attr, func in release_methods:
                if hasattr(cls.chat_handler, attr):
                    try:
                        func(getattr(cls.chat_handler, attr))
                        print(f"【资源释放】成功释放ChatHandler资源")
                    except Exception as e:
                        print(f"【提示】释放ChatHandler资源失败（忽略）：{e}")
                    break
        
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        
        if all:
            cls.clean_state()
    
    @classmethod
    def get_chat_handler_cls(cls, chat_handler_name):
        try:
            if chat_handler_name == "None":
                return None
            elif chat_handler_name == "LLaVA-1.5":
                return Llava15ChatHandler
            elif chat_handler_name == "LLaVA-1.6":
                return Llava16ChatHandler
            elif chat_handler_name == "Moondream2":
                return MoondreamChatHandler
            elif chat_handler_name == "nanoLLaVA":
                return NanoLlavaChatHandler
            elif chat_handler_name == "llama3-Vision-Alpha":
                return Llama3VisionAlphaChatHandler
            elif chat_handler_name == "MiniCPM-v2.6":
                return MiniCPMv26ChatHandler
            elif chat_handler_name == "Gemma3" and Gemma3ChatHandler is not None:
                return Gemma3ChatHandler
            elif chat_handler_name == "Qwen2.5-VL" and Qwen25VLChatHandler is not None:
                return Qwen25VLChatHandler
            elif "Qwen3-VL" in chat_handler_name and Qwen3VLChatHandler is not None:
                return Qwen3VLChatHandler
            elif chat_handler_name == "GLM-4.6V" and GLM46VChatHandler is not None:
                return GLM46VChatHandler
            elif chat_handler_name == "GLM-4.1V-Thinking" and GLM41VChatHandler is not None:
                return GLM41VChatHandler
            elif chat_handler_name == "LFM2-VL" and LFM2VLChatHandler is not None:
                return LFM2VLChatHandler
            else:
                print(f"【提示】未找到匹配的ChatHandler：{chat_handler_name}，使用默认处理")
                return None
        except Exception as e:
            print(f"【错误】获取ChatHandler失败：{e}")
            return None
    
    @classmethod
    def init_chat_handler(cls, handler_cls, mmproj_path, chat_handler_name, image_max_tokens, image_min_tokens):
        if handler_cls is None:
            return None
        
        try:
            import inspect
            print(f"【调试】初始化ChatHandler：{handler_cls.__name__}")
            
            # 针对Qwen3-VL的特殊处理
            if "Qwen3-VL" in chat_handler_name:
                print(f"【调试】Qwen3-VL特殊处理")
                
                # 尝试多种备选方案
                backup_handlers = [
                    ("Qwen3VLChatHandler", handler_cls),
                    ("Llava15ChatHandler", None),
                    ("Llava16ChatHandler", None),
                    ("Llama3VisionAlphaChatHandler", None),
                    ("MiniCPMv26ChatHandler", None),
                    ("MoondreamChatHandler", None)
                ]
                
                for handler_name, backup_cls in backup_handlers:
                    try:
                        # 如果是第一个处理（Qwen3-VL），直接使用提供的handler_cls
                        # 否则，动态导入备份ChatHandler
                        if backup_cls is None:
                            from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler, MoondreamChatHandler
                            backup_cls = locals()[handler_name]
                        
                        if backup_cls is None:
                            continue
                            
                        print(f"【调试】尝试使用{backup_cls.__name__}初始化")
                        
                        # 构建初始化参数
                        init_params = {"verbose": False}
                        if mmproj_path:
                            # 检查参数名
                            try:
                                sig = inspect.signature(backup_cls.__init__)
                                if "mmproj" in sig.parameters:
                                    init_params["mmproj"] = mmproj_path
                                elif "clip_model_path" in sig.parameters:
                                    init_params["clip_model_path"] = mmproj_path
                                print(f"【调试】添加参数：{list(init_params.keys())}")
                            except Exception:
                                # 如果无法获取签名，尝试两种参数名
                                try:
                                    init_params["mmproj"] = mmproj_path
                                except Exception:
                                    init_params.pop("mmproj", None)
                                    init_params["clip_model_path"] = mmproj_path
                        
                        # 初始化ChatHandler
                        chat_handler = backup_cls(**init_params)
                        print(f"【成功】已使用{backup_cls.__name__}初始化ChatHandler！")
                        return chat_handler
                    except Exception as e:
                        print(f"【错误】{handler_name}初始化失败：{type(e).__name__}: {e}")
                        # 不打印完整堆栈，避免输出过多
                        continue
            
            # 通用ChatHandler初始化
            print(f"【调试】通用ChatHandler初始化")
            
            # 获取ChatHandler构造函数的参数签名
            sig = inspect.signature(handler_cls.__init__)
            init_params = {"verbose": False}
            
            # 检查并添加可用的参数
            if "clip_model_path" in sig.parameters or "mmproj" in sig.parameters:
                if mmproj_path:
                    # 根据不同ChatHandler的参数名选择合适的参数
                    if "clip_model_path" in sig.parameters:
                        init_params["clip_model_path"] = mmproj_path
                        print(f"【调试】添加参数：clip_model_path={mmproj_path}")
                    else:
                        init_params["mmproj"] = mmproj_path
                        print(f"【调试】添加参数：mmproj={mmproj_path}")
            
            if "image_max_tokens" in sig.parameters and "image_min_tokens" in sig.parameters:
                if image_max_tokens > 0:
                    init_params["image_max_tokens"] = image_max_tokens
                    print(f"【调试】添加参数：image_max_tokens={image_max_tokens}")
                if image_min_tokens > 0:
                    init_params["image_min_tokens"] = image_min_tokens
                    print(f"【调试】添加参数：image_min_tokens={image_min_tokens}")
            
            print(f"【调试】最终初始化参数：{init_params}")
            chat_handler = handler_cls(**init_params)
            print(f"【成功】通用ChatHandler初始化成功：{handler_cls.__name__}")
            return chat_handler
        except Exception as e:
            print(f"【错误】初始化ChatHandler失败：{e}")
            # 提供友好的错误信息和解决方案
            raise ValueError(f"【错误】初始化ChatHandler失败：{chat_handler_name}\n" \
                          f"可能的原因：\n" \
                          f"1. llama-cpp-python版本过低，不支持Qwen3-VL\n" \
                          f"2. MMProj文件与ChatHandler不兼容\n" \
                          f"3. 模型可能需要更新的llama-cpp-python版本\n" \
                          f"\n解决建议：\n" \
                          f"- 更新llama-cpp-python到最新版本\n" \
                          f"- 尝试使用其他ChatHandler（如LLaVA-1.5）\n" \
                          f"- 检查MMProj文件是否与模型匹配") from e
    
    @classmethod
    def load_model(cls, config):
        try:
            # 首先释放旧的模型资源，避免资源冲突
            print(f"【模型加载】开始加载新模型，正在释放旧资源...")
            cls.clean()
            
            model = config["model"]
            enable_mmproj = config["enable_mmproj"]
            mmproj = config["mmproj"]
            chat_handler_name = config["chat_handler"]
            n_ctx = config["n_ctx"]
            n_gpu_layers = config["n_gpu_layers"]
            vram_limit = config["vram_limit"]
            image_min_tokens = config["image_min_tokens"]
            image_max_tokens = config["image_max_tokens"]
            
            # 构建模型路径
            model_path = os.path.join(folder_paths.models_dir, 'LLM', model)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在：{model_path}")
            
            # 构建MMProj路径
            mmproj_path = None
            if enable_mmproj and mmproj != "None":
                mmproj_path = os.path.join(folder_paths.models_dir, 'LLM', mmproj)
                if not os.path.exists(mmproj_path):
                    raise FileNotFoundError(f"MMProj文件不存在：{mmproj_path}")
            
            # 获取模型格式
            model_ext = os.path.splitext(model)[1].lower()
            if model_ext not in [".gguf", ".gguf", ".safetensors"]:
                print(f"【提示】模型格式可能不支持：{model_ext}，将尝试加载")
            
            # 获取并初始化ChatHandler
            handler_cls = cls.get_chat_handler_cls(chat_handler_name)
            cls.chat_handler = cls.init_chat_handler(handler_cls, mmproj_path, chat_handler_name, image_max_tokens, image_min_tokens)
            
            if enable_mmproj:
                if chat_handler_name == "None":
                    raise ValueError(f"【错误】启用了MMProj但未选择ChatHandler，请选择对应的ChatHandler（如Qwen3-VL）")
                
                if handler_cls is None:
                    raise ValueError(f"【错误】无法找到ChatHandler：{chat_handler_name}，请确保选择了正确的ChatHandler")
                
                if cls.chat_handler is None:
                    # 尝试使用默认的视觉ChatHandler作为备选
                    print(f"【调试】尝试使用默认视觉ChatHandler作为备选...")
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        cls.chat_handler = Llava15ChatHandler(mmproj=mmproj_path, verbose=False)
                        print(f"【成功】已使用Llava15ChatHandler作为备选")
                    except Exception as e:
                        print(f"【错误】备选ChatHandler也初始化失败：{e}")
                        raise ValueError(f"【错误】初始化ChatHandler失败：{chat_handler_name}，请检查MMProj文件和ChatHandler是否兼容\n" \
                                        f"可能的原因：\n" \
                                        f"1. llama-cpp-python版本过低，不支持Qwen3-VL\n" \
                                        f"2. MMProj文件与ChatHandler不兼容\n" \
                                        f"3. ChatHandler类需要特定的参数配置") from e
                
                print(f"【成功】已启用MMProj并初始化ChatHandler：{type(cls.chat_handler).__name__}")
            
            # 加载LLM模型前的智能检查
            print(f"【模型加载】开始加载LLM模型：{model}（上下文：{n_ctx}，GPU层数：{n_gpu_layers}）")
            
            # 计算推荐的最大GPU层数
            recommended_gpu_layers = n_gpu_layers
            if HARDWARE_INFO["has_cuda"] and n_gpu_layers > 0:
                # 根据显存大小计算推荐的GPU层数
                gpu_vram = HARDWARE_INFO["gpu_vram_total"]
                
                # 估算模型在GPU中占用的显存（包括上下文）
                estimated_vram_usage = 0
                try:
                    model_file_size = os.path.getsize(model_path) / (1024 ** 3)  # GB
                    estimated_vram_usage = model_file_size * 1.8  # 考虑模型加载到显存后的膨胀和上下文开销
                    
                    if enable_mmproj and mmproj != "None":
                        mmproj_file_size = os.path.getsize(os.path.join(folder_paths.models_dir, 'LLM', mmproj)) / (1024 ** 3)
                        estimated_vram_usage += mmproj_file_size * 1.2
                    
                    # 预留1.5GB给系统和其他进程
                    available_vram = gpu_vram - 1.5
                    
                    if estimated_vram_usage > available_vram and n_gpu_layers == -1:
                        # 如果模型需要的显存超过可用显存，自动降低GPU层数
                        recommended_gpu_layers = int(n_gpu_layers * (available_vram / estimated_vram_usage))
                        recommended_gpu_layers = max(0, recommended_gpu_layers)
                        
                        print(f"【显存警告】模型预计需要{estimated_vram_usage:.2f}GB显存，可用{available_vram:.2f}GB")
                        print(f"【智能建议】建议将GPU层数调整为{recommended_gpu_layers}层")
                        print(f"【提示】您可以通过降低n_ctx、减少max_tokens或使用更小的模型来解决显存问题")
                except Exception as e:
                    print(f"【提示】显存估算失败，使用默认设置：{e}")
            
            # 构建模型参数
            llama_kwargs = {
                "model_path": model_path,
                "chat_handler": cls.chat_handler,
                "n_gpu_layers": recommended_gpu_layers,
                "n_ctx": n_ctx,
                "verbose": False,
                "n_threads": os.cpu_count() or 8,
                "n_threads_batch": os.cpu_count() or 16,
                "low_vram": HARDWARE_INFO["is_low_perf"],  # 低性能设备自动启用低显存模式
                "tensor_split": None,
            }
            
            # 尝试加载模型，失败时提供降级策略
            try:
                cls.llm = Llama(**llama_kwargs)
                print(f"【模型加载】LLM模型加载成功！（格式：{model_ext}）")
            except Exception as e:
                error_msg = str(e)
                
                # 分析错误类型，提供针对性建议
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    # 显存不足错误
                    print(f"【显存错误】加载模型失败：{error_msg}")
                    print(f"【智能建议】")
                    print(f"  1. 降低n_gpu_layers值（当前：{recommended_gpu_layers}）")
                    print(f"  2. 减少n_ctx值（当前：{n_ctx}）")
                    print(f"  3. 使用更小的模型或更高压缩率的量化版本")
                    print(f"  4. 关闭mmproj（如果不需要多模态功能）")
                    
                    # 尝试降级加载（纯CPU）
                    print(f"【尝试降级】尝试使用纯CPU模式加载模型...")
                    llama_kwargs["n_gpu_layers"] = 0
                    llama_kwargs["low_vram"] = True
                    
                    try:
                        cls.llm = Llama(**llama_kwargs)
                        print(f"【模型加载】LLM模型已使用纯CPU模式加载成功！")
                        print(f"【提示】CPU模式推理速度会较慢，建议使用更小的模型以提高速度")
                    except Exception as fallback_e:
                        raise RuntimeError(f"【错误】加载LLM模型失败（包括降级尝试）：{fallback_e}") from fallback_e
                else:
                    # 其他类型错误
                    raise RuntimeError(f"【错误】加载LLM模型失败：{e}") from e
            
            cls.current_config = config
        except Exception as e:
            print(f"【错误】加载模型失败：{e}")
            raise
