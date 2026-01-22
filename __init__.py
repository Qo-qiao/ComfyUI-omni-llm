# -*- coding: utf-8 -*-
"""
ComfyUI-llama-cpp-vlmforQo 节点包
"""

# 导入所有节点
from .llama_cpp_model_loader import llama_cpp_model_loader
from .llama_cpp_instruct_adv import llama_cpp_instruct_adv
from .llama_cpp_parameters import llama_cpp_parameters
from .llama_cpp_clean_states import llama_cpp_clean_states
from .llama_cpp_unload_model import llama_cpp_unload_model
from .json_to_bbox import json_to_bbox

# 节点映射关系，ComfyUI通过这个字典识别节点
NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_instruct_adv": llama_cpp_instruct_adv,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "json_to_bbox": json_to_bbox,
}

# 节点显示名称映射，在ComfyUI界面中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_instruct_adv": "Llama-cpp Image Inference",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "json_to_bbox": "JSON to Bounding Box",
}

# 导出所有映射关系
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
