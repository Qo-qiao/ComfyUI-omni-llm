# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm 节点包

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

import sys
import os

plugin_dir = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录和nodes目录到路径
sys.path.insert(0, plugin_dir)
nodes_dir = os.path.join(plugin_dir, "nodes")
sys.path.insert(0, nodes_dir)

# 添加插件自带的site-packages目录到路径（优先使用插件自带的依赖，避免版本冲突）
site_packages_dir = os.path.join(plugin_dir, "site-packages")
if os.path.exists(site_packages_dir):
    sys.path.insert(0, site_packages_dir)

# 导入所有节点模块
from llama_cpp_model_loader import llama_cpp_model_loader
from llama_cpp_unified_inference import llama_cpp_unified_inference
from llama_cpp_parameters import llama_cpp_parameters
from llama_cpp_clean_states import llama_cpp_clean_states
from llama_cpp_unload_model import llama_cpp_unload_model
from llama_cpp_tts_loader import llama_cpp_tts_loader
from multi_model_tts import multi_model_tts
from llama_cpp_asr_loader import llama_cpp_asr_loader
from forced_aligner_loader import forced_aligner_loader
from forced_aligner_inference import forced_aligner_inference
from json_to_bbox import json_to_bbox
from multi_image_input import MultiImageInput
from video_loader import VideoLoader
from tts_align import tts_align

# 节点映射关系，ComfyUI通过这个字典识别节点
NODE_CLASS_MAPPINGS = {
    "llama_cpp_model_loader": llama_cpp_model_loader,
    "llama_cpp_parameters": llama_cpp_parameters,
    "llama_cpp_clean_states": llama_cpp_clean_states,
    "llama_cpp_unload_model": llama_cpp_unload_model,
    "llama_cpp_tts_loader": llama_cpp_tts_loader,
    "multi_model_tts": multi_model_tts,
    "llama_cpp_asr_loader": llama_cpp_asr_loader,
    "forced_aligner_loader": forced_aligner_loader,
    "forced_aligner_inference": forced_aligner_inference,
    "json_to_bbox": json_to_bbox,
    "multi_image_input": MultiImageInput,
    "llama_cpp_unified_inference": llama_cpp_unified_inference,
    "VideoLoader": VideoLoader,
    "tts_align": tts_align,
}

# 节点显示名称映射，在ComfyUI界面中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "llama_cpp_model_loader": "Llama-cpp Model Loader",
    "llama_cpp_parameters": "Llama-cpp Parameters",
    "llama_cpp_clean_states": "Llama-cpp Clean States",
    "llama_cpp_unload_model": "Llama-cpp Unload Model",
    "llama_cpp_tts_loader": "TTS Model Loader",
    "multi_model_tts": "Multi-Model TTS",
    "llama_cpp_asr_loader": "Llama-cpp ASR Model Loader",
    "forced_aligner_loader": "Forced Aligner Model Loader",
    "forced_aligner_inference": "Forced Aligner Inference",
    "json_to_bbox": "JSON to Bounding Box",
    "multi_image_input": "Multi-Image Input (Story Creation)",
    "llama_cpp_unified_inference": "Llama CPP Unified Inference",
    "VideoLoader": "Video Loader",
    "tts_align": "TTS Align",
}

# 导出所有映射关系
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 版本信息
VERSION = "3.1.0"