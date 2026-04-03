# -*- coding: utf-8 -*-
"""
推理引擎工厂模块

根据模型类型自动创建对应的推理引擎实例。提供统一的引擎创建接口，
支持多种模型系列（Omni、VL等）的动态加载。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

from typing import Dict
from .base_engine import BaseInferenceEngine
from .omni_engines import QwenOmniInferenceEngine
from .vl_engines import GLM4VInferenceEngine


class InferenceEngineFactory:
    """推理引擎工厂 - 根据模型类型创建对应的引擎"""
    
    @staticmethod
    def create_engine(model_info: Dict) -> BaseInferenceEngine:
        """创建对应的推理引擎"""
        subtype = model_info.get("subtype", "default")
        
        if subtype in ["qwen35", "qwen25_omni"]:
            return QwenOmniInferenceEngine(model_info)

        elif subtype == "glm4v":
            return GLM4VInferenceEngine(model_info)
        else:
            return BaseInferenceEngine(model_info)
