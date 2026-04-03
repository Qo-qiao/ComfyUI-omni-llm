# -*- coding: utf-8 -*-
"""
推理引擎模块包

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

from .model_detector import ModelTypeDetector
from .base_engine import BaseInferenceEngine, convert_audio_to_wav_bytes, create_audio_data_uri
from .omni_engines import QwenOmniInferenceEngine
from .vl_engines import GLM4VInferenceEngine
from .engine_factory import InferenceEngineFactory

__all__ = [
    "ModelTypeDetector",
    "BaseInferenceEngine",
    "QwenOmniInferenceEngine",

    "GLM4VInferenceEngine",
    "InferenceEngineFactory",
    "convert_audio_to_wav_bytes",
    "create_audio_data_uri"
]
