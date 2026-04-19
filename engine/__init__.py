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
from .vram_layers import (
    get_layer_count,
    calculate_vram_layers,
    get_file_size_gb,
    get_model_info,
    get_tensor_count,
    estimate_vram_for_safetensors,
    get_safetensors_model_info,
    calculate_safetensors_vram_layers,
)
from .hook_utils import (
    StderrFilter,
    filter_stderr,
    MultiLevelFilter,
    filtered_stderr,
    ModelUnloadHook,
    install_model_unload_hook,
    uninstall_model_unload_hook,
    apply_acceleration_hooks,
)
from .progress import (
    cqdm,
    create_progress_bar,
)
from .turboquant_module import (
    TurboQuantKVCache,
    TurboQuantCompressorV2,
    TurboQuantManager,
    TurboQuantMSE,
    TurboQuantProd,
    get_global_manager,
    init_turboquant,
    clear_all_caches,
)
from .airllm_turboquant_integration import (
    AirLLMTurboQuantConfig,
    AirLLMTurboQuantWrapper,
    TurboQuantKVCacheManager,
    LayerwiseAirLLMWithTurboQuant,
    create_turboquant_wrapper,
    integrate_turboquant_to_airllm,
    TurboQuantAttentionHook,
)

__all__ = [
    "ModelTypeDetector",
    "BaseInferenceEngine",
    "QwenOmniInferenceEngine",
    "GLM4VInferenceEngine",
    "InferenceEngineFactory",
    "convert_audio_to_wav_bytes",
    "create_audio_data_uri",
    "get_layer_count",
    "calculate_vram_layers",
    "get_file_size_gb",
    "get_model_info",
    "get_tensor_count",
    "estimate_vram_for_safetensors",
    "get_safetensors_model_info",
    "calculate_safetensors_vram_layers",
    "StderrFilter",
    "filter_stderr",
    "MultiLevelFilter",
    "filtered_stderr",
    "ModelUnloadHook",
    "install_model_unload_hook",
    "uninstall_model_unload_hook",
    "apply_acceleration_hooks",
    "cqdm",
    "create_progress_bar",
    "TurboQuantKVCache",
    "TurboQuantCompressorV2",
    "TurboQuantManager",
    "TurboQuantMSE",
    "TurboQuantProd",
    "get_global_manager",
    "init_turboquant",
    "clear_all_caches",
    "AirLLMTurboQuantConfig",
    "AirLLMTurboQuantWrapper",
    "TurboQuantKVCacheManager",
    "LayerwiseAirLLMWithTurboQuant",
    "create_turboquant_wrapper",
    "integrate_turboquant_to_airllm",
    "TurboQuantAttentionHook",
]
