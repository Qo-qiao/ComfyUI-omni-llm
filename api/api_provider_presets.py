# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm API Provider Presets
API 提供商预设配置
"""

# 用户保存的自定义配置（由 api_config 节点自动写入）
SAVED_CONFIGS = {}

# 每个提供商支持的模型列表
PRESET_MODELS = {
    "OpenAI": [],
    "Anthropic": [],
    "Grok": [],
    "Google": [],
    "DeepSeek": [],
    "阿里云": [],
    "火山引擎": [],
    "MiniMax": [],
    "Kimi": [],
    "自定义": [],
}

# 每个提供商的默认配置
DEFAULT_PROVIDER_CONFIGS = {
    "OpenAI": {
        "api_base": "https://api.openai.com/v1",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "Anthropic": {
        "api_base": "https://api.anthropic.com/v1",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/messages",
    },
    "Grok": {
        "api_base": "https://api.x.ai/v1",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "Google": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "DeepSeek": {
        "api_base": "https://api.deepseek.com",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "阿里云": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "火山引擎": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "MiniMax": {
        "api_base": "https://api.minimaxi.com/v1",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "Kimi": {
        "api_base": "https://api.moonshot.cn/v1",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
    "自定义": {
        "api_base": "",
        "model_id": "",
        "api_key": "",
        "context_limit": 1048576,
        "api_endpoint": "/chat/completions",
    },
}
