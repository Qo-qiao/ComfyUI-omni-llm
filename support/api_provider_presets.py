# -*- coding: utf-8 -*-
"""
API提供商预设配置
存储各种API提供商的默认配置，方便用户修改API密钥

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

PRESET_MODELS = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "Google": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-2.0-pro"],
    "Moonshot AI": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    "Kimi": ["kimi-k2.5"],
    "GLM": ["glm-4.5-flash", "glm-4.5-plus", "glm-4.5-air"],
    "MiniMax": ["abab5.5-chat", "abab5.5-turbo"],
    "Qwen": [
        "qwen2.5-vl-72b-instruct",
        "qwen-vl-max",
        "qwen-vl-plus",
        "qwen-3.5-plus",
        "qwen-3.5-flash",
        "qwen2.5-vl-14b-instruct"
    ],
    "Meta": ["llama-3.2-3b-instant", "llama-3.2-70b-versatile", "llama-3.2-90b-vision-pro"],
    "Ollama": ["llama3.2:latest", "gemma2:latest", "mistral:latest", "phi3:latest", "qwen2:latest"],
    "llms-py": ["local-model"],
    "llama-cpp-python": ["Qwen2.5-Omni-7B", "Qwen3-VL-8B", "MiniCPM-V-4.5", "LLaVA-1.6-vicuna-7b", "Moondream2"],
    "vllm-omni": ["vllm-model"],
    "自定义": ["自定义输入"],
}

DEFAULT_PROVIDER_CONFIGS = {
    "OpenAI": {
        "api_base": "https://api.openai.com/v1",
        "api_key": "",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": "gpt-3.5-turbo"
    },
    "Anthropic": {
        "api_base": "https://api.anthropic.com/v1",
        "api_key": "",
        "max_tokens": 4096,
        "temperature": 0.7,
        "model_id": "claude-3-5-sonnet-20240620"
    },
    "Google": {
        "api_base": "https://generativelanguage.googleapis.com/v1",
        "api_key": "",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "gemini-1.5-flash-latest"
    },
    "Moonshot AI": {
        "api_base": "https://api.moonshot.cn/v1",
        "api_key": "",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "moonshot-v1-8k"
    },
    "Kimi": {
        "api_base": "https://api.moonshot.cn/v1",
        "api_key": "",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "kimi-k2.5"
    },
    "GLM": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "glm-4.5-flash"
    },
    "MiniMax": {
        "api_base": "https://api.minimax.chat/v1",
        "api_key": "",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "abab5.5-chat"
    },
    "Qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": "qwen2.5-vl-72b-instruct"
    },
    "Ollama": {
        "api_base": "http://localhost:11434/v1",
        "api_key": "",
        "max_tokens": 2048,
        "temperature": 0.8,
        "model_id": "llama3.2:latest"
    },
    "llms-py": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": "local-model"
    },
    "llama-cpp-python": {
        "api_base": "http://localhost:8080/v1",
        "api_key": "",
        "max_tokens": 4096,
        "temperature": 0.7,
        "model_id": "Qwen2.5-Omni-7B"
    },
    "vllm-omni": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "vllm-model"
    },
    "自定义": {
        "api_base": "",
        "api_key": "",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": ""
    }
}
