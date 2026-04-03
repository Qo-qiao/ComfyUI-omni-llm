// ComfyUI omni-llm 前端配置
// 已修复：Qwen 模型列表 + 图像API格式兼容 + 实时预览信息更新
const PRESET_MODELS = {
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
};

const DEFAULT_PROVIDER_CONFIGS = {
    "OpenAI": {
        "api_base": "https://api.openai.com/v1",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": "gpt-3.5-turbo"
    },
    "Anthropic": {
        "api_base": "https://api.anthropic.com/v1",
        "max_tokens": 4096,
        "temperature": 0.7,
        "model_id": "claude-3-5-sonnet-20240620"
    },
    "Google": {
        "api_base": "https://generativelanguage.googleapis.com/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "gemini-1.5-flash-latest"
    },
    "Moonshot AI": {
        "api_base": "https://api.moonshot.cn/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "moonshot-v1-8k"
    },
    "Kimi": {
        "api_base": "https://api.moonshot.cn/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "kimi-k2.5"
    },
    "GLM": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "glm-4.5-flash"
    },
    "MiniMax": {
        "api_base": "https://api.minimax.chat/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "abab5.5-chat"
    },
    "Qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": "qwen2.5-vl-72b-instruct"
    },
    "Ollama": {
        "api_base": "http://localhost:11434/v1",
        "max_tokens": 2048,
        "temperature": 0.8,
        "model_id": "llama3.2:latest"
    },
    "llms-py": {
        "api_base": "http://localhost:8000/v1",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": "local-model"
    },
    "llama-cpp-python": {
        "api_base": "http://localhost:8080/v1",
        "max_tokens": 4096,
        "temperature": 0.7,
        "model_id": "Qwen2.5-Omni-7B"
    },
    "vllm-omni": {
        "api_base": "http://localhost:8000/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "model_id": "vllm-model"
    },
    "自定义": {
        "api_base": "",
        "max_tokens": 1024,
        "temperature": 0.7,
        "model_id": ""
    }
};

// 辅助函数：重置提供商切换后的字段（设置对应提供商的默认值）
function resetProviderFields(provider, widgets, touched) {
    let cfg = DEFAULT_PROVIDER_CONFIGS[provider] || DEFAULT_PROVIDER_CONFIGS["自定义"];
    console.log(`重置提供商配置: ${provider}`, cfg);
    
    // 重置touched状态
    Object.keys(touched).forEach(key => {
        touched[key] = false;
    });
    
    // 设置API地址
    if (widgets.api_base) {
        const newApiBase = cfg.api_base || "";
        console.log(`设置API地址: ${newApiBase}`);
        widgets.api_base.value = newApiBase;
        if (widgets.api_base.setValue) {
            widgets.api_base.setValue(newApiBase);
        }
    }
    
    // 设置最大token数
    if (widgets.max_tokens) {
        const newMaxTokens = cfg.max_tokens !== undefined ? cfg.max_tokens : "";
        console.log(`设置最大token数: ${newMaxTokens}`);
        widgets.max_tokens.value = newMaxTokens;
        if (widgets.max_tokens.setValue) {
            widgets.max_tokens.setValue(newMaxTokens);
        }
    }
    
    // 设置温度
    if (widgets.temperature) {
        const newTemperature = cfg.temperature !== undefined ? cfg.temperature : "";
        console.log(`设置温度: ${newTemperature}`);
        widgets.temperature.value = newTemperature;
        if (widgets.temperature.setValue) {
            widgets.temperature.setValue(newTemperature);
        }
    }
    
    // 设置模型名称
    if (widgets.model_name) {
        const newModelName = cfg.model_id || "";
        console.log(`设置模型名称: ${newModelName}`);
        widgets.model_name.value = newModelName;
        if (widgets.model_name.setValue) {
            widgets.model_name.setValue(newModelName);
        }
    }
    
    // 清空API密钥
    if (widgets.api_key) {
        console.log(`清空API密钥`);
        widgets.api_key.value = "";
        if (widgets.api_key.setValue) {
            widgets.api_key.setValue("");
        }
    }

    // provider 字段本身应正确显示当前选中项
    if (widgets.api_provider) {
        widgets.api_provider.value = provider;
        if (widgets.api_provider.setValue) {
            widgets.api_provider.setValue(provider);
        }
    }
}

// 辅助函数：获取当前有效配置
function getEffectiveConfig(widgets, touched) {
    const provider = widgets.api_provider ? widgets.api_provider.value : "OpenAI";
    const cfg = DEFAULT_PROVIDER_CONFIGS[provider] || DEFAULT_PROVIDER_CONFIGS["自定义"];

    // 获取控件的当前值
    let modelNameRaw = "";
    let apiBaseRaw = "";
    let apiKeyRaw = "";
    let maxTokensRaw = "";
    let temperatureRaw = "";

    if (widgets.model_name) modelNameRaw = widgets.model_name.value || "";
    if (widgets.api_base) apiBaseRaw = widgets.api_base.value || "";
    if (widgets.api_key) apiKeyRaw = widgets.api_key.value || "";
    if (widgets.max_tokens) maxTokensRaw = widgets.max_tokens.value || "";
    if (widgets.temperature) temperatureRaw = widgets.temperature.value || "";

    // 优先使用用户输入的值，如果为空则使用预设值
    const model_id = modelNameRaw || cfg.model_id;
    const api_base = apiBaseRaw || cfg.api_base;
    const api_key = apiKeyRaw || "";
    
    // 处理数字类型（空字符串或undefined时使用预设值）
    const max_tokens = maxTokensRaw !== "" && maxTokensRaw !== undefined ? Number(maxTokensRaw) : cfg.max_tokens;
    const temperature = temperatureRaw !== "" && temperatureRaw !== undefined ? Number(temperatureRaw) : cfg.temperature;

    return {
        provider,
        model_id,
        api_base,
        api_key,
        max_tokens,
        temperature,
    };
}


// 生成预览文本（与 Python 节点中的格式保持一致）
function generatePreview(config) {
    const provider = config.provider || "OpenAI";
    const modelName = config.model_id || "";
    const apiBase = config.api_base || "";
    let apiKey = config.api_key || "";
    const maxTokens = config.max_tokens || 1024;
    const temperature = config.temperature || 0.7;

    // 对 API Key 进行掩码处理（与 Python 一致）
    let maskedKey = "";
    if (apiKey.length > 8) {
        maskedKey = apiKey.slice(0, 4) + "***" + apiKey.slice(-4);
    } else if (apiKey.length > 0) {
        maskedKey = "***";
    } else {
        maskedKey = "(未设置)";
    }

    const preview = `===== API 配置 =====\n模型：${modelName}\n厂商：${provider}\n地址：${apiBase}\n密钥：${maskedKey}\n最大Token：${maxTokens}\n温度：${temperature}\n=====================`;

    return preview;
}

// 注册 ComfyUI 扩展
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "omni.llm.config",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 仅处理我们的配置节点
        if (nodeData.name !== "llama_cpp_api_model_config") return;

        // 在节点创建时绑定动态更新逻辑
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // 收集节点上所有相关的 widget
            const widgets = {};
            for (let w of this.widgets) {
                widgets[w.name] = w;
            }

            // 确保 preview_info 控件存在（它是可选输入，一般都会存在）
            if (!widgets.preview_info) {
                console.warn("未找到 preview_info 控件，预览信息将无法自动更新");
                return result;
            }

            // 是否已手动修改值（true: 用户已选择/输入，不跟随 provider 直接覆盖）
            const touched = {
                model_name: false,
                api_base: false,
                api_key: false,
                max_tokens: false,
                temperature: false,
            };

            // 定义刷新预览的函数
            const refreshPreview = () => {
                const effective = getEffectiveConfig(widgets, touched);
                const previewText = generatePreview(effective);
                if (!widgets.preview_info) return;
                widgets.preview_info.value = previewText;
                // 强制重绘控件
                if (widgets.preview_info.callback) widgets.preview_info.callback(widgets.preview_info.value);
            };

            // 防御性轮询：即使 onChange 机制失效也能保持动态更新
            let lastPreview = null;
            const previewWatcher = setInterval(() => {
                const effective = getEffectiveConfig(widgets, touched);
                const newPreview = generatePreview(effective);
                if (newPreview !== lastPreview) {
                    lastPreview = newPreview;
                    refreshPreview();
                }
            }, 200);

            // 节点删除时清理定时器
            const originalOnRemove = this.onNodeRemoved;
            this.onNodeRemoved = function () {
                clearInterval(previewWatcher);
                if (originalOnRemove) originalOnRemove.apply(this, arguments);
            };

            // 为相关控件绑定变更事件
            const relevantWidgets = ["api_provider", "api_base", "api_key", "max_tokens", "temperature", "model_name"];
            for (let name of relevantWidgets) {
                const w = widgets[name];
                if (w) {
                    const originalOnChange = w.onChange;
                    w.onChange = (value) => {
                        console.log(`控件 ${name} 变更为:`, value);
                        if (originalOnChange) originalOnChange(value);

                        if (name === "api_provider") {
                            console.log(`切换提供商为: ${value}`);
                            resetProviderFields(value, widgets, touched);
                            // 切换提供商后立即刷新预览
                            setTimeout(() => {
                                refreshPreview();
                            }, 50);
                        } else {
                            touched[name] = true;
                            refreshPreview();
                        }
                    };
                }
            }

            // 初始化预览
            refreshPreview();

            return result;
        };
    },
    async setup() {
        console.log("omni-llm 配置已加载 ✅ （预览实时更新已启用）");
    }
});