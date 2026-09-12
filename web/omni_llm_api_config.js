/**
 * omni_llm_api_config.js
 * OmniLLM API 配置节点前端逻辑
 * 功能：提供 API 配置的预设管理（加载、保存、删除）和实时预览
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "omni_llm_api_config";
const log = (...a) => console.log("[OmniLLM-APIConfig]", ...a);

const PROVIDER_DEFAULTS = {
    "OpenAI": { api_base: "https://api.openai.com/v1", context_limit: 1048576 },
    "Anthropic": { api_base: "https://api.anthropic.com/v1", context_limit: 1048576 },
    "Grok": { api_base: "https://api.x.ai/v1", context_limit: 1048576 },
    "Google": { api_base: "https://generativelanguage.googleapis.com/v1beta/openai", context_limit: 1048576 },
    "DeepSeek": { api_base: "https://api.deepseek.com", context_limit: 1048576 },
    "阿里云": { api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1", context_limit: 1048576 },
    "火山引擎": { api_base: "https://ark.cn-beijing.volces.com/api/v3", context_limit: 1048576 },
    "MiniMax": { api_base: "https://api.minimaxi.com/v1", context_limit: 1048576 },
    "Kimi": { api_base: "https://api.moonshot.cn/v1", context_limit: 1048576 },
};

// 加载外部 CSS 文件
const timestamp = new Date().getTime();
const cssLink = document.createElement("link");
cssLink.rel = "stylesheet";
cssLink.type = "text/css";
cssLink.href = new URL(`./omni_llm_api_config.css?v=${timestamp}`, import.meta.url).href;
document.head.appendChild(cssLink);

// 隐藏 ComfyUI 原生 widget
function hideWidget(widget) {
    if (!widget) return;
    widget.type = "converted-widget:hidden";
    widget.hidden = true;
    widget.options ||= {};
    widget.options.hidden = true;
    widget.options.hideInPanel = true;
    widget.computeSize = () => [0, -4];
    widget.serializeValue = async () => widget.value;
    if (widget.inputEl) widget.inputEl.style.display = "none";
    if (widget.element) widget.element.style.display = "none";
}

// 从后端获取已保存的配置列表
async function fetchSavedConfigs() {
    try {
        const r = await api.fetchApi("/omni_llm/api/saved_configs");
        const d = await r.json();
        return d.configs || {};
    } catch (e) {
        log("fetchSavedConfigs error:", e);
        return {};
    }
}

// 设置 widget 值并触发回调
function setWidgetValue(node, name, value) {
    const w = node.widgets?.find((w) => w.name === name);
    if (w) {
        w.value = value;
        w.callback?.(value);
    }
}

// 获取 widget 值
function getWidgetValue(node, name) {
    const w = node.widgets?.find((w) => w.name === name);
    return w ? w.value : undefined;
}

// 从 widget 构建配置对象（自动合并提供商默认值）
function buildConfigFromWidgets(node) {
    const provider = getWidgetValue(node, "api_provider") || "自定义";
    const apiBase = getWidgetValue(node, "api_base") || "";
    let contextLimit = getWidgetValue(node, "context_limit") || 0;
    const defaults = PROVIDER_DEFAULTS[provider] || {};
    return {
        provider: provider,
        base_url: apiBase || defaults.api_base || "",
        api_key: getWidgetValue(node, "api_key") || "",
        model: getWidgetValue(node, "model_name") || "",
        context_limit: contextLimit || defaults.context_limit || 0,
        timeout: getWidgetValue(node, "timeout") || 300,
        max_history_rounds: getWidgetValue(node, "max_history_rounds") || 100,
        max_edge: getWidgetValue(node, "max_edge") || 1024,
        preserve_thinking: getWidgetValue(node, "preserve_thinking") || false,
    };
}

// 保存配置到后端
async function saveConfig(name, config) {
    const r = await api.fetchApi("/omni_llm/api/save_config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, config }),
    });
    return await r.json();
}

// 从后端删除配置
async function deleteConfig(name) {
    const r = await api.fetchApi("/omni_llm/api/delete_config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    return await r.json();
}

// 刷新预设下拉列表
async function refreshPresetDropdown(sel) {
    const configs = await fetchSavedConfigs();
    const names = Object.keys(configs);
    sel.innerHTML = "";
    if (names.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "（暂无保存的预设）";
        sel.appendChild(opt);
    } else {
        for (const n of names) {
            const opt = document.createElement("option");
            opt.value = n;
            opt.textContent = n;
            sel.appendChild(opt);
        }
    }
    sel.__configs = configs;
}

// 初始化 API 配置面板
function setupApiConfig(node) {
    if (node.__acReady) return;
    if (typeof node.addDOMWidget !== "function") {
        setTimeout(() => setupApiConfig(node), 200);
        return;
    }
    node.__acReady = true;

    // 隐藏后端控件
    for (const name of ["save_preset", "preset_name", "load_preset", "delete_preset"]) {
        const w = node.widgets?.find((w) => w.name === name);
        if (w) hideWidget(w);
    }

    // 创建统一 DOM 面板（配置控件 + 预览框合并为单个 flex 容器，参考 ComfyUI-prompt-storage 实现）
    const panel = document.createElement("div");
    panel.className = "ac-panel";
    panel.innerHTML = `
        <div class="ac-row">
            <label>已保存</label>
            <select class="ac-preset-sel"><option value="">加载预设...</option></select>
        </div>
        <div class="ac-bar">
            <button class="ac-btn primary ac-load-btn" title="加载选中的预设">加载</button>
            <button class="ac-btn ac-save-btn" title="保存当前配置">保存</button>
            <button class="ac-btn danger ac-del-btn" title="删除选中的预设">删除</button>
        </div>
        <div class="ac-info">选择预设后点「加载」自动填充上方字段</div>
        <div class="ac-preview-box">
            <div class="ac-preview-header">实时预览</div>
            <pre class="ac-preview-content"></pre>
        </div>
    `;

    for (const ev of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel"]) {
        panel.addEventListener(ev, (e) => e.stopPropagation());
    }

    const sel = panel.querySelector(".ac-preset-sel");
    const btnLoad = panel.querySelector(".ac-load-btn");
    const btnSave = panel.querySelector(".ac-save-btn");
    const btnDel = panel.querySelector(".ac-del-btn");
    const info = panel.querySelector(".ac-info");
    const previewContent = panel.querySelector(".ac-preview-content");

    refreshPresetDropdown(sel);

    // 更新预览内容
    function updatePreview() {
        const provider = getWidgetValue(node, "api_provider") || "";
        const model = getWidgetValue(node, "model_name") || "";
        const apiBase = getWidgetValue(node, "api_base") || "";
        const apiKey = getWidgetValue(node, "api_key") || "";
        let contextLimit = getWidgetValue(node, "context_limit") || 0;
        const timeout = getWidgetValue(node, "timeout") || 300;
        const maxHistoryRounds = getWidgetValue(node, "max_history_rounds") || 100;
        const maxEdge = getWidgetValue(node, "max_edge") || 1024;
        const preserveThinking = getWidgetValue(node, "preserve_thinking") || false;

        // 提供商默认上下文
        const providerDefaults = {
            "OpenAI": 1048576, "Anthropic": 1048576, "Grok": 1048576,
            "Google": 1048576, "DeepSeek": 1048576, "阿里云": 1048576,
            "火山引擎": 1048576, "MiniMax": 1048576, "Kimi": 1048576,
        };
        if (contextLimit === 0 && providerDefaults[provider]) {
            contextLimit = providerDefaults[provider];
        }

        let showKey = "***";
        if (apiKey && apiKey.length > 8) {
            showKey = apiKey.slice(0, 4) + "***" + apiKey.slice(-4);
        } else if (apiKey && apiKey.length > 0) {
            showKey = apiKey.slice(0, 2) + "***";
        }

        const ctxLimit = contextLimit > 0 ? contextLimit : "0";
        const lines = [
            `提供商: ${provider}`,
            `模型: ${model || "（未设置）"}`,
            `地址: ${apiBase || "（未设置）"}`,
            `密钥: ${showKey}`,
            `上下文上限: ${ctxLimit}`,
            `超时: ${timeout}秒`,
            `最大历史轮数: ${maxHistoryRounds}`,
            `图片最大边长: ${maxEdge}`,
            `保留思考过程: ${preserveThinking ? "是" : "否"}`,
        ];

        // 添加警告
        const warnings = [];
        if (!apiBase) warnings.push("⚠ API 地址为空");
        if (!apiKey) warnings.push("⚠ API 密钥为空");
        if (!model) warnings.push("⚠ 模型名称为空");

        if (warnings.length > 0) {
            lines.push("", ...warnings);
        }

        previewContent.textContent = lines.join("\n");
    }

    // 监听所有 widget 变化
    function bindWidgetCallbacks() {
        const watchNames = ["api_provider", "model_name", "api_base", "api_key", "context_limit", "timeout", "max_history_rounds", "max_edge", "preserve_thinking"];
        for (const name of watchNames) {
            const w = node.widgets?.find((w) => w.name === name);
            if (w) {
                const originalCallback = w.callback;
                w.callback = (value) => {
                    originalCallback?.(value);
                    updatePreview();
                };
            }
        }
    }

    btnLoad.onclick = async () => {
        const name = sel.value;
        if (!name) { info.textContent = "请先选择一个预设"; return; }
        const configs = sel.__configs || {};
        const cfg = configs[name];
        if (!cfg) { info.textContent = "预设数据异常"; return; }
        setWidgetValue(node, "api_provider", cfg.provider || "自定义");
        setWidgetValue(node, "model_name", cfg.model_id || "");
        setWidgetValue(node, "api_base", cfg.api_base || "");
        setWidgetValue(node, "api_key", cfg.api_key || "");
        setWidgetValue(node, "context_limit", cfg.context_limit || 0);
        setWidgetValue(node, "timeout", cfg.timeout || 300);
        setWidgetValue(node, "max_history_rounds", cfg.max_history_rounds || 100);
        setWidgetValue(node, "max_edge", cfg.max_edge || 1024);
        setWidgetValue(node, "preserve_thinking", cfg.preserve_thinking || false);
        info.textContent = `✅ 已加载「${name}」`;
        node.setDirtyCanvas?.(true, true);
    };

    btnSave.onclick = async () => {
        const name = prompt("输入预设名称（留空自动命名）：", "");
        if (name === null) return;
        const config = buildConfigFromWidgets(node);
        const autoName = name.trim() || `${getWidgetValue(node, "api_provider") || "Custom"}_${config.model || "model"}`;
        const result = await saveConfig(autoName, config);
        if (result.ok) {
            info.textContent = `✅ 已保存「${autoName}」`;
            await refreshPresetDropdown(sel);
            sel.value = autoName;
        } else {
            info.textContent = `❌ 保存失败: ${result.error}`;
        }
    };

    btnDel.onclick = async () => {
        const name = sel.value;
        if (!name) { info.textContent = "请先选择要删除的预设"; return; }
        if (!confirm(`确定删除预设「${name}」？`)) return;
        const result = await deleteConfig(name);
        if (result.ok) {
            info.textContent = `🗑 已删除「${name}」`;
            await refreshPresetDropdown(sel);
        } else {
            info.textContent = `❌ 删除失败: ${result.error}`;
        }
    };

    node.addDOMWidget("api_config_panel", "div", panel, {
        getValue() { return ""; },
        setValue(v) { },
    });
    // 确保节点有正确的大小
    const [w, h] = node.size || [0, 0];
    const newW = Math.max(w, 360);
    const newH = Math.max(h, 380);
    // 只有当大小不同时才设置，避免不必要的重绘
    if (w !== newW || h !== newH) {
        node.setSize([newW, newH]);
    }
    node.minHeight = 380;
    node.minWidth = 360;

    // 初始化预览并绑定回调
    updatePreview();
    bindWidgetCallbacks();

    log("setup complete", node.id);
}

// 注册扩展：监听节点创建和加载
app.registerExtension({
    name: "OmniLlm.APIConfig",
    nodeCreated(node) {
        if (node.type === NODE_CLASS || node.comfyClass === NODE_CLASS) {
            setTimeout(() => setupApiConfig(node), 150);
        }
    },
    loadedGraphNode(node) {
        if (node.type === NODE_CLASS || node.comfyClass === NODE_CLASS) {
            setTimeout(() => setupApiConfig(node), 150);
        }
    },
});

log("extension loaded");
