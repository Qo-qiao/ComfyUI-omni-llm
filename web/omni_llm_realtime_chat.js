/**
 * omni_llm_realtime_chat.js
 * OmniLLM 实时对话节点前端逻辑
 * 功能：内嵌 DOM 聊天窗，支持 Skill 多阶段交互、图片/视频上传、上下文管理
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "omni_llm_realtime_chat";
// 常量定义
const CHAT_MIN_HEIGHT = 300;
const CHAT_NODE_CHROME_HEIGHT = 120;
const CHAT_MAX_HEIGHT = 800;
const CHAT_WIDGET_PADDING = 10;
const CHAT_FONT_SIZE_DEFAULT = 15;
const CHAT_FONT_SIZE_MIN = 11;
const CHAT_FONT_SIZE_MAX = 28;
let activeImagePreview = null;
let activeImagePreviewKeyHandler = null;

// 加载外部 CSS 文件
const timestamp = new Date().getTime();
const link = document.createElement("link");
link.rel = "stylesheet";
link.type = "text/css";
link.href = new URL(`./omni_llm_realtime_chat.css?v=${timestamp}`, import.meta.url).href;
document.head.appendChild(link);

// ===== 工具函数 =====

// 取数组第一个值或原值
function firstValue(value) {
    return Array.isArray(value) ? value[0] : value;
}

// 解析对话历史 JSON 或纯文本格式
function parseHistory(raw) {
    if (!raw || typeof raw !== "string") return [];
    const trimmed = raw.trim();
    if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
        try {
            const value = JSON.parse(trimmed);
            let items = [];
            if (Array.isArray(value)) {
                items = value;
            } else if (value && typeof value === "object") {
                items = value.messages || value.history || value.conversation || [];
            }
            return items.filter((item) =>
                item &&
                (item.role === "user" || item.role === "assistant") &&
                typeof item.content === "string"
            );
        } catch (_) {}
    }
    const messages = [];
    const lines = trimmed.split("\n");
    let currentRole = null;
    let currentContent = [];
    for (const line of lines) {
        const roleMatch = line.match(/^【(用户|助手|user|assistant)】/i);
        if (roleMatch) {
            if (currentRole && currentContent.length > 0) {
                messages.push({
                    role: currentRole === "用户" || currentRole === "user" ? "user" : "assistant",
                    content: currentContent.join("\n").trim()
                });
            }
            currentRole = roleMatch[1];
            currentContent = [];
        } else if (currentRole && !line.startsWith("=") && !line.startsWith("-")) {
            currentContent.push(line);
        }
    }
    if (currentRole && currentContent.length > 0) {
        messages.push({
            role: currentRole === "用户" || currentRole === "user" ? "user" : "assistant",
            content: currentContent.join("\n").trim()
        });
    }
    return messages.filter(m => m.content);
}

function validHistoryRaw(raw) {
    if (typeof raw !== "string") return null;
    try {
        return Array.isArray(JSON.parse(raw || "")) ? raw : null;
    } catch (_) {
        return null;
    }
}

// 解析图片/视频附件
function parseImages(raw) {
    try {
        const value = JSON.parse(raw || "[]");
        if (!Array.isArray(value)) return [];
        return value.filter((item) =>
            item &&
            typeof (item.filename ?? item.name) === "string" &&
            (item.filename ?? item.name)
        ).map((item) => ({
            filename: item.filename ?? item.name,
            subfolder: item.subfolder || "",
            type: "input",
            media_type: item.media_type || "image",
        }));
    } catch (_) {
        return [];
    }
}

// 判断是否为视频文件
function isVideoFile(file) {
    return file && file.type && file.type.startsWith("video/");
}

function getMediaTypeFromFile(file) {
    if (isVideoFile(file)) return "video";
    return "image";
}

// 解析流程状态
function parseFlowState(raw) {
    try {
        const value = JSON.parse(raw || "{}");
        return value && typeof value === "object" ? value : {};
    } catch (_) {
        return {};
    }
}

// 解析上下文状态
function parseContextState(raw) {
    if (raw && typeof raw === "object") return raw;
    try {
        const value = JSON.parse(raw || "{}");
        return value && typeof value === "object" ? value : {};
    } catch (_) {
        return {};
    }
}

// 格式化 token 数量显示
function formatTokenCount(value) {
    const tokens = Math.max(0, Number(value) || 0);
    if (tokens < 1000) return String(Math.round(tokens));
    const scaled = tokens / 1000;
    return `${scaled >= 10 ? scaled.toFixed(0) : scaled.toFixed(1)}k`;
}

// 格式化消息时间
function formatMessageTime(value) {
    const timestamp = Number(value);
    if (!Number.isFinite(timestamp) || timestamp <= 0) return "";
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return "";
    const pad = (part) => String(part).padStart(2, "0");
    const now = new Date();
    const sameDay =
        date.getFullYear() === now.getFullYear() &&
        date.getMonth() === now.getMonth() &&
        date.getDate() === now.getDate();
    const clock = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
    return sameDay ? clock : `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${clock.slice(0, 5)}`;
}

// 解析选项列表
function parseOptions(raw) {
    try {
        const value = JSON.parse(raw || "[]");
        return Array.isArray(value) ? value.filter((item) => typeof item === "string" && item.trim()) : [];
    } catch (_) {
        return [];
    }
}

function isHistoryJson(raw) {
    try {
        return Array.isArray(JSON.parse(raw || ""));
    } catch (_) {
        return false;
    }
}

// 上传聊天媒体文件（图片/视频）
async function uploadChatMedia(file, index) {
    const isVideo = isVideoFile(file);
    const safeName = String(file.name || (isVideo ? "video.mp4" : "image.png")).replace(/[^a-zA-Z0-9._-]+/g, "_");
    const uploadName = `omni_chat_${Date.now()}_${index}_${safeName}`;
    const body = new FormData();
    body.append("image", file, uploadName);
    body.append("type", "input");
    body.append("subfolder", "omni_llm_chat");
    body.append("overwrite", "false");

    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response?.ok) throw new Error(`${isVideo ? "视频" : "图片"}上传失败 (${response?.status || "unknown"})`);
    const result = await response.json();
    return {
        filename: result.name || uploadName,
        subfolder: result.subfolder || "omni_llm_chat",
        type: "input",
        media_type: getMediaTypeFromFile(file),
    };
}

// 隐藏后端 widget（数值由聊天窗维护）
function hideBackendWidget(widget) {
    if (!widget) return;
    widget.type = `converted-widget:omni-llm-chat-${widget.name}`;
    widget.computeSize = () => [0, 0];
    widget.serializeValue = async () => widget.value;
    widget.hidden = true;
    widget.advanced = true;
    if (widget.inputEl) widget.inputEl.style.display = "none";
    if (widget.element) widget.element.style.display = "none";
    if (widget.container) widget.container.style.display = "none";
}

// 创建 DOM 元素
function createElement(tag, className, text = "") {
    const element = document.createElement(tag);
    element.className = className;
    if (text) element.textContent = text;
    return element;
}

// 构建媒体预览 URL
function buildChatMediaUrl(mediaRef) {
    if (!mediaRef?.filename) return "";
    const params = new URLSearchParams({
        filename: String(mediaRef.filename),
        type: String(mediaRef.type || "input"),
        subfolder: String(mediaRef.subfolder || ""),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function isVideoMedia(mediaRef) {
    return mediaRef?.media_type === "video";
}

function closeImagePreview() {
    if (activeImagePreviewKeyHandler) {
        document.removeEventListener("keydown", activeImagePreviewKeyHandler);
        activeImagePreviewKeyHandler = null;
    }
    activeImagePreview?.remove();
    activeImagePreview = null;
}

// 打开媒体全屏预览
function openMediaPreview(url, label, isVideo = false) {
    closeImagePreview();

    const overlay = createElement("div", "omni-llm-chat__preview");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", label || (isVideo ? "视频预览(Video Preview)" : "图片预览(Image Preview)"));

    let mediaElement;
    if (isVideo) {
        mediaElement = createElement("video", "omni-llm-chat__preview-video");
        mediaElement.src = url;
        mediaElement.controls = true;
        mediaElement.autoplay = true;
        mediaElement.loop = false;
        mediaElement.muted = false;
    } else {
        mediaElement = createElement("img", "omni-llm-chat__preview-image");
        mediaElement.src = url;
        mediaElement.alt = label || "图片预览(Image Preview)";
    }

    const closeButton = createElement("button", "omni-llm-chat__preview-close", "×");
    closeButton.type = "button";
    closeButton.title = `关闭${isVideo ? "视频" : "图片"}预览(Close Preview)`;
    closeButton.setAttribute("aria-label", `关闭${isVideo ? "视频" : "图片"}预览(Close Preview)`);
    closeButton.addEventListener("click", closeImagePreview);
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) closeImagePreview();
    });
    for (const eventName of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel"]) {
        overlay.addEventListener(eventName, (event) => event.stopPropagation());
    }

    activeImagePreviewKeyHandler = (event) => {
        if (event.key === "Escape") closeImagePreview();
    };
    document.addEventListener("keydown", activeImagePreviewKeyHandler);
    overlay.append(mediaElement, closeButton);
    document.body.append(overlay);
    activeImagePreview = overlay;
    closeButton.focus();
}

// 创建消息中的媒体画廊
function createMessageMedia(mediaRefs) {
    const validMedia = (Array.isArray(mediaRefs) ? mediaRefs : [])
        .map((mediaRef) => ({ mediaRef, url: buildChatMediaUrl(mediaRef), isVideo: isVideoMedia(mediaRef) }))
        .filter((item) => item.url);
    if (!validMedia.length) return null;

    const gallery = createElement("div", "omni-llm-chat__message-images");
    validMedia.forEach(({ mediaRef, url, isVideo }, index) => {
        const previewButton = createElement("button", "omni-llm-chat__message-image-link");
        previewButton.type = "button";
        const mediaLabel = isVideo ? `视频${index + 1}` : `图${index + 1}`;
        const mediaLabelEn = isVideo ? `Vid ${index + 1}` : `Img ${index + 1}`;
        previewButton.title = `预览${mediaLabel}：${mediaRef.filename}(Preview ${mediaLabelEn})`;
        previewButton.setAttribute("aria-label", `预览${mediaLabel}(Preview ${mediaLabelEn})`);
        previewButton.addEventListener("click", (event) => {
            event.stopPropagation();
            openMediaPreview(url, `${mediaLabel}(${mediaLabelEn})`, isVideo);
        });

        if (isVideo) {
            const video = createElement("video", "omni-llm-chat__message-video");
            video.src = url;
            video.preload = "metadata";
            video.muted = true;
            video.playsInline = true;
            previewButton.append(video, createElement("span", "omni-llm-chat__message-image-label", `${mediaLabel}(${mediaLabelEn})`));
        } else {
            const image = createElement("img", "omni-llm-chat__message-image");
            image.src = url;
            image.alt = `${mediaLabel}(${mediaLabelEn})`;
            image.loading = "lazy";
            image.decoding = "async";
            previewButton.append(image, createElement("span", "omni-llm-chat__message-image-label", `${mediaLabel}(${mediaLabelEn})`));
        }
        gallery.append(previewButton);
    });
    return gallery;
}

// 创建消息内容（支持代码块渲染）
function createMessageContent(text, onCopy) {
    const content = createElement("div", "omni-llm-chat__message-content");
    const source = String(text || "");
    const fence = /```([^\n`]*)\n([\s\S]*?)```/g;
    let cursor = 0;
    let match;

    while ((match = fence.exec(source)) !== null) {
        if (match.index > cursor) content.append(document.createTextNode(source.slice(cursor, match.index)));
        const pre = createElement("pre", "omni-llm-chat__code");
        const language = match[1].trim();
        const codeText = match[2].replace(/\n$/, "");
        const codeHeader = createElement("div", "omni-llm-chat__code-header");
        if (language) codeHeader.append(createElement("span", "omni-llm-chat__code-language", language));
        
        const scrollButtons = createElement("div", "omni-llm-chat__code-scroll-buttons");
        const scrollUpBtn = createElement("button", "omni-llm-chat__code-scroll-btn omni-llm-chat__code-scroll-up", "↑");
        scrollUpBtn.type = "button";
        scrollUpBtn.title = "向上滚动(Scroll Up)";
        scrollUpBtn.setAttribute("aria-label", "向上滚动(Scroll Up)");
        const scrollDownBtn = createElement("button", "omni-llm-chat__code-scroll-btn omni-llm-chat__code-scroll-down", "↓");
        scrollDownBtn.type = "button";
        scrollDownBtn.title = "向下滚动(Scroll Down)";
        scrollDownBtn.setAttribute("aria-label", "向下滚动(Scroll Down)");
        scrollButtons.append(scrollUpBtn, scrollDownBtn);
        
        const copyCodeButton = createElement("button", "omni-llm-chat__code-copy", "⧉");
        copyCodeButton.type = "button";
        copyCodeButton.title = "复制代码块(Copy Code)";
        copyCodeButton.setAttribute("aria-label", "复制代码块(Copy Code)");
        copyCodeButton.addEventListener("click", (event) => {
            event.stopPropagation();
            onCopy?.(codeText);
        });
        codeHeader.append(scrollButtons, copyCodeButton);
        pre.append(codeHeader);
        const code = document.createElement("code");
        code.textContent = codeText;
        pre.append(code);
        
        scrollUpBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            pre.scrollBy({ top: -60, behavior: "smooth" });
        });
        scrollDownBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            pre.scrollBy({ top: 60, behavior: "smooth" });
        });
        
        content.append(pre);
        cursor = fence.lastIndex;
    }

    if (cursor < source.length) content.append(document.createTextNode(source.slice(cursor)));
    return content;
}

// ===== 构建只包含聊天节点的 prompt =====

// 判断是否为 prompt 链接
function isPromptLink(value, output) {
    if (!Array.isArray(value) || value.length !== 2) return false;
    const sourceId = value[0];
    const outputSlot = value[1];
    const validSource =
        typeof sourceId === "number" ||
        (typeof sourceId === "string" && /^\d+$/.test(sourceId));
    return (
        validSource &&
        typeof outputSlot === "number" &&
        Number.isFinite(outputSlot) &&
        Boolean(output?.[String(sourceId)] ?? output?.[Number(sourceId)])
    );
}

// 收集所有上游节点 ID
function collectPromptLinks(value, output, result = new Set()) {
    if (isPromptLink(value, output)) {
        result.add(String(value[0]));
        return result;
    }
    if (Array.isArray(value)) {
        for (const item of value) collectPromptLinks(item, output, result);
    } else if (value && typeof value === "object") {
        for (const item of Object.values(value)) collectPromptLinks(item, output, result);
    }
    return result;
}

// 构建仅包含聊天节点及其上游的 prompt
async function buildChatOnlyPrompt(node, chatInputs = null) {
    const prompt = await app.graphToPrompt();
    const output = prompt?.output;
    const targetId = String(node.id);
    if (!output || !(output[targetId] ?? output[Number(targetId)])) {
        throw new Error("当前聊天节点不在可执行提示中，请检查模型连接。(Chat node not in prompt, check model connections.)");
    }

    const keep = new Set();
    const addWithAncestors = (nodeId) => {
        const id = String(nodeId);
        if (keep.has(id)) return;
        const apiNode = output[id] ?? output[Number(id)];
        if (!apiNode) return;
        keep.add(id);
        for (const sourceId of collectPromptLinks(apiNode.inputs || {}, output)) {
            addWithAncestors(sourceId);
        }
    };
    addWithAncestors(targetId);

    const scopedOutput = {};
    for (const [id, apiNode] of Object.entries(output)) {
        if (keep.has(String(id))) scopedOutput[id] = apiNode;
    }
    const targetNode = scopedOutput[targetId] ?? scopedOutput[Number(targetId)];
    if (targetNode?.inputs && chatInputs) {
        Object.assign(targetNode.inputs, chatInputs);
    }
    prompt.output = scopedOutput;
    return prompt;
}

// ===== 主聊天节点设置 =====

function setupChatNode(node) {
    // 初始化历史数据
    node.properties ||= {};

    const userWidget = node.widgets?.find((widget) => widget.name === "user_message");
    const historyWidget = node.widgets?.find((widget) => widget.name === "chat_history_json");
    const requestWidget = node.widgets?.find((widget) => widget.name === "request_id");
    const currentImagesWidget = node.widgets?.find((widget) => widget.name === "current_images_json");
    if (!userWidget || !historyWidget || !requestWidget || !currentImagesWidget || typeof node.addDOMWidget !== "function") return;

    if (!isHistoryJson(historyWidget.value)) {
        const recoveredHistory = validHistoryRaw(node.__omniLlmLastValidHistoryRaw);
        historyWidget.value = recoveredHistory ?? "[]";
        requestWidget.value = "";
        currentImagesWidget.value = "[]";
    }
    let lastValidHistoryRaw = validHistoryRaw(historyWidget.value) ?? "[]";
    node.__omniLlmLastValidHistoryRaw = lastValidHistoryRaw;

    // 隐藏后端通信用的 widget
    // 后端通信用的隐藏 widget（数值由聊天窗维护，界面不展示）
    hideBackendWidget(userWidget);
    hideBackendWidget(historyWidget);
    hideBackendWidget(requestWidget);
    hideBackendWidget(currentImagesWidget);
    const flowWidget = node.widgets?.find((widget) => widget.name === "flow_state_json");
    if (flowWidget) hideBackendWidget(flowWidget);
    const optionsWidget = node.widgets?.find((widget) => widget.name === "options_json");
    if (optionsWidget) hideBackendWidget(optionsWidget);
    const maxRoundsWidget = node.widgets?.find((w) => w.name === "max_rounds");
    if (maxRoundsWidget) hideBackendWidget(maxRoundsWidget);
    const maxEdgeWidget = node.widgets?.find((w) => w.name === "max_edge");
    if (maxEdgeWidget) hideBackendWidget(maxEdgeWidget);
    const preserveThinkingWidget = node.widgets?.find((w) => w.name === "preserve_thinking");
    if (preserveThinkingWidget) hideBackendWidget(preserveThinkingWidget);
    const seedWidget = node.widgets?.find((w) => w.name === "seed");
    if (seedWidget) hideBackendWidget(seedWidget);
    const nativeVideoWidget = node.widgets?.find((w) => w.name === "native_video");
    if (nativeVideoWidget) hideBackendWidget(nativeVideoWidget);
    const saveConversationWidget = node.widgets?.find((w) => w.name === "save_conversation");
    if (saveConversationWidget) hideBackendWidget(saveConversationWidget);
    const sessionIdWidget = node.widgets?.find((w) => w.name === "session_id");
    if (sessionIdWidget) hideBackendWidget(sessionIdWidget);
    // system_prompt: 隐藏 STRING widget（用手动 DOM 输入框代替）
    const systemPromptWidget = node.widgets?.find((w) => w.name === "system_prompt");
    if (systemPromptWidget) hideBackendWidget(systemPromptWidget);

    // 构建聊天界面 DOM 结构
    const root = createElement("div", "omni-llm-chat");
    
    // 主内容区域（横向布局：左侧内容 + 右侧历史面板）
    const mainContent = createElement("div", "omni-llm-chat__main");
    const leftPanel = createElement("div", "omni-llm-chat__left-panel");
    
    const systemPromptContainer = createElement("div", "omni-llm-chat__system-prompt");
    const systemPromptHeader = createElement("div", "omni-llm-chat__system-prompt-header");
    const systemPromptToggle = createElement("button", "omni-llm-chat__system-prompt-toggle");
    systemPromptToggle.innerHTML = "&#9660;";
    systemPromptToggle.title = "展开/折叠系统提示词(Expand/Collapse)";
    const systemPromptLabel = createElement("span", "omni-llm-chat__system-prompt-label", "系统提示词(System Prompt)");
    systemPromptHeader.append(systemPromptToggle, systemPromptLabel);
    const systemPromptInput = createElement("textarea", "omni-llm-chat__system-prompt-input");
    systemPromptContainer.append(systemPromptHeader, systemPromptInput);
    const messages = createElement("div", "omni-llm-chat__messages");
    const flow = createElement("div", "omni-llm-chat__flow");
    const flowSummary = createElement("div", "omni-llm-chat__flow-summary");
    const flowTools = createElement("div", "omni-llm-chat__flow-tools");
    const stage = createElement("span", "omni-llm-chat__stage", "未开始(Idle)");
    const skillLabel = createElement("span", "omni-llm-chat__skill", "普通对话(Normal)");
    const contextMeter = createElement("div", "omni-llm-chat__context");
    const contextRing = createElement("div", "omni-llm-chat__context-ring");
    const contextPercent = createElement("span", "omni-llm-chat__context-percent", "--");
    const contextMeta = createElement("div", "omni-llm-chat__context-meta");
    const contextTokens = createElement("span", "omni-llm-chat__context-tokens", "已用约 --(Used ~--)");
    const contextRounds = createElement("span", "omni-llm-chat__context-rounds", "轮数 --/--(Rds --/--)");
    const contextNote = createElement("span", "omni-llm-chat__context-note", "上下文估算(Ctx Est.)");
    const fontSizeControl = createElement("div", "omni-llm-chat__font-size");
    const decreaseFontButton = createElement("button", "omni-llm-chat__font-button", "−");
    const fontSizeValue = createElement("span", "omni-llm-chat__font-value");
    const increaseFontButton = createElement("button", "omni-llm-chat__font-button", "+");
    const unloadButton = createElement("button", "omni-llm-chat__unload", "卸载模型(Unload)");
    const options = createElement("div", "omni-llm-chat__options");
    const composer = createElement("div", "omni-llm-chat__composer");
    const inputContainer = createElement("div", "omni-llm-chat__input-container");
    const toolbar = createElement("div", "omni-llm-chat__toolbar");
    const insertImageButton = createElement("button", "omni-llm-chat__toolbar-btn", "+");
    const clearButton = createElement("button", "omni-llm-chat__toolbar-btn", "清空(Clear)");
    const regenerateButton = createElement("button", "omni-llm-chat__toolbar-btn", "重新生成(Regenerate)");
    const historyButton = createElement("button", "omni-llm-chat__toolbar-btn omni-llm-chat__history-btn");
    historyButton.type = "button";
    historyButton.title = "查看历史对话(View History)";
    historyButton.setAttribute("aria-label", "查看历史对话(View History)");
    historyButton.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"/></svg>`;
    const saveCheckboxContainer = createElement("div", "omni-llm-chat__save-container");
    const saveCheckbox = createElement("input", "omni-llm-chat__save-checkbox");
    saveCheckbox.type = "checkbox";
    saveCheckbox.id = `omni-llm-save-${node.id}`;
    saveCheckbox.checked = saveConversationWidget?.value === true || saveConversationWidget?.value === "true";
    const saveCheckboxLabel = createElement("label", "omni-llm-chat__save-label");
    saveCheckboxLabel.htmlFor = saveCheckbox.id;
    saveCheckboxLabel.textContent = "保存对话(Save)";
    saveCheckboxLabel.title = "勾选后自动将对话内容保存到 cache 文件夹(Check to auto-save conversation)";
    saveCheckboxContainer.append(saveCheckbox, saveCheckboxLabel);
    const inputRow = createElement("div", "omni-llm-chat__input-row");
    const attachments = createElement("div", "omni-llm-chat__attachments");
    const input = createElement("textarea", "omni-llm-chat__input");
    const actions = createElement("div", "omni-llm-chat__actions");
    const sendButton = createElement("button", "omni-llm-chat__button");
    const fileInput = document.createElement("input");
    const status = createElement("div", "omni-llm-chat__status", "准备就绪(Ready)");
    
    // 右侧历史面板
    const historySidebar = createElement("div", "omni-llm-chat__history-sidebar");
    const historySidebarHeader = createElement("div", "omni-llm-chat__history-sidebar-header");
    const historySidebarTitle = createElement("span", "omni-llm-chat__history-sidebar-title", "历史对话");
    const historySidebarClose = createElement("button", "omni-llm-chat__history-sidebar-close");
    historySidebarClose.innerHTML = "&times;";
    historySidebarClose.title = "关闭(Close)";
    historySidebarHeader.append(historySidebarTitle, historySidebarClose);
    
    const historySearchContainer = createElement("div", "omni-llm-chat__history-search");
    const historySearchInput = createElement("input", "omni-llm-chat__history-search-input");
    historySearchInput.type = "text";
    historySearchInput.placeholder = "搜索历史对话...";
    historySearchInput.setAttribute("aria-label", "搜索历史对话");
    historySearchContainer.append(historySearchInput);
    
    const historyList = createElement("div", "omni-llm-chat__history-list");
    historySidebar.append(historySidebarHeader, historySearchContainer, historyList);

    input.placeholder = "发消息或按住空格说话...";
    fileInput.type = "file";
    fileInput.accept = "image/*,video/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    sendButton.type = "button";
    insertImageButton.type = "button";
    insertImageButton.title = "添加图片或视频(Add Image or Video)";
    insertImageButton.setAttribute("aria-label", "添加图片或视频(Add Image or Video)");
    clearButton.type = "button";
    regenerateButton.type = "button";
    decreaseFontButton.type = "button";
    increaseFontButton.type = "button";
    decreaseFontButton.title = "减小聊天字体(Smaller Font)";
    decreaseFontButton.setAttribute("aria-label", "减小聊天字体(Smaller Font)");
    increaseFontButton.title = "增大聊天字体(Larger Font)";
    increaseFontButton.setAttribute("aria-label", "增大聊天字体(Larger Font)");
    fontSizeValue.setAttribute("aria-live", "polite");
    fontSizeControl.setAttribute("role", "group");
    fontSizeControl.setAttribute("aria-label", "聊天字体大小(Font Size)");
    flowTools.setAttribute("role", "toolbar");
    flowTools.setAttribute("aria-label", "聊天工具(Tools)");
    unloadButton.type = "button";
    unloadButton.title = "卸载 Omni LLM 模型（释放显存）(Unload Model)";
    unloadButton.setAttribute("aria-label", "卸载 Omni LLM 模型(Unload Model)");
    regenerateButton.title = "重新生成上一条助手回复(Regenerate)";
    sendButton.title = "发送消息(Send)";
    
    // 设置系统提示词输入框初始值和事件监听
    systemPromptInput.placeholder = "输入系统提示词，定义AI的角色和行为...";
    systemPromptInput.value = systemPromptWidget?.value || "你是一个有帮助的AI助手。";
    systemPromptInput.addEventListener("input", () => {
        if (systemPromptWidget) {
            systemPromptWidget.value = systemPromptInput.value;
        }
    });
    
    // 监听widget值变化（当外部节点连接时更新DOM输入框）
    if (systemPromptWidget) {
        const originalCallback = systemPromptWidget.callback;
        systemPromptWidget.callback = function(value) {
            if (originalCallback) originalCallback.call(this, value);
            if (typeof value === "string") {
                systemPromptInput.value = value;
            } else if (value && typeof value === "object") {
                // 处理 OMNI_LLM_PRESET 类型
                const text = value.prompt || value.content || value.system_prompt || "";
                systemPromptInput.value = text;
            }
        };
    }
    
    // 折叠功能（默认折叠）
    let isSystemPromptExpanded = false;
    systemPromptContainer.classList.add("collapsed");
    systemPromptToggle.innerHTML = "&#9654;";
    systemPromptInput.style.display = "none";
    
    systemPromptToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        isSystemPromptExpanded = !isSystemPromptExpanded;
        if (isSystemPromptExpanded) {
            systemPromptContainer.classList.remove("collapsed");
            systemPromptToggle.innerHTML = "&#9660;";
            systemPromptInput.style.display = "block";
        } else {
            systemPromptContainer.classList.add("collapsed");
            systemPromptToggle.innerHTML = "&#9654;";
            systemPromptInput.style.display = "none";
        }
    });
    
    // 点击标题栏也可以展开/折叠
    systemPromptHeader.addEventListener("click", () => {
        isSystemPromptExpanded = !isSystemPromptExpanded;
        if (isSystemPromptExpanded) {
            systemPromptContainer.classList.remove("collapsed");
            systemPromptToggle.innerHTML = "&#9660;";
            systemPromptInput.style.display = "block";
        } else {
            systemPromptContainer.classList.add("collapsed");
            systemPromptToggle.innerHTML = "&#9654;";
            systemPromptInput.style.display = "none";
        }
    });
    
    // 初始化 session_id
    if (!sessionIdWidget?.value) {
        sessionIdWidget.value = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    }
    actions.append(sendButton);
    toolbar.append(insertImageButton, regenerateButton, clearButton, historyButton, saveCheckboxContainer, actions);
    inputRow.append(attachments, input);
    inputContainer.append(inputRow, toolbar);
    composer.append(systemPromptContainer, inputContainer);
    contextRing.append(contextPercent);
    contextMeta.append(contextTokens, contextRounds, contextNote);
    contextMeter.append(contextRing, contextMeta);
    fontSizeControl.append(decreaseFontButton, fontSizeValue, increaseFontButton);
    flowSummary.append(stage, skillLabel);
    flowTools.append(fontSizeControl, contextMeter, unloadButton);
    flow.append(flowSummary, flowTools);
    
    leftPanel.append(flow, messages, options, composer, status, fileInput);
    mainContent.append(leftPanel, historySidebar);
    root.append(mainContent);

    for (const eventName of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel"]) {
        root.addEventListener(eventName, (event) => event.stopPropagation());
    }

    // 字体大小控制
    const applyChatFontSize = (value, markDirty = false) => {
        const numericValue = Number(value);
        const nextValue = Math.min(
            CHAT_FONT_SIZE_MAX,
            Math.max(
                CHAT_FONT_SIZE_MIN,
                Number.isFinite(numericValue) ? Math.round(numericValue) : CHAT_FONT_SIZE_DEFAULT
            )
        );
        node.properties.omniLlmChatFontSize = nextValue;
        root.style.setProperty("--omni-llm-chat-font-size", `${nextValue}px`);
        fontSizeValue.textContent = String(nextValue);
        fontSizeValue.title = `当前聊天字体：${nextValue}px(Current: ${nextValue}px)`;
        decreaseFontButton.disabled = nextValue <= CHAT_FONT_SIZE_MIN;
        increaseFontButton.disabled = nextValue >= CHAT_FONT_SIZE_MAX;
        if (markDirty) node.graph?.setDirtyCanvas?.(true, true);
    };
    decreaseFontButton.addEventListener("click", () => {
        applyChatFontSize(Number(node.properties.omniLlmChatFontSize) - 1, true);
    });
    increaseFontButton.addEventListener("click", () => {
        applyChatFontSize(Number(node.properties.omniLlmChatFontSize) + 1, true);
    });
    applyChatFontSize(node.properties.omniLlmChatFontSize);

    // ===== 历史对话弹窗 =====
    let historyListData = [];
    let selectedHistoryFile = null;
    let isHistorySidebarVisible = false;

    const toggleHistorySidebar = () => {
        isHistorySidebarVisible = !isHistorySidebarVisible;
        if (isHistorySidebarVisible) {
            historySidebar.classList.add("omni-llm-chat__history-sidebar--visible");
            historyButton.classList.add("omni-llm-chat__history-btn--active");
            loadHistoryList();
        } else {
            historySidebar.classList.remove("omni-llm-chat__history-sidebar--visible");
            historyButton.classList.remove("omni-llm-chat__history-btn--active");
        }
    };

    const loadHistoryList = async () => {
        historyList.textContent = "加载中...";
        try {
            const response = await api.fetchApi("/omni_llm/history/list");
            const data = await response.json();
            if (data.ok && data.files) {
                historyListData = data.files;
                renderHistoryList();
            } else {
                historyList.textContent = "无历史对话";
            }
        } catch (err) {
            historyList.textContent = "加载失败";
        }
    };

    const renderHistoryList = () => {
        historyList.replaceChildren();
        const filterText = historySearchInput.value.toLowerCase();
        const filtered = historyListData.filter((f) => 
            !filterText || f.filename.toLowerCase().includes(filterText)
        );
        if (!filtered.length) {
            historyList.textContent = historyListData.length ? "无匹配结果" : "无历史对话";
            return;
        }
        filtered.forEach((file) => {
            const item = createElement("div", "omni-llm-chat__history-item");
            const icon = createElement("span", "omni-llm-chat__history-item-icon");
            icon.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
            const info = createElement("div", "omni-llm-chat__history-item-info");
            const name = createElement("span", "omni-llm-chat__history-item-name", file.filename);
            const meta = createElement("span", "omni-llm-chat__history-item-meta", file.mtime_str);
            info.append(name, meta);
            const delBtn = createElement("button", "omni-llm-chat__history-item-del");
            delBtn.innerHTML = "❌";
            delBtn.title = "删除此对话";
            delBtn.addEventListener("click", async (e) => {
                e.stopPropagation();
                if (!confirm(`确定删除对话 ${file.filename} ?`)) return;
                try {
                    const resp = await api.fetchApi("/omni_llm/history/delete", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ filename: file.filename }),
                    });
                    const result = await resp.json();
                    if (result.ok) {
                        historyListData = historyListData.filter((f) => f.filename !== file.filename);
                        if (selectedHistoryFile === file.filename) {
                            selectedHistoryFile = null;
                        }
                        renderHistoryList();
                    }
                } catch (_) {}
            });
            item.append(icon, info, delBtn);
            
            item.addEventListener("click", async () => {
                historyList.querySelectorAll(".omni-llm-chat__history-item").forEach((el) => {
                    el.classList.remove("omni-llm-chat__history-item--selected");
                });
                item.classList.add("omni-llm-chat__history-item--selected");
                selectedHistoryFile = file.filename;
                await loadHistoryContent(file.filename);
            });
            historyList.append(item);
        });
    };

    let historyLoadingAbort = null;

    const loadHistoryContent = async (filename) => {
        if (historyLoadingAbort) {
            historyLoadingAbort.abort();
            historyLoadingAbort = null;
        }
        
        historyLoadingAbort = new AbortController();
        status.textContent = `加载中: ${filename}`;
        status.dataset.state = "loading";
        
        try {
            const response = await api.fetchApi(`/omni_llm/history/content?filename=${encodeURIComponent(filename)}`, {
                signal: historyLoadingAbort.signal
            });
            const data = await response.json();
            if (data.ok && data.content) {
                const rawContent = data.content.trim();
                let history = parseHistory(rawContent);
                if (history.length === 0) {
                    try {
                        const parsed = JSON.parse(rawContent);
                        if (Array.isArray(parsed) && parsed.length > 0) {
                            history = parsed.filter(item => item && typeof item === "object");
                        } else if (parsed && typeof parsed === "object") {
                            const messages = parsed.messages || parsed.history || parsed.conversation;
                            if (Array.isArray(messages)) {
                                history = messages.filter(item => item && typeof item === "object");
                            }
                        }
                    } catch (e) {
                        console.warn("[OmniLLM] 解析历史内容失败:", e);
                    }
                }
                if (history.length > 0) {
                    const normalized = history.map(item => ({
                        role: item.role || "assistant",
                        content: typeof item.content === "string" ? item.content : JSON.stringify(item.content || ""),
                        created_at: item.created_at || item.timestamp || Date.now(),
                        images: item.images || [],
                        token_count: item.token_count || 0
                    }));
                    commitHistoryRaw(JSON.stringify(normalized));
                    render();
                    status.textContent = `已加载: ${filename}`;
                    status.dataset.state = "idle";
                } else {
                    console.warn("[OmniLLM] 历史内容解析为空, 原始内容:", rawContent.substring(0, 200));
                    status.textContent = "历史内容为空";
                    status.dataset.state = "error";
                }
            }
        } catch (err) {
            if (err.name !== "AbortError") {
                status.textContent = "加载失败";
                status.dataset.state = "error";
            }
        } finally {
            historyLoadingAbort = null;
        }
    };

    historyButton.addEventListener("click", toggleHistorySidebar);
    historySidebarClose.addEventListener("click", toggleHistorySidebar);
    historySearchInput.addEventListener("input", renderHistoryList);

    // 提交历史数据（带安全检查）
    const commitHistoryRaw = (raw, { allowEmptyRegression = false } = {}) => {
        const validRaw = validHistoryRaw(raw);
        if (validRaw === null) return false;
        if (
            !allowEmptyRegression &&
            parseHistory(validRaw).length === 0 &&
            parseHistory(lastValidHistoryRaw).length > 0
        ) return false;
        historyWidget.value = validRaw;
        lastValidHistoryRaw = validRaw;
        node.__omniLlmLastValidHistoryRaw = validRaw;
        return true;
    };

    const protectedHistory = () => {
        if (!commitHistoryRaw(historyWidget.value)) {
            historyWidget.value = lastValidHistoryRaw;
            status.textContent = "检测到历史数据异常，已恢复上一次有效对话(History error, restored)";
            status.dataset.state = "error";
        }
        return parseHistory(lastValidHistoryRaw);
    };

    const copyText = async (value) => {
        if (!value) {
            status.textContent = "暂无可复制内容(Nothing to copy)";
            status.dataset.state = "error";
            return false;
        }
        try {
            await navigator.clipboard.writeText(value);
        } catch (_) {
            const textarea = document.createElement("textarea");
            textarea.value = value;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            document.body.append(textarea);
            textarea.select();
            document.execCommand("copy");
            textarea.remove();
        }
        status.textContent = "已复制这条消息(Copied)";
        status.dataset.state = "idle";
        return true;
    };

    // 渲染消息列表
    const render = () => {
        const history = protectedHistory();
        messages.replaceChildren();
        if (!history.length) {
            messages.append(createElement("div", "omni-llm-chat__empty", "暂无对话，输入消息开始(No messages yet)"));
            return;
        }

        history.forEach((item, index) => {
            const imageCount = Array.isArray(item.images) ? item.images.length : 0;
            const block = createElement(
                "div",
                `omni-llm-chat__message omni-llm-chat__message--${item.role}`
            );
            const messageActions = createElement("div", "omni-llm-chat__message-actions");
            const messageMeta = createElement("div", "omni-llm-chat__message-meta");
            const messageControls = createElement("div", "omni-llm-chat__message-controls");
            const tokenCount = Number(item.token_count);
            if (Number.isFinite(tokenCount) && tokenCount >= 0) {
                const tokenLabel = createElement(
                    "span",
                    "omni-llm-chat__message-tokens",
                    `${Math.round(tokenCount)} tokens`
                );
                tokenLabel.title = imageCount
                    ? "包含文本、消息模板开销和图片视觉 token 估算(Incl text, template, vision tokens)"
                    : "使用当前模型 tokenizer 统计，并包含少量消息模板开销(Tokenizer count with template overhead)";
                messageMeta.append(tokenLabel);
            }
            const formattedTime = formatMessageTime(item.created_at);
            if (formattedTime) {
                const timeLabel = createElement("span", "omni-llm-chat__message-time", formattedTime);
                timeLabel.title = new Date(Number(item.created_at)).toLocaleString();
                messageMeta.append(timeLabel);
            }
            const copyMessageButton = createElement("button", "omni-llm-chat__message-copy", "⧉");
            copyMessageButton.type = "button";
            copyMessageButton.title = "复制这条消息(Copy Message)";
            copyMessageButton.setAttribute("aria-label", "复制这条消息(Copy Message)");
            copyMessageButton.addEventListener("click", (event) => {
                event.stopPropagation();
                copyText(item.content);
            });
            messageControls.append(copyMessageButton);
            messageActions.append(messageMeta, messageControls);
            block.append(createElement(
                "span",
                "omni-llm-chat__role",
                item.role === "user" ? (imageCount ? `用户 · 图片${imageCount}(User · ${imageCount} img)` : "用户(User)") : "助手(Assistant)"
            ));
            const mediaGallery = item.role === "user" ? createMessageMedia(item.images) : null;
            if (mediaGallery) block.append(mediaGallery);
            block.append(createMessageContent(item.content, copyText), messageActions);
            messages.append(block);
        });
        messages.scrollTop = messages.scrollHeight;
    };

    // 渲染流程状态
    const renderFlow = () => {
        const state = parseFlowState(flowWidget?.value);
        stage.textContent = String(state.stage || "未开始(Idle)");
        stage.title = stage.textContent;
        skillLabel.textContent = state.skill_name || state.skill || "普通对话(Normal)";
        skillLabel.title = skillLabel.textContent;
        options.replaceChildren();
        const optionValues = parseOptions(optionsWidget?.value || "[]");
        optionValues.forEach((value) => {
            const button = createElement("button", "omni-llm-chat__option", value);
            button.type = "button";
            button.title = "发送此选项(Send this option)";
            button.addEventListener("click", () => {
                if (node.__omniLlmChatBusy) return;
                input.value = value;
                send();
            });
            options.append(button);
        });
    };

    // 渲染上下文用量
    const renderContext = () => {
        const state = parseContextState(node.properties.omniLlmContextState);
        const usedTokens = Math.max(0, Number(state.used_tokens) || 0);
        const promptBudget = Math.max(0, Number(state.prompt_budget) || 0);
        const contextLimit = Math.max(0, Number(state.context_limit) || 0);
        const outputReserve = Math.max(0, Number(state.output_reserve) || 0);
        const trimmedMessages = Math.max(0, Number(state.trimmed_messages) || 0);
        const currentRounds = Math.max(0, Number(state.current_rounds) || 0);
        const maxRounds = Math.max(0, Number(state.max_rounds) || 0);
        const remainingTokens = Math.max(0, Number(state.remaining_tokens) || 0);

        if (!promptBudget || !contextLimit) {
            contextPercent.textContent = "--";
            contextTokens.textContent = "已用约 --(Used ~--)";
            contextRounds.textContent = "轮数 --/--(Rds --/--)";
            contextNote.textContent = "剩余约 --(Rem ~--)";
            contextRing.style.background = "conic-gradient(#5d9f80 0deg, #45494f 0deg)";
            contextMeter.title = "完成一次回复后显示上下文占用估算(Shows after first reply)";
            return;
        }

        const rawPercent = usedTokens / promptBudget * 100;
        const displayPercent = Math.max(0, Math.round(rawPercent));
        const ringPercent = Math.min(100, Math.max(0, rawPercent));
        const color = rawPercent >= 90 ? "#d66f6f" : rawPercent >= 75 ? "#d4a653" : "#5d9f80";
        contextPercent.textContent = `${displayPercent}%`;
        contextTokens.textContent = `已用约 ${formatTokenCount(usedTokens)}(Used ~${formatTokenCount(usedTokens)})`;
        contextRounds.textContent = `轮数 ${currentRounds}/${maxRounds || "--"}(Rds ${currentRounds}/${maxRounds || "--"})`;
        contextNote.textContent = trimmedMessages > 0
            ? `剩余约 ${formatTokenCount(remainingTokens)} · 裁${trimmedMessages}(Rem ${formatTokenCount(remainingTokens)} · trim${trimmedMessages})`
            : `剩余约 ${formatTokenCount(remainingTokens)}(Rem ~${formatTokenCount(remainingTokens)})`;
        contextRing.style.background = `conic-gradient(${color} ${ringPercent * 3.6}deg, #45494f 0deg)`;
        contextMeter.title = [
            `当前已使用约 ${Math.round(usedTokens)} tokens`,
            `当前剩余约 ${Math.round(remainingTokens)} tokens`,
            `模型上下文上限 ${Math.round(contextLimit)} tokens`,
            `已预留输出 ${Math.round(outputReserve)} tokens`,
            `当前保留历史 ${Math.round(currentRounds)} / ${Math.round(maxRounds)} 轮`,
            trimmedMessages > 0 ? `本轮因上下文不足裁剪了 ${trimmedMessages} 条历史消息(Ctx trim: ${trimmedMessages})` : "本轮未裁剪历史消息(No trim)",
        ].join("\n");
    };

    // 渲染附件列表
    const renderAttachments = () => {
        const media = parseImages(currentImagesWidget.value);
        attachments.replaceChildren();
        media.forEach((mediaRef, index) => {
            const chip = createElement("span", "omni-llm-chat__attachment");
            chip.title = mediaRef.filename;
            const isVideo = isVideoMedia(mediaRef);
            const mediaLabel = isVideo ? "视频" : "图片";
            const mediaLabelEn = isVideo ? "Vid" : "Img";
            const label = createElement("span", "", `${mediaLabel}${index + 1}(${mediaLabelEn} ${index + 1})`);
            const removeButton = createElement("button", "omni-llm-chat__attachment-remove", "×");
            removeButton.type = "button";
            removeButton.title = `移除${mediaLabel}${index + 1}(Remove ${mediaLabelEn} ${index + 1})`;
            removeButton.addEventListener("click", () => {
                const next = parseImages(currentImagesWidget.value);
                next.splice(index, 1);
                currentImagesWidget.value = JSON.stringify(next);
                renderAttachments();
                node.graph?.setDirtyCanvas?.(true, true);
            });
            chip.append(label, removeButton);
            attachments.append(chip);
        });
    };

    // 重新生成上一条回复
    const regenerateLastReply = () => {
        if (node.__omniLlmChatBusy) return;
        const history = protectedHistory();
        const assistantIndex = history.length - 1;
        const userIndex = assistantIndex - 1;
        if (
            assistantIndex < 1 ||
            history[assistantIndex]?.role !== "assistant" ||
            history[userIndex]?.role !== "user"
        ) return;

        const assistantMessage = history[assistantIndex];
        const userMessage = history[userIndex];
        commitHistoryRaw(
            JSON.stringify(history.slice(0, userIndex)),
            { allowEmptyRegression: true }
        );
        input.value = userMessage.content;
        currentImagesWidget.value = JSON.stringify(userMessage.images || []);
        if (flowWidget) {
            const fallbackState = parseFlowState(flowWidget.value);
            fallbackState.final_result = "";
            fallbackState.stage = "重新生成(Regenerate)";
            flowWidget.value = JSON.stringify(assistantMessage.flow_before || fallbackState);
        }
        if (optionsWidget) optionsWidget.value = "[]";
        render();
        renderFlow();
        renderAttachments();
        send();
    };

    // 设置忙碌状态
    const setBusy = (busy, message = busy ? "正在生成...(Generating...)" : "准备就绪(Ready)", state = busy ? "busy" : "idle") => {
        node.__omniLlmChatBusy = busy;
        sendButton.disabled = busy;
        insertImageButton.disabled = busy;
        clearButton.disabled = busy;
        regenerateButton.disabled = busy;
        input.disabled = busy;
        options.querySelectorAll("button").forEach((button) => { button.disabled = busy; });
        status.textContent = message;
        status.dataset.state = state;
    };

    // 发送消息
    const send = async () => {
        const text = input.value.trim();
        if (!text || node.__omniLlmChatBusy) return;

        protectedHistory();
        userWidget.value = text;
        requestWidget.value = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        if (systemPromptWidget) {
            systemPromptWidget.value = systemPromptInput.value;
        }
        setBusy(true);
        node.graph?.setDirtyCanvas?.(true, true);

        try {
            const prompt = await buildChatOnlyPrompt(node, {
                "user_message": text,
                "chat_history_json": lastValidHistoryRaw,
                "request_id": requestWidget.value,
                "current_images_json": currentImagesWidget.value,
                "flow_state_json": flowWidget?.value || "{}",
                "options_json": optionsWidget?.value || "[]",
                "save_conversation": saveCheckbox.checked,
                "session_id": sessionIdWidget?.value || "",
                "system_prompt": systemPromptInput.value,
            });
            await api.queuePrompt(0, prompt);
            status.textContent = "已加入队列...(Queued...)";
        } catch (error) {
            setBusy(false, `加入队列失败：${error?.message || error}(Queue failed)`, "error");
        }
    };

    // 绑定按钮事件
    sendButton.addEventListener("click", send);
    regenerateButton.addEventListener("click", regenerateLastReply);

    // 卸载模型按钮
    unloadButton.addEventListener("click", async () => {
        if (node.__omniLlmUnloadBusy) return;
        node.__omniLlmUnloadBusy = true;
        unloadButton.disabled = true;
        status.textContent = "正在卸载 Omni LLM 模型...(Unloading...)";
        status.dataset.state = "busy";
        try {
            const response = await api.fetchApi("/omni_llm/unload", { method: "POST" });
            let payload = {};
            try {
                payload = await response.json();
            } catch (_) {
                payload = {};
            }
            if (!response.ok || payload.ok === false) {
                if (response.status === 409) {
                    throw new Error("当前有运行中或排队任务，请等待完成后再卸载模型(Task running, wait to unload)");
                }
                throw new Error(payload.error || `HTTP ${response.status}`);
            }
            status.textContent = payload.unloaded ? "Omni LLM 模型已卸载(Model unloaded)" : "当前没有已加载的 Omni LLM 模型(No model loaded)";
            status.dataset.state = "idle";
        } catch (error) {
            status.textContent = `卸载失败：${error?.message || error}(Unload failed)`;
            status.dataset.state = "error";
        } finally {
            node.__omniLlmUnloadBusy = false;
            unloadButton.disabled = false;
        }
    });
    // 文件选择回调
    insertImageButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = "";
        if (!files.length || node.__omniLlmChatBusy) return;

        const fileCount = files.length;
        const imageCount = files.filter(f => !isVideoFile(f)).length;
        const videoCount = files.filter(f => isVideoFile(f)).length;
        let statusMsg = "正在上传...(Uploading...)";
        if (imageCount > 0 && videoCount > 0) {
            statusMsg = `正在上传 ${imageCount} 张图片和 ${videoCount} 个视频...(Uploading ${imageCount} img, ${videoCount} vid...)`;
        } else if (videoCount > 0) {
            statusMsg = `正在上传 ${videoCount} 个视频...(Uploading ${videoCount} video(s)...)`;
        } else {
            statusMsg = `正在上传 ${imageCount} 张图片...(Uploading ${imageCount} image(s)...)`;
        }
        setBusy(true, statusMsg);
        try {
            const current = parseImages(currentImagesWidget.value);
            const startIndex = current.length;
            for (let index = 0; index < files.length; index += 1) {
                current.push(await uploadChatMedia(files[index], startIndex + index));
            }
            currentImagesWidget.value = JSON.stringify(current);
            renderAttachments();
            let successMsg = `已插入 ${fileCount} 个文件(${fileCount} file(s) added)`;
            if (imageCount > 0 && videoCount > 0) {
                successMsg = `已插入 ${imageCount} 张图片和 ${videoCount} 个视频(${imageCount} img, ${videoCount} vid added)`;
            } else if (videoCount > 0) {
                successMsg = `已插入 ${videoCount} 个视频(${videoCount} video(s) added)`;
            } else {
                successMsg = `已插入 ${imageCount} 张图片(${imageCount} image(s) added)`;
            }
            setBusy(false, successMsg);
            node.graph?.setDirtyCanvas?.(true, true);
            input.focus();
        } catch (error) {
            setBusy(false, `插入失败：${error?.message || error}(Insert failed)`, "error");
        }
    });
    // 复选框事件监听
    saveCheckbox.addEventListener("change", () => {
        if (saveConversationWidget) {
            saveConversationWidget.value = saveCheckbox.checked;
        }
        node.graph?.setDirtyCanvas?.(true, true);
    });
    
    // 清空对话
    clearButton.addEventListener("click", () => {
        // 如果正在加载历史，取消加载
        if (historyLoadingAbort) {
            historyLoadingAbort.abort();
            historyLoadingAbort = null;
            status.textContent = "加载已取消(Cancelled)";
            status.dataset.state = "idle";
            return;
        }
        
        // 如果勾选了保存对话，先保存当前对话
        if (saveCheckbox.checked) {
            const history = parseHistory(lastValidHistoryRaw);
            if (history.length > 0) {
                try {
                    // 调用保存逻辑
                    const sessionToSave = sessionIdWidget?.value || "";
                    // 发送保存请求到后端
                    api.fetchApi("/omni_llm/save_conversation", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ history, session_id: sessionToSave }),
                    }).catch(err => console.warn("保存对话失败:", err));
                } catch (e) {
                    console.warn("保存对话失败:", e);
                }
            }
        }
        
        commitHistoryRaw("[]", { allowEmptyRegression: true });
        userWidget.value = "";
        requestWidget.value = `${Date.now()}-clear`;
        currentImagesWidget.value = "[]";
        if (flowWidget) flowWidget.value = "{}";
        if (optionsWidget) optionsWidget.value = "[]";
        node.properties.omniLlmContextState = {};
        input.value = "";
        
        // 重置 session_id 以新建文件
        if (sessionIdWidget) {
            sessionIdWidget.value = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        }
        
        render();
        renderFlow();
        renderContext();
        renderAttachments();
        setBusy(false, "会话已清空(Cleared)");
        node.graph?.setDirtyCanvas?.(true, true);
    });
    // Enter 发送，Shift+Enter 换行
    input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            send();
        }
    });

    // 注册 DOM widget
const chatPanelHeight = (size = node.size) => {
    const nodeHeight = Number(size?.[1] ?? node.size?.[1] ?? 560);
    const clampedNodeHeight = Math.min(nodeHeight, CHAT_MAX_HEIGHT + CHAT_NODE_CHROME_HEIGHT);
    return Math.max(CHAT_MIN_HEIGHT, clampedNodeHeight - CHAT_NODE_CHROME_HEIGHT);
};
const domWidget = node.addDOMWidget("omni_llm_chat", "omni_llm_chat", root, {
    getMinHeight: () => CHAT_MIN_HEIGHT,
    getMaxHeight: () => CHAT_MAX_HEIGHT,
    getHeight: () => chatPanelHeight(),
    hideOnZoom: false,
    serialize: false,
});

    const updateChatLayout = (size = node.size) => {
        root.style.height = `${chatPanelHeight(size)}px`;
        root.style.minHeight = `${CHAT_MIN_HEIGHT}px`;
        node.graph?.setDirtyCanvas?.(true, true);
    };

    domWidget.computeSize = (width) => [
        Math.max(360, width || node.size?.[0] || 440),
        chatPanelHeight(),
    ];
    domWidget.afterResize = () => updateChatLayout();
    const domWidgetIndex = node.widgets.indexOf(domWidget);
    if (domWidgetIndex > 0) {
        node.widgets.splice(domWidgetIndex, 1);
        node.widgets.unshift(domWidget);
    }

    const originalOnResize = node.onResize;
    node.onResize = function (size) {
        const result = originalOnResize?.apply(this, arguments);
        updateChatLayout(size || this.size);
        return result;
    };

    // 处理执行结果回调
    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        originalOnExecuted?.apply(this, arguments);
        const sent = Boolean(firstValue(output?.sent));
        const previousHistory = parseHistory(lastValidHistoryRaw);
        const rawHistory = firstValue(output?.chat_history_json);
        const candidateRaw = validHistoryRaw(rawHistory);
        const candidateHistory = candidateRaw === null ? null : parseHistory(candidateRaw);
        let historyError = "";
        if (candidateHistory === null) {
            if (sent || typeof rawHistory === "string") {
                historyError = "返回的历史数据异常，已保留发送前的对话(History error, kept previous)";
            }
        } else if (candidateHistory.length === 0 && (sent || previousHistory.length > 0)) {
            historyError = "返回了异常空历史，已保留发送前的对话(Empty history, kept previous)";
        } else {
            commitHistoryRaw(candidateRaw);
        }
        const rawFlow = firstValue(output?.flow_state_json);
        if (flowWidget && typeof rawFlow === "string") flowWidget.value = rawFlow;
        const rawOptions = firstValue(output?.options_json);
        if (optionsWidget) optionsWidget.value = typeof rawOptions === "string" ? rawOptions : "[]";
        const rawContextState = firstValue(output?.context_state_json);
        if (typeof rawContextState === "string") {
            node.properties.omniLlmContextState = parseContextState(rawContextState);
        }
        if (sent && !historyError) {
            userWidget.value = "";
            currentImagesWidget.value = "[]";
            input.value = "";
        }
        render();
        renderFlow();
        renderContext();
        renderAttachments();
        if (historyError) {
            setBusy(false, historyError, "error");
        } else {
            setBusy(false);
        }
        this.graph?.setDirtyCanvas?.(true, true);
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        window.setTimeout(() => {
            applyChatFontSize(this.properties?.omniLlmChatFontSize);
            render();
            renderFlow();
            renderContext();
            renderAttachments();
        }, 0);
        return result;
    };

    // 监听执行错误
    const handleExecutionFailure = () => {
        if (!node.__omniLlmChatBusy) return;
        setBusy(false, "生成失败，请查看 ComfyUI 日志(Generation failed, check logs)", "error");
    };
    api.addEventListener("execution_error", handleExecutionFailure);
    api.addEventListener("execution_interrupted", handleExecutionFailure);

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        api.removeEventListener("execution_error", handleExecutionFailure);
        api.removeEventListener("execution_interrupted", handleExecutionFailure);
        return originalOnRemoved?.apply(this, arguments);
    };

    node.setSize([
        Math.max(node.size?.[0] || 0, 390),
        Math.max(node.size?.[1] || 0, 560),
    ]);
    node.minWidth = 390;
    window.setTimeout(() => {
        updateChatLayout();
        render();
        renderFlow();
        renderContext();
        renderAttachments();
    }, 0);
}

app.registerExtension({
    name: "OmniLlm.RealtimeChat",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupChatNode(this);
            return r;
        };
    },
});
