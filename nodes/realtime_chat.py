# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Realtime Chat Node

实时对话节点 + Skill 多阶段协议状态机。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import sys
import re
import io
import json
import time
import base64

# 添加项目根目录到路径（保持与其他节点模块一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from common import (
    LLAMA_CPP_STORAGE,
    mm,
    folder_paths,
    BaseInferenceEngine,
)

# Skill 相关工具（节点模块按顶层模块加载，保持与 model_loader 等一致）
from skill_loader import get_skill, read_reference, read_skill_body

# ---------------------------------------------------------------- 服务器路由
def _call_api_chat_completion(api_base_url: str, api_key: str, api_model: str, messages: list, params: dict) -> str:
    """调用外部 API（OpenAI 兼容格式）"""
    import requests as _requests
    
    if not api_key:
        raise RuntimeError("API 密钥为空，请在 API 配置节点中填写 api_key")
    if not api_base_url:
        raise RuntimeError("API 地址为空，请在 API 配置节点中填写 api_base")
    if not api_model:
        raise RuntimeError("模型名称为空，请在 API 配置节点中填写 model_name")
    
    # 智能拼接 URL：避免重复 /v1
    base = api_base_url.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    elif "/v1/" in base or base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": api_model,
        "messages": messages,
        "max_tokens": params.get("max_tokens", 1024),
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.9),
        "stream": False,
    }
    if params.get("seed", -1) >= 0:
        payload["seed"] = params["seed"]
    
    timeout = int(params.get("timeout", 300))

    def _do_request():
        try:
            return _requests.post(url, headers=headers, json=payload, timeout=timeout)
        except _requests.ProxyError:
            # 系统代理不可用（如 Clash 未启动）时绕过代理直连重试（国内 API 通常无需代理）
            try:
                return _requests.post(url, headers=headers, json=payload, timeout=timeout,
                                      proxies={"http": None, "https": None})
            except _requests.ConnectionError as e:
                raise RuntimeError(f"网络连接失败，无法访问 {api_base_url}（已尝试直连和系统代理）。请检查网络、VPN/代理、防火墙设置。") from e
        except _requests.ConnectionError as e:
            raise RuntimeError(f"网络连接失败，无法访问 {api_base_url}。请检查网络、VPN/代理、防火墙设置。") from e
        except _requests.Timeout as e:
            raise RuntimeError(f"API 请求超时（{timeout}秒）。请检查网络或稍后重试。") from e
        except _requests.RequestException as e:
            raise RuntimeError(f"API 请求异常：{e}") from e

    # 推理模型（如 deepseek-flash）的思考 token 也计入 max_tokens，复杂任务可能导致
    # 思考过程占满配额、正文 content 为空（finish_reason=length），此时自动翻倍配额重试
    max_tokens_cap = 8192
    while True:
        resp = _do_request()

        if resp.status_code != 200:
            error_msg = resp.text[:500]
            status_hints = {
                400: f"请求参数错误：{error_msg}",
                401: "API 密钥无效或已过期，请检查 api_key",
                402: "API 余额不足，请充值后重试",
                403: "没有 API 访问权限",
                404: f"接口不存在：{url}",
                429: "请求过于频繁，请稍后重试",
                500: "API 服务端异常，请稍后重试",
                502: "API 服务暂时不可用，请稍后重试",
                503: "API 服务繁忙，请稍后重试",
            }
            hint = status_hints.get(resp.status_code, f"HTTP {resp.status_code}")
            raise RuntimeError(f"API 错误 {resp.status_code}：{hint}")

        try:
            result = resp.json()
        except Exception:
            raise RuntimeError(f"API 响应解析失败：{resp.text[:300]}")

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"API 返回格式异常，无 choices 字段：{str(result)[:300]}")

        first_choice = choices[0]
        content = first_choice.get("message", {}).get("content")
        if content is None:
            content = first_choice.get("text", "")

        current_max_tokens = payload["max_tokens"]
        if (not content) and first_choice.get("finish_reason") == "length" \
                and current_max_tokens < max_tokens_cap:
            payload["max_tokens"] = min(current_max_tokens * 2, max_tokens_cap)
            print(f"【实时对话】思考过程占满 max_tokens={current_max_tokens} 导致正文为空，"
                  f"自动提升至 {payload['max_tokens']} 重试...")
            continue

        return content or ""

# ---------------------------------------------------------------- 服务器路由
try:
    from aiohttp import web
    from server import PromptServer
    _SERVER_AVAILABLE = True
except Exception as _server_import_error:  # pragma: no cover - 非 ComfyUI 环境
    web = None
    PromptServer = None
    _SERVER_AVAILABLE = False
    _SERVER_IMPORT_ERROR = str(_server_import_error)

_UNLOAD_ROUTE_REGISTERED = False


def _register_unload_route():
    """注册 /omni_llm/unload 路由（仅注册一次，供前端聊天窗卸载按钮调用）"""
    global _UNLOAD_ROUTE_REGISTERED
    if _UNLOAD_ROUTE_REGISTERED or not _SERVER_AVAILABLE:
        return
    try:
        @PromptServer.instance.routes.post("/omni_llm/unload")
        async def _unload_omni_llm_model(request):
            prompt_queue = getattr(PromptServer.instance, "prompt_queue", None)
            if prompt_queue is not None:
                running, queued = prompt_queue.get_current_queue_volatile()
                if running or queued:
                    return web.json_response(
                        {
                            "ok": False,
                            "error": "ComfyUI 有正在运行或排队中的任务，请等待队列空闲后再卸载模型。"
                        },
                        status=409,
                    )
            was_loaded = LLAMA_CPP_STORAGE.llm is not None
            LLAMA_CPP_STORAGE.clean()
            print(
                f"【实时对话】前端请求卸载 LLM 模型：{'已卸载' if was_loaded else '没有已加载的模型'}",
                flush=True,
            )
            return web.json_response({"ok": True, "unloaded": was_loaded})

        _UNLOAD_ROUTE_REGISTERED = True
    except Exception as e:
        print(f"【实时对话】注册卸载路由失败（忽略）：{e}")


_register_unload_route()

# ---------------------------------------------------------------- API 配置路由
_API_CONFIG_ROUTES_REGISTERED = False

def _register_api_config_routes():
    """注册 /omni_llm/api/saved_configs 路由（供前端获取已保存的 API 配置）"""
    global _API_CONFIG_ROUTES_REGISTERED
    if _API_CONFIG_ROUTES_REGISTERED or not _SERVER_AVAILABLE:
        return
    try:
        @PromptServer.instance.routes.get("/omni_llm/api/saved_configs")
        async def _get_saved_configs(request):
            import importlib
            api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api")
            preset_path = os.path.join(api_dir, "api_provider_presets.py")
            if not os.path.exists(preset_path):
                return web.json_response({"configs": {}})
            try:
                ns = {}
                with open(preset_path, "r", encoding="utf-8") as f:
                    exec(f.read(), ns)
                configs = ns.get("SAVED_CONFIGS", {})
                return web.json_response({"configs": configs})
            except Exception as e:
                return web.json_response({"configs": {}, "error": str(e)})

        @PromptServer.instance.routes.post("/omni_llm/api/save_config")
        async def _save_config(request):
            try:
                data = await request.json()
                name = data.get("name", "").strip()
                config = data.get("config", {})
                if not name:
                    return web.json_response({"ok": False, "error": "预设名称为空"}, status=400)
                
                # 合并提供商默认值
                provider = config.get("provider", "自定义")
                _provider_defaults = {
                    "OpenAI": {"api_base": "https://api.openai.com/v1", "context_limit": 1048576},
                    "Anthropic": {"api_base": "https://api.anthropic.com/v1", "context_limit": 1048576},
                    "Grok": {"api_base": "https://api.x.ai/v1", "context_limit": 1048576},
                    "Google": {"api_base": "https://generativelanguage.googleapis.com/v1beta/openai", "context_limit": 1048576},
                    "DeepSeek": {"api_base": "https://api.deepseek.com", "context_limit": 1048576},
                    "阿里云": {"api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "context_limit": 1048576},
                    "火山引擎": {"api_base": "https://ark.cn-beijing.volces.com/api/v3", "context_limit": 1048576},
                    "MiniMax": {"api_base": "https://api.minimaxi.com/v1", "context_limit": 1048576},
                    "Kimi": {"api_base": "https://api.moonshot.cn/v1", "context_limit": 1048576},
                }
                defaults = _provider_defaults.get(provider, {})
                api_base = config.get("base_url", "") or defaults.get("api_base", "")
                context_limit = config.get("context_limit", 0) or defaults.get("context_limit", 0)

                api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api")
                preset_path = os.path.join(api_dir, "api_provider_presets.py")
                
                with open(preset_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                entry_lines = [
                    f'    "{name}": {{',
                    f'        "provider": "{provider}",',
                    f'        "api_base": "{api_base}",',
                    f'        "model_id": "{config.get("model", "")}",',
                    f'        "api_key": "{config.get("api_key", "")}",',
                    f'        "context_limit": {context_limit},',
                    f'        "timeout": {config.get("timeout", 300)},',
                    f'        "max_history_rounds": {config.get("max_history_rounds", 100)},',
                    f'        "max_edge": {config.get("max_edge", 1024)},',
                    f'        "preserve_thinking": {config.get("preserve_thinking", False)},',
                    f'    }},',
                ]
                saved_entry = "\n".join(entry_lines)
                
                import re as _re
                m = _re.search(r'SAVED_CONFIGS\s*=\s*\{', content)
                if m:
                    start = m.end()
                    depth = 1
                    i = start
                    while i < len(content) and depth > 0:
                        if content[i] == "{": depth += 1
                        elif content[i] == "}": depth -= 1
                        i += 1
                    end = i
                    old_inner = content[start:end - 1].strip()
                    pat = _re.compile(rf'\s*"{_re.escape(name)}"\s*:\s*\{{[^}}]*\}},')
                    new_inner = pat.sub("", old_inner).strip()
                    if new_inner:
                        new_inner = new_inner + "\n" + saved_entry
                    else:
                        new_inner = saved_entry
                    new_block = "SAVED_CONFIGS = {\n" + new_inner + "\n}"
                    content = content[:m.start()] + new_block + content[end:]
                else:
                    content = content.rstrip() + "\n\nSAVED_CONFIGS = {\n" + saved_entry + "\n}\n"
                
                with open(preset_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                return web.json_response({"ok": True, "name": name})
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/omni_llm/api/delete_config")
        async def _delete_config(request):
            try:
                data = await request.json()
                name = data.get("name", "").strip()
                if not name:
                    return web.json_response({"ok": False, "error": "预设名称为空"}, status=400)
                
                api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api")
                preset_path = os.path.join(api_dir, "api_provider_presets.py")
                
                with open(preset_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                import re as _re
                m = _re.search(r'SAVED_CONFIGS\s*=\s*\{', content)
                if not m:
                    return web.json_response({"ok": False, "error": "没有找到 SAVED_CONFIGS"}, status=404)
                
                start = m.end()
                depth = 1
                i = start
                while i < len(content) and depth > 0:
                    if content[i] == "{": depth += 1
                    elif content[i] == "}": depth -= 1
                    i += 1
                end = i
                old_inner = content[start:end - 1].strip()
                pat = _re.compile(rf'\s*"{_re.escape(name)}"\s*:\s*\{{[^}}]*\}},')
                new_inner = pat.sub("", old_inner).strip()
                new_block = "SAVED_CONFIGS = {\n" + new_inner + "\n}" if new_inner else "SAVED_CONFIGS = {}"
                content = content[:m.start()] + new_block + content[end:]
                
                with open(preset_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                return web.json_response({"ok": True, "name": name})
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=500)

        _API_CONFIG_ROUTES_REGISTERED = True
    except Exception as e:
        print(f"【API 配置】注册路由失败（忽略）：{e}")


_register_api_config_routes()

# ---------------------------------------------------------------- 对话保存路由
_SAVE_CONVERSATION_ROUTES_REGISTERED = False

def _register_save_conversation_routes():
    """注册 /omni_llm/save_conversation 路由（供前端清空时保存对话）"""
    global _SAVE_CONVERSATION_ROUTES_REGISTERED
    if _SAVE_CONVERSATION_ROUTES_REGISTERED or not _SERVER_AVAILABLE:
        return
    try:
        @PromptServer.instance.routes.post("/omni_llm/save_conversation")
        async def _save_conversation(request):
            try:
                data = await request.json()
                history = data.get("history", [])
                session_id = data.get("session_id", "")
                
                if not history:
                    return web.json_response({"ok": True, "message": "没有对话内容需要保存"})
                
                # 调用保存函数
                filepath = _save_conversation_to_file(history, session_id)
                return web.json_response({"ok": True, "filepath": filepath})
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=500)
        
        _SAVE_CONVERSATION_ROUTES_REGISTERED = True
    except Exception as e:
        print(f"【对话保存】注册路由失败（忽略）：{e}")


_register_save_conversation_routes()

# ---------------------------------------------------------------- 对话保存与历史路由
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def _ensure_cache_dir():
    """确保 cache 目录存在"""
    if not os.path.exists(_CACHE_DIR):
        os.makedirs(_CACHE_DIR, exist_ok=True)


_ensure_cache_dir()
print(f"【历史对话】_SERVER_AVAILABLE={_SERVER_AVAILABLE}, _CACHE_DIR={_CACHE_DIR}")

if _SERVER_AVAILABLE:
    @PromptServer.instance.routes.get("/omni_llm/history/list")
    async def _list_history(request):
        try:
            _ensure_cache_dir()
            files = []
            for filename in os.listdir(_CACHE_DIR):
                if filename.startswith("chat_") and filename.endswith(".txt"):
                    filepath = os.path.join(_CACHE_DIR, filename)
                    stat = os.stat(filepath)
                    files.append({
                        "filename": filename,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    })
            files.sort(key=lambda x: x["mtime"], reverse=True)
            return web.json_response({"ok": True, "files": files})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @PromptServer.instance.routes.get("/omni_llm/history/content")
    async def _get_history_content(request):
        try:
            filename = request.query.get("filename", "")
            if not filename or not filename.startswith("chat_") or not filename.endswith(".txt"):
                return web.json_response({"ok": False, "error": "无效的文件名"}, status=400)
            filepath = os.path.join(_CACHE_DIR, filename)
            if not os.path.exists(filepath):
                return web.json_response({"ok": False, "error": "文件不存在"}, status=404)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return web.json_response({"ok": True, "content": content, "filename": filename})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    @PromptServer.instance.routes.post("/omni_llm/history/delete")
    async def _delete_history(request):
        try:
            data = await request.json()
            filename = data.get("filename", "")
            if not filename or not filename.startswith("chat_") or not filename.endswith(".txt"):
                return web.json_response({"ok": False, "error": "无效的文件名"}, status=400)
            filepath = os.path.join(_CACHE_DIR, filename)
            if not os.path.exists(filepath):
                return web.json_response({"ok": False, "error": "文件不存在"}, status=404)
            os.remove(filepath)
            return web.json_response({"ok": True, "filename": filename})
        except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=500)
    print("【历史对话】路由注册成功: /omni_llm/history/list, /omni_llm/history/content, /omni_llm/history/delete")

# ---------------------------------------------------------------- 常量与协议
_DEFAULT_CHAT_SYSTEM_PROMPT = "你是一个有帮助的AI助手。"
SKILL_STATE_TAG = re.compile(r"<omni_llm_state>\s*(\{.*?\})\s*</omni_llm_state>", re.DOTALL)
_JPEG_QUALITY = 88


def _save_conversation_to_file(history: list, session_id: str):
    """将对话历史保存到文本文件
    
    Args:
        history: 对话历史列表
        session_id: 会话ID，用于文件命名
    """
    if not history:
        return
    
    _ensure_cache_dir()
    
    # 生成文件名：使用 session_id 或时间戳
    if session_id:
        # 清理文件名中的非法字符
        safe_session_id = re.sub(r'[<>:"/\\|?*]', '_', session_id)
        filename = f"chat_{safe_session_id}.txt"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.txt"
    
    filepath = os.path.join(_CACHE_DIR, filename)
    
    # 构建文本内容
    lines = []
    lines.append("=" * 60)
    lines.append(f"对话记录 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")
    
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        created_at = item.get("created_at", 0)
        
        if role == "user":
            role_label = "用户"
        elif role == "assistant":
            role_label = "助手"
        else:
            role_label = role
        
        # 格式化时间
        time_str = ""
        if created_at > 0:
            try:
                time_str = f" ({time.strftime('%H:%M:%S', time.localtime(created_at / 1000))})"
            except Exception:
                pass
        
        lines.append(f"【{role_label}】{time_str}")
        lines.append("-" * 40)
        lines.append(content)
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("")
    
    # 写入文件（追加模式，同一会话的内容追加到同一文件）
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"【对话保存】已保存到: {filepath}")
    return filepath

SKILL_EXECUTION_PROTOCOL = """
你正在通过 ComfyUI 的本地 Skill 执行器工作。严格遵循下方当前 Skill，并遵守以下交互协议：
1. 只完成当前 Skill 能在文本对话中完成的工作。Skill 提到画布、媒体生成、联网工具或 Hub agent 时，不得声称已经执行；应输出对应方案、提示词或说明当前需要连接的 ComfyUI 节点。
2. 信息不足或到达确认门时，先提问并等待用户。每次只推进当前阶段，不得替用户确认。
3. 回复正文之后必须追加一个状态标记，且标记必须是回复的最后内容：
<omni_llm_state>{"stage":"当前阶段","options":["选项1","选项2"],"load_references":[],"final":false}</omni_llm_state>
4. 需要用户选择时，options 提供 2 到 6 个可直接作为用户回复的完整选项；开放问题可以使用空数组。
5. Skill 要求读取 reference 时，如果该文件尚未出现在“已加载 references”，必须先把相对路径写入 load_references。执行器会加载文件并让你重新回答，不要猜测文件内容。
6. 只有已经交付当前 Skill 要求的最终文本产物时才设置 final=true。最终产物必须完整写在状态标记之前。
7. 使用简体中文交流和输出；协议字段、固定字段、标签以及用户要求原样保留的内容除外。
""".strip()

# ---------------------------------------------------------------- 图片工具
def _resize_pil_to_max_edge(pil: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return pil
    w, h = pil.size
    long_edge = max(w, h)
    if long_edge <= max_edge:
        return pil
    scale = max_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), resample=Image.BICUBIC)


def _encode_pil_as_jpeg(pil: Image.Image) -> bytes:
    if pil.mode in ("RGBA", "LA") or "transparency" in pil.info:
        rgba = pil.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        pil = background.convert("RGB")
    elif pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")

    buf = io.BytesIO()
    try:
        pil.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True, progressive=True)
    except Exception:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _image_path_to_data_uri(image_path: str, max_edge: int) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到对话图片：{image_path}")

    with Image.open(image_path) as pil:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = _resize_pil_to_max_edge(pil, max_edge)
        image_bytes = _encode_pil_as_jpeg(pil)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


def _image_ref_to_data_uri(image_ref: dict, max_edge: int) -> str:
    input_root = os.path.realpath(folder_paths.get_input_directory())
    image_path = os.path.realpath(
        os.path.join(input_root, image_ref.get("subfolder", ""), image_ref["filename"])
    )
    try:
        is_inside_input = os.path.commonpath([input_root, image_path]) == input_root
    except ValueError:
        is_inside_input = False
    if not is_inside_input:
        raise ValueError("图片路径超出 ComfyUI input 目录。")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"找不到对话图片：{image_path}")
    return _image_path_to_data_uri(image_path, int(max_edge))


def _video_ref_to_content(video_ref: dict, max_edge: int, max_frames: int = 8, native_video: bool = False) -> list:
    """从视频文件中提取内容
    
    Args:
        video_ref: 视频引用 {filename, subfolder, type, media_type}
        max_edge: 图片最大边长
        max_frames: 最大帧数（提取帧模式）
        native_video: 是否发送原生视频（base64）
    
    Returns:
        content_items: OpenAI 格式的 content 列表
    """
    input_root = os.path.realpath(folder_paths.get_input_directory())
    video_path = os.path.realpath(
        os.path.join(input_root, video_ref.get("subfolder", ""), video_ref["filename"])
    )
    try:
        is_inside_input = os.path.commonpath([input_root, video_path]) == input_root
    except ValueError:
        is_inside_input = False
    if not is_inside_input:
        raise ValueError("视频路径超出 ComfyUI input 目录。")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"找不到对话视频：{video_path}")

    content_items = []
    
    # 原生视频模式：直接发送 base64 视频
    if native_video:
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
            
            # 根据文件扩展名确定 MIME 类型
            ext = os.path.splitext(video_ref["filename"])[1].lower()
            mime_map = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".flv": "video/x-flv",
                ".wmv": "video/x-ms-wmv",
                ".m4v": "video/x-m4v",
            }
            mime_type = mime_map.get(ext, "video/mp4")
            
            content_items.append({
                "type": "video_url",
                "video_url": {"url": f"data:{mime_type};base64,{video_b64}"}
            })
            return content_items
        except Exception as e:
            print(f"【视频处理】原生视频编码失败，回退到帧提取：{e}")
    
    # 帧提取模式：从视频中提取关键帧
    try:
        import av
        container = av.open(video_path)
        video_stream = None
        for stream in container.streams:
            if isinstance(stream, av.video.stream.VideoStream):
                video_stream = stream
                break
        if video_stream is None:
            container.close()
            return content_items

        total_frames = video_stream.frames or 0
        if total_frames <= 0:
            for frame in container.decode(video_stream):
                total_frames += 1
            container.seek(0)
            video_stream = None
            for stream in container.streams:
                if isinstance(stream, av.video.stream.VideoStream):
                    video_stream = stream
                    break

        if total_frames <= 0:
            container.close()
            return content_items

        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]

        frame_count = 0
        for frame in container.decode(video_stream):
            if frame_count in indices:
                img = frame.to_ndarray(format="rgb24")
                pil = Image.fromarray(img)
                pil = _resize_pil_to_max_edge(pil, max_edge)
                image_bytes = _encode_pil_as_jpeg(pil)
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            frame_count += 1
            if frame_count > indices[-1]:
                break

        container.close()
    except Exception as e:
        print(f"【视频处理】提取视频帧失败：{e}")
    return content_items


def _video_ref_to_data_uri(video_ref: dict, max_edge: int) -> str:
    """将视频第一帧转换为 data URI（用于预览或简单场景）"""
    import av
    input_root = os.path.realpath(folder_paths.get_input_directory())
    video_path = os.path.realpath(
        os.path.join(input_root, video_ref.get("subfolder", ""), video_ref["filename"])
    )
    try:
        is_inside_input = os.path.commonpath([input_root, video_path]) == input_root
    except ValueError:
        is_inside_input = False
    if not is_inside_input:
        raise ValueError("视频路径超出 ComfyUI input 目录。")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"找不到对话视频：{video_path}")

    try:
        container = av.open(video_path)
        video_stream = None
        for stream in container.streams:
            if isinstance(stream, av.video.stream.VideoStream):
                video_stream = stream
                break
        if video_stream is None:
            container.close()
            raise ValueError("视频中未找到视频流")

        frame = next(container.decode(video_stream))
        container.close()
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        pil = _resize_pil_to_max_edge(pil, max_edge)
        image_bytes = _encode_pil_as_jpeg(pil)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"
    except Exception as e:
        print(f"【视频处理】提取视频首帧失败：{e}")
        return ""


# ---------------------------------------------------------------- 输出清洗
def _clean_reasoning_markers(text):
    """清理回复中的思考内容与残留标记（think 块 + 原生 thinking 结束标记）"""
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    cleaned = text
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, count=1, flags=re.DOTALL | re.IGNORECASE)

    for marker in ("<|end_of_thinking|>", "<|end_of_solution|>", "<|finish_reason|>"):
        if marker in cleaned:
            parts = cleaned.split(marker)
            if len(parts) >= 2 and parts[-1].strip():
                cleaned = parts[-1].strip()
                break

    for marker in (
        "<thinking>", "</thinking>",
        "<think>", "</think>",
        "<analysis>", "</analysis>",
        "<|end_of_thinking|>",
        "<|end_of_solution|>",
        "<|finish_reason|>",
    ):
        cleaned = cleaned.replace(marker, "")

    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------- 历史 / 消息
def _normalize_image_ref(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    filename = os.path.basename(str(item.get("filename") or item.get("name") or "").strip())
    subfolder = str(item.get("subfolder") or "").replace("\\", "/").strip("/")
    image_type = str(item.get("type") or "input").strip().lower()
    media_type = str(item.get("media_type") or "image").strip().lower()
    if not filename or image_type != "input":
        return None
    if any(part in ("", ".", "..") for part in subfolder.split("/")) and subfolder:
        return None
    return {"filename": filename, "subfolder": subfolder, "type": "input", "media_type": media_type}


def _parse_image_list(raw_images) -> list:
    if isinstance(raw_images, str):
        if not raw_images.strip():
            return []
        try:
            raw_images = json.loads(raw_images)
        except json.JSONDecodeError as exc:
            raise ValueError(f"待发送图片数据损坏，无法解析：{exc}") from exc
    if not isinstance(raw_images, list):
        return []

    images = []
    for item in raw_images:
        normalized = _normalize_image_ref(item)
        if normalized is not None:
            images.append(normalized)
    return images


def _parse_chat_history(raw_history: str) -> list:
    if not raw_history or not raw_history.strip():
        return []

    try:
        data = json.loads(raw_history)
    except json.JSONDecodeError as exc:
        raise ValueError(f"对话历史数据损坏，无法解析：{exc}") from exc

    if not isinstance(data, list):
        raise ValueError("对话历史格式无效，应为消息列表。请在节点中清空会话后重试。")

    history = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            message = {"role": role, "content": content}
            images = _parse_image_list(item.get("images")) if role == "user" else []
            if images:
                message["images"] = images
            try:
                token_count = int(item.get("token_count"))
            except (TypeError, ValueError):
                token_count = -1
            if token_count >= 0:
                message["token_count"] = token_count
            try:
                created_at = int(item.get("created_at"))
            except (TypeError, ValueError):
                created_at = 0
            if created_at > 0:
                message["created_at"] = created_at
            if role == "assistant" and isinstance(item.get("flow_before"), dict):
                flow_before = item["flow_before"]
                message["flow_before"] = {
                    "skill": str(flow_before.get("skill") or ""),
                    "skill_name": str(flow_before.get("skill_name") or "")[:80],
                    "stage": str(flow_before.get("stage") or "未开始")[:40],
                    "loaded_references": [
                        str(reference)
                        for reference in flow_before.get("loaded_references", [])
                        if isinstance(reference, str)
                    ],
                    "final_result": str(flow_before.get("final_result") or ""),
                }
            history.append(message)
    return history


def _trim_by_rounds(history: list, max_rounds: int) -> list:
    max_messages = max(1, int(max_rounds)) * 2
    trimmed = history[-max_messages:]
    while trimmed and trimmed[0]["role"] == "assistant":
        trimmed.pop(0)
    return trimmed


# ---------------------------------------------------------------- Token 估算
def _estimate_text_tokens(llm, text: str) -> int:
    if not text:
        return 0
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return max(1, len(text.encode("utf-8")) // 3)


def _estimate_message_tokens(llm, message: dict) -> int:
    return (
        _estimate_text_tokens(llm, str(message.get("content") or ""))
        + 8
        + len(message.get("images") or []) * 2048
    )


def _estimate_messages_tokens(llm, messages: list) -> int:
    return sum(_estimate_message_tokens(llm, message) for message in messages) + 16


def _parse_request_time_ms(request_id: str) -> int:
    now_ms = int(time.time() * 1000)
    try:
        candidate = int(str(request_id or "").split("-", 1)[0])
    except (TypeError, ValueError):
        return now_ms
    if 946684800000 <= candidate <= now_ms + 300000:
        return candidate
    return now_ms


def _compute_context_budget(max_tokens: int, n_ctx: int):
    output_reserve = min(max(32, int(max_tokens)), max(32, int(n_ctx) - 512))
    prompt_budget = max(256, int(n_ctx) - output_reserve - 128)
    return output_reserve, prompt_budget


def _trim_by_context(
    llm,
    history: list,
    system_text: str,
    user_text: str,
    max_tokens: int,
    n_ctx: int,
    current_image_count: int = 0,
) -> list:
    """按 n_ctx 预算裁剪历史，始终从最旧的完整一问一答开始删除"""
    _output_reserve, prompt_budget = _compute_context_budget(max_tokens, n_ctx)

    prefix = []
    if system_text:
        prefix.append({"role": "system", "content": system_text})
    suffix = [{"role": "user", "content": user_text}]
    if current_image_count > 0:
        suffix[0]["images"] = [{}] * int(current_image_count)

    trimmed = list(history)
    while trimmed and _estimate_messages_tokens(llm, prefix + trimmed + suffix) > prompt_budget:
        trimmed.pop(0)
        if trimmed and trimmed[0]["role"] == "assistant":
            trimmed.pop(0)

    required_tokens = _estimate_messages_tokens(llm, prefix + trimmed + suffix)
    if required_tokens > prompt_budget:
        raise ValueError(
            f"当前系统提示词、Skill reference 和用户消息约需 {required_tokens} tokens，"
            f"超过可用输入上下文 {prompt_budget}。请提高模型上下文长度或使用更短的 Skill。"
        )
    return trimmed


def _build_context_state(
    llm,
    system_text: str,
    history: list,
    max_tokens: int,
    n_ctx: int,
    max_rounds: int,
    trimmed_messages: int = 0,
) -> dict:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(history)
    used_tokens = _estimate_messages_tokens(llm, messages)
    output_reserve, prompt_budget = _compute_context_budget(max_tokens, n_ctx)
    percent = (used_tokens / prompt_budget * 100.0) if prompt_budget > 0 else 0.0
    return {
        "used_tokens": int(used_tokens),
        "prompt_budget": int(prompt_budget),
        "context_limit": int(n_ctx),
        "output_reserve": int(output_reserve),
        "remaining_tokens": max(0, int(prompt_budget) - int(used_tokens)),
        "percent": round(percent, 1),
        "trimmed_messages": max(0, int(trimmed_messages)),
        "current_rounds": sum(1 for message in history if message.get("role") == "user"),
        "max_rounds": max(1, int(max_rounds)),
        "estimated": True,
    }


# ---------------------------------------------------------------- 模型同步
def _get_n_ctx(storage, llm) -> int:
    config = getattr(storage, "current_config", None)
    if isinstance(config, dict) and config.get("n_ctx"):
        return int(config["n_ctx"])
    try:
        method = getattr(llm, "n_ctx", None)
        if callable(method):
            return int(method())
        if isinstance(method, int):
            return method
    except Exception:
        pass
    return 8192


def _sync_loaded_llm(storage):
    """校验共享存储中已加载的模型；模型对象始终来自现有模型加载器全局存储"""
    llm = getattr(storage, "llm", None)
    if llm is None:
        raise RuntimeError("模型未加载：请先运行『Omni LLM Model Loader』加载模型。")
    return llm


def _reset_llm_state(llm) -> None:
    try:
        ctx = getattr(llm, "_ctx", None)
        if ctx is not None and hasattr(ctx, "memory_clear"):
            ctx.memory_clear(True)
    except Exception:
        pass
    try:
        hybrid_cache_mgr = getattr(llm, "_hybrid_cache_mgr", None)
        if hybrid_cache_mgr is not None and hasattr(hybrid_cache_mgr, "clear"):
            hybrid_cache_mgr.clear()
    except Exception:
        pass
    try:
        batch = getattr(llm, "_batch", None)
        if batch is not None and hasattr(batch, "reset"):
            batch.reset()
    except Exception:
        pass


# ---------------------------------------------------------------- 推理调用
_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = BaseInferenceEngine({})
    return _ENGINE


def _extract_reply(result) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except Exception:
        return str(result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _call_chat_completion(llm, messages: list, params: dict) -> dict:
    _reset_llm_state(llm)
    return _get_engine().create_chat_completion(llm, messages, params)


def _build_user_content(text: str, images: list, max_edge: int, max_frames: int = 8, native_video: bool = False):
    if not images:
        return text
    content = [{"type": "text", "text": text}]
    for media_ref in images:
        media_type = media_ref.get("media_type", "image")
        if media_type == "video":
            video_content = _video_ref_to_content(media_ref, max_edge, max_frames, native_video)
            content.extend(video_content)
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_ref_to_data_uri(media_ref, max_edge)},
                }
            )
    return content


def _build_model_history(history: list, max_edge: int, native_video: bool = False) -> list:
    messages = []
    for item in history:
        images = item.get("images") or []
        if item["role"] == "user" and images:
            messages.append({"role": "user", "content": _build_user_content(item["content"], images, max_edge, native_video=native_video)})
        else:
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


# ---------------------------------------------------------------- Skill 状态
def _default_flow_state() -> dict:
    return {"skill": "", "skill_name": "", "stage": "未开始", "loaded_references": [], "final_result": ""}


def _parse_flow_state(raw_state: str) -> dict:
    state = _default_flow_state()
    if raw_state and str(raw_state).strip():
        try:
            value = json.loads(raw_state)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            state["skill"] = str(value.get("skill") or "")
            state["skill_name"] = str(value.get("skill_name") or "")[:80]
            state["stage"] = str(value.get("stage") or "未开始")[:40]
            state["loaded_references"] = [
                str(item) for item in value.get("loaded_references", []) if isinstance(item, str)
            ]
            state["final_result"] = str(value.get("final_result") or "")
    if state["skill"] and not state["skill_name"]:
        skill = get_skill(state["skill"])
        if skill is not None:
            state["skill_name"] = skill["name"]
    return state


def _parse_skill_reply(reply: str):
    """解析回复正文与 <omni_llm_state> 状态标记"""
    matches = list(SKILL_STATE_TAG.finditer(reply or ""))
    if not matches:
        return (reply or "").strip(), {}
    match = matches[-1]
    try:
        state = json.loads(match.group(1))
    except json.JSONDecodeError:
        return SKILL_STATE_TAG.sub("", reply).strip(), {}
    if not isinstance(state, dict):
        state = {}
    text = (reply[: match.start()] + reply[match.end() :]).strip()
    return text, state


def _normalize_options(value) -> list:
    if not isinstance(value, list):
        return []
    options = []
    for item in value[:6]:
        text = str(item or "").strip()
        if text:
            options.append(text[:240])
    return options


def _parse_options_json(raw_options: str) -> list:
    try:
        return _normalize_options(json.loads(raw_options or "[]"))
    except json.JSONDecodeError:
        return []


def _auto_select_skill(llm, skills: list, user_text: str) -> str:
    if not skills:
        raise ValueError("Skill加载器没有发现可用 Skill，请把 Skill 放入插件的 skills 目录。")
    catalogue = "\n".join(
        f'- {item["id"]}: {item["name"]}；{item["description"][:500]}' for item in skills
    )
    messages = [
        {
            "role": "system",
            "content": "根据用户任务选择唯一最匹配的 Skill。只输出 Skill ID，不解释，不添加标点。",
        },
        {"role": "user", "content": f"可用 Skills：\n{catalogue}\n\n用户任务：\n{user_text}"},
    ]
    params = {
        "max_tokens": 80,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 0,
        "min_p": 0.0,
    }
    result = _call_chat_completion(llm, messages, params)
    selected = _clean_reasoning_markers(_extract_reply(result)).strip().strip("`'\".,，。 ")
    valid_ids = {item["id"] for item in skills}
    if selected in valid_ids:
        return selected
    for skill_id in valid_ids:
        if skill_id in selected:
            return skill_id
    raise ValueError(f"自动选择 Skill 失败，模型返回：{selected[:120]}。请在 Skill加载器中手动选择。")


def _build_skill_system_prompt(base_system: str, skill: dict, flow_state: dict) -> str:
    loaded = []
    for relative_path in flow_state["loaded_references"]:
        if relative_path not in skill["references"]:
            continue
        loaded.append(
            f"\n\n===== reference: {relative_path} =====\n{read_reference(skill, relative_path)}"
        )
    catalogue = "\n".join(f"- {path}" for path in skill["references"]) or "- 无"
    loaded_names = "、".join(flow_state["loaded_references"]) or "无"
    parts = [base_system.strip()]
    parts.append(
        f"当前 Skill：{skill['name']} ({skill['id']})\n"
        f"当前流程阶段：{flow_state['stage']}\n"
        f"可用 references：\n{catalogue}\n"
        f"已加载 references：{loaded_names}\n\n"
        f"===== {skill['skill_file']} =====\n{read_skill_body(skill)}"
    )
    parts.extend(loaded)
    parts.append(SKILL_EXECUTION_PROTOCOL)
    return "\n\n".join(part for part in parts if part)


# ---------------------------------------------------------------- 构建返回
def _build_return(
    history: list,
    reply: str,
    final_result: str = "",
    flow_state: dict | None = None,
    options=None,
    sent: bool = False,
    context_state: dict | None = None,
):
    history_json = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    state = flow_state or _default_flow_state()
    ui = {
        "chat_history_json": [history_json],
        "assistant_reply": [reply],
        "flow_state_json": [json.dumps(state, ensure_ascii=False, separators=(",", ":"))],
        "flow_stage": [state.get("stage", "未开始")],
        "options_json": [json.dumps(_normalize_options(options), ensure_ascii=False)],
        "sent": [bool(sent)],
    }
    if context_state is not None:
        ui["context_state_json"] = [json.dumps(context_state, ensure_ascii=False, separators=(",", ":"))]
    return {"ui": ui, "result": ()}


# ---------------------------------------------------------------- 默认设置
_CHAT_PARAMS_KEYS = (
    "max_tokens",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "temperature",
    "repeat_penalty",
    "frequency_penalty",
    "presence_penalty",
    "mirostat_mode",
    "mirostat_eta",
    "mirostat_tau",
    "seed",
    "reasoning_budget",
)


def _default_chat_settings() -> dict:
    return {
        "system_prompt": "",
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 30,
        "min_p": 0.05,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": -1,
        "reasoning_budget": -1,
    }


# ---------------------------------------------------------------- 节点
class omni_llm_realtime_chat:
    """Omni LLM 实时对话（内嵌聊天窗 + Skill 多阶段协议）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_message": ("STRING", {"default": "", "multiline": True}),
                "chat_history_json": ("STRING", {"default": "[]", "multiline": True}),
                "request_id": ("STRING", {"default": ""}),
                "current_images_json": ("STRING", {"default": "[]", "multiline": True}),
                "flow_state_json": ("STRING", {"default": "{}", "multiline": True}),
                "options_json": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "llama_model": ("LLAMACPPMODEL", {"tooltip": "本地模型（API模式可不连接）"}),
                "api_config": (
                    "OMNI_LLM_API_CONFIG",
                    {"tooltip": "可选：接入 API 配置节点，使用外部 API 推理"},
                ),
                "parameters": (
                    "LLAMACPPARAMS",
                    {"tooltip": "可选：接入参数节点，覆盖采样参数、种子、图片边长、思考模式等"},
                ),
                "skill_loader": (
                    "OMNI_LLM_SKILL",
                    {"tooltip": "可选：接入 Skill加载器，启用 Skill 驱动的多阶段对话"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "你是一个有帮助的AI助手。",
                        "multiline": True,
                        "tooltip": "系统提示词：定义AI的角色和行为，可直接输入或接入预设模板节点",
                    },
                ),
                "system_prompt_preset": (
                    "OMNI_LLM_PRESET",
                    {"tooltip": "可选：接入预设模板节点，覆盖上方手动输入的系统提示词"},
                ),
                "native_video": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "原生视频模式：发送 base64 视频文件（需模型支持视频理解）\n关闭则提取视频帧作为图片发送（兼容所有模型）",
                }),
                "save_conversation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "保存对话：勾选后自动将对话内容保存到 cache 文件夹中的文本文件",
                }),
                "session_id": ("STRING", {
                    "default": "",
                    "tooltip": "会话ID：用于跟踪当前对话，清空对话后会自动新建文件",
                }),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "run"
    CATEGORY = "omni-llm"
    OUTPUT_NODE = True

    def run(
        self,
        llama_model=None,
        user_message="",
        chat_history_json="[]",
        request_id="",
        current_images_json="[]",
        flow_state_json="{}",
        options_json="[]",
        parameters=None,
        skill_loader=None,
        api_config=None,
        system_prompt="",
        system_prompt_preset=None,
        native_video=False,
        save_conversation=False,
        session_id="",
    ):
        request_created_at = _parse_request_time_ms(request_id)

        settings = _default_chat_settings()
        if isinstance(parameters, dict):
            settings.update(
                {key: value for key, value in parameters.items() if key in _CHAT_PARAMS_KEYS}
            )
        
        # 处理 system_prompt（优先使用 OMNI_LLM_PRESET 预设节点，其次使用手动输入）
        resolved_prompt = None
        # 优先：OMNI_LLM_PRESET 预设节点
        if isinstance(system_prompt_preset, dict):
            preset_text = system_prompt_preset.get("prompt", "") or system_prompt_preset.get("content", "") or system_prompt_preset.get("system_prompt", "")
            if preset_text and str(preset_text).strip():
                resolved_prompt = str(preset_text).strip()
        # 其次：手动输入
        if not resolved_prompt:
            if isinstance(system_prompt, dict):
                preset_text = system_prompt.get("prompt", "") or system_prompt.get("content", "") or system_prompt.get("system_prompt", "")
                if preset_text and str(preset_text).strip():
                    resolved_prompt = str(preset_text).strip()
            elif isinstance(system_prompt, str) and system_prompt.strip():
                resolved_prompt = system_prompt.strip()
        if resolved_prompt:
            settings["system_prompt"] = resolved_prompt
        
        # 从 api_config 节点提取 API 参数
        api_base_url = ""
        api_key = ""
        api_model = ""
        api_timeout = 300
        api_context_limit = 0
        api_max_history_rounds = 100
        api_max_edge = 1024
        api_preserve_thinking = False
        if isinstance(api_config, dict):
            api_base_url = api_config.get("base_url", "")
            api_key = api_config.get("api_key", "")
            api_model = api_config.get("model", "")
            api_timeout = int(api_config.get("timeout", 300))
            api_context_limit = int(api_config.get("context_limit", 0))
            api_max_history_rounds = int(api_config.get("max_history_rounds", 100))
            api_max_edge = int(api_config.get("max_edge", 1024))
            api_preserve_thinking = bool(api_config.get("preserve_thinking", False))

        max_rounds = int(settings.get("max_history_rounds", api_max_history_rounds))
        max_tokens = int(settings["max_tokens"])
        max_edge = int(settings.get("max_edge", api_max_edge))
        history = _trim_by_rounds(_parse_chat_history(chat_history_json), max_rounds)
        current_images = _parse_image_list(current_images_json)
        user_text = (user_message or "").strip()
        flow_state = _parse_flow_state(flow_state_json)
        flow_state_before = {
            **flow_state,
            "loaded_references": list(flow_state["loaded_references"]),
        }

        # 无新消息：不推理，仅返回最近一条助手回复（幂等，供完整图重跑）
        if not user_text:
            last_reply = next(
                (item["content"] for item in reversed(history) if item["role"] == "assistant"),
                "",
            )
            return _build_return(
                history,
                last_reply,
                flow_state.get("final_result", ""),
                flow_state,
                _parse_options_json(options_json),
            )

        # 检测API模式：有API配置且无本地模型时使用API
        has_api_config = isinstance(api_config, dict) and api_config.get("base_url") and api_config.get("api_key") and api_config.get("model")
        use_api_mode = has_api_config and llama_model is None

        if use_api_mode:
            # API模式：不需要本地模型
            llm = None
            n_ctx = api_context_limit if api_context_limit > 0 else 1048576
        else:
            # 本地模型模式
            if llama_model is None:
                raise RuntimeError("未连接模型且未配置API，请先加载模型或配置API节点。")
            llm = _sync_loaded_llm(llama_model)
            n_ctx = _get_n_ctx(llama_model, llm)
        
        # 仅在 API 模式下使用 API 配置的上下文上限
        if use_api_mode and api_context_limit > 0:
            n_ctx = api_context_limit

        for history_item in history:
            if "token_count" not in history_item:
                history_item["token_count"] = _estimate_message_tokens(llm, history_item) if llm is not None else 0

        system_text = str(settings["system_prompt"] or "").strip()
        skill = None
        if isinstance(skill_loader, dict):
            selected_id = str(skill_loader.get("selected") or "").strip()
            if selected_id and flow_state.get("skill") and flow_state["skill"] != selected_id:
                flow_state = _default_flow_state()
            if selected_id:
                flow_state["skill"] = selected_id
                skill = get_skill(selected_id)
            elif not flow_state.get("skill"):
                if llm is not None:
                    skill_id = _auto_select_skill(llm, skill_loader.get("skills") or [], user_text)
                else:
                    # API模式：使用第一个可用skill
                    available_skills = skill_loader.get("skills") or []
                    skill_id = available_skills[0]["id"] if available_skills else ""
                flow_state["skill"] = skill_id
                skill = get_skill(skill_id)
            else:
                skill = get_skill(flow_state["skill"])
            if skill is None:
                raise ValueError("Skill 已不存在，请刷新 Skill加载器并重新选择。")
            flow_state["skill_name"] = skill["name"]
            # Skill 节点优先：忽略用户自定义系统提示词，仅使用 Skill 提示词
            system_text = _build_skill_system_prompt("", skill, flow_state)
        elif not system_text:
            system_text = _DEFAULT_CHAT_SYSTEM_PROMPT

        history_image_count = sum(len(item.get("images") or []) for item in history)
        if (history_image_count or current_images) and llama_model is not None and getattr(llama_model, "chat_handler", None) is None:
            raise RuntimeError("图片对话需要加载对应的视觉投影 mmproj（请在模型加载器启用多模态）。")

        history_before_context_trim = len(history)
        if llm is not None:
            model_history = _trim_by_context(
                llm,
                history,
                system_text,
                user_text,
                max_tokens,
                n_ctx,
                current_image_count=len(current_images),
            )
        else:
            model_history = list(history)
        trimmed_message_count = history_before_context_trim - len(model_history)

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.extend(_build_model_history(model_history, max_edge, native_video))
        messages.append({"role": "user", "content": _build_user_content(user_text, current_images, max_edge, native_video=native_video)})

        params = {
            "max_tokens": max_tokens,
            "temperature": float(settings["temperature"]),
            "top_p": float(settings["top_p"]),
            "top_k": int(settings["top_k"]),
            "min_p": float(settings["min_p"]),
            "typical_p": float(settings["typical_p"]),
            "repeat_penalty": float(settings["repeat_penalty"]),
            "frequency_penalty": float(settings["frequency_penalty"]),
            "presence_penalty": float(settings["presence_penalty"]),
            "mirostat_mode": int(settings.get("mirostat_mode", 0) or 0),
            "mirostat_eta": float(settings.get("mirostat_eta", 0.1) or 0.1),
            "mirostat_tau": float(settings.get("mirostat_tau", 5.0) or 5.0),
            "reasoning_budget": int(settings["reasoning_budget"]),
            "timeout": api_timeout,
        }
        seed = settings["seed"]
        if seed is not None and int(seed) >= 0:
            params["seed"] = int(seed)

        reply = ""
        skill_state = {}
        for attempt in range(2):
            # 使用 API 或本地模型（本地模型优先）
            if use_api_mode:
                result = _call_api_chat_completion(api_base_url, api_key, api_model, messages, params)
            else:
                result = _call_chat_completion(llm, messages, params)
            raw_reply = _extract_reply(result)
            if not bool(settings.get("preserve_thinking", api_preserve_thinking)):
                raw_reply = _clean_reasoning_markers(raw_reply)
            reply, skill_state = _parse_skill_reply(raw_reply.lstrip().removeprefix(": ").strip())
            if skill is None:
                break
            requested = []
            for item in skill_state.get("load_references", []):
                if (
                    isinstance(item, str)
                    and item in skill["references"]
                    and item not in flow_state["loaded_references"]
                ):
                    requested.append(item)
            if not requested or attempt == 1:
                break
            flow_state["loaded_references"].extend(requested)
            system_text = _build_skill_system_prompt(
                str(settings["system_prompt"] or "").strip(), skill, flow_state
            )
            history_before_reference_trim = len(model_history)
            if llm is not None:
                model_history = _trim_by_context(
                    llm,
                    model_history,
                    system_text,
                    user_text,
                    max_tokens,
                    n_ctx,
                    current_image_count=len(current_images),
                )
            trimmed_message_count += history_before_reference_trim - len(model_history)
            messages = [{"role": "system", "content": system_text}]
            messages.extend(_build_model_history(model_history, max_edge, native_video))
            messages.append(
                {"role": "user", "content": _build_user_content(user_text, current_images, max_edge, native_video=native_video)}
            )

        mm.throw_exception_if_processing_interrupted()

        user_history_item = {
            "role": "user",
            "content": user_text,
            "created_at": request_created_at,
        }
        if current_images:
            user_history_item["images"] = current_images
        user_history_item["token_count"] = _estimate_message_tokens(llm, user_history_item) if llm is not None else 0
        assistant_history_item = {
            "role": "assistant",
            "content": reply,
            "flow_before": flow_state_before,
            "token_count": _estimate_message_tokens(llm, {"role": "assistant", "content": reply}) if llm is not None else 0,
            "created_at": int(time.time() * 1000),
        }
        for history_item in history:
            history_item.pop("flow_before", None)
        history.extend([user_history_item, assistant_history_item])
        history = _trim_by_rounds(history, max_rounds)

        if skill is not None:
            flow_state["stage"] = str(
                skill_state.get("stage") or flow_state.get("stage") or "进行中"
            )[:40]
            options = _normalize_options(skill_state.get("options"))
            if bool(skill_state.get("final")):
                flow_state["final_result"] = reply
            final_result = flow_state.get("final_result", "")
        else:
            options = []
            final_result = ""

        context_history = model_history + [user_history_item, assistant_history_item]
        if llm is not None:
            context_state = _build_context_state(
                llm,
                system_text,
                context_history,
                max_tokens,
                n_ctx,
                max_rounds,
                trimmed_messages=trimmed_message_count,
            )
        else:
            # API模式：简化上下文状态
            output_reserve, prompt_budget = _compute_context_budget(max_tokens, n_ctx)
            context_state = {
                "used_tokens": 0,
                "prompt_budget": int(prompt_budget),
                "context_limit": int(n_ctx),
                "output_reserve": int(output_reserve),
                "remaining_tokens": int(prompt_budget),
                "percent": 0.0,
                "trimmed_messages": 0,
                "current_rounds": sum(1 for message in history if message.get("role") == "user"),
                "max_rounds": max(1, int(max_rounds)),
                "estimated": True,
            }
        context_state["current_rounds"] = sum(1 for message in history if message.get("role") == "user")
        
        # 保存对话到文件
        if save_conversation:
            try:
                _save_conversation_to_file(history, session_id)
            except Exception as e:
                print(f"【对话保存】保存失败: {e}")
        
        return _build_return(
            history,
            reply,
            final_result,
            flow_state,
            options,
            sent=True,
            context_state=context_state,
        )


NODE_CLASS_MAPPINGS = {
    "omni_llm_realtime_chat": omni_llm_realtime_chat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_realtime_chat": "Omni LLM Realtime Chat",
}
