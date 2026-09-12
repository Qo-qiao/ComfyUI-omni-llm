# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm API Config Node

API 配置节点，用于实时对话节点

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import re
import sys

# 导入 api/ 目录下的 provider presets
_api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api")
if _api_dir not in sys.path:
    sys.path.insert(0, _api_dir)
from api_provider_presets import PRESET_MODELS, DEFAULT_PROVIDER_CONFIGS


def _load_saved_configs():
    """从 api_provider_presets.py 读取 SAVED_CONFIGS"""
    try:
        preset_path = os.path.join(_api_dir, "api_provider_presets.py")
        if not os.path.exists(preset_path):
            return {}
        ns = {}
        with open(preset_path, "r", encoding="utf-8") as f:
            exec(f.read(), ns)
        return ns.get("SAVED_CONFIGS", {})
    except Exception:
        return {}


def _save_config_to_file(preset_name: str, config: dict):
    """将配置追加/更新到 api_provider_presets.py 的 SAVED_CONFIGS"""
    preset_path = os.path.join(_api_dir, "api_provider_presets.py")
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 新条目
        entry_lines = [
            f'    "{preset_name}": {{',
            f'        "provider": "{config.get("provider", "自定义")}",',
            f'        "api_base": "{config["base_url"]}",',
            f'        "model_id": "{config["model"]}",',
            f'        "api_key": "{config["api_key"]}",',
            f'        "context_limit": {config.get("context_limit", 0)},',
            f'        "timeout": {config.get("timeout", 300)},',
            f'        "max_history_rounds": {config.get("max_history_rounds", 100)},',
            f'        "max_edge": {config.get("max_edge", 1024)},',
            f'        "preserve_thinking": {config.get("preserve_thinking", False)},',
            f'    }},',
        ]
        saved_entry = "\n".join(entry_lines)

        # 找 SAVED_CONFIGS = { ... }
        m = re.search(r'SAVED_CONFIGS\s*=\s*\{', content)
        if not m:
            # 文件末尾追加
            content = content.rstrip() + "\n\nSAVED_CONFIGS = {\n" + saved_entry + "\n}\n"
            with open(preset_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        # 找到整个 dict 的结束位置（匹配花括号）
        start = m.end()
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        end = i  # } 的位置+1

        old_inner = content[start:end - 1].strip()

        # 删除同名旧条目
        pat = rf'\s*"{re.escape(preset_name)}"\s*:\s*\{{[^}}]*\}},'
        new_inner = re.sub(pat, "", old_inner).strip()

        if new_inner:
            new_inner = new_inner + "\n" + saved_entry
        else:
            new_inner = saved_entry

        new_block = "SAVED_CONFIGS = {\n" + new_inner + "\n}"
        content = content[:m.start()] + new_block + content[end:]

        with open(preset_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"【API 配置】已保存预设「{preset_name}」")
        return True
    except Exception as e:
        print(f"【API 配置】保存失败：{e}")
        return False


def _delete_config_from_file(preset_name: str):
    """从 api_provider_presets.py 删除指定预设"""
    preset_path = os.path.join(_api_dir, "api_provider_presets.py")
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            content = f.read()

        m = re.search(r'SAVED_CONFIGS\s*=\s*\{', content)
        if not m:
            return False, "没有找到 SAVED_CONFIGS"

        start = m.end()
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        end = i

        old_inner = content[start:end - 1].strip()
        pat = rf'\s*"{re.escape(preset_name)}"\s*:\s*\{{[^}}]*\}},'
        new_inner = re.sub(pat, "", old_inner).strip()

        if new_inner == old_inner:
            return False, f"未找到预设「{preset_name}」"

        new_block = "SAVED_CONFIGS = {\n" + new_inner + "\n}" if new_inner else "SAVED_CONFIGS = {}"
        content = content[:m.start()] + new_block + content[end:]

        with open(preset_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"【API 配置】已删除预设「{preset_name}」")
        return True, f"已删除「{preset_name}」"
    except Exception as e:
        return False, str(e)


def _build_saved_list_text(saved: dict) -> str:
    """构建已保存配置的预览文本"""
    if not saved:
        return "（暂无保存的配置）"
    lines = []
    for name, cfg in saved.items():
        key = cfg.get("api_key", "")
        show_key = key[:4] + "***" + key[-4:] if len(key) > 8 else "***"
        lines.append(f"• {name}\n  {cfg.get('api_base', '')} | {cfg.get('model_id', '')} | {show_key}")
    return "\n".join(lines)


def _detect_provider(api_base: str, model_name: str = "") -> str:
    """根据 API 地址反向匹配提供商名称"""
    base = api_base.lower().rstrip("/")
    for name, cfg in DEFAULT_PROVIDER_CONFIGS.items():
        if name == "自定义":
            continue
        if cfg.get("api_base", "") and base.startswith(cfg["api_base"].rstrip("/").lower()):
            return name
    return "自定义"


class omni_llm_api_config:
    """Omni LLM API 配置节点"""

    @classmethod
    def INPUT_TYPES(cls):
        saved = _load_saved_configs()
        saved_names = list(saved.keys())
        providers = list(DEFAULT_PROVIDER_CONFIGS.keys())

        preset_options = saved_names if saved_names else ["（暂无）"]
        if "（暂无）" not in preset_options:
            preset_options.insert(0, "（暂无）")
        preset_default = preset_options[0] if preset_options else "（暂无）"

        return {
            "required": {
                "api_provider": (providers, {"default": "OpenAI", "tooltip": "选择 API 提供商"}),
                "model_name": ("STRING", {"default": "", "tooltip": "模型名称（留空使用提供商默认，推荐手动填写）"}),
                "api_base": ("STRING", {"default": "", "tooltip": "API 地址（留空使用提供商默认，推荐手动填写）"}),
                "api_key": ("STRING", {"default": "", "tooltip": "API 密钥"}),
            },
            "optional": {
                "context_limit": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 1024, "tooltip": "上下文上限（0=自动（1M token），用于历史裁剪与上下文圆环显示）"}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 1200, "step": 10, "tooltip": "API 请求超时时间（秒）"}),
                "max_history_rounds": ("INT", {"default": 100, "min": 1, "max": 500, "step": 1, "tooltip": "多轮对话保留的历史轮数，超出部分会被裁剪"}),
                "max_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64, "tooltip": "输入图片最大边长（像素），超出会等比缩放"}),
                "preserve_thinking": ("BOOLEAN", {"default": False, "tooltip": "是否保留模型的思考过程（thinking block）到对话历史中"}),
                "save_preset": ("BOOLEAN", {"default": False, "tooltip": "✅ 勾选后执行节点即保存当前配置"}),
                "preset_name": ("STRING", {"default": "", "tooltip": "保存的预设名称（留空自动用 提供商_模型）"}),
                "load_preset": (preset_options, {"default": preset_default, "tooltip": "选择已保存的配置加载（自动覆盖上方所有字段）"}),
                "delete_preset": ("BOOLEAN", {"default": False, "tooltip": "🗑 勾选后执行节点即删除当前选择的预设"}),
            },
        }

    RETURN_TYPES = ("OMNI_LLM_API_CONFIG",)
    RETURN_NAMES = ("api_config",)
    FUNCTION = "process"
    CATEGORY = "omni-llm"
    OUTPUT_NODE = True

    def process(self, api_provider, model_name, api_base, api_key,
                context_limit=0, timeout=300,
                max_history_rounds=100, max_edge=1024, preserve_thinking=False,
                save_preset=False, preset_name="", load_preset="（暂无）",
                delete_preset=False):
        saved = _load_saved_configs()

        # ── 删除 ──
        if delete_preset and load_preset and load_preset in saved:
            ok, msg = _delete_config_from_file(load_preset)
            saved = _load_saved_configs()  # 刷新
            if ok:
                print(f"【API 配置】{msg}")
            else:
                print(f"【API 配置】删除失败：{msg}")

        # ── 加载预设（仅当 load_preset 在当前已保存列表中时才加载）──
        if load_preset and load_preset != "（暂无）" and load_preset in saved:
            p = saved[load_preset]
            api_provider = p.get("provider", "自定义")
            model_name = p.get("model_id", "")
            api_base = p.get("api_base", "")
            api_key = p.get("api_key", "")
            context_limit = int(p.get("context_limit", 0))
            timeout = int(p.get("timeout", 300))
            max_history_rounds = int(p.get("max_history_rounds", 100))
            max_edge = int(p.get("max_edge", 1024))
            preserve_thinking = bool(p.get("preserve_thinking", False))
            # 补全旧预设缺失的字段（用提供商默认值）
            if context_limit == 0:
                ref_cfg = DEFAULT_PROVIDER_CONFIGS.get(
                    api_provider,
                    DEFAULT_PROVIDER_CONFIGS.get("自定义", {}),
                )
                context_limit = int(ref_cfg.get("context_limit", 1048576))
            print(f"【API 配置】已加载预设「{load_preset}」")

        # ── 合并默认值 ──
        cfg = DEFAULT_PROVIDER_CONFIGS.get(api_provider, DEFAULT_PROVIDER_CONFIGS["自定义"])
        model_id = model_name.strip() or cfg.get("model_id", "")
        base_url = api_base.strip() or cfg.get("api_base", "")
        key = api_key.strip()
        ctx_limit_default = cfg.get("context_limit", 1048576)
        if context_limit == 0:
            context_limit = ctx_limit_default

        warnings = []
        if not base_url:
            warnings.append("⚠ API 地址为空")
        if not key:
            warnings.append("⚠ API 密钥为空")
        if not model_id:
            warnings.append("⚠ 模型名称为空")

        config = {
            "provider": api_provider,
            "base_url": base_url,
            "api_key": key,
            "model": model_id,
            "context_limit": context_limit,
            "timeout": timeout,
            "max_history_rounds": max_history_rounds,
            "max_edge": max_edge,
            "preserve_thinking": preserve_thinking,
        }

        # ── 保存 ──
        save_status = ""
        if save_preset:
            name = preset_name.strip() or f"{api_provider}_{model_id}"
            if _save_config_to_file(name, config):
                save_status = f"\n✅ 已保存「{name}」"
                saved = _load_saved_configs()
            else:
                save_status = "\n❌ 保存失败"

        show_key = key[:4] + "***" + key[-4:] if len(key) > 8 else (key[:2] + "***" if key else "***")
        preview = f"提供商: {api_provider}\n模型: {model_id}\n地址: {base_url}\n密钥: {show_key}\n上下文上限: {context_limit}\n超时: {timeout}秒\n历史轮数: {max_history_rounds}\n图片边长: {max_edge}\n保留思考: {'是' if preserve_thinking else '否'}"
        if warnings:
            preview += "\n" + "\n".join(warnings)
        if save_status:
            preview += save_status

        return (config, {"ui": {"preview": [preview]}})


NODE_CLASS_MAPPINGS = {
    "omni_llm_api_config": omni_llm_api_config,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_api_config": "Omni LLM API Config",
}
