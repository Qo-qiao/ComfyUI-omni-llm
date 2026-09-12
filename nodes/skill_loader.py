# -*- coding: utf-8 -*-
"""
ComfyUI-omni-llm Skill Loader Node

从插件 skills/ 目录发现并暴露 Skill（SKILL.md / SKILL.cn.md + meta.yaml + references/），
供实时对话节点进行 Skill 驱动的多阶段对话推理。

参考 comfyUI-llama-TE 的 skill_loader 设计移植，标识符使用英文以保持节点命名统一。

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import os
import re

try:
    from aiohttp import web
    from server import PromptServer
    _SERVER_AVAILABLE = True
except Exception:
    web = None
    PromptServer = None
    _SERVER_AVAILABLE = False


SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")
AUTO_SELECT = "自动选择"


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as file:
        return file.read()


def _parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}

    values = {}
    lines = text[3:end].splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^([\w-]+):\s*(.*)$", lines[index])
        if not match:
            index += 1
            continue
        key, value = match.groups()
        if value in ("|", ">"):
            index += 1
            parts = []
            while index < len(lines) and (not lines[index].strip() or lines[index][:1].isspace()):
                parts.append(lines[index].strip())
                index += 1
            values[key] = " ".join(part for part in parts if part)
            continue
        values[key] = value.strip().strip("\"'")
        index += 1
    return values


def _read_meta_value(skill_dir: str, key: str) -> str:
    path = os.path.join(skill_dir, "meta.yaml")
    if not os.path.isfile(path):
        return ""
    for line in _read_text(path).splitlines():
        match = re.match(rf"^{re.escape(key)}:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip("\"'")
    return ""


def _list_references(skill_dir: str) -> list:
    reference_dir = os.path.join(skill_dir, "references")
    if not os.path.isdir(reference_dir):
        return []
    files = []
    for root, _, names in os.walk(reference_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() not in (".md", ".txt", ".yaml", ".yml", ".json"):
                continue
            relative = os.path.relpath(os.path.join(root, name), skill_dir).replace("\\", "/")
            files.append(relative)
    return sorted(files)


def discover_skills(lang: str = "zh") -> list:
    """扫描 skills 目录，返回可用的 Skill 元数据列表
    
    Args:
        lang: 语言选择，zh=中文, en=英文
    """
    if not os.path.isdir(SKILLS_DIR):
        return []

    skills = []
    for skill_id in sorted(os.listdir(SKILLS_DIR)):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", skill_id):
            continue
        skill_dir = os.path.join(SKILLS_DIR, skill_id)
        default_path = os.path.join(skill_dir, "SKILL.md")
        chinese_path = os.path.join(skill_dir, "SKILL.cn.md")
        if not os.path.isfile(default_path) and not os.path.isfile(chinese_path):
            continue

        # 读取中英文版本的元数据
        name_zh = _read_meta_value(skill_dir, "display-name-zh") or skill_id
        name_en = _read_meta_value(skill_dir, "display-name-en") or skill_id
        desc_zh = _read_meta_value(skill_dir, "desc-cn") or ""
        desc_en = _read_meta_value(skill_dir, "desc-en") or ""
        
        # 读取标签（逗号分隔）
        tag_str = _read_meta_value(skill_dir, "tag-cn" if lang == "zh" else "tag-en") or ""
        tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        if not tags:
            tags = ["未分类"] if lang == "zh" else ["Uncategorized"]
        
        # 根据语言选择输出对应的字段
        if lang == "en":
            name = name_en
            description = desc_en
            skill_file = "SKILL.md" if os.path.isfile(default_path) else "SKILL.cn.md"
        else:
            name = name_zh
            description = desc_zh
            skill_file = "SKILL.cn.md" if os.path.isfile(chinese_path) else "SKILL.md"
        
        label = f"{name} [{skill_id}]" if name != skill_id else skill_id
        
        skills.append(
            {
                "id": skill_id,
                "name": name,
                "source_name": skill_id,
                "display_name": name,
                "display_source": "metadata",
                "label": label,
                "description": description,
                "tags": tags,
                "skill_file": skill_file,
                "references": _list_references(skill_dir),
                "issues": [],
                "needs_optimization": False,
            }
        )
    return skills


def get_skill(skill_id: str, lang: str = "zh"):
    """按 skill_id 获取 Skill 元数据"""
    return next((skill for skill in discover_skills(lang) if skill["id"] == skill_id), None)


def get_skill_catalog():
    """返回完整 Skill 目录（含分类列表），供前端滑动列表使用"""
    skills_zh = discover_skills("zh")
    skills_en = discover_skills("en")
    
    # 提取分类列表
    categories_zh = ["全部"]
    categories_en = ["All"]
    for s in skills_zh:
        for tag in s.get("tags", ["未分类"]):
            if tag not in categories_zh:
                categories_zh.append(tag)
    for s in skills_en:
        for tag in s.get("tags", ["Uncategorized"]):
            if tag not in categories_en:
                categories_en.append(tag)
    if "未分类" not in categories_zh:
        categories_zh.append("未分类")
    if "Uncategorized" not in categories_en:
        categories_en.append("Uncategorized")
    
    return {
        "skills_zh": skills_zh,
        "skills_en": skills_en,
        "categories_zh": categories_zh,
        "categories_en": categories_en
    }


def read_skill_body(skill: dict, lang: str = "zh") -> str:
    """读取 Skill 正文（根据语言选择 SKILL.md 或 SKILL.cn.md）"""
    skill_dir = os.path.join(SKILLS_DIR, skill["id"])
    default_path = os.path.join(skill_dir, "SKILL.md")
    chinese_path = os.path.join(skill_dir, "SKILL.cn.md")
    
    if lang == "zh" and os.path.isfile(chinese_path):
        return _read_text(chinese_path)
    elif lang == "en" and os.path.isfile(default_path):
        return _read_text(default_path)
    # 回退：优先中文，其次英文
    if os.path.isfile(chinese_path):
        return _read_text(chinese_path)
    return _read_text(default_path)


def read_reference(skill: dict, relative_path: str) -> str:
    """读取 Skill 的 reference 文件（仅允许 Skill 目录内部）"""
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    if normalized not in skill["references"]:
        raise ValueError(f"Skill reference 不存在：{normalized}")
    skill_dir = os.path.realpath(os.path.join(SKILLS_DIR, skill["id"]))
    path = os.path.realpath(os.path.join(skill_dir, normalized))
    if os.path.commonpath([skill_dir, path]) != skill_dir:
        raise ValueError("Skill reference 路径超出 Skill 目录。")
    return _read_text(path)


class omni_llm_skill_loader:
    """Skill 加载器：自动选择或固定选择一个 Skill"""

    @classmethod
    def INPUT_TYPES(cls):
        choices = [AUTO_SELECT] + [skill["label"] for skill in discover_skills("zh")]
        return {
            "required": {
                "skill": (
                    choices,
                    {
                        "default": AUTO_SELECT,
                        "tooltip": "自动选择会根据首次任务匹配 Skill；也可以固定选择一个 Skill。",
                    },
                ),
                "language": (
                    ["zh", "en"],
                    {
                        "default": "zh",
                        "tooltip": "输出语言：zh=中文, en=English",
                    },
                ),
            }
        }

    RETURN_TYPES = ("OMNI_LLM_SKILL",)
    RETURN_NAMES = ("skill_loader",)
    FUNCTION = "run"
    CATEGORY = "omni-llm"

    def run(self, skill, language="zh"):
        # 同时获取中英文列表，用于匹配（widget 值始终为中文格式）
        skills = discover_skills(language)
        skills_zh = discover_skills("zh") if language != "zh" else skills
        selected = ""
        selected_skill = None
        if skill != AUTO_SELECT:
            # 先在当前语言列表中匹配
            selected_skill = next(
                (item for item in skills if item["label"] == skill or item["id"] == skill), None
            )
            # 如果没匹配到，尝试在中文列表中匹配（widget 值为中文格式）
            if selected_skill is None and language != "zh":
                zh_match = next(
                    (item for item in skills_zh if item["label"] == skill or item["id"] == skill), None
                )
                if zh_match:
                    # 从当前语言列表中找到对应的 skill
                    selected_skill = next(
                        (item for item in skills if item["id"] == zh_match["id"]), None
                    )
            if selected_skill is None:
                raise ValueError(f"找不到 Skill：{skill}，请刷新节点后重新选择。")
            selected = selected_skill["id"]
        elif skills:
            # 自动选择模式：默认选中第一个
            selected_skill = skills[0]
            selected = selected_skill["id"]
        
        # 只输出选中 skill 的信息
        return ({"selected": selected, "skill": selected_skill, "language": language},)


if _SERVER_AVAILABLE:
    @PromptServer.instance.routes.get("/omni_llm/skill/catalog")
    async def _skill_catalog(request):
        return web.json_response(get_skill_catalog())

    @PromptServer.instance.routes.get("/omni_llm/skill/content")
    async def _skill_content(request):
        skill_id = request.query.get("id", "")
        lang = request.query.get("lang", "zh")
        skill = get_skill(skill_id, lang)
        if not skill:
            return web.json_response({"error": "Skill not found"}, status=404)
        try:
            content = read_skill_body({"id": skill_id}, lang)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"id": skill_id, "content": content})

    @PromptServer.instance.routes.post("/omni_llm/skill/import")
    async def _skill_import(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        name = str(data.get("name", "")).strip()
        content = str(data.get("content", ""))
        if not name or not content:
            return web.json_response({"error": "Name and content required"}, status=400)
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        skill_dir = os.path.join(SKILLS_DIR, safe_id)
        os.makedirs(skill_dir, exist_ok=True)
        skill_file = os.path.join(skill_dir, "SKILL.md")
        with open(skill_file, "w", encoding="utf-8") as f:
            f.write(content)
        meta_file = os.path.join(skill_dir, "meta.yaml")
        if not os.path.isfile(meta_file):
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write(f"display-name-zh: {name}\ntag-cn: 未分类\nsummary-cn: {name}\n")
        return web.json_response({"ok": True, "id": safe_id})

    @PromptServer.instance.routes.post("/omni_llm/skill/import_file")
    async def _skill_import_file(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        file_path = str(data.get("path", "")).strip()
        if not file_path or not os.path.exists(file_path):
            return web.json_response({"error": "路径无效"}, status=400)
        
        imported = []
        errors = []
        
        def copy_skill_dir(src_dir):
            """复制整个 skill 目录保持原始结构，必须包含 SKILL.md 或 SKILL.cn.md"""
            try:
                # 检查是否包含 SKILL.md 或 SKILL.cn.md
                skill_md = os.path.join(src_dir, "SKILL.md")
                skill_cn_md = os.path.join(src_dir, "SKILL.cn.md")
                if not os.path.exists(skill_md) and not os.path.exists(skill_cn_md):
                    errors.append(f"{os.path.basename(src_dir)}: 文件夹内必须包含 SKILL.md 或 SKILL.cn.md 文件")
                    return
                
                dir_name = os.path.basename(src_dir)
                safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", dir_name)
                dst_dir = os.path.join(SKILLS_DIR, safe_id)
                
                if os.path.exists(dst_dir):
                    import shutil
                    shutil.rmtree(dst_dir)
                
                import shutil
                shutil.copytree(src_dir, dst_dir)
                imported.append(safe_id)
                print(f"【Skill导入】已导入文件夹: {dir_name} -> {safe_id}")
            except Exception as e:
                errors.append(f"{os.path.basename(src_dir)}: {str(e)}")
        
        if os.path.isdir(file_path):
            copy_skill_dir(file_path)
        else:
            return web.json_response({"error": "请输入文件夹路径"}, status=400)
        
        return web.json_response({
            "ok": len(imported) > 0,
            "imported": imported,
            "errors": errors,
            "total": len(imported),
            "error": errors[0] if errors else None
        })

    @PromptServer.instance.routes.post("/omni_llm/skill/update")
    async def _skill_update(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        skill_id = str(data.get("id", "")).strip()
        lang = str(data.get("lang", "zh")).strip()
        
        if not skill_id:
            return web.json_response({"error": "ID required"}, status=400)
        
        skill = get_skill(skill_id)
        if not skill:
            return web.json_response({"error": "Skill not found"}, status=404)
        
        skill_dir = os.path.join(SKILLS_DIR, skill_id)
        
        # 根据语言获取对应的字段值
        if lang == "zh":
            name = str(data.get("name_zh", "")).strip()
            category = str(data.get("tags_zh", "")).strip()
            desc = str(data.get("desc_zh", "")).strip()
            content = str(data.get("content_zh", ""))
            name_key = "display-name-zh"
            tag_key = "tag-cn"
            desc_key = "desc-cn"
        else:
            name = str(data.get("name_en", "")).strip()
            category = str(data.get("tags_en", "")).strip()
            desc = str(data.get("desc_en", "")).strip()
            content = str(data.get("content_en", ""))
            name_key = "display-name-en"
            tag_key = "tag-en"
            desc_key = "desc-en"
        
        try:
            # 更新内容文件
            if content:
                if lang == "zh":
                    content_file = os.path.join(skill_dir, "SKILL.cn.md")
                else:
                    content_file = os.path.join(skill_dir, "SKILL.md")
                with open(content_file, "w", encoding="utf-8") as f:
                    f.write(content)
            
            # 更新 meta.yaml
            meta_file = os.path.join(skill_dir, "meta.yaml")
            meta_lines = []
            if os.path.isfile(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_lines = f.readlines()
            
            # 构建需要更新的字段
            updates = {}
            if name:
                updates[name_key] = name
            if category:
                updates[tag_key] = category
            if desc:
                updates[desc_key] = desc
            
            # 处理现有行
            found_keys = set()
            new_lines = []
            for line in meta_lines:
                key = line.strip().split(":")[0] if ":" in line else ""
                if key in updates:
                    new_lines.append(f"{key}: {updates[key]}\n")
                    found_keys.add(key)
                else:
                    new_lines.append(line)
            
            # 添加未找到的字段
            for key, value in updates.items():
                if key not in found_keys:
                    new_lines.append(f"{key}: {value}\n")
            
            # 写入文件
            with open(meta_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            
            print(f"【Skill更新】已更新: {skill_id} ({lang})")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        
        return web.json_response({"ok": True})

    @PromptServer.instance.routes.post("/omni_llm/skill/delete")
    async def _skill_delete(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        skill_id = str(data.get("id", "")).strip()
        if not skill_id:
            return web.json_response({"error": "ID required"}, status=400)
        
        skill_dir = os.path.join(SKILLS_DIR, skill_id)
        if not os.path.isdir(skill_dir):
            return web.json_response({"error": "Skill not found"}, status=404)
        
        try:
            import shutil
            shutil.rmtree(skill_dir)
            print(f"【Skill删除】已删除: {skill_id}")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        
        return web.json_response({"ok": True})


NODE_CLASS_MAPPINGS = {
    "omni_llm_skill_loader": omni_llm_skill_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "omni_llm_skill_loader": "Omni LLM Skill Loader",
}
