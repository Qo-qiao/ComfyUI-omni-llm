/**
 * omni_llm_skill_loader.js
 * OmniLLM Skill 加载器节点前端逻辑
 * 功能：提供 Skill 的搜索、筛选、编辑、导入等交互界面
 * 依赖：ComfyUI 的 app.js 和 api.js
 */

// 导入 ComfyUI 核心模块
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// 节点类名常量，用于匹配后端注册的节点类型
const NODE_CLASS = "omni_llm_skill_loader";

// 调试日志工具函数，统一前缀便于过滤
const log = (...a) => console.log("[OmniLLM-SkillLoader]", ...a);

// 加载外部 CSS 文件
const timestamp = new Date().getTime();
const cssLink = document.createElement("link");
cssLink.rel = "stylesheet";
cssLink.type = "text/css";
cssLink.href = new URL(`./omni_llm_skill_loader.css?v=${timestamp}`, import.meta.url).href;
document.head.appendChild(cssLink);

/**
 * 导入 Skill 文件夹
 * 使用系统文件夹选择对话框
 * @param {Function} onDone - 导入完成后的回调函数，用于刷新列表
 */
function importSkillFolder(onDone) {
    const folderInput = document.createElement("input");
    folderInput.type = "file";
    folderInput.webkitdirectory = true;
    folderInput.style.display = "none";
    document.body.appendChild(folderInput);

    folderInput.onchange = async () => {
        const files = Array.from(folderInput.files);
        if (!files.length) {
            folderInput.remove();
            return;
        }

        // 获取选中的文件夹路径（取第一个文件的路径）
        const firstFile = files[0];
        const relativePath = firstFile.webkitRelativePath;
        const folderName = relativePath.split('/')[0];

        // 找到包含 SKILL.md 或 SKILL.cn.md 的文件夹
        const hasSkillMd = files.some(f => {
            const path = f.webkitRelativePath;
            return (path.endsWith('/SKILL.md') || path.endsWith('/SKILL.cn.md')) && path.startsWith(folderName + '/');
        });

        if (!hasSkillMd) {
            alert("所选文件夹必须包含 SKILL.md 或 SKILL.cn.md 文件\nThe selected folder must contain SKILL.md or SKILL.cn.md file");
            folderInput.remove();
            return;
        }

        // 获取文件夹的完整路径（通过第一个文件的路径推断）
        // 由于浏览器安全限制，我们无法直接获取绝对路径
        // 需要用户手动输入路径
        const path = prompt("请输入文件夹的完整路径:\nPlease enter the full folder path:", `E:\\skills\\${folderName}`);
        if (!path) {
            folderInput.remove();
            return;
        }

        try {
            const r = await api.fetchApi("/omni_llm/skill/import_file", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path })
            });
            const result = await r.json();
            if (result.ok) {
                alert("✅ 成功导入 " + result.total + " 个 Skill");
                onDone();
            } else {
                alert("❌ " + (result.error || "导入失败"));
            }
        } catch (e) {
            alert("❌ 导入失败: " + e.message);
        }

        folderInput.remove();
    };

    folderInput.click();
}

/**
 * 显示 Skill 编辑弹窗
 * 支持编辑名称、分类、内容，以及删除操作
 * @param {Object} skill - 要编辑的 Skill 对象
 * @param {Function} onDone - 编辑完成后的回调函数
 * @param {Array} categories - 所有分类列表，用于自动补全
 */
function showEditModal(skill, onDone, categories, lang) {
    // 创建弹窗遮罩层
    const overlay = document.createElement("div");
    overlay.className = "sl-modal-overlay";
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

    // 创建弹窗主体
    const modal = document.createElement("div");
    modal.className = "sl-modal";

    // 弹窗标题
    const header = document.createElement("div");
    header.className = "sl-modal-header";
    header.innerText = lang === "zh" ? "编辑 Skill" : "Edit Skill";
    modal.appendChild(header);

    // 弹窗内容区
    const content = document.createElement("div");
    content.className = "sl-modal-content";

    const fields = [];
    const isZh = lang === "zh";

    const fieldDefs = [
        { label: isZh ? "技能名称(中文)" : "Skill Name(English)", type: "input", placeholder: isZh ? "输入中文名称" : "Enter English name", value: skill.name || "", fieldKey: "name" },
        { label: isZh ? "分类(中文)" : "Category(English)", type: "input", placeholder: isZh ? "输入分类，多个用逗号分隔" : "Enter categories, separated by commas", value: (skill.tags || []).join(", "), fieldKey: "tags" },
        { label: isZh ? "技能介绍(中文)" : "Introduction(English)", type: "textarea", placeholder: isZh ? "输入技能介绍" : "Enter skill introduction", value: skill.description || "", rows: 2, className: "intro", fieldKey: "description" },
        { label: isZh ? "技能内容(中文)" : "Content(English)", type: "textarea", placeholder: isZh ? "输入技能内容" : "Enter skill content", value: "", rows: 15, className: "content", fieldKey: "content" }
    ];

    fieldDefs.forEach(field => {
        const fieldGroup = document.createElement("div");
        fieldGroup.className = "sl-modal-field";

        const label = document.createElement("label");
        label.innerText = field.label;
        fieldGroup.appendChild(label);

        if (field.type === "textarea") {
            const textarea = document.createElement("textarea");
            textarea.className = "sl-modal-textarea" + (field.className ? " " + field.className : "");
            textarea.value = field.value;
            textarea.placeholder = field.placeholder;
            textarea.rows = field.rows || 5;
            fieldGroup.appendChild(textarea);
            field.element = textarea;
        } else {
            const input = document.createElement("input");
            input.type = "text";
            input.className = "sl-modal-input";
            input.value = field.value;
            input.placeholder = field.placeholder;
            fieldGroup.appendChild(input);
            field.element = input;
        }

        content.appendChild(fieldGroup);
        fields.push(field);
    });

    modal.appendChild(content);

    // 分类自动补全功能
    // 从分类列表中过滤掉"全部"选项
    const allCategories = (categories || []).filter(c => c && c !== "全部");
    let catDropdown = null;
    const categoryInput = fields[1].element;

    /**
     * 显示分类自动补全下拉菜单
     * @param {string} filter - 过滤关键词
     */
    function showCatDropdown(filter = "") {
        // 移除旧的下拉菜单
        if (catDropdown) catDropdown.remove();
        catDropdown = document.createElement("div");
        catDropdown.className = "sl-popup-category";
        let shown = 0;

        // 遍历分类，匹配过滤条件
        allCategories.forEach(cat => {
            if (filter && !cat.toLowerCase().includes(filter.toLowerCase())) return;
            if (shown >= 8) return;
            const opt = document.createElement("div");
            opt.className = "sl-category-option";
            opt.textContent = cat;
            // 鼠标按下时选中分类
            opt.onmousedown = (e) => {
                e.preventDefault();
                categoryInput.value = cat;
                catDropdown.remove();
                catDropdown = null;
            };
            catDropdown.appendChild(opt);
            shown++;
        });

        // 显示下拉菜单
        if (shown > 0) {
            const rect = categoryInput.getBoundingClientRect();
            catDropdown.style.top = rect.bottom + 2 + "px";
            catDropdown.style.left = rect.left + "px";
            document.body.appendChild(catDropdown);
        }
    }

    // 分类输入框事件绑定
    categoryInput.onfocus = () => showCatDropdown(categoryInput.value);
    categoryInput.oninput = () => showCatDropdown(categoryInput.value);
    categoryInput.onblur = () => setTimeout(() => { if (catDropdown) { catDropdown.remove(); catDropdown = null; } }, 200);

    // 从后端加载 Skill 内容（根据语言加载对应的 SKILL.md 或 SKILL.cn.md）
    api.fetchApi(`/omni_llm/skill/content?id=${encodeURIComponent(skill.id)}&lang=${lang}`)
        .then((r) => r.json())
        .then((d) => { fields[3].element.value = d.content || ""; })
        .catch(() => {});

    // 底部按钮
    const footer = document.createElement("div");
    footer.className = "sl-modal-footer";

    const btnDefs = [
        { text: "删除(Delete)", danger: true, onClick: async () => {
            if (!confirm("确定要删除这个 Skill 吗？(Confirm delete?)")) return;
            try {
                const r = await api.fetchApi("/omni_llm/skill/delete", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ id: skill.id })
                });
                if (r.ok) { overlay.remove(); onDone(); } else {
                    const e = await r.json().catch(() => ({}));
                    alert("删除失败(Delete failed): " + (e.error || r.status));
                }
            } catch (e) { alert("删除失败(Delete failed): " + e.message); }
        }},
        { text: "取消(Cancel)", onClick: () => overlay.remove() },
        { text: "保存(Save)", primary: true, onClick: async () => {
            const name = fields[0].element.value.trim();
            const category = fields[1].element.value.trim();
            const desc = fields[2].element.value.trim();
            const content = fields[3].element.value;

            if (!name) {
                alert(lang === "zh" ? "请输入技能名称" : "Enter skill name");
                return;
            }

            // 根据当前语言构建保存数据
            const saveData = { id: skill.id, lang: lang };
            if (lang === "zh") {
                saveData.name_zh = name;
                saveData.tags_zh = category;
                saveData.desc_zh = desc;
                saveData.content_zh = content;
            } else {
                saveData.name_en = name;
                saveData.tags_en = category;
                saveData.desc_en = desc;
                saveData.content_en = content;
            }

            try {
                const r = await api.fetchApi("/omni_llm/skill/update", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(saveData)
                });
                if (r.ok) { overlay.remove(); onDone(); } else {
                    const e = await r.json().catch(() => ({}));
                    alert(lang === "zh" ? "保存失败: " : "Save failed: " + (e.error || r.status));
                }
            } catch (e) { alert(lang === "zh" ? "保存失败: " : "Save failed: " + e.message); }
        }}
    ];

    btnDefs.forEach(btn => {
        const button = document.createElement("button");
        let cls = "sl-btn";
        if (btn.primary) cls += " sl-btn-primary";
        if (btn.danger) cls += " sl-btn-danger";
        button.className = cls;
        button.innerText = btn.text;
        button.onclick = btn.onClick;
        footer.appendChild(button);
    });

    modal.appendChild(footer);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

/**
 * 创建 Skill 面板主界面
 * 包含搜索栏、语言切换、分类筛选、Skill 列表、底部按钮
 * @param {Object} node - ComfyUI 节点实例
 * @returns {Object} 包含 DOM widget 的对象
 */
function createSkillPanel(node) {
    // 隐藏原始 skill COMBO 控件
    const skillWidget = node.widgets?.find((w) => w.name === "skill");
    if (skillWidget) {
        skillWidget.type = "converted-widget:hidden";
        skillWidget.hidden = true;
        skillWidget.options ||= {};
        skillWidget.options.hidden = true;
        skillWidget.options.hideInPanel = true;
        skillWidget.computeSize = () => [0, -4];
        skillWidget.serializeValue = async () => skillWidget.value;
        if (skillWidget.inputEl) skillWidget.inputEl.style.display = "none";
        if (skillWidget.element) skillWidget.element.style.display = "none";
    }

    // 查找或创建语言选择 widget（隐藏）
    let langWidget = node.widgets?.find((w) => w.name === "language");
    if (!langWidget) {
        langWidget = node.addWidget("combo", "language", "zh", (v) => {}, { values: ["zh", "en"] });
    }
    langWidget.type = "converted-widget:hidden";
    langWidget.hidden = true;
    langWidget.computeSize = () => [0, -4];

    const root = document.createElement("div");
    root.className = "sl-panel";
    // 使用绝对定位填充节点内容区，高度自适应节点大小（参考 ComfyUI-prompt-storage 实现）
    root.style.cssText = "position:absolute;left:0;right:0;top:0;bottom:0;";

    for (const ev of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "wheel"]) {
        root.addEventListener(ev, (e) => e.stopPropagation());
    }

    // 头部区域（搜索框 + 筛选按钮）
    const header = document.createElement("div");
    header.className = "sl-header";

    // 语言切换按钮
    const langBtn = document.createElement("button");
    langBtn.className = "sl-lang-btn";
    langBtn.textContent = "EN";
    langBtn.title = "切换语言(Toggle Language)";

    // 搜索行（语言切换 + 搜索框 + 筛选按钮）
    const searchRow = document.createElement("div");
    searchRow.className = "sl-search-row";

    // 搜索栏
    const searchBar = document.createElement("div");
    searchBar.className = "sl-search-bar";
    const searchInput = document.createElement("input");
    searchInput.className = "sl-search-input";
    searchInput.placeholder = "搜索模板...(Search)";
    searchBar.appendChild(searchInput);

    // 筛选按钮
    const filterBtn = document.createElement("button");
    filterBtn.className = "sl-filter-btn";
    filterBtn.innerHTML = "▼";
    filterBtn.title = "筛选分类(Filter)";
    const filterDropdown = document.createElement("div");
    filterDropdown.className = "sl-filter-dropdown";
    filterDropdown.style.display = "none";

    searchRow.append(langBtn, searchBar, filterBtn);
    header.append(searchRow);

    // Skill 列表
    const listContainer = document.createElement("div");
    listContainer.className = "sl-list-container";

    // 底部按钮栏
    const bar = document.createElement("div");
    bar.className = "sl-bar";
    const btnImport = document.createElement("button");
    btnImport.className = "sl-btn primary";
    btnImport.textContent = "导入(Import)";
    const btnEdit = document.createElement("button");
    btnEdit.className = "sl-btn";
    btnEdit.textContent = "编辑(Edit)";
    bar.append(btnImport, btnEdit);

    root.append(header, listContainer, bar);

    let catalog = { skills_zh: [], skills_en: [], categories_zh: ["全部"], categories_en: ["All"] };
    let lang = "zh"; // 当前语言：zh=中文, en=英文
    let activeCat = lang === "zh" ? "全部" : "All"; // 当前选中的分类
    let selectedId = skillWidget?.value || ""; // 当前选中的 Skill ID（用于跨语言匹配）
    let allSkills = []; // 当前语言的所有 skills

    /**
     * 语言切换按钮点击事件
     * 切换中英文显示，更新搜索框占位符和分类
     */
    langBtn.onclick = () => {
        lang = lang === "zh" ? "en" : "zh";
        langBtn.textContent = lang === "zh" ? "EN" : "中文";
        searchInput.placeholder = lang === "zh" ? "搜索模板..." : "Search template...";
        activeCat = lang === "zh" ? "全部" : "All";
        // 同步更新隐藏的 language widget
        if (langWidget) {
            langWidget.value = lang;
        }
        renderFilterDropdown();
        renderList();
    };

    // 筛选按钮下拉菜单
    let menuOpen = false;
    /**
     * 更新筛选按钮显示文本
     * 激活时显示分类名称+向上箭头，未激活时显示向下箭头
     */
    function updateFilterBtnText() {
        const allCat = lang === "zh" ? "全部" : "All";
        filterBtn.innerHTML = activeCat !== allCat ? `${activeCat} ▲` : "▼";
        filterBtn.classList.toggle("active", activeCat !== allCat);
    }
    /**
     * 关闭下拉菜单
     */
    function closeDropdown() {
        menuOpen = false;
        filterDropdown.style.display = "none";
        if (filterDropdown.parentNode) filterDropdown.parentNode.removeChild(filterDropdown);
    }
    filterBtn.onclick = (e) => {
        e.stopPropagation();
        menuOpen = !menuOpen;
        if (menuOpen) {
            if (!filterDropdown.parentNode) document.body.appendChild(filterDropdown);
            filterDropdown.style.display = "block";
            const rect = filterBtn.getBoundingClientRect();
            filterDropdown.style.top = rect.bottom + 5 + "px";
            filterDropdown.style.left = rect.left + "px";
        } else {
            closeDropdown();
        }
    };
    // 点击选项时关闭
    filterDropdown.addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("click", () => closeDropdown());

    /**
     * 从后端加载 Skill 目录数据
     * 获取所有 Skill 和分类信息
     */
    async function loadCatalog() {
        try {
            const r = await api.fetchApi("/omni_llm/skill/catalog");
            if (r.ok) catalog = await r.json();
            log("catalog:", (catalog.skills_zh || []).length, "skills");
            renderFilterDropdown();
            renderList();
        } catch (e) { log("catalog error:", e); }
    }

    /**
     * 渲染分类筛选下拉菜单
     * 根据后端返回的分类列表生成选项
     */
    function renderFilterDropdown() {
        const cats = lang === "zh" ? (catalog.categories_zh || ["全部"]) : (catalog.categories_en || ["All"]);
        filterDropdown.innerHTML = "";
        cats.forEach((c) => {
            const item = document.createElement("div");
            item.className = "sl-filter-option" + (c === activeCat ? " active" : "");
            item.textContent = c;
            item.onclick = (e) => {
                e.stopPropagation();
                activeCat = c;
                filterDropdown.querySelectorAll(".sl-filter-option").forEach((i) => i.classList.remove("active"));
                item.classList.add("active");
                updateFilterBtnText();
                closeDropdown();
                renderList();
            };
            filterDropdown.appendChild(item);
        });
        updateFilterBtnText();
    }

    /**
     * 渲染 Skill 列表
     * 根据搜索关键词、分类筛选、语言筛选过滤并显示 Skill
     */
    function renderList() {
        const q = searchInput.value.trim().toLowerCase();
        const allCat = lang === "zh" ? "全部" : "All";
        const defaultTag = lang === "zh" ? "未分类" : "Uncategorized";
        
        // 根据语言选择对应的 skills 列表
        allSkills = lang === "zh" ? (catalog.skills_zh || []) : (catalog.skills_en || []);
        
        // 过滤 Skill 列表
        const skills = allSkills.filter((s) => {
            // 获取当前语言的标签
            const tags = s.tags || [defaultTag];
            if (activeCat !== allCat && !tags.includes(activeCat)) return false;
            if (!q) return true;
            // 搜索时使用当前语言的名称和描述
            const name = s.name || "";
            const description = s.description || "";
            const tagsStr = tags.join(" ").toLowerCase();
            return name.toLowerCase().includes(q) || (description || "").toLowerCase().includes(q) || tagsStr.includes(q);
        });

        // 渲染列表容器
        listContainer.innerHTML = "";

        // 显示空状态提示
        if (!skills.length) {
            listContainer.innerHTML = lang === "zh" ? '<div class="sl-list-empty">未找到匹配的 Skill</div>' : '<div class="sl-list-empty">No matching skills found</div>';
            return;
        }

        // 遍历 Skill 创建卡片
        skills.forEach((skill) => {
            const name = skill.name || skill.id;
            const tags = skill.tags || [defaultTag];
            const description = skill.description || "";
            
            // 生成分类标签 HTML
            const tagsHtml = tags.map(t => `<span class="sl-list-card-tag">${t}</span>`).join("");
            // 创建 Skill 卡片
            const card = document.createElement("div");
            card.className = "sl-list-card" + (skill.id === selectedId ? " sel" : "");
            card.innerHTML = `
                <div class="sl-list-card-head">
                    <div class="sl-list-card-name">${name}</div>
                </div>
                <div class="sl-list-card-body">
                    <div class="sl-list-card-tags">${tagsHtml}</div>
                    <div class="sl-list-card-desc">${description || (lang === "zh" ? "暂无描述" : "No description")}</div>
                </div>
            `;
            // 卡片点击事件：选中 Skill
            card.onclick = () => {
                // 移除其他卡片的选中状态
                listContainer.querySelectorAll(".sl-list-card.sel").forEach((c) => c.classList.remove("sel"));
                card.classList.add("sel");
                selectedId = skill.id;
                if (skillWidget) {
                    // 始终使用中文 label 格式（与后端 INPUT_TYPES 生成的选项匹配）
                    const zhSkill = (catalog.skills_zh || []).find(s => s.id === skill.id);
                    const labelForWidget = zhSkill ? zhSkill.label : skill.label;
                    skillWidget.value = labelForWidget;
                    skillWidget.callback?.(labelForWidget);
                }
                node.setDirtyCanvas?.(true, true);
                node.graph?.setDirtyCanvas?.(true, true);
                log("selected:", skill.id);
            };

            // 悬停预览功能：鼠标悬停时显示详情浮窗
            const detail = skill.description || "";
            // 创建浮窗对象
            if (detail.length > 0) {
                let tooltip = null;
                let mouseInTooltip = false;

                // 鼠标进入卡片时显示浮窗
                card.addEventListener("mouseenter", () => {
                    if (!tooltip) {
                        tooltip = document.createElement("div");
                        tooltip.className = "sl-tooltip";
                        const detailHtml = detail.replace(/\n/g, "<br>") || "";
                        const tagsHtml = tags.map(t => `<span style="display:inline-block;background:#388e3c;color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;margin-right:4px;">${t}</span>`).join("");
                        let contentHtml = `<div style="margin-bottom:8px;font-weight:500;color:#fff;font-size:14px;">${name}</div>`;
                        contentHtml += `<div style="color:#888;font-size:11px;margin-bottom:8px;">${lang === "zh" ? "分类" : "Category"}: ${tagsHtml}</div>`;
                        if (detailHtml) {
                            contentHtml += `<div style="border-top:1px solid #333;padding-top:8px;color:#aaa;font-size:12px;">${detailHtml}</div>`;
                        }
                        tooltip.innerHTML = contentHtml;
                        document.body.appendChild(tooltip);

                        const rect = card.getBoundingClientRect();
                        const tooltipWidth = 380;
                        if (rect.right + tooltipWidth + 10 > window.innerWidth) {
                            tooltip.style.left = `${rect.left - tooltipWidth - 10}px`;
                        } else {
                            tooltip.style.left = `${rect.right + 10}px`;
                        }
                        tooltip.style.top = `${rect.top}px`;

                        setTimeout(() => tooltip.classList.add("show"), 10);

                        tooltip.addEventListener("mouseenter", () => {
                            mouseInTooltip = true;
                        });

                        tooltip.addEventListener("mouseleave", () => {
                            mouseInTooltip = false;
                            setTimeout(() => {
                                if (!mouseInTooltip && tooltip) {
                                    tooltip.remove();
                                    tooltip = null;
                                }
                            }, 150);
                        });
                    }
                });

                card.addEventListener("mouseleave", () => {
                    setTimeout(() => {
                        if (!mouseInTooltip && tooltip) {
                            tooltip.remove();
                            tooltip = null;
                        }
                    }, 150);
                });
            }

            listContainer.appendChild(card);
        });
    }

    searchInput.oninput = () => renderList();

    btnImport.onclick = () => importSkillFolder(() => loadCatalog());
    btnEdit.onclick = async () => {
        if (!selectedId) {
            alert(lang === "zh" ? "请先选择一个 Skill" : "Please select a skill first");
            return;
        }
        const skill = allSkills.find((s) => s.id === selectedId);
        if (!skill) {
            alert(lang === "zh" ? "找不到该 Skill" : "Skill not found");
            return;
        }
        const categories = lang === "zh" ? catalog.categories_zh : catalog.categories_en;
        showEditModal(skill, () => loadCatalog(), categories, lang);
    };

    loadCatalog();

    return { widget: root };
}

app.registerExtension({
    name: "OmniLlm.SkillLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const dom = createSkillPanel(this);
            this.addDOMWidget("skill_panel", "div", dom.widget, {
                getValue() { return ""; },
                setValue(v) { },
            });

            const [w, h] = this.size || [0, 0];
            this.setSize([Math.max(w, 360), Math.max(h, 340)]);
            this.minHeight = 340;
            this.minWidth = 360;

            log("node created with DOM widget", this.id);
            return r;
        };
    }
});

log("extension loaded");
