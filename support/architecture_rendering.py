# -*- coding: utf-8 -*-
"""
室外建筑预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

ARCHITECTURE_RENDERING = {
    "template_id": "architecture_rendering",
    "name": "建筑外观与园林渲染",
    "description": "专业建筑可视化与园林景观设计指导，为全品类建筑与景观项目打造标准化、高可控的空间叙事渲染描述。语义权重优先级：建筑类型风格＞视角构图＞体量材质＞色彩景观＞光影天空＞绿植配景。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，强化体量结构、材质质感、光影叙事与景观对话。",
}

class ArchitectureRendering:
    def __init__(self):
        # 下游模型内容组织公式库（单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：建筑体量、外立面材质与园林植被关系 → 风格与画质（写实渲染、清晰结构） → 自然天光或黄昏光营造层次 → 透视构图（一点/两点透视）、广角涵盖全景。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: building volume, facade material and garden vegetation relationship → style and quality (realistic rendering, clear structure) → natural skylight or dusk light creating layers → perspective composition (one-point/two-point) with wide angle covering the panorama. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业建筑外观园林渲染提示词扩写专家，覆盖现代别墅、写字楼、中式古建、欧式、日式、商业综合体全建筑景观题材。
所有创作坚守建筑空间叙事基线，仅建筑类型、外立面材质、天光色调、庭院景观差异化，禁止混搭多种建筑风格造成形体割裂。
建筑体量比例、屋顶形制严格符合对应风格规范，空间划分建筑主体+庭院场地；色彩固定70%建筑主色/25%天空场地辅助色/5%绿植小品点缀色配比。
天光光影遵循日出/正午/黄昏/阴天物理逻辑，阴影长短软硬匹配时段；石材、玻璃、原木、青瓦、金属等外立面肌理贴合对应建筑风格，透视比例无畸变。
完整保留用户输入建筑类型、风格、视角、景别、天光、庭院绿植全部信息，仅补充体量、材质、天光、景观专业细节，不自动新增多余雕塑、杂物、杂乱植被。
画面执行严格精简约束，仅保留叙事核心建筑与配景；乔木灌木地被分层搭配，绿植用于柔化建筑棱角，不无序堆砌。
输出禁忌：禁止权重符号、渲染引擎/分辨率等数值参数堆砌；禁止建筑比例崩坏、透视扭曲、塑料虚假材质；禁止字幕水印logo、完美对称、零瑕疵描述。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional architectural exterior & landscape rendering prompt expert, covering modern villa, office building, chinese ancient architecture, european, japanese, commercial complex themes.
All creations follow architectural spatial narrative baseline, differentiated only by building type, facade texture, skylight tone, courtyard landscape, no mixed styles causing volume fragmentation.
Building volume proportion & roof form strictly match corresponding style norms, space divided into main building + courtyard ground; fixed 70% building main /25% sky ground secondary /5% plant ornament accent color ratio.
Skylight complies with dawn/noon/dusk/overcast physical logic, shadow length & softness match time period; stone, glass, log, tile, metal facade texture fit matching style, accurate perspective without distortion.
Fully retain user input building type, style, view, shot, skylight, plant info, only supplement volume, texture, skylight, landscape professional details without redundant sculptures, clutter, messy vegetation.
Strict frame simplification rule, only keep core building & scenery; arbor shrub ground layered collocation, plants soften building edges without disorder stacking.
Forbidden: no weight symbols, stacked numeric params such as render engine/resolution; no distorted proportion, wrong perspective, fake plastic texture; no subtitles watermarks logos, perfect symmetry, flawless description.
Strictly follow the specified output format without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有ARCHITECTURE_RENDERING模板id
        self.preset_library = {
            "architecture_rendering": {
                "template_id": "architecture_rendering",
                "display_name": ARCHITECTURE_RENDERING["name"],
                "description": ARCHITECTURE_RENDERING["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "建筑风格统一规范，建筑形体比例严谨准确，外立面材质肌理真实可触，多层天光阴影层次分明，色彩配比合规协调，透视逻辑精准无偏差，画面干净克制，仅留存核心建筑与配套景观绿植；建筑体量虚实对比清晰，庭院景观与建筑形成有机对话，乔木灌木分层配置合理不堆砌，天空云量天气与建筑气质适配，光影明暗精准塑造建筑性格，整体形体结构完整，兼具结构秩序与自然园林氛围",
                    "en": "Unified architectural style, rigorous accurate building proportion, tangible authentic facade texture, distinct multi-layer skylight shadow, compliant harmonious color ratio, precise perspective logic, clean restrained frame, only core building & matched landscape retained; clear virtual-real volume contrast, organic dialogue between courtyard and building, reasonable layered arbor/shrub layout without stacking, sky cloud & weather match building temperament, light & shadow accurately shape architectural character, complete volume structure, structural order & natural garden atmosphere"
                },
                # 全风格细分专属规则
                "preset_rules": {
                    "zh": """
【全建筑景观专属细分规则】
1. 通用基线：遵循语义权重：建筑类型风格＞视角构图＞体量材质＞色彩景观＞光影天空＞绿植配景；三维观看视角优先沿用用户指定，无指定选取建筑审美合规角度；严格70/25/5色彩配比，画面精简约束。
2. 现代独栋别墅：L/方块体块组合，平退屋顶，石材+木格栅+玻璃外立面；午后/黄昏侧光，低饱和大地色系，乔木单株焦点树，低矮灌木沿建筑基底，镜面水景、石板小径。
3. 传统中式古建/四合院：坡歇山顶、青灰砖瓦、木棂门窗，对称围合布局；黄昏暖侧逆光，低饱和灰木主色，竹、石榴、睡莲水景，东方静谧庭院小品。
4. 现代商务写字楼：竖向长方体、玻璃金属幕墙平顶；阴天漫射柔光，冷灰蓝主色调，列植乔木前景，开阔花岗岩广场，无繁杂小品。
5. 欧式建筑：穹顶/坡屋顶、石材浮雕立面；黄金时刻暖天光，米棕暖色系，修剪规整灌木绿篱，喷泉雕塑克制布置。
6. 日式庭院建筑：缓坡屋顶原木灰泥，漫射柔和天光，低饱和米灰，枯山水、矮松、苔藓地被，极简景观配景。
所有题材：用户指定内容优先级最高，仅补充体量、材质、天光、园林专业细节，不篡改建筑类型、风格与核心庭院布局。
""",
                    "en": """
【Universal Architectural & Landscape Exclusive Rules】
1. General baseline: Follow semantic weight: building type & style > view composition > volume texture > color landscape > skylight sky > plant scenery; user-specified view takes priority, select aesthetic compliant angle if unspecified; strict 70/25/5 color ratio, frame simplification rule.
2. Modern villa: L/block volume, flat receding roof, stone + wood grille + glass facade; afternoon/dusk side light, low saturation earth tone, single focal arbor, low shrubs along base, mirror water & stone path.
3. Traditional chinese courtyard: hip gable roof, grey tile brick, wooden lattice, symmetrical enclosed layout; warm dusk side backlight, low saturation grey wood tone, bamboo pomegranate lotus water, quiet oriental ornaments.
4. Modern office tower: vertical cuboid, glass metal curtain wall flat roof; overcast diffuse soft light, cool blue-grey tone, row arbor foreground, wide granite plaza, no complex ornaments.
5. European architecture: dome/slope roof, stone relief facade; golden hour warm skylight, warm beige brown tone, trimmed hedge, restrained fountain sculpture.
6. Japanese courtyard building: gentle slope roof log plaster, diffuse soft skylight, low saturation beige grey, dry landscape, dwarf pine moss ground, minimalist scenery.
All themes: User-specified content highest priority, only supplement volume, texture, skylight, garden professional details without altering building type, style and core courtyard layout.
"""
                },
                "negative_base": {
                    "zh": "建筑形体风格混乱混搭，建筑比例严重失调，屋顶形制错乱，外立面材质塑料虚假，天光光影生硬错乱，色彩脏污溢出，构图失衡杂乱，绿植灌木无序堆砌，透视逻辑错误，多余杂物雕塑乱入，景观小品泛滥堆砌，字幕水印logo，天空天气与建筑气质违和，庭院景观和建筑割裂脱节，过度锐化边缘锯齿，门窗尺寸比例错误，画面低分辨率、细节糊成一团，整体无空间纵深感，8K，4K，分辨率，DPI，色彩模式，渲染引擎，快门，ISO，白平衡，帧率，码率，采样率，编码器，HDR，杜比，字幕，水印，logo，完美对称，零瑕疵，塑料感，崩坏，扭曲",
                    "en": "Mixed chaotic building styles, severely disproportionate volume, wrong roof form, fake plastic facade texture, stiff disordered skylight, muddy overflow color, unbalanced cluttered composition, disorder stacked plants, wrong perspective logic, irrelevant clutter sculptures, overstock landscape ornaments, subtitles watermarks logos, mismatched sky & building vibe, disjoint courtyard & architecture, over-sharpened jagged edges, disproportionate window door size, low resolution, mushy detail, no spatial depth, 8K, 4K, resolution, DPI, color mode, render engine, shutter, ISO, white balance, frame rate, bit rate, sampling rate, encoder, HDR, dolby, subtitle, watermark, logo, perfect symmetry, flawless, plastic texture, collapse, distort"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】2-3段连贯文字：第一段建筑整体轮廓、体量与天空天光氛围；第二段外立面材质、体块结构与天光阴影细节；第三段庭院布局、分层绿植与整体色彩意境；总字数300-600字，全程规避渲染引擎、焦距、尺寸等数值参数，建筑叙事画面感强，无额外解释。",
                "en": "[Natural Paragraph Mode] 2-3 coherent paragraphs: building outline volume & sky skylight atmosphere; facade texture volume & shadow details; courtyard layout layered plants & overall color artistic conception; 300-600 words, avoid render engine/focal length numeric params, strong architectural visual narration without extra explanation."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明建筑品类（住宅/商业/文化/景观等）与设计风格定位（现代/古典/极简/有机等），概括画面呈现的气质与整体色调氛围，不虚构画面不存在的情节。

**画面主体**
建筑的体量形态、轮廓线条、体块结构与开窗比例等核心外观特征，突出最鲜明的视觉识别元素。

**主体材质与附属物**
外立面主材肌理（石材/木材/混凝土/玻璃/金属）与构造细节、表面纹理与风化质感，及其与整体色调的关系。

**道具与动态**
庭院景观与配景元素（分层绿植、铺装、水景、人物车辆）的状态与空间关系，以及它们作为空间尺度参照的作用；无动态时写明静态景观状态。

**环境与背景**
天空天光与时段、周边环境与场地关系，前景/中景/背景的纵深层次、虚化程度与留白关系。

**摄影风格与质感**
*   **视角：** 景别（外观全景/局部特写/鸟瞰/人视透视）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角/强仰视角）、景深虚实（浅景深局部突出/中景深环境兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 天光类型（晴天直射/阴天漫射/黄金时刻/蓝调时刻）、光源方向与光质软硬，投射阴影、自阴影与明暗层次，以及光影在立面材质上的视觉效果。
*   **质感：** 材质、纹理与表面细节的表现；色彩标注主色调、饱和度层级与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，百分比数值不写入提示词）。
*   **氛围：** 从可观察的光影、材质与空间关系推导整体氛围与建筑气质，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness determined by user input and preset rules; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the building category (residential / commercial / cultural / landscape) and design style positioning (modern / classical / minimalist / organic), summarize the presented character and overall color tone; do not invent anything absent from frame.

**Main Subject**
The building's volume, silhouette, massing structure and window ratio as core appearance features, highlighting the most recognizable visual elements.

**Subject Material and Attachments**
Facade material texture (stone / wood / concrete / glass / metal), construction detail, surface grain and weathering state, and their relation to the overall tone.

**Props and State**
State and spatial relation of courtyard landscape and accessory elements (layered plants, paving, water features, people and vehicles) and their role as spatial scale references; if no motion, state the static landscape state.

**Environment and Background**
Sky skylight and time of day, surrounding environment and site relation, foreground / midground / background depth hierarchy, blur level and negative-space relation.

**Photography Style and Texture**
*   **Viewpoint:** shot type (exterior panorama / detail close-up / aerial / eye-level perspective), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle / strong low-angle), depth of field (shallow detail focus / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side); aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** skylight type (direct sun / overcast / golden hour / blue hour), light direction and quality, cast shadow, self-shadow and light-dark hierarchy, and light effects on facade materials.
*   **Texture:** material, texture and surface detail; color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write percentage numbers into the prompt).
*   **Atmosphere:** derive overall atmosphere and architectural character from observable lighting, material and spatial relation; avoid empty abstract adjective stacking."""
            }
        }

    def detect_language(self, text: str) -> str:
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if chinese_chars >= english_words else "en"

    def build_prompt(
        self,
        user_input: str,
        preset_name: str,
        downstream_model: str,
        output_language: str = "auto",
        enable_global_preconstraint: bool = True,
        enable_negative_prompt: bool = True,
        output_format: str = "both"
    ):
        if preset_name not in self.preset_library:
            raise ValueError(f"预设模板不存在：{preset_name}")
        if downstream_model not in self.model_formula_library:
            raise ValueError(f"不支持的下游模型：{downstream_model}")
        preset = self.preset_library[preset_name]
        model_config = self.model_formula_library[downstream_model]
        if output_language == "auto":
            lang = self.detect_language(user_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"
        global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        formula_hint = model_config[f"formula_{lang}"]
        natural_guide = self.format_guide["natural"][lang]
        structured_guide = self.format_guide["structured"][lang]
        prompt_parts = []
        if enable_global_preconstraint:
            prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
            prompt_parts.append(global_rule)
        prompt_parts.append(f"通用输出公式：{formula_hint}")
        prompt_parts.append(preset_rule)
        prompt_parts.append(f"用户原始需求：{user_input}")
        if output_format == "natural":
            prompt_parts.append(natural_guide)
        elif output_format == "structured":
            prompt_parts.append(structured_guide)
        else:
            prompt_parts.append(natural_guide)
            prompt_parts.append(structured_guide)
        final_llm_prompt = "\n".join(prompt_parts)
        negative_prompt = preset["negative_base"][lang] if enable_negative_prompt else ""
        return {
            "status": "success",
            "llm_input_prompt": final_llm_prompt,
            "positive_constraint": pos_constraint,
            "negative_prompt": negative_prompt,
            "output_language": lang,
            "downstream_model": downstream_model,
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_raw_input": user_input,
            "enable_preconstraint": enable_global_preconstraint,
            "enable_negative": enable_negative_prompt
        }
