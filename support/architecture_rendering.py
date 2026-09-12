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
        # 下游生图模型内容组织公式库
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体建筑天空氛围光影 → 建筑体量外立面 → 材质景观绿植 → 远景留白。侧重建筑空间叙事，弱化细碎关键词堆砌，建筑画面宏大高级。",
                "formula_en": "Content order: overall building & sky atmosphere lighting → building volume facade → texture landscape plants → distant blank space. Focus on architectural spatial narration."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：建筑与园林（风格、材质、体量）→ 写实渲染与材质 → 晨昏光或天光、通透氛围 → 透视或鸟瞰、纳入植被水景",
                "formula_en": "Content order: architecture and garden (style, material, volume) → realistic rendering and material → dawn-dusk or skylight, transparent atmosphere → perspective or bird's-eye view, incorporating vegetation and water features."
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：建筑与园林主体（风格、材质、体量）→ 写实渲染与材质表现 → 晨昏光或天光、开阔通透氛围 → 透视或鸟瞰构图、纳入植被与水景（需渲染文字直接写入，支持中英双语）。",
                "formula_en": "Content order: architecture and garden subject (style, material, volume) → realistic rendering with material expression → dawn-dusk or skylight, open transparent atmosphere → perspective or bird's-eye composition, incorporating vegetation and water features (write any rendered text directly, supports Chinese and English)."
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：建筑体量、外立面材质与园林植被关系 → 风格与画质（写实渲染、清晰结构） → 自然天光或黄昏光营造层次 → 透视构图（一点/两点透视）、广角涵盖全景 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: building volume, facade material and garden vegetation relationship → style & quality (realistic rendering, clear structure) → natural skylight or dusk light creating layers → perspective composition (one-point/two-point), wide angle covering panorama → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：全天光氛围基调 → 建筑尺度形体情绪 → 外立面材质景观细节 → 铺装植被肌理 → 极简远景布景（密集关键词，中英术语并列），建筑材质高度细致，材质肌理真实呈现",
                "formula_en": "Content order: global daylight atmosphere tone → building scale and form emotion → facade material and landscape details → pavement and vegetation texture → minimalist distant set (dense keywords, Chinese-English terms in parallel), highly detailed architectural material, realistic texture presentation"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体建筑风格基调 → 舒展建筑形体层次 → 统一材质景观质感 → 简约远景留白",
                "formula_en": "Content order: overall architectural style tone → relaxed building form layers → unified material and landscape texture → simple distant blank."
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：建筑透视外立面材质、体量 → 远近景观层次 → 天光明暗过渡、晨昏光 → 庭院铺装细节 → 轻量化远景（密集关键词，中英术语并列）",
                "formula_en": "Content order: building perspective and facade material, volume → near-far landscape layers → skylight shadow transition, dawn-dusk light → courtyard pavement details → lightweight distant view (dense keywords, Chinese-English terms in parallel)"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：建筑体量、外立面材质与园林植被 → 风格与画质（写实渲染、清晰结构） → 自然天光或黄昏光层次 → 透视构图、广角涵盖全景 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: building volume, facade material and garden vegetation → style & quality (realistic rendering, clear structure) → natural skylight or dusk light layers → perspective composition, wide angle covering panorama → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：建筑立面与园林景观 → 建筑表现图风格、材质真实与绿化细节 → 晨昏自然光、开阔氛围 → 透视全景、黄金分割 → 强调结构准确无扭曲、避免比例失调。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词。",
                "formula_en": "Content order: building facade and garden landscape → architectural visualization style, realistic material and greenery detail → dawn-dusk natural light, open atmosphere → perspective panorama, golden ratio → emphasize accurate structure without distortion, avoid proportion imbalance. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt."
            },
            "LongCat_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：主体衣着与特质描写 → 神态与动作刻画 → 环境与背景交代 → 光线与氛围渲染 → 景别与构图说明。纯中文长自然语言描述效果最佳，需渲染文字用引号包裹。",
                "formula_en": "Content order: subject clothing & traits → expression & action → environment & background → light & atmosphere → shot & composition. Long Chinese natural language describes best; wrap any rendered text in quotation marks."
            },
            "HiDream-O1-Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：建筑立面与园林景观 → 场景与构图（透视全景黄金分割）→ 光影与氛围（晨昏自然光开阔）→ 画种/摄影风格（建筑表现图）→ 需渲染文字用引号包裹。",
                "formula_en": "Content order: building facade and garden landscape → scene & composition (perspective panorama, golden ratio) → light & atmosphere (dawn-dusk natural light, open) → art/photography style (architectural visualization) → wrap rendered text in quotes."
            }
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
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional architectural exterior & landscape rendering prompt expert, covering modern villa, office building, chinese ancient architecture, european, japanese, commercial complex themes.
All creations follow architectural spatial narrative baseline, differentiated only by building type, facade texture, skylight tone, courtyard landscape, no mixed styles causing volume fragmentation.
Building volume proportion & roof form strictly match corresponding style norms, space divided into main building + courtyard ground; fixed 70% building main /25% sky ground secondary /5% plant ornament accent color ratio.
Skylight complies with dawn/noon/dusk/overcast physical logic, shadow length & softness match time period; stone, glass, log, tile, metal facade texture fit matching style, accurate perspective without distortion.
Fully retain user input building type, style, view, shot, skylight, plant info, only supplement volume, texture, skylight, landscape professional details without redundant sculptures, clutter, messy vegetation.
Strict frame simplification rule, only keep core building & scenery; arbor shrub ground layered collocation, plants soften building edges without disorder stacking.
Forbidden: no weight symbols, stacked numeric params such as render engine/resolution; no distorted proportion, wrong perspective, fake plastic texture; no subtitles watermarks logos, perfect symmetry, flawless description.
Strictly output two formats without extra comments.
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
1. 通用基线：遵循语义权重：建筑类型风格＞视角构图＞体量材质＞色彩景观＞光影天空＞绿植配景；三维观看视角优先沿用用户指定，无指定选取建筑审美合规角度；严格70/25/5色彩配比，画面精简约束；禁用["8K", "4K", "分辨率", "DPI", "色彩模式", "渲染引擎", "快门", "ISO", "白平衡", "帧率", "码率", "采样率", "编码器", "HDR", "杜比", "字幕", "水印", "logo", "完美对称", "零瑕疵", "塑料感", "崩坏", "扭曲"]。
2. 现代独栋别墅：L/方块体块组合，平退屋顶，石材+木格栅+玻璃外立面；午后/黄昏侧光，低饱和大地色系，乔木单株焦点树，低矮灌木沿建筑基底，镜面水景、石板小径。
3. 传统中式古建/四合院：坡歇山顶、青灰砖瓦、木棂门窗，对称围合布局；黄昏暖侧逆光，低饱和灰木主色，竹、石榴、睡莲水景，东方静谧庭院小品。
4. 现代商务写字楼：竖向长方体、玻璃金属幕墙平顶；阴天漫射柔光，冷灰蓝主色调，列植乔木前景，开阔花岗岩广场，无繁杂小品。
5. 欧式建筑：穹顶/坡屋顶、石材浮雕立面；黄金时刻暖天光，米棕暖色系，修剪规整灌木绿篱，喷泉雕塑克制布置。
6. 日式庭院建筑：缓坡屋顶原木灰泥，漫射柔和天光，低饱和米灰，枯山水、矮松、苔藓地被，极简景观配景。
所有题材：用户指定内容优先级最高，仅补充体量、材质、天光、园林专业细节，不篡改建筑类型、风格与核心庭院布局。
""",
                    "en": """
【Universal Architectural & Landscape Exclusive Rules】
1. General baseline: Follow semantic weight: building type & style > view composition > volume texture > color landscape > skylight sky > plant scenery; user-specified view takes priority, select aesthetic compliant angle if unspecified; strict 70/25/5 color ratio, frame simplification rule; forbidden words list: 8K,4K,resolution,DPI,color mode,render engine,shutter,ISO,white balance,frame rate,bit rate,sampling rate,encoder,HDR,dolby,subtitle,watermark,logo,perfect symmetry,flawless,plastic texture,collapse,distort.
2. Modern villa: L/block volume, flat receding roof, stone + wood grille + glass facade; afternoon/dusk side light, low saturation earth tone, single focal arbor, low shrubs along base, mirror water & stone path.
3. Traditional chinese courtyard: hip gable roof, grey tile brick, wooden lattice, symmetrical enclosed layout; warm dusk side backlight, low saturation grey wood tone, bamboo pomegranate lotus water, quiet oriental ornaments.
4. Modern office tower: vertical cuboid, glass metal curtain wall flat roof; overcast diffuse soft light, cool blue-grey tone, row arbor foreground, wide granite plaza, no complex ornaments.
5. European architecture: dome/slope roof, stone relief facade; golden hour warm skylight, warm beige brown tone, trimmed hedge, restrained fountain sculpture.
6. Japanese courtyard building: gentle slope roof log plaster, diffuse soft skylight, low saturation beige grey, dry landscape, dwarf pine moss ground, minimalist scenery.
All themes: User-specified content highest priority, only supplement volume, texture, skylight, garden professional details without altering building type, style and core courtyard layout.
"""
                },
                "negative_base": {
                    "zh": "建筑形体风格混乱混搭，建筑比例严重失调，屋顶形制错乱，外立面材质塑料虚假，天光光影生硬错乱，色彩脏污溢出，构图失衡杂乱，绿植灌木无序堆砌，透视逻辑错误，多余杂物雕塑乱入，景观小品泛滥堆砌，字幕水印logo，天空天气与建筑气质违和，庭院景观和建筑割裂脱节，过度锐化边缘锯齿，门窗尺寸比例错误，画面低分辨率模糊，整体无空间纵深感",
                    "en": "Mixed chaotic building styles, severely disproportionate volume, wrong roof form, fake plastic facade texture, stiff disordered skylight, muddy overflow color, unbalanced cluttered composition, disorder stacked plants, wrong perspective logic, irrelevant clutter sculptures, overstock landscape ornaments, subtitles watermarks logos, mismatched sky & building vibe, disjoint courtyard & architecture, over-sharpened jagged edges, disproportionate window door size, blurry low-res frame, no spatial depth"
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
                "zh": """【结构化模式】严格顺序输出：
1.建筑品类与体量特征
2.风格与设计定位
3.外立面主材肌理
4.三维度镜头视角与构图
   - 画面比例：竖版建筑图（4:5/3:4）/ 横版全景图（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：外观全景 / 局部特写 / 鸟瞰 / 人视透视，对应建筑叙事重心
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注建筑展现效果
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 强仰视角（仰望建筑），对应建筑气势
   - 景深氛围：浅景深局部突出 / 中景深环境兼顾 / 深景深全景清晰，标注虚实层次
5.景观环境与庭院布局
   - 分层绿植：乔木/灌木/地被/草坪
   - 铺装材质：石材/木材/砾石/透水砖
   - 水景元素：喷泉/泳池/镜面水池/溪流
6.天光与阴影细节
   - 天光类型：晴天直射/阴天漫射/黄金时刻/蓝调时刻
   - 阴影层次：投射阴影/自阴影/环境光遮蔽
   - 材质反光：玻璃反射/金属光泽/石材漫射
7.整体风格与色彩意境
   - 设计风格定位：现代/古典/极简/有机/参数化
   - 色彩意境：材质本色/环境融合/冷暖对比
   - 光影特效：天光氛围/阴影细节/光影斑驳/动态光斑/边缘发光/柔化光晕/明暗渐变过渡/HDR高动态
8.画面品质与建筑感
   - 写实程度：照片级/渲染级/手绘感
   - 质感细节：材质肌理/光影层次/空间纵深
   - 整体建筑感：宏伟/精致/现代/历史
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许定性描述镜头空间效果，禁用数值参数：
- 外观全景：24mm-35mm广角，f/8-f/11光圈，深景深全景清晰，展现建筑全貌
- 局部特写：85mm-200mm长焦，f/2.8-f/4光圈，突出材质肌理与构造细节
- 鸟瞰全景：超广角/无人机视角，f/8-f/11光圈，展现建筑与环境关系
- 人视透视：35mm-50mm标准镜头，f/4-f/5.6光圈，模拟人眼视角的真实感受""",
                "en": """[Structured Mode] Output strictly in this order:
1. Building category and volume features
2. Style and design positioning
3. Facade main material texture
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical building (4:5/3:4) / horizontal panorama (16:9/3:2) / square (1:1)
   - Distance (shot type): exterior panorama / detail close-up / aerial / eye-level perspective, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe building display effect
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / strong low-angle (looking up), describe building presence
   - Depth of field: shallow DOF detail focus / medium DOF environment balanced / deep DOF full sharpness
5. Landscape and courtyard layout
   - Layered plants: trees/shrubs/ground cover/lawn
   - Paving material: stone/wood/gravel/permeable brick
   - Water features: fountain/pool/mirror pond/stream
6. Skylight and shadow details
   - Skylight type: direct sun/overcast/golden hour/blue hour
   - Shadow layers: cast shadow/self-shadow/ambient occlusion
   - Material reflection: glass reflection/metallic gloss/stone diffuse
7. Overall style and color artistic conception
   - Design style positioning: modern/classical/minimalist/organic/parametric
   - Color artistic conception: material natural color/environment integration/warm-cool contrast
   - Lighting effects: skylight atmosphere / shadow details / dappled light / dynamic light spots / rim glow / soft haze / gradient transition / HDR
8. Image quality and architectural feel
   - Realism level: photo-real / rendering / hand-drawn feel
   - Texture details: material texture / light layers / spatial depth
   - Overall architectural feel: grand/exquisite/modern/historical
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative description of lens spatial effects, no numeric parameters:
- Exterior panorama: 24mm-35mm wide-angle, f/8-f/11 aperture, deep DOF full sharpness, showcasing building full view
- Detail close-up: 85mm-200mm telephoto, f/2.8-f/4 aperture, highlighting material texture and construction details
- Aerial panorama: super wide-angle/drone view, f/8-f/11 aperture, showcasing building-environment relationship
- Eye-level perspective: 35mm-50mm standard lens, f/4-f/5.6 aperture, simulating human eye perspective"""
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
        prompt_parts.append(f"下游模型内容组织公式：{formula_hint}")
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
