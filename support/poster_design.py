# -*- coding: utf-8 -*-
"""
海报设计预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

POSTER_DESIGN = {
    "template_id": "poster_design",
    "name": "海报设计",
    "description": "专业海报设计指导，为商业、电影、活动、包装打造高可控视觉叙事描述。语义权重优先级：主题目标情感→风格类型调性→三维视角构图→色彩配比→主视觉质感→文字层级。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，强化构图叙事、文字层级、材质质感与电影级镜头语言。覆盖现代极简、国潮、电影感、复古等风格赛道，支持人物与静物海报双赛道适配。",
}

class PosterDesign:
    def __init__(self):
        # 下游生图模型内容组织公式库
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体海报主题情感光影 → 主视觉主体造型 → 材质印刷质感 → 文字留白区域。侧重海报信息叙事，弱化细碎关键词堆砌，画面商业高级。",
                "formula_en": "Content order: overall poster theme emotion lighting → main visual subject shape → printing texture → text blank area. Focus on poster information narration."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：海报主题与主体（人物/产品/标题文字）→ 设计感与印刷级画质 → 对比光或霓虹氛围 → 强中心或对角线构图、留白版式",
                "formula_en": "Content order: poster theme and subject (character/product/title text) → design sense and print-grade quality → contrast light or neon atmosphere → strong centered or diagonal composition, blank layout"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：海报主题与主体（人物/产品/标题文字）→ 设计感与印刷级画质 → 对比光或霓虹氛围 → 强中心或对角线构图、留白与版式（需渲染文字直接写入，支持中英双语）。",
                "formula_en": "Content order: poster theme and subject (character/product/title text) → design sense and print-grade quality → contrast light or neon atmosphere → strong centered or diagonal composition, blank and layout (write any rendered text directly, supports Chinese and English)"
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：主视觉主体与情绪、文字层级（标题/标语直接写入提示词） → 风格与画质（平面化或插画风、强对比） → 色彩与光影烘托主题氛围 → 中心或对角线构图、留白排版 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: main visual subject and emotion, text hierarchy (title/slogan written directly into prompt) → style & quality (flat or illustration style, strong contrast) → color and light supporting theme atmosphere → centered or diagonal composition, blank layout → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：全局电影光影基调 → 主体情绪表达 → 印刷材质细节 → 字体面料肌理 → 极简文字布景（密集关键词，中英术语并列），印刷材质高度细致，画面质感逼真呈现",
                "formula_en": "Content order: global cinematic lighting tone → subject emotion expression → print material details → font and fabric texture → minimalist text set (dense keywords, Chinese-English terms in parallel), highly detailed print material, realistic image quality presentation"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整张海报基础风格基调 → 舒展主体造型 → 统一印刷质感 → 简约文字留白",
                "formula_en": "Content order: whole poster basic style tone → relaxed subject shape → unified print texture → simple text blank"
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：画面透视主视觉肌理、文字层级 → 远近虚实层次 → 光影冷暖过渡 → 字体层级细节 → 轻量化留白（密集关键词，中英术语并列）",
                "formula_en": "Content order: perspective main visual texture, text hierarchy → near-far virtual layers → light warm-cold transition → font hierarchy details → lightweight blank (dense keywords, Chinese-English terms in parallel)"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：海报整体主题气质 → 主视觉材质细节 → 专业分层布光 → 专属字体排版 → 极简留白。整体色调统一，信息层级细腻，海报沉浸叙事强烈。",
                "formula_en": "Content order: overall poster temperament → main visual texture details → professional layered lighting → exclusive font layout → minimalist blank. Unified tone, delicate information hierarchy."
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：海报主视觉与标题主体 → 平面设计风格、配色与排版质感 → 高对比光影、视觉张力 → 中心构图、留白与字体区 → 强调信息明确无杂乱、避免元素堆叠。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词。",
                "formula_en": "Content order: poster main visual and title subject → graphic design style, color and typography texture → high contrast light, visual tension → centered composition, blank and type zone → emphasize clear information without clutter, avoid element stacking. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt."
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
                "formula_zh": "内容组织顺序：海报主视觉与标题主体 → 场景与构图（中心构图留白字体区）→ 光影与氛围（高对比光影张力）→ 画种/摄影风格（平面设计风）→ 需渲染文字用引号包裹。",
                "formula_en": "Content order: poster main visual and title → scene & composition (centered composition, blank type zone) → light & atmosphere (high contrast light, tension) → art/photography style (graphic design) → wrap rendered text in quotes."
            }
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业全品类海报设计提示词扩写专家，覆盖商业广告、电影、文化活动、礼盒包装、国潮复古全海报题材，分人物/静物双赛道。
所有创作坚守海报信息叙事基线，仅主题情感、画风、镜头、文字排版差异化，禁止多风格混搭割裂画面。
画面遵循主体70%、环境文字30分配，色彩严格70主/25辅/5点缀配比；文字作为设计元素，层级清晰不遮挡主视觉。
光影匹配电影镜头逻辑，区分电影悬疑/国潮暖光/商业柔光质感；纸质、布料、金属、烫金等印刷肌理贴合海报风格。
完整保留用户输入海报类型、画幅、视角、主视觉、文字信息，仅补充镜头、材质、字体、纹样专业细节，不自动新增无关装饰杂物。
画面执行精简约束，仅留存叙事与信息核心元素，文字克制不拥挤堆砌。
输出禁忌：禁止权重符号、分辨率/字号/DPI等数值参数堆砌；禁止画风混乱、透视扭曲、塑料虚假质感；禁止字幕水印logo、完美对称、零瑕疵描述。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional full-category poster prompt expert, covering commercial ads, films, cultural events, gift packaging, guochao vintage posters, character & object dual tracks.
All creations follow poster information narration baseline, differentiated by theme, painting style, lens and text layout, no mixed styles.
70% main subject, 30% background & text ratio, fixed 70/25/5 color ratio; text acts as design element with clear hierarchy without covering subject.
Light complies with cinematic lens rules, distinguish suspense/guochao/commercial light; paper, fabric, metal, hot stamping printing texture match poster style.
Fully retain user input poster type, frame, view, subject, text info, only add lens, texture, font, pattern details without clutter.
Strict simplification rule, only keep core narrative & info elements, avoid crowded text.
Forbidden: no weight symbols, numeric params like resolution/font size/DPI; no chaotic styles, distorted perspective, fake plastic texture; no watermark/logo, perfect symmetry, flawless description.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有POSTER_DESIGN模板id
        self.preset_library = {
            "poster_design": {
                "template_id": "poster_design",
                "display_name": POSTER_DESIGN["name"],
                "description": POSTER_DESIGN["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "海报风格统一连贯，构图叙事逻辑严谨，70/25/5色彩配比和谐分层，主视觉材质印刷质感真实，镜头透视精准无畸变；文字层级清晰有序，字体与画面自然融合不突兀；电影海报具备镜头张力，国潮传统纹样与现代排版自然融合，商业海报印刷细腻无过度锐化；光影冷暖过渡柔和，画面干净聚焦主体，仅保留核心叙事元素，留白参与视觉节奏，兼具信息传递力与艺术质感",
                    "en": "Unified poster style, rigorous composition logic, harmonious 70/25/5 color layers, authentic printing texture, accurate lens perspective; clear text hierarchy, fonts blend naturally with frame; movie posters own lens tension, guochao combines traditional patterns & modern layout, commercial print delicate without over-sharpening; soft light transition, clean subject-focused frame, blank space adjusts visual rhythm, balance info delivery & artistic texture"
                },
                # 全风格细分专属规则
                "preset_rules": {
                    "zh": """
【全海报专属细分规则】
1. 通用基线：语义权重：主题目标情感→风格调性→三维构图→色彩配比→主视觉→文字层级；用户画幅视角优先，无则选合规镜头；70/25/5色彩，精简约束；禁用["8K", "4K", "分辨率", "DPI", "色彩模式", "字号", "尺寸", "帧率", "码率", "采样率", "编码器", "HDR", "杜比", "字幕", "水印", "logo", "完美对称", "零瑕疵", "塑料感", "崩坏", "扭曲"]。
2. 字体样式限制：整张海报最多使用3种字体，标题字体+副标题/信息字体+点缀字体；字体风格需与海报调性统一，禁止风格冲突的字体混搭；字体与画面自然融合，不突兀抢戏。
3. 文字颜色限制：文字颜色从70/25/5主色系中选取，最多使用2-3种文字颜色；主标题可用点缀色突出，副标题/正文用辅助色或中性色；文字与背景需保持足够对比度，确保可读性。
4. 字号层级限制：整张海报最多3-4级字号层级，主标题（最大）→副标题/标语→辅助信息/正文→点缀文字（最小）；层级清晰，视觉引导明确，禁止字号混乱。
5. 排版构图规则：文字对齐方式统一（左对齐/居中/右对齐），禁止无规律散落；文字与主视觉保持适当间距，不遮挡核心主体；留白参与视觉节奏，文字区域与图像区域形成虚实对比；文字作为设计元素融入画面，而非简单叠加。
6. 现代极简商业海报：柔和漫射柔光，干净平涂/摄影质感，纤细无衬线字体，低饱和纯色背景，文字克制少，留白充足，产品/人物居中。
7. 国潮活动海报：书法标题、传统祥云缠枝纹样，红金米主色调，工笔/平涂结合，书法字体搭配简约信息字，大面积国风留白。
8. 电影悬疑海报：低角度仰拍镜头，高对比冷调光影，粗体风化标题，大面积云层/暗背景，文字极简退让不抢主体。
9. 复古胶片海报：暖褪色颗粒肌理，粗衬线复古字体，柔和黄昏柔光，低饱和旧色调，年代装饰元素克制点缀。
10. 礼盒包装海报：细腻哑光纸质，柔和漫射光，纤细精致字体，低饱和高级配色，静物产品居中，少量烫金点缀。
所有题材：用户原始需求优先级最高，仅补充镜头、材质、字体、纹样专业细节，不改动海报主题与核心排版。
""",
                    "en": """
【Universal Poster Exclusive Rules】
1. General baseline: Weight order: theme emotion > style tone > 3D composition > color ratio > main visual > text hierarchy; user frame view priority; fixed color ratio & simplification rule; forbidden words list: 8K,4K,resolution,DPI,color mode,font size,size,frame rate,bit rate,sampling rate,encoder,HDR,dolby,subtitle,watermark,logo,perfect symmetry,flawless,plastic texture,collapse,distort.
2. Minimal commercial poster: soft diffuse light, flat/photoreal texture, thin sans-serif font, low saturation solid background, sufficient blank space.
3. Guochao event poster: calligraphy headline, traditional cloud patterns, red-gold palette, meticulous flat mix, calligraphy + simple info fonts, chinese blank layout.
4. Suspense movie poster: low-angle lens, high contrast cool light, rough weathered bold font, large dark background, minimal text.
5. Vintage film poster: faded grain texture, vintage serif font, warm dusk soft light, low saturation retro tone, restrained vintage ornaments.
6. Gift package poster: matte paper texture, soft diffuse light, delicate thin font, low saturation premium palette, centered product, subtle hot stamping accents.
All themes: User demand highest priority, only add lens/texture/font/pattern details without altering core layout.
"""
                },
                "negative_base": {
                    "zh": "多种风格混乱混搭，色彩脏污溢出，构图失衡主体偏移，装饰杂物冗余堆砌，文字排版拥挤层级混乱，字体风格冲突，透视错误，塑料虚假印刷质感，光影生硬断层，边缘锯齿过度锐化，字幕水印logo，高饱和杂乱撞色，人物神态僵硬，国潮元素生硬拼贴，画面无留白节奏，信息杂乱抢夺视觉焦点",
                    "en": "Mixed chaotic styles, muddy overflow color, unbalanced composition, stacked clutter, crowded messy text, conflicting fonts, wrong perspective, fake plastic print texture, stiff disjoint light, jagged over-sharpening, watermark/logo, messy saturated color, stiff character expressions, forced guochao elements, no blank rhythm, distracting cluttered info"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】2-3段连贯文字：首段海报画幅构图、整体风格与情绪基调；第二段主视觉镜头、材质光影与色彩分层；第三段字体排版、信息层级与留白节奏；总字数300-600，规避字号/分辨率等数字参数，设计叙事语言无额外解释。",
                "en": "[Natural Paragraph Mode] 2-3 coherent paragraphs: poster frame composition & overall tone; main visual lens texture & color layers; font layout info hierarchy & blank rhythm; 300-600 words, no numeric print params, design narration without extra notes."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
1.海报品类与设计风格
2.视觉风格与情绪定位
3.主视觉核心主体
4.三维度镜头视角与构图
   - 画面比例：竖版海报（4:5/3:4）/ 横版海报（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：主体特写 / 半身构图 / 全身动态 / 场景全景，对应叙事重心
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注主体展现效果
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 强仰视角，对应画面张力
   - 景深氛围：浅景深主体突出 / 中景深环境兼顾 / 深景深全景清晰，标注虚实层次
5.文字排版与信息层级
   - 主标题/副标题/辅助信息：字体风格/大小层级/位置关系
   - 画面融合关系：文字与图像的遮挡/透叠/留白
   - 视觉动线引导：阅读顺序/信息优先级
6.色彩配比与饱和度调性
   - 主色70% / 辅助色25% / 点缀色5%
   - 饱和度调性：高饱和冲击/中饱和自然/低饱和高级
   - 色彩情感：暖调活力/冷调高级/中性稳重
7.材质光影与光影特效
   - 主视觉光影：光源方向/光质软硬/光影层次
   - 材质光影效果：金属反光/纸张纹理/布料质感
   - 光影特效：光影斑驳/动态光斑/边缘发光/柔化光晕/胶片颗粒感/明暗渐变过渡
8.画面品质与视觉冲击力
   - 设计完成度：排版精度/色彩统一/细节处理
   - 质感细节：材质肌理/光影层次/空间纵深
   - 整体视觉冲击力：吸引力/记忆点/传播力
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许定性描述设计工具与效果，禁用数值参数：
- 电影海报：大画幅，高对比度，戏剧性光影，文字与图像深度融合
- 商业海报：标准画幅，色彩鲜明，信息层级清晰，视觉焦点明确
- 文艺海报：留白构图，低饱和色调，诗意氛围，文字与图像呼应
- 活动海报：动态构图，高饱和色彩，活力氛围，信息突出醒目""",
                "en": """[Structured Mode] Output strictly in this order:
1. Poster category and design style
2. Visual style and emotion positioning
3. Main visual core subject
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical poster (4:5/3:4) / horizontal poster (16:9/3:2) / square (1:1)
   - Distance (shot type): subject close-up / half-body / full-body dynamic / scene panorama, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe subject display effect
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / strong low-angle, describe frame tension
   - Depth of field: shallow DOF subject focus / medium DOF environment balanced / deep DOF full sharpness
5. Typography and information hierarchy
   - Title/subtitle/info: font style/size hierarchy/position relationship
   - Image integration: text-image overlap/overlay/blank space
   - Visual flow guide: reading order/information priority
6. Color ratio and saturation tone
   - 70% main / 25% auxiliary / 5% accent
   - Saturation tone: high saturation impact / medium natural / low saturation high-end
   - Color emotion: warm vitality / cool high-end / neutral stability
7. Material lighting and lighting effects
   - Main visual lighting: light direction/light quality/light layers
   - Material light effects: metallic reflection/paper texture/fabric texture
   - Lighting effects: dappled light / dynamic light spots / rim glow / soft haze / highlight bloom / film grain / particles / gradient transition
8. Image quality and visual impact
   - Design completion: typography precision/color unity/detail handling
   - Texture details: material texture / light layers / spatial depth
   - Overall visual impact: attraction/memory point/spreadability
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative description of design tools and effects, no numeric parameters:
- Movie poster: large format, high contrast, dramatic lighting, deep text-image integration
- Commercial poster: standard format, vivid colors, clear information hierarchy, defined visual focus
- Artistic poster: blank composition, low saturation tone, poetic atmosphere, text-image echo
- Event poster: dynamic composition, high saturation colors, vibrant atmosphere, prominent information"""
            }
        }

    def detect_language(self, text: str):
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if chinese_chars >= english_words else "en"

    def build_prompt(
        self,
        user_input,
        preset_name,
        downstream_model,
        output_language="auto",
        enable_global_preconstraint=True,
        enable_negative_prompt=True,
        output_format="both"
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
