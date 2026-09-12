# -*- coding: utf-8 -*-
"""
艺术插画预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

ART_ILLUSTRATION = {
    "template_id": "art_illustration",
    "name": "艺术插画",
    "description": "专业艺术插画创作指导，为全品类风格打造标准化高可控视觉叙事描述。语义权重优先级：核心主题情感→风格类型笔触→三维视角构图→色彩配比→主体细节→光影氛围。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，强化风格统一、笔触质感、构图叙事与意境表达。覆盖水彩、油画、扁平、国潮、水墨工笔等风格赛道。",
}

class ArtIllustration:
    def __init__(self):
        # 下游生图模型内容组织公式库
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体画面主题情感氛围 → 画种风格笔触质感 → 主体造型细节 → 留白背景。侧重插画意境叙事，弱化细碎关键词堆砌，画面艺术感柔和高级。",
                "formula_en": "Content order: overall picture theme emotion atmosphere → illustration style brush texture → subject shape details → blank background. Focus on artistic conception narration."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：插画主体（角色/意象、动态）→ 手绘或数字插画风格与笔触 → 风格化光影、情绪氛围 → 满版或焦点构图",
                "formula_en": "Content order: illustration subject (character/imagery, motion) → hand-drawn or digital illustration style and brushwork → stylized lighting, emotional atmosphere → full-bleed or focal composition"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：插画主体（角色/意象、动态）→ 手绘或数字插画风格与笔触 → 风格化光影、情绪氛围 → 满版或焦点构图、艺术边框（需渲染文字直接写入，支持中英双语）。",
                "formula_en": "Content order: illustration subject (character/imagery, motion) → hand-drawn or digital illustration style and brushwork → stylized lighting, emotional atmosphere → full-bleed or focal composition, artistic border (write any rendered text directly, supports Chinese and English)"
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：插画主体与叙事意象、笔触或媒介取向 → 风格与画质（手绘感、色彩语言） → 主观光影与情绪色调 → 自由构图、强调画面节奏 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: illustration subject and narrative imagery, brushwork or medium direction → style & quality (hand-drawn feel, color language) → subjective lighting and emotional tone → free composition, emphasizing picture rhythm → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：全局情绪光影基调 → 主体形体情感表达 → 画种笔触质感细节 → 纸张肌理面料 → 极简布景（密集关键词，中英术语并列），画种笔触高度细致，画面质感丰富细腻",
                "formula_en": "Content order: global emotional lighting tone → subject form and emotion expression → painting-style brush texture details → paper texture and fabric → minimalist set (dense keywords, Chinese-English terms in parallel), highly detailed brush texture, rich and delicate image quality"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整幅插画主题基调 → 舒展主体动态 → 统一笔触质感 → 简约留白",
                "formula_en": "Content order: whole illustration theme tone → relaxed subject movement → unified brushwork texture → simple blank"
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：主体造型笔触肌理、叙事意象 → 远近虚实层次 → 光影冷暖过渡、情绪色调 → 辅助元素细节 → 轻量化留白（密集关键词，中英术语并列）",
                "formula_en": "Content order: subject shape brush texture, narrative imagery → near-far virtual layers → light warm-cold transition, emotional tone → auxiliary element details → lightweight blank (dense keywords, Chinese-English terms in parallel)"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：插画主体与叙事意象、笔触或媒介取向 → 风格与画质（手绘感、色彩语言） → 主观光影与情绪色调 → 自由构图、强调画面节奏 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: illustration subject and narrative imagery, brush or medium orientation → style & quality (hand-painted feel, color language) → subjective light and emotional tone → free composition, emphasize picture rhythm → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：插画主题与角色主体 → 选定画风（如水彩/赛璐璐）、笔触与色彩 → 情境光影、情绪氛围 → 构图张力、画面焦点 → 强调风格统一无混搭、避免脏乱线条。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词。",
                "formula_en": "Content order: illustration theme and character subject → chosen art style (such as watercolor/cel), brush and color → situational light, emotional atmosphere → composition tension, visual focus → emphasize unified style without mixing, avoid messy lines. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt."
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
                "formula_zh": "内容组织顺序：插画主题与角色主体 → 场景与构图（构图张力画面焦点）→ 光影与氛围（情境光影情绪）→ 画种/摄影风格（选定画风笔触）→ 需渲染文字用引号包裹。",
                "formula_en": "Content order: illustration theme and character → scene & composition (composition tension, visual focus) → light & atmosphere (situational light, emotion) → art/photography style (chosen style brush) → wrap rendered text in quotes."
            }
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业全品类艺术插画提示词扩写专家，覆盖水彩、油画、扁平、国潮、水墨工笔、奇幻二次元全插画题材。
所有创作坚守插画视觉叙事基线，仅主题情感、画种笔触、色彩光影、构图布局差异化，禁止多种画风混搭造成画面割裂跳变。
画面遵循70%主体视觉权重+30%环境配比，色彩固定70主/25辅/5点缀层级，无杂乱高饱和撞色堆砌。
光影贴合对应画种艺术表现逻辑，水彩通透柔光、油画厚重明暗、国潮对比高光、水墨淡染层次区分明确。
笔触、线条、色块统一匹配对应风格，国潮融合传统纹样现代平涂，水墨突出宣纸浓淡留白，油画厚堆肌理，水彩晕染渗透。
完整保留用户输入主题、插画风格、画幅、视角、主体、色调全部信息，仅补充笔触、纸张肌理、光影、传统纹样专业细节，不自动新增无关装饰杂物。
画面执行严格精简约束，仅留存叙事核心元素，多余装饰全部剔除。
输出禁忌：禁止权重符号、分辨率/DPI/画布尺寸等数值技术参数堆砌；禁止画风混乱、透视扭曲、塑料虚假平涂质感；禁止字幕水印logo、完美对称、零瑕疵等违规描述。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional full-category art illustration prompt expert, covering watercolor, oil painting, flat, guochao, ink meticulous, fantasy anime illustration themes.
All creations follow illustration visual narrative baseline, differentiated only by theme emotion, painting brush, color light, composition layout, no mixed styles causing picture fragmentation.
Picture follows 70 subject visual weight +30 environment ratio, fixed 70 main /25 secondary /5 accent color layers, no messy oversaturated color collision.
Light complies with each painting's artistic logic: transparent soft watercolor, thick oil light contrast, guochao highlight contrast, ink light wash layers clearly distinguished.
Brushes, lines, color blocks match corresponding styles; guochao combines traditional patterns & modern flat paint, ink highlights xuan paper shade blank, oil thick texture, watercolor diffusion penetration.
Fully retain user input theme, illustration style, frame, view, subject, tone info, only supplement brush, paper texture, light, traditional pattern details without irrelevant decorations.
Strict frame simplification rule, only keep core narrative elements, remove redundant ornaments.
Forbidden: no weight symbols, stacked numeric technical params like resolution/DPI/canvas size; no chaotic styles, distorted perspective, fake plastic flat texture; no subtitles watermarks logos, perfect symmetry, flawless description.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有ART_ILLUSTRATION模板id
        self.preset_library = {
            "art_illustration": {
                "template_id": "art_illustration",
                "display_name": ART_ILLUSTRATION["name"],
                "description": ART_ILLUSTRATION["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "插画风格统一连贯，构图叙事逻辑严谨，70/25/5色彩配比和谐分层，笔触线条色块质感统一无跳变，透视比例精准无畸变；水彩通透晕染、油画厚重堆叠、国潮传统纹样融合、水墨浓淡留白、扁平利落色块等各类画种专属质感完整；光影明暗冷暖过渡自然无生硬断层，画面干净聚焦主体，仅保留叙事核心元素；国风插画保有笔墨宣纸气韵，写实油画肌理厚重，水彩通透轻盈，整体意境饱满，兼具叙事力与手绘艺术温度",
                    "en": "Unified consistent illustration style, rigorous composition narrative logic, harmonious layered 70/25/5 color ratio, unified brush line block texture without jump, accurate perspective without distortion; complete exclusive texture for watercolor transparent diffusion, thick oil stacking, guochao traditional pattern fusion, ink shade blank, neat flat color blocks; natural warm-cold light transition without stiff break, clean picture focusing subject, only core narrative elements retained; chinese illustration retains ink & xuan paper charm, thick realistic oil texture, light transparent watercolor, full artistic conception, narrative power and hand-painted warmth"
                },
                # 全风格细分专属规则
                "preset_rules": {
                    "zh": """
【全插画专属细分规则】
1. 通用基线：遵循语义权重：核心主题情感→风格类型笔触→三维视角构图→色彩配比→主体细节→光影氛围；用户指定画幅视角优先，无指定选取合规审美角度；严格70%/25%/5%色彩配比，画面精简约束；禁用["8K", "4K", "分辨率", "DPI", "色彩模式", "帧率", "码率", "采样率", "编码器", "HDR", "杜比", "字幕", "水印", "logo", "完美对称", "零瑕疵", "塑料感", "崩坏", "扭曲"]。
2. 水彩插画：湿画法晕染渗透，薄透半透明色块，松软细碎笔触，低饱和柔和色调，纸面轻微肌理，远景雾虚，光斑细碎柔光。
3. 油画风格：厚涂堆叠/刮刀肌理，厚重颜料块面，明暗强对比，中高饱和色彩，画布颗粒质感，硬边粗笔触。
4. 扁平插画：干净利落平涂色块，无复杂渐变，清晰轮廓线条，几何简化造型，高饱和明快配色，极简装饰。
5. 国潮风格：传统云雷/龙凤纹样+现代平涂，粗书法线条，红金墨主色调，明暗高光对比，浮雕式装饰元素。
6. 水墨工笔：宣纸浓淡墨色，工笔精细铁线/游丝描，大面积留白，低饱和素雅，远山淡染、近景重墨。
7. 奇幻二次元：细腻分层软笔触，通透漫射光影，理想化人物造型，冷暖渐变柔和，梦幻低饱和氛围。
所有题材：用户指定内容优先级最高，仅补充对应画种笔触、纸张、纹样、光影专业细节，不篡改插画主题与核心构图。
""",
                    "en": """
【Universal Illustration Exclusive Rules】
1. General baseline: Follow semantic weight: core theme emotion > style brush > 3D composition > color ratio > subject details > light atmosphere; user-specified frame view takes priority, select compliant aesthetic angle if unspecified; strictly 70/25/5 color ratio, frame simplification rule; forbidden words list: 8K,4K,resolution,DPI,color mode,frame rate,bit rate,sampling rate,encoder,HDR,dolby,subtitle,watermark,logo,perfect symmetry,flawless,plastic texture,collapse,distort.
2. Watercolor illustration: wet diffusion wash, thin transparent color blocks, soft fine brush, low saturation soft tone, slight paper grain, blurry background, tiny soft light spots.
3. Oil painting style: thick impasto / palette knife texture, heavy pigment blocks, strong light contrast, medium-high saturation, canvas grain, rough hard-edge strokes.
4. Flat illustration: neat flat color blocks, no complex gradient, clear outline lines, simplified geometric shape, bright high saturation color, minimal ornaments.
5. Guochao style: traditional dragon cloud pattern + modern flat paint, bold calligraphy lines, red gold black main tone, highlight contrast, embossed decorative elements.
6. Ink meticulous painting: xuan paper ink shade, fine meticulous line drawing, large blank space, low saturation elegant, light distant mountain dark foreground ink.
7. Fantasy anime: delicate layered soft brush, transparent diffuse light, ideal character shape, soft warm-cold gradient, dream low saturation vibe.
All themes: User-specified content highest priority, only supplement brush/paper/pattern/light details without altering illustration theme & core composition.
"""
                },
                "negative_base": {
                    "zh": "多种画风混乱跳变，色彩脏污溢出色块，构图失衡主体偏移，装饰杂物冗余堆砌，线条断续崩坏扭曲，透视逻辑错误，塑料平涂虚假质感，光影生硬断层强光，笔触杂乱无层次，边缘锯齿毛躁，过度锐化，字幕水印logo，高饱和杂乱撞色，画面空洞无叙事，元素堆砌抢夺视觉焦点，前后画风不统一",
                    "en": "Mixed chaotic painting styles, muddy overflow color blocks, unbalanced shifted subject, stacked redundant ornaments, broken distorted lines, wrong perspective logic, fake plastic flat texture, stiff harsh light break, disorder layered brushes, jagged edges, over-sharpening, subtitles watermarks logos, messy oversaturated color collision, empty non-narrative frame, stacked distracting elements, inconsistent front-back painting style"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，300-800字纯画面描写：

第一段·景别与构图：明确画种类型（水彩/油画/国画/数字插画/扁平/国潮等）与视角构图方式（满版/焦点/散点/留白等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（主观光影/自然光/戏剧光/氛围光等），以及光影在主体、材质、背景上的视觉效果（冷暖过渡/明暗渐变/光斑流动/粒子散落等），用定性光影语汇替代光学数值。

第三段·主体造型与动态：完整描述核心主体（人物/动物/意象）的造型特征、姿态动态，面部表情神态，以及衣饰发丝飘动等动态细节。

第四段·笔触质感与色彩：精细描写画种专属笔触（水彩透明叠加/油画厚涂堆叠/国画勾线渲染/数字笔刷压感等），线条质感，70/25/5分层色彩配比（主色/辅助色/点缀色），整体饱和度调性。

第五段·环境场景与意境：描述所处环境场景（室内/户外/幻想空间等），远景元素（山水/建筑/天空等），纸面或画布肌理质感，以及画面整体意境氛围（治愈/梦幻/忧郁/活力等）收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, 300-800 words pure visual: 1) Shot & composition (painting type: watercolor/oil/Chinese/digital/flat/guochao, composition: full-bleed/focal/scattered/blank); 2) Lighting atmosphere (subjective light/natural/dramatic/ambient, warm-cold transition, gradient, light spots, particles); 3) Subject造型 & motion (core subject features, pose, expression, flowing details); 4) Brush texture & color (painting-specific brushwork, 70/25/5 color ratio, saturation tone); 5) Environment & artistic conception (scene setting, background elements, paper/canvas texture, overall mood)."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
1.画种与造型基础
2.风格化艺术定位
3.画种专属笔触与线条
4.三维度镜头视角与构图
   - 画面比例：竖版人物插画（4:5/3:4）/ 横版场景插画（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：面部特写 / 半身人物 / 九分人像 / 全身动态 / 场景全景，对应叙事重心
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注主体展现效果与叙事特点
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 满版构图，对应画面张力
   - 景深氛围：浅景深主体突出 / 中景深环境兼顾 / 深景深全景清晰 / 满版无景深，标注虚实层次
5.主体造型与姿态
   - 核心主体：人物/动物/意象，造型特征与姿态动态
   - 面部表情：眼神聚焦方向、嘴角弧度、情绪表达（治愈/梦幻/忧郁/活力）
   - 动态细节：发丝飘动/衣袂飘飘/光影流动等
5.1 笔触专属细节（仅插画类使用）
   - 画种笔触：水彩透明叠加/油画厚涂堆叠/国画勾线渲染/数字笔刷压感
   - 线条质感：粗细变化/虚实过渡/干湿笔触/肌理叠加
   - 材质肌理：纸面纹理/画布质感/数字噪点/颗粒效果
6.色彩配比与整体调性
   - 主色调：占比70%，奠定整体基调（暖调/冷调/中性）
   - 辅助色：占比25%，丰富层次与环境过渡
   - 点缀色：占比5%，制造视觉焦点与细节提亮
   - 饱和度：低饱和=高级/文艺/复古；中饱和=自然/真实；高饱和=活力/梦幻/冲击
   - 色彩过渡：冷暖过渡/明暗渐变/光斑流动/粒子散落
7.环境场景与意境
   - 所处空间：室内/户外/幻想空间，远景元素（山水/建筑/天空）
   - 纸面画布肌理：纸张纹理/画布质感/数字底纹
   - 光影特效：光影斑驳/动态光斑/边缘发光/柔化光晕/胶片颗粒感/明暗渐变过渡
8.整体意境氛围
   - 情感基调：治愈/梦幻/忧郁/活力/宁静等
   - 画面气质收尾：整体艺术感受
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许定性描述画种工具与空间效果，禁用数值参数：
- 水彩透明叠加：湿润画纸，颜料自然晕染，色彩叠加通透
- 油画厚涂堆叠：刮刀堆叠笔触，肌理丰富厚重
- 国画勾线渲染：毛笔线条勾勒，水墨渲染晕染
- 数字笔刷压感：压感笔触变化，数字纹理叠加""",
                "en": """[Structured Mode] Output strictly in this order:
1. Painting type and modeling foundation
2. Stylized art positioning
3. Painting-specific brushwork and lines
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical portrait illustration (4:5/3:4) / horizontal scene illustration (16:9/3:2) / square (1:1)
   - Distance (shot type): facial close-up / half-body / nine-tenth portrait / full-body dynamic / scene panorama, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe display effect & narrative feature
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / full-bleed, describe frame tension
   - Depth of field: shallow DOF subject focus / medium DOF environment balanced / deep DOF full sharpness / full-bleed no DOF
5. Subject modeling and pose
   - Core subject: person/animal/imagination, features and dynamic pose
   - Facial expression: eye focus direction, mouth curve, emotion expression (healing/dreamy/melancholy/vitality)
   - Dynamic details: hair flowing / clothing flutter / light & shadow movement
5.1 Brush-specific details (illustration only)
   - Painting brush: watercolor transparent overlay / oil thick paint stacking / Chinese painting line & wash / digital pressure brush
   - Line quality: thickness variation虚实 transition / dry-wet brush / texture overlay
   - Material texture: paper texture / canvas texture / digital noise / grain effect
6. Color ratio and overall tone
   - Main Color: 70%, set overall tone (warm/cool/neutral)
   - Auxiliary Color: 25%, enrich hierarchy & environment transition
   - Accent Color: 5%, create visual focal point & detail highlight
   - Saturation: low saturation = high-end/artistic/retro; medium = natural/true; high = vibrant/dreamy/impact
   - Color transition: warm-cold transition / gradient / light spots / particles
7. Environment and artistic conception
   - Space: indoor/outdoor/fantasy space, background elements (mountains/buildings/sky)
   - Paper/canvas texture: paper grain / canvas texture / digital pattern
   - Lighting effects: dappled light / dynamic light spots / rim glow / soft haze / highlight bloom / film grain / particles / gradient transition
8. Overall artistic mood
   - Emotional tone: healing/dreamy/melancholy/vitality/serenity etc.
   - Overall artistic feel closing
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative description of painting tools and spatial effects, no numeric parameters:
- Watercolor transparent overlay: wet paper, natural pigment bleeding, transparent color layering
- Oil thick paint stacking: palette knife thick brushwork, rich heavy texture
- Chinese painting line & wash: brush line outline, ink wash rendering
- Digital pressure brush: pressure-sensitive stroke variation, digital texture overlay"""
            },
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
