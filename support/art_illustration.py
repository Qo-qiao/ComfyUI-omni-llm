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
        # 下游模型内容组织公式库（单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：插画主体与叙事意象、笔触或媒介取向 → 风格与画质（手绘感、色彩语言） → 主观光影与情绪色调 → 自由构图、强调画面节奏。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: illustration subject with narrative imagery, brushwork or medium direction → style and quality (hand-drawn feel, color language) → subjective lighting and emotional tone → free composition emphasizing visual rhythm. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业全品类艺术插画提示词扩写专家，覆盖水彩、油画、扁平、国潮、水墨工笔、奇幻二次元全插画题材。
所有创作坚守插画视觉叙事基线，仅主题情感、画种笔触、色彩光影、构图布局差异化，禁止多种画风混搭造成画面割裂跳变。
主体占据画面主要面积与视觉中心，环境作为叙事补充，色彩固定70主/25辅/5点缀层级，无杂乱高饱和撞色堆砌。
光影贴合对应画种艺术表现逻辑，水彩通透柔光、油画厚重明暗、国潮对比高光、水墨淡染层次区分明确。
笔触、线条、色块统一匹配对应风格，国潮融合传统纹样现代平涂，水墨突出宣纸浓淡留白，油画厚堆肌理，水彩晕染渗透。
完整保留用户输入主题、插画风格、画幅、视角、主体、色调全部信息，仅补充笔触、纸张肌理、光影、传统纹样专业细节，不自动新增无关装饰杂物。
画面执行严格精简约束，仅留存叙事核心元素，多余装饰全部剔除。
输出禁忌：禁止权重符号、分辨率/DPI/画布尺寸等数值技术参数堆砌；禁止画风混乱、透视扭曲、塑料虚假平涂质感；禁止字幕水印logo、完美对称、零瑕疵等违规描述。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
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
Strictly follow the specified output format without extra comments.
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
1. 通用基线：遵循语义权重：核心主题情感→风格类型笔触→三维视角构图→色彩配比→主体细节→光影氛围；用户指定画幅视角优先，无指定选取合规审美角度；严格70%/25%/5%色彩配比，画面精简约束。
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
1. General baseline: Follow semantic weight: core theme emotion > style brush > 3D composition > color ratio > subject details > light atmosphere; user-specified frame view takes priority, select compliant aesthetic angle if unspecified; strictly 70/25/5 color ratio, frame simplification rule.
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
                    "zh": "多种画风混乱跳变，色彩脏污溢出色块，构图失衡主体偏移，装饰杂物冗余堆砌，线条断续崩坏扭曲，透视逻辑错误，写实场景误用平涂，光影生硬断层强光，笔触杂乱无层次，边缘锯齿毛躁，过度锐化，字幕水印logo，高饱和杂乱撞色，画面空洞无叙事，元素堆砌抢夺视觉焦点，前后画风不统一，8K，4K，分辨率，DPI，色彩模式，帧率，码率，采样率，编码器，HDR，杜比，字幕，水印，logo，完美对称，零瑕疵，塑料感，崩坏，扭曲",
                    "en": "Mixed chaotic painting styles, muddy overflow color blocks, unbalanced shifted subject, stacked redundant ornaments, broken distorted lines, wrong perspective logic, flat paint misused in realistic scene, stiff harsh light break, disorder layered brushes, jagged edges, over-sharpening, subtitles watermarks logos, messy oversaturated color collision, empty non-narrative frame, stacked distracting elements, inconsistent front-back painting style, 8K, 4K, resolution, DPI, color mode, frame rate, bit rate, sampling rate, encoder, HDR, dolby, subtitle, watermark, logo, perfect symmetry, flawless, plastic texture, collapse, distort"
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

第四段·笔触质感与色彩：精细描写画种专属笔触（水彩透明叠加/油画厚涂堆叠/国画勾线渲染/数字笔刷压感等），线条质感，主色铺底、辅助色过渡、点缀色提亮的定性色彩分层，整体饱和度调性。

第五段·环境场景与意境：描述所处环境场景（室内/户外/幻想空间等），远景元素（山水/建筑/天空等），纸面或画布肌理质感，以及画面整体意境氛围（治愈/梦幻/忧郁/活力等）收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, 300-800 words pure visual: 1) Shot & composition (painting type: watercolor/oil/Chinese/digital/flat/guochao, composition: full-bleed/focal/scattered/blank); 2) Lighting atmosphere (subjective light/natural/dramatic/ambient, warm-cold transition, gradient, light spots, particles); 3) Subject pose & motion (core subject features, pose, expression, flowing details); 4) Brush texture & color (painting-specific brushwork, qualitative color layering with dominant base, secondary transition and accent highlight, saturation tone); 5) Environment & artistic conception (scene setting, background elements, paper/canvas texture, overall mood)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明画种（水彩/油画/国画/数字插画/扁平/国潮等）与风格化艺术定位，概括画面讲述的瞬间、意境与整体色调氛围，不虚构画面不存在的情节。

**画面主体**
核心主体（人物/动物/意象）的造型特征、姿态动态与面部神态，突出最鲜明的视觉识别元素。

**主体材质与附属物**
画种专属笔触与线条（水彩透明叠加/油画厚涂堆叠/国画勾线渲染/数字笔刷压感）、衣饰与表面纹理图案的状态，及其与整体色调的关系。

**道具与动态**
主体与道具的互动细节、发丝衣袂飘动等动态表现；无动态时写明静态构图状态。

**环境与背景**
所处空间（室内/户外/幻想空间）与远景元素（山水/建筑/天空）、纸面或画布肌理，前景/背景的层次、留白与虚实关系。

**摄影风格与质感**
*   **视角：** 景别（面部特写/半身/全身动态/场景全景）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角/满版构图）、景深虚实（浅景深主体突出/中景深环境兼顾/深景深全景清晰/满版无景深）及各自带来的视觉感受。
*   **构图：** 构图方式（满版/焦点/散点/留白/三分法/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 光源类型（主观光/自然光/戏剧光/氛围光等）、方向与光质软硬，明暗层次，以及光影在主体、材质与背景上的视觉效果（冷暖过渡/明暗渐变等）。
*   **质感：** 笔触、线条与纸面画布肌理的表现；色彩标注主色调、饱和度层级与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，百分比数值不写入提示词）。
*   **氛围：** 从可观察的造型、光影与色彩推导整体意境氛围，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness determined by user input and preset rules; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the painting type (watercolor / oil / Chinese ink / digital illustration / flat / guochao) and stylized art positioning, summarize the captured moment, artistic conception and overall color tone; do not invent anything absent from frame.

**Main Subject**
The form features, dynamic pose and facial demeanor of the core subject (person / animal / imagery), highlighting the most recognizable visual elements.

**Subject Material and Attachments**
Painting-specific brushwork and lines (watercolor transparent overlay / oil thick paint stacking / Chinese line and wash / digital pressure brush), costume and surface texture patterns, and their relation to the overall tone.

**Props and State**
Interaction detail between subject and props, flowing dynamics such as hair and robes in motion; if no motion, state the static composition state.

**Environment and Background**
The space (indoor / outdoor / fantasy) and distant elements (mountains / buildings / sky), paper or canvas texture, foreground-background hierarchy, negative space and virtual-real relation.

**Photography Style and Texture**
*   **Viewpoint:** shot type (facial close-up / half-body / full-body dynamic / scene panorama), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle / full-bleed), depth of field (shallow subject focus / medium environment balanced / deep full sharpness / full-bleed no DOF) and the visual feeling each brings.
*   **Composition:** composition type (full-bleed / focal / scattered / negative space / rule-of-thirds / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side); aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** light source type (subjective light / natural light / dramatic light / ambient light), direction and quality, light-dark hierarchy, and light effects on subject, material and background (warm-cold transition / gradient, etc.).
*   **Texture:** brushwork, lines and paper/canvas grain; color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write percentage numbers into the prompt).
*   **Atmosphere:** derive overall artistic mood from observable form, lighting and color; avoid empty abstract adjective stacking."""
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
