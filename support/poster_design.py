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
        # 下游模型内容组织公式库（单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：主视觉主体与情绪、文字层级（标题/标语直接写入提示词） → 风格与画质（平面化或插画风、强对比） → 色彩与光影烘托主题氛围 → 中心或对角线构图、留白排版。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: key visual subject and mood, text hierarchy (write titles/slogans directly into the prompt) → style and quality (graphic or illustrative, strong contrast) → color and lighting supporting the theme → centered or diagonal composition with white-space layout. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
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
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional full-category poster prompt expert, covering commercial ads, films, cultural events, gift packaging, guochao vintage posters, character & object dual tracks.
All creations follow poster information narration baseline, differentiated by theme, painting style, lens and text layout, no mixed styles.
70% main subject, 30% background & text ratio, fixed 70/25/5 color ratio; text acts as design element with clear hierarchy without covering subject.
Light complies with cinematic lens rules, distinguish suspense/guochao/commercial light; paper, fabric, metal, hot stamping printing texture match poster style.
Fully retain user input poster type, frame, view, subject, text info, only add lens, texture, font, pattern details without clutter.
Strict simplification rule, only keep core narrative & info elements, avoid crowded text.
Forbidden: no weight symbols, numeric params like resolution/font size/DPI; no chaotic styles, distorted perspective, fake plastic texture; no watermark/logo, perfect symmetry, flawless description.
Strictly follow the specified output format without extra comments.
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
1. 通用基线：语义权重：主题目标情感→风格调性→三维构图→色彩配比→主视觉→文字层级；用户画幅视角优先，无则选合规镜头；70/25/5色彩，精简约束。
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
1. General baseline: Weight order: theme emotion > style tone > 3D composition > color ratio > main visual > text hierarchy; user frame view priority; fixed color ratio & simplification rule.
2. Font style limit: use at most 3 font families across the poster — headline font + subtitle/info font + accent font; font style must match the poster tone, no mixing of conflicting fonts; fonts blend naturally with the frame without stealing attention.
3. Text color limit: pick text colors from the 70/25/5 palette, using at most 2-3 text colors; the main headline may use the accent color for emphasis, subtitles/body use the secondary color or neutral tones; keep enough contrast between text and background for readability.
4. Font size hierarchy limit: at most 3-4 size levels per poster — main headline (largest) → subtitle/slogan → supporting info/body → accent text (smallest); clear hierarchy with obvious visual guidance, no chaotic font sizes.
5. Layout composition rules: consistent text alignment (left / centered / right), no irregular scattering; keep proper spacing between text and key visual so the core subject is not covered; blank space participates in visual rhythm, text areas and image areas form a solid-void contrast; text is integrated into the frame as a design element rather than simply overlaid.
6. Minimal commercial poster: soft diffuse light, flat/photoreal texture, thin sans-serif font, low saturation solid background, sufficient blank space.
7. Guochao event poster: calligraphy headline, traditional cloud patterns, red-gold palette, meticulous flat mix, calligraphy + simple info fonts, chinese blank layout.
8. Suspense movie poster: low-angle lens, high contrast cool light, rough weathered bold font, large dark background, minimal text.
9. Vintage film poster: faded grain texture, vintage serif font, warm dusk soft light, low saturation retro tone, restrained vintage ornaments.
10. Gift package poster: matte paper texture, soft diffuse light, delicate thin font, low saturation premium palette, centered product, subtle hot stamping accents.
All themes: User demand highest priority, only add lens/texture/font/pattern details without altering core layout.
"""
                },
                "negative_base": {
                    "zh": "多种风格混乱混搭，色彩脏污溢出，构图失衡主体偏移，装饰杂物冗余堆砌，文字排版拥挤层级混乱，字体风格冲突，透视错误，塑料虚假印刷质感，光影生硬断层，边缘锯齿过度锐化，字幕水印logo，高饱和杂乱撞色，人物神态僵硬，国潮元素生硬拼贴，画面无留白节奏，信息杂乱抢夺视觉焦点，8K，4K，分辨率，DPI，色彩模式，字号，尺寸，帧率，码率，采样率，编码器，HDR，杜比，字幕，水印，logo，完美对称，零瑕疵，塑料感，崩坏，扭曲",
                    "en": "Mixed chaotic styles, muddy overflow color, unbalanced composition, stacked clutter, crowded messy text, conflicting fonts, wrong perspective, fake plastic print texture, stiff disjoint light, jagged over-sharpening, watermark/logo, messy saturated color, stiff character expressions, forced guochao elements, no blank rhythm, distracting cluttered info, 8K, 4K, resolution, DPI, color mode, font size, size, frame rate, bit rate, sampling rate, encoder, HDR, dolby, subtitle, watermark, logo, perfect symmetry, flawless, plastic texture, collapse, distort"
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
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明海报品类（电影/商业/文艺/活动等）与设计风格、情绪定位，概括画面主题与整体色调氛围，不虚构画面不存在的情节。

**画面主体**
主视觉核心主体（人物/产品/图形符号等）的造型特征、姿态与状态，突出最鲜明的视觉识别元素。

**主体材质与附属物**
主体的材质肌理与印刷质感、图形装饰元素、纹理叠加状态，及其与整体色调的关系。

**道具与动态**
画面辅助元素的摆放与视觉动线引导，文字排版与信息层级（主标题/副标题/辅助信息的字体风格、位置关系、遮挡透叠与留白）；无动态时写明静态版式状态。

**环境与背景**
背景处理（纯色/渐变/图像/留白）、负空间节奏与图文叠压关系，前景/中景/背景的层次与虚化程度。

**摄影风格与质感**
*   **视角：** 景别（主体特写/半身构图/全身动态/场景全景）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角）、景深虚实（浅景深主体突出/中景深环境兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）与文字区域的相对位置；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 主光类型与光源方向、光质软硬（硬光/柔光/散射光），明暗层次，以及光影在主体与材质上的视觉效果（金属反光/纸张纹理/布料质感）。
*   **质感：** 材质、纹理与印刷表面细节的表现；色彩标注主色调、饱和度调性与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，百分比数值不写入提示词）。
*   **氛围：** 从可观察的构图、光影与色彩推导整体氛围与情绪意境，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness determined by user input and preset rules; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the poster category (movie / commercial / artistic / event) and design style with emotional positioning, summarize the frame theme and overall color tone; do not invent anything absent from frame.

**Main Subject**
The form features, posture and state of the main visual core subject (person / product / graphic symbol), highlighting the most recognizable visual elements.

**Subject Material and Attachments**
Material texture and print quality of the subject, graphic decorative elements, texture overlay state, and their relation to the overall tone.

**Props and State**
Placement of supporting elements and visual-flow guidance, typography and information hierarchy (font style, position relation, overlap and negative space of title / subtitle / supporting info); if no motion, state the static layout state.

**Environment and Background**
Background treatment (solid / gradient / image / negative space), negative-space rhythm and text-image overlap relation, foreground / midground / background hierarchy and blur level.

**Photography Style and Texture**
*   **Viewpoint:** shot type (subject close-up / half-body composition / full-body dynamic / scene panorama), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle), depth of field (shallow subject focus / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side) and the relative position of text areas; aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type and direction, light quality (hard / soft / diffused), light-dark hierarchy, and light effects on subject and material (metallic reflection / paper texture / fabric texture).
*   **Texture:** material, texture and print surface detail; color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write percentage numbers into the prompt).
*   **Atmosphere:** derive overall atmosphere and mood from observable composition, lighting and color; avoid empty abstract adjective stacking."""
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
