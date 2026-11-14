# -*- coding: utf-8 -*-
"""
电商场景与产品摄影预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

ECOMMERCE = {
    "template_id": "ecommerce",
    "name": "电商场景与产品摄影",
    "description": "专业电商视觉与产品摄影指导，为全品类电商产品打造高转化视觉描述。语义权重优先级：产品类型卖点→三维视角构图→场景氛围→色彩配比→道具搭配→光影质感。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，强化产品质感、光影高级感与购买驱动力。覆盖美妆、3C、食品、首饰、家居等品类赛道。",
}

class Ecommerce:
    def __init__(self):
        # 下游模型内容组织公式库（单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：产品主体、材质与核心卖点（名称/卖点可写入提示词） → 风格与画质（高清晰度、商业级打光） → 干净布光突出质感与反光 → 中心构图或 45° 展示、纯色或场景背景。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: product subject, material and key selling points (names/selling points may be written into the prompt) → style and quality (high clarity, commercial-grade lighting) → clean lighting highlighting texture and reflections → centered composition or 45-degree display on a solid or scene background. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业电商产品摄影提示词扩写专家，覆盖美妆、3C数码、食品、首饰、家居全电商商品赛道，适配小红书/抖音/天猫多平台。
所有创作坚守商品转化叙事基线，仅产品品类、场景、道具、光影差异化，禁止多品类元素混搭造成画面杂乱。
商品永久为视觉绝对主体，遵循70%商品+30%场景道具权重；色彩固定70主/25辅/5点缀配比，杜绝高饱和撞色堆砌。
光影贴合实物拍摄逻辑，玻璃通透柔光、金属细腻反光、膏体柔和漫射、布艺温润漫光分品类区分。
道具仅衬托产品，不可抢夺视觉焦点，材质匹配商品定位（轻奢/ins/科技/复古）。
完整保留用户输入产品、平台、视角、场景、卖点全部信息，仅补充布光、材质、道具专业细节，不自动新增多余摆件杂物。
画面执行精简约束，仅留存叙事衬托核心道具，无冗余装饰。
输出禁忌：禁止权重符号、快门/ISO/白平衡/分辨率等数值参数堆砌；禁止产品变形畸变、塑料虚假材质；禁止杂乱堆砌、水印logo、完美零瑕疵描述。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are professional e-commerce product photography prompt expert, covering beauty, 3C, food, jewelry, home goods for all shopping platforms.
All creation follow commodity conversion baseline, differentiated by product type, scene, props, lighting, no mixed clutter.
Product is absolute subject, 70 product /30 scene weight; fixed 70 main /25 secondary /5 accent color ratio, no oversaturated collision.
Light matches real shooting logic, glass soft light, metal fine reflection, cream diffuse light categorized.
Props only foil product without stealing focus, texture match product positioning.
Fully retain user input product/platform/view info, only add light/texture details without extra clutter.
Strict simplification rule, only keep necessary foil props.
Forbidden: weight symbols, shutter/ISO/white balance/resolution numeric params; distorted goods, fake plastic texture, clutter, watermark, flawless description.
Strictly follow the specified output format without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有ECOMMERCE模板id
        self.preset_library = {
            "ecommerce": {
                "template_id": "ecommerce",
                "display_name": ECOMMERCE["name"],
                "description": ECOMMERCE["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "商品为画面绝对核心，形体比例规整无畸变，各类材质肌理真实贴合物理特性；70/25/5色彩配比和谐分层，光影过渡柔和无生硬断层；构图聚焦产品核心卖点，道具克制虚化不抢视觉；美妆膏体绵密通透、3C磨砂金属细腻、首饰金属宝石内敛反光、家居布艺温润；场景氛围匹配平台调性，营销信息自然融入不突兀，整体画面高级干净，材质纹理清晰、接缝工整、光源方向统一",
                    "en": "Product absolute core, regular proportion without distortion, authentic physical texture; harmonious layered color ratio, soft light transition; composition focus on selling points, restrained blurred props; smooth cream, fine matte metal, subtle jewelry reflection, soft home fabric; scene fit platform tone, natural marketing info, clean premium frame with clear material texture, neat seams and consistent light direction"
                },
                # 全品类细分专属规则
                "preset_rules": {
                    "zh": """
【全电商品类细分规则】
1. 通用基线：语义权重：产品类型卖点→三维视角构图→场景氛围→色彩配比→道具搭配→光影质感；用户拍摄视角优先，无则选用商品合规角度；严格70/25/5色彩配比，画面精简约束。
2. 美妆护肤：玻璃/亚克力通透柔光，膏体绵密哑光，低饱和ins浅调，大理石/棉麻简约道具，单侧柔和侧光，突出质地可视卖点。
3. 3C数码：磨砂金属低反光，冷调中性色，纯色织物极简背景，多角度均匀柔光，对称规整构图，凸显精密做工。
4. 食品饮品：温润漫射天光，原木/陶瓷道具，暖柔和色调，微距展现肌理，虚化少量果蔬衬托自然新鲜。
5. 首饰配饰：45°微距侧光，金属细腻反光，丝绸绒布基底，低饱和暖调，少量绿植弱化金属冷硬。
6. 家居用品：大面积柔和漫光，棉麻实木道具，低饱和大地色系，平视舒适视角，突出居家温润氛围。
所有题材：用户需求优先级最高，仅补充布光、材质、搭配专业细节，不改动产品与核心卖点。
""",
                    "en": """
【Universal E-commerce Category Rules】
1. General baseline: Weight order: product selling points > 3D composition > scene > color ratio > props > lighting; user view priority, fixed color ratio.
2. Beauty: transparent glass soft light, matte cream, low saturation ins tone, marble/cotton props, side light show texture.
3. 3C digital: matte low-reflection metal, neutral cool tone, solid fabric background, even soft light, neat symmetry.
4. Food: warm diffuse skylight, wood/ceramic props, warm tone, macro texture, blurry fruit foil.
5. Jewelry: 45° macro side light, fine metal reflection, silk base, low warm tone, tiny green plants.
6. Home goods: full diffuse light, cotton wood props, earth tone, flat view, warm living vibe.
All themes: user demand highest priority, only add light/texture details without altering product selling points.
"""
                },
                "negative_base": {
                    "zh": "产品形体扭曲变形，材质塑料虚假，光影杂乱过曝死黑，色彩脏溢高饱和撞色，道具堆砌喧宾夺主，构图失衡焦点偏移，透视比例错误，低分辨率、细节糊成一团，多余杂物摆件，水印logo字幕，光影生硬断层，营销贴纸突兀，过度锐化边缘锯齿，反光失真不自然，背景杂乱抢夺产品视觉焦点，8K，4K，分辨率，DPI，色彩模式，快门，ISO，白平衡，帧率，码率，采样率，编码器，HDR，杜比，字幕，水印，logo，完美对称，零瑕疵，塑料感，崩坏，扭曲",
                    "en": "Distorted product, fake plastic texture, overexposed dead shadow, muddy oversaturated color, overwhelming props, unbalanced composition, wrong perspective, low resolution, mushy detail, clutter, watermark, stiff light, abrupt marketing sticker, over-sharp jagged edges, unnatural reflection, distracting background, 8K, 4K, resolution, DPI, color mode, shutter, ISO, white balance, frame rate, bit rate, sampling rate, encoder, HDR, dolby, subtitle, watermark, logo, perfect symmetry, flawless, plastic texture, collapse, distort"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】2-3段连贯文字：首段场景平台整体布光氛围；第二段产品形体、核心卖点与材质肌理；第三段道具搭配、色彩分层与购买氛围感；总字数300-600，全程规避焦距光圈等数值参数，电商转化叙事画面感，无额外解释。",
                "en": "[Natural Paragraph Mode] 2-3 coherent paragraphs: scene platform lighting; product shape selling texture; prop color shopping vibe; 300-600 words, no focal/aperture numeric params, e-commerce narration."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明商品品类与平台风格定位（白底商品图/场景氛围图/模特穿搭图等），概括画面呈现的瞬间与整体色调氛围，不虚构画面不存在的情节。

**画面主体**
商品本体的形态轮廓、结构特征与核心卖点，突出最鲜明的视觉识别元素。

**主体材质与附属物**
商品表面材质肌理（光滑/磨砂/编织纹理/皮革毛孔/金属光泽）、图案印花、随附配件与包装标签的状态，及其与主体色调的关系。

**道具与动态**
陪衬道具的材质选择、摆放位置与主次关系，商品的使用状态或静态陈列姿态；无动态时写明静态陈列状态。

**环境与背景**
场景平台与背景的色调明暗、简洁程度，前景/中景/背景的空间层次、虚化程度与商品的衬托关系。

**摄影风格与质感**
*   **视角：** 景别（微距材质特写/标准产品/场景融入/模特穿搭/全景展示）、水平视角（正面/四分之三斜侧/正侧面/俯拍）、垂直俯仰（小俯视角/平视/小仰视角）、景深虚实（浅景深主体突出/中景深环境兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 主光类型（柔光箱/侧光/逆光/顶光等）、光源方向与光质软硬（硬光/柔光/散射光），商品高光、中间调与阴影的层次，以及光影在商品与道具上的视觉效果。
*   **质感：** 商品材质、纹理与表面细节的表现；色彩标注主色调、饱和度层级与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，百分比数值不写入提示词）。
*   **氛围：** 从可观察的材质、光影与陈列推导整体氛围与品质感，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness determined by user input and preset rules; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the product category and platform style positioning (white-background product shot / scene atmosphere image / model wear shot), summarize the presented moment and overall color tone; do not invent anything absent from frame.

**Main Subject**
The product's silhouette, structural features and core selling points, highlighting the most recognizable visual elements.

**Subject Material and Attachments**
Product surface texture (smooth / matte / woven texture / leather pores / metallic gloss), printed patterns, bundled accessories and packaging labels, and their relation to the subject's tone.

**Props and State**
Material choice, placement and hierarchy of complementary props, the product's in-use state or static display posture; if no motion, state the static display state.

**Environment and Background**
Tone, brightness and simplicity of the scene platform and background, foreground / mid-ground / background hierarchy, blur level and how they set off the product.

**Photography Style and Texture**
*   **Viewpoint:** shot type (macro material close-up / standard product / scene integration / model wear / full display), horizontal angle (front / three-quarter / profile / overhead), vertical pitch (slight high-angle / eye-level / slight low-angle), depth of field (shallow subject focus / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side); aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type (softbox / side light / backlight / top light), light direction and quality (hard / soft / diffused), product highlight-midtone-shadow hierarchy, and light effects on product and props.
*   **Texture:** product material, texture and surface detail; color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write percentage numbers into the prompt).
*   **Atmosphere:** derive overall atmosphere and quality feel from observable material, lighting and display; avoid empty abstract adjective stacking."""
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
