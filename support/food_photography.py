# -*- coding: utf-8 -*-
"""
美食摄影预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

FOOD_PHOTOGRAPHY = {
    "template_id": "food_photography",
    "name": "美食摄影",
    "description": "专业美食摄影指导，为全品类美食打造高可控食欲感视觉描述。语义权重优先级：美食类型焦点→三维视角构图→场景氛围→色彩配比→食物细节质感→光影食欲设计。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，以「不完美的真实感」为核心，通过轻微焦痕、酱汁流淌、自然热气强化手工可信度。覆盖甜点、中式、西餐、饮品、小吃等品类赛道。",
}

class FoodPhotography:
    def __init__(self):
        # 下游生图模型内容组织公式库
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体美食场景食欲光影 → 食物主体形态质感 → 餐具简约留白。侧重美食食欲叙事，弱化细碎关键词堆砌，画面治愈高级。",
                "formula_en": "Content order: overall food scene appetite lighting → food shape texture → tableware blank. Focus on appetite narration."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：美食主体（食材、摆盘、器皿）→ 诱人写实与油润质感 → 顶光或侧光、食欲氛围 → 俯拍或 45° 特写",
                "formula_en": "Content order: food subject (ingredients, plating, tableware) → appetizing realism with glossy texture → top light or side light, appetizing atmosphere → top-down or 45-degree close-up (no independent negative channel; supports multi-reference image editing)"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：美食主体（食材、摆盘、器皿）→ 诱人写实与油润质感 → 顶光或侧光、温暖食欲氛围 → 俯拍或 45° 特写、背景虚化（需渲染文字直接写入，支持中英双语）。",
                "formula_en": "Content order: food subject (ingredients, plating, tableware) → appetizing realism with glossy texture → top light or side light, warm appetizing atmosphere → top-down or 45-degree close-up, blurred background (write any rendered text directly, supports Chinese and English). Negative prompt provided by preset template."
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：菜品主体、食材质感与摆盘 → 风格与画质（诱人色泽、高清细节） → 顶部柔光或侧逆光突出油脂与蒸汽 → 俯拍或 45° 近景、简洁背景 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: dish subject, ingredient texture and plating → style & quality (appetizing color, HD details) → top soft light or side backlight highlighting oil sheen and steam → top-down or 45-degree close-up, simple background → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：全局食欲光影基调 → 食物诱人特质 → 食材肌理细节 → 碗盘面料 → 极简布景（密集关键词，中英术语并列），食材质感高度细致，色泽诱人细节丰富",
                "formula_en": "Content order: global appetite lighting tone → appetizing food qualities → ingredient texture details → tableware and fabric → minimalist set (dense keywords, Chinese-English terms in parallel), highly detailed food texture, rich color and appetizing details"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整张美食基础基调 → 舒展食材形态 → 统一食物真实质感 → 简约餐具留白",
                "formula_en": "Content order: whole food basic tone → relaxed ingredient form → unified real food texture → simple tableware blank"
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：食物透视肌理、食材摆盘 → 远近餐具层次 → 光影冷暖过渡、暖调布光 → 配料细节 → 轻量化留白（密集关键词，中英术语并列）",
                "formula_en": "Content order: food perspective texture, ingredient plating → near-far tableware layers → light warm-cold transition, warm lighting → ingredient details → lightweight blank (dense keywords, Chinese-English terms in parallel)"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：美食整体食欲气质 → 食材肌理细节 → 专业分层布光 → 品类专属餐具 → 极简留白。整体色调统一，食欲细节细腻，美食沉浸感强烈。",
                "formula_en": "Content order: overall food appetite temperament → ingredient texture → professional layered lighting → category exclusive tableware → minimalist blank. Unified tone, strong food immersion."
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：菜肴与餐具主体 → 食物精修风格、色泽与蒸汽质感 → 顶光与侧补光、诱人氛围 → 45度俯拍、构图聚焦 → 强调食欲真实无干瘪、避免色泽失真。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词。",
                "formula_en": "Content order: dish and tableware subject → food retouch style, color and steam texture → top light and side fill, appetizing atmosphere → 45-degree top-down shot, focused composition → emphasize real appetite without dryness, avoid color distortion. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt."
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
                "formula_zh": "内容组织顺序：菜肴与餐具主体 → 场景与构图（45度俯拍聚焦）→ 光影与氛围（顶光侧补光诱人）→ 画种/摄影风格（食物精修）→ 需渲染文字用引号包裹。",
                "formula_en": "Content order: dish and tableware subject → scene & composition (45-degree top-down focus) → light & atmosphere (top light, side fill, appetizing) → art/photography style (food retouch) → wrap rendered text in quotes."
            }
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业美食摄影提示词扩写专家，覆盖甜点烘焙、中式料理、西餐、饮品咖啡、小吃全美食赛道。
所有创作坚守食欲叙事基线，仅食材品类、拍摄视角、餐具、光影差异化，禁止多类食材杂乱混搭。
画面遵循70%食物视觉主体，餐具环境占30；色彩固定70主/25辅/5点缀配比，低饱和暖调提升食欲。
光影贴合美食拍摄逻辑，甜点柔光绵密、中餐暖油光、西餐逆光焦香、饮品通透反光；统一保留手工不完美肌理（焦痕/酱汁/气泡）。
完整保留用户输入美食品类、画幅、视角、核心卖点全部信息，仅补充食材肌理、布光、餐具专业细节，不自动新增多余配菜摆件。
画面执行精简约束，仅留存核心食物与必要衬托餐具，杜绝过度完美3D假质感。
输出禁忌：禁止权重符号、快门/ISO/白平衡/分辨率等数值参数堆砌；禁止食材变形、塑料虚假肌理；禁止水印logo、完美零瑕疵描述。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional food photography prompt expert, covering dessert, chinese food, western dish, coffee, snacks.
All creations follow appetite narration baseline, differentiated by food type, view, tableware, lighting, no messy mixed ingredients.
70% food main subject, 30% tableware & background; fixed 70 main /25 secondary /5 accent warm color ratio.
Light matches food shooting rule: soft light for cream, warm oil light for chinese, backlight for roast, transparent for drinks; retain handmade imperfect texture (burn marks/sauce/bubbles).
Fully retain user food, frame, view, selling points, only add texture/light/tableware details without extra side dishes.
Strict simplification rule, only core food & necessary tableware, no fake perfect CG texture.
Forbidden: weight symbols, shutter/ISO/white balance/resolution numeric params; distorted food, fake plastic texture; watermark/logo, flawless description.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有FOOD_PHOTOGRAPHY模板id
        self.preset_library = {
            "food_photography": {
                "template_id": "food_photography",
                "display_name": FOOD_PHOTOGRAPHY["name"],
                "description": FOOD_PHOTOGRAPHY["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "美食为绝对视觉主体，食材形态比例规整无畸变；70/25/5色彩配比和谐暖调，光影过渡柔和自然；天然手工不完整细节（焦痕、流淌酱汁、细小气泡、不均切面）完整保留；酥脆/绵密/油亮/通透肌理真实贴合食材物理属性；餐具克制虚化不抢焦点，场景氛围匹配美食调性，热气油脂光泽真实自然，画面干净克制，充满烟火治愈食欲感",
                    "en": "Food absolute main subject, regular ingredient shape without distortion; harmonious warm color ratio, soft light transition; reserved handmade imperfection (burn marks, flowing sauce, tiny bubbles, uneven cut); crispy/creamy/oily/transparent texture match physical logic; blurred tableware no distraction, natural steam & grease glow, clean warm appetite frame"
                },
                # 全美食细分专属规则
                "preset_rules": {
                    "zh": """
【全美食专属细分规则】
1. 通用基线：语义权重：美食类型焦点→三维视角构图→场景氛围→色彩配比→食材质感→食欲光影；用户画幅视角优先，严格70/25/5色彩配比，精简约束；禁用["8K", "4K", "分辨率", "DPI", "色彩模式", "快门", "ISO", "白平衡", "帧率", "码率", "采样率", "编码器", "HDR", "杜比", "字幕", "水印", "logo", "完美对称", "零瑕疵", "塑料感", "崩坏", "扭曲"]。
2. 甜点烘焙：柔和单侧窗光，绵密奶油肌理，轻微烤焦边缘，果粒自然不均，浅木/粗陶餐具，低饱和暖柔色调。
3. 中式料理：前侧暖柔光，油亮酱汁流淌，表皮自然龟裂纹，陶瓷深碗，烟火暖棕主色，少量香料点缀。
4. 西餐炭烤：后侧轮廓逆光，网格炭烤焦痕，半透明油脂，深色石板，中饱和肉色调，迷迭香少量搭配。
5. 饮品咖啡：平视柔光，细腻奶泡不规则拉花，杯沿奶渍，浅木桌面，低饱和棕米配色。
6. 小吃零食：漫射天光，酥脆碎边，调味粉不均撒放，简约纸盘，暖黄接地气色调。
所有题材：用户需求优先级最高，仅补充食材、布光、餐具细节，不篡改核心美食与食欲卖点。
""",
                    "en": """
【Universal Food Exclusive Rules】
1. General baseline: Weight order: food focus > 3D composition > scene > color ratio > texture > appetite light; user frame priority, fixed color ratio; forbidden list: 8K,4K,resolution,DPI,color mode,shutter,ISO,white balance,frame rate,bit rate,sampling rate,encoder,HDR,dolby,subtitle,watermark,logo,perfect symmetry,flawless,plastic texture,collapse,distort.
2. Dessert: soft side window light, creamy texture, slight baked edge, uneven fruit, wood/ceramic tableware, warm low saturation.
3. Chinese food: front warm soft light, flowing glossy sauce, natural crack skin, dark ceramic bowl, warm brown tone, minor spices.
4. Western roast: back rim light, grill burn marks, translucent grease, dark stone plate, medium meat tone, rosemary foil.
5. Coffee drink: flat soft light, uneven latte art, milk stain on cup, light wood table, brown beige palette.
6. Snack: diffuse skylight, crispy broken edge, uneven seasoning, simple paper tray, warm earth tone.
All themes: user demand highest priority, only add food/light/tableware details without altering core selling points.
"""
                },
                "negative_base": {
                    "zh": "食材扭曲变形，肌理塑料CG假质感，光影过曝死黑，色彩脏灰杂乱，餐具堆砌抢主体，构图失衡焦点偏移，透视错误，低分辨率模糊，多余配菜杂物，装饰冗余，水印logo，热气僵硬不自然，油脂虚假反光，过度锐化锯齿，画面完美无手工痕迹，3D渲染虚假光滑感",
                    "en": "Distorted food, fake CG plastic texture, overexposed shadow, muddy color, overwhelming tableware, unbalanced composition, wrong perspective, blurry low-res, extra side dishes, redundant decor, watermark/logo, stiff steam, fake grease reflection, over-sharp jagged edges, flawless CG smooth surface"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】2-3段连贯文字：首段画幅构图与整体治愈食欲氛围；第二段食材形态、手工不完美肌理与餐具搭配；第三段光影冷暖、70/25/5色彩与食欲感受；总字数300-600，全程规避焦距光圈等数值参数，美食食欲叙事画面感，无额外解释。",
                "en": "[Natural Paragraph Mode] 2-3 coherent paragraphs: frame & warm appetite atmosphere; food handmade texture & tableware; light color & tasting feeling; 300-600 words, no focal/aperture numeric params, food narration only."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
1.美食品类与食材形态
2.风格与拍摄定位
3.食物主体与核心食欲焦点
4.三维度镜头视角与构图
   - 画面比例：竖版美食图（4:5/3:4）/ 横版场景图（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：微距食材 / 标准美食 / 餐桌全景 / 烹饪过程，对应美食叙事重心
   - 水平视角维度：俯拍（90°）/ 45°角 / 平视，标注美食展现效果
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角，对应美食视觉张力
   - 景深氛围：浅景深主体突出 / 中景深环境兼顾 / 深景深全景清晰，标注虚实层次
5.食材细节与手工肌理
   - 酥脆/绵密/流心/拉丝等质感表现
   - 手工不完美肌理：不规则边缘/自然裂纹/手工痕迹
   - 色彩分层：主色70% / 辅助色25% / 点缀色5%
6.餐具搭配与摆盘
   - 碗盘材质：陶瓷/玻璃/木质/金属/石板
   - 餐具摆放：位置关系/高低层次/疏密节奏
   - 色彩呼应：餐具与食物色调协调
7.光影氛围与专业布光
   - 主光类型：侧光/逆光/顶光/柔光箱
   - 光源方向：正侧光45°/90°侧光/逆光轮廓/窗光
   - 光质软硬：硬光（清晰边缘阴影）/柔光（渐变过渡阴影）
   - 光影特效：暖调食欲光/柔化光晕/光影斑驳/动态光斑/边缘发光/胶片颗粒感/明暗渐变过渡
8.场景氛围与食欲感受
   - 用餐场景：餐厅/厨房/户外/咖啡厅
   - 整体色调：暖调/冷调，与美食风格统一
   - 食欲感受：治愈/温馨/精致/日常
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许定性描述镜头空间效果，禁用数值参数：
- 微距食材特写：微距镜头，极浅景深，突出食材纹理与手工肌理
- 45°角标准美食：85mm中长焦，f/2.8-f/4光圈，经典美食角度
- 俯拍餐桌全景：50mm标准镜头，f/4-f/5.6光圈，桌面全景展示
- 平视美食场景：50mm标准镜头，平视角度，环境氛围融入""",
                "en": """[Structured Mode] Output strictly in this order:
1. Food category and ingredient form
2. Style and shooting positioning
3. Food subject and core appetite focus
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical food (4:5/3:4) / horizontal scene (16:9/3:2) / square (1:1)
   - Distance (shot type): macro ingredient / standard food / table panorama / cooking process, mark narrative focus
   - Horizontal view: overhead (90°) / 45° angle / eye-level, describe food display effect
   - Vertical pitch: slight high-angle / eye-level / slight low-angle, describe food visual tension
   - Depth of field: shallow DOF subject focus / medium DOF environment balanced / deep DOF full sharpness
5. Ingredient details and handmade texture
   - Crispy/creamy/flowing/pulling texture expression
   - Handmade imperfections: irregular edges/natural cracks/handcraft traces
   - Color layering: 70% main / 25% auxiliary / 5% accent
6. Tableware matching and plating
   - Tableware material: ceramic/glass/wood/metal/slate
   - Tableware placement: position relationship/height layers/density rhythm
   - Color echo: tableware and food tone coordination
7. Lighting atmosphere and professional lighting
   - Key light type: side light/backlight/top light/softbox
   - Light direction: 45° side / 90° side / backlit outline / window light
   - Light quality: hard (clear edge shadow) / soft (gradual transition)
   - Lighting effects: warm appetite light / highlight bloom / soft haze / dappled light / dynamic light spots / rim glow / film grain / particles / gradient transition
8. Scene atmosphere and appetite feeling
   - Dining scene: restaurant/kitchen/outdoor/café
   - Overall tone: warm/cool, unified with food style
   - Appetite feeling: healing/warm/exquisite/daily
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative description of lens spatial effects, no numeric parameters:
- Macro ingredient close-up: macro lens, extreme shallow DOF, highlighting ingredient texture and handmade details
- 45° standard food shot: 85mm medium telephoto, f/2.8-f/4 aperture, classic food angle
- Overhead table panorama: 50mm standard lens, f/4-f/5.6 aperture, full table display
- Eye-level food scene: 50mm standard lens, eye-level angle, environment atmosphere integration"""
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
