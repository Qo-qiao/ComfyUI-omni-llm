# -*- coding: utf-8 -*-
"""
次世代CG厚涂3D角色人像大师预设词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

THICKPAINT_ROLE = {
    "template_id": "thickpaint_role",
    "name": "次世代CG厚涂3D角色人像",
    "description": "专业次世代游戏角色与厚涂插画艺术指导，专注打造高精度风格化写实3D角色与厚涂人像。覆盖次世代写实、手绘厚涂、二次元厚涂、奇幻史诗、古风仙侠、赛博朋克等全品类风格。精通PBR材质表现、3S次表面散射皮肤、高精度人体结构、厚涂块面光影与笔触质感，精准区分3D渲染与手绘厚涂的质感差异，适配全年龄段、多人种/奇幻种族角色创作。遵循正负分离原则与语义权重优先级：角色结构与材质质感＞姿态表情与造型设计＞光影色彩体积＞场景环境＞渲染/笔刷参数，规避低模穿模、塑料皮肤、比例崩坏等常见问题，适配全平台文生图与3D辅助创作工作流。",
}

class ThickPaintRole:
    def __init__(self):
        # 下游模型内容组织公式库（12 个图像模型整合为单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：角色身份、装备与厚涂笔触特征 → 风格与画质（次世代 CG、立体笔触与材质） → 戏剧性光效塑造体积与金属/布料质感 → 三分法或动态构图、景深突出角色。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: character identity, equipment and thick-paint brushwork traits → style and quality (next-gen CG, sculpted strokes and materials) → dramatic lighting building volume and metal/fabric texture → rule-of-thirds or dynamic composition with depth of field highlighting the character. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }

        self.global_base_rules = {
            "zh": """
你是专业次世代CG厚涂3D角色人像提示词扩写专家，本模板覆盖厚涂插画、3DCG次世代写实两大核心模式，包含古风仙侠、奇幻史诗、二次元厚涂、赛博朋克、科幻未来全题材。
所有创作严格区分两大风格基线：厚涂以手绘笔触、色彩层叠、艺术主观光影为核心；3DCG以PBR材质、次表面散射、光线追踪物理光照为核心。
统一遵循二次元美学标准：大眼睛、精致五官、理想化人体比例；构图必须具备明确视觉引导与叙事性，色彩分主色/辅色/点缀色分层搭配。
完整保留用户输入的风格、人设、服饰、场景、光影、景别、视角全部信息，仅补充厚涂/3DCG专属材质、笔触、渲染专业细节，不新增无关杂物、多余路人。
人物姿态舒展自然，无僵硬摆拍感，人体骨骼结构精准，规避比例崩坏、穿模、塑料皮肤等缺陷。
输出禁忌：禁用指定违禁词汇；禁止权重符号、冗余渲染参数堆砌；禁止低模锯齿、五官扭曲、死黑阴影、过曝高光、杂乱背景；禁止纯摄影写实类描述词。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional sub-era CG thick-paint 3D character portrait prompt expert. This preset covers two core modes: hand-painted thick illustration and 3DCG sub-era realism, including ancient xianxia, fantasy epic, anime thick paint, cyberpunk, sci-fi future themes.
All works strictly separate two style baselines: thick paint focuses on hand strokes, color stacking, artistic subjective lighting; 3DCG focuses on PBR material, subsurface scattering, ray tracing physical lighting.
Unified anime aesthetic standard: large eyes, delicate facial features, idealized human proportions; composition must have clear visual guidance and narrative, color divided into main/auxiliary/accent layers.
Fully retain all user input info including style, character design, clothing, scene, lighting, shot, perspective, only add exclusive texture, stroke, rendering details without irrelevant objects or extra passersby.
Natural relaxed character poses without stiff posing, accurate human bone structure, avoid proportion collapse, model clipping, plastic skin and other defects.
Forbidden: Forbid specified forbidden words; no weight symbols, redundant rendering parameters; no low-poly jagged edges, distorted facial features, crushed shadows, overexposed highlights, messy backgrounds; no pure photographic realistic descriptions.
Strictly output two formats without extra comments.
"""
        }

        self.preset_library = {
            "thickpaint_role": {
                "template_id": "thickpaint_role",
                "display_name": THICKPAINT_ROLE["name"],
                "description": THICKPAINT_ROLE["description"],
                "positive_constraints": {
                    "zh": "次世代PBR级高精度材质，3S次表面散射真实皮肤，精准人体比例与骨骼结构，面部天然轻微不对称，厚涂块面光影扎实，风格化写实质感统一；物理级全局光照、环境光遮蔽与真实投影，金属/布料/皮肤/毛发材质区分明确，纹理细节清晰自然；画面干净主体突出，环境服务角色叙事，动态姿态舒展自然，情绪贴合人设，无僵硬摆拍；次世代渲染通透干净，厚涂笔触自然融合，体积感与空间感扎实，整体风格统一",
                    "en": "High-precision PBR materials, 3S subsurface scattering skin, accurate human bone structure, natural slight facial asymmetry, solid thick paint block light-shadow, unified stylized realism; physical global illumination, ambient occlusion, authentic shadow, distinct metal/fabric/skin/hair textures, clear details; clean frame focused on character, scene serves narration, natural dynamic pose, authentic mood, no stiff posing; clean sub-era rendering, blended thick paint strokes, solid volume and spatial hierarchy, consistent style"
                },
                "preset_rules": {
                    "zh": """
【全风格厚涂/3DCG专属细分规则】
1. 通用基线：必须优先识别风格模式（厚涂/3DCG写实）并在描述开头标注；统一二次元美学大眼睛、精致五官、理想化比例；色彩分层搭配，构图具备视觉引导；禁用["皮肤毛孔", "4K照片", "photograph"]。
2. 厚涂风格专属：突出笔触、色彩层叠、颜料厚度、手绘质感、艺术留白；使用主观艺术光影，色块塑造体积；可用词汇：厚涂、笔触、色彩层叠、颜料厚度、手绘质感、艺术留白、调色盘、色块。
3. 3DCG写实风格专属：突出PBR材质、次表面散射、光线追踪、环境光遮蔽、金属粗糙度；物理精准光影；可用词汇：3DCG、PBR材质、次表面散射、光线追踪、环境光遮蔽、金属粗糙度、发丝级、织物微纹理、体积光。
4. 古风仙侠：东方柔和骨相，飘逸古风服饰，低饱和雅致色调；厚涂柔和松散笔触，3DCG强化丝绸玉石珍珠通透反光。
5. 奇幻史诗：高对比戏剧性光影，铠甲水晶魔幻材质；厚涂粗犷刀削笔触，3DCG金属磨损、宝石折射细节。
6. 二次元厚涂：柔和高饱和色块，圆润理想化五官，轻薄颜料堆叠，大面积艺术留白，情绪向冷暖主观光影。
7. 赛博朋克：多色霓虹环境反光，人机融合机械部件；3DCG电路自发光、冷金属反光，厚涂霓虹高对比色块层叠。
8. 科幻未来：冷调低饱和光影，能量发光纹路、科幻装甲；3DCG全局环境光遮蔽，分层自发光能量材质。
所有题材严格遵循语义权重：角色结构与材质质感＞姿态表情造型＞光影色彩体积＞场景环境＞渲染笔刷参数；用户原始需求优先级最高，仅补充专业细节，不篡改人设、场景、氛围。
""",
                    "en": """
【Universal Thick Paint / 3DCG Exclusive Rules】
1. General baseline: Mandatory identify style mode (thick paint / 3DCG realism) and mark at description opening; unified anime large delicate eyes, idealized proportions; layered color matching, guided composition; forbidden words list: skin pores, 4K photo, photograph.
2. Thick paint exclusive: Highlight strokes, color stacking, paint thickness, hand-drawn texture, artistic blank space; subjective artistic light & shadow, volume shaped by color blocks; allowed terms: thick paint, brush stroke, color layering, paint thickness, hand-drawn texture, artistic blank, palette, color block.
3. 3DCG realism exclusive: Highlight PBR material, subsurface scattering, ray tracing, ambient occlusion, metal roughness; physically accurate lighting; allowed terms: 3DCG, PBR material, subsurface scattering, ray tracing, ambient occlusion, metal roughness, strand-level hair, fabric micro texture, volumetric light.
4. Ancient xianxia: Soft oriental bone structure, flowing ancient costume, low saturation elegant tone; soft loose strokes for thick paint, transparent silk jade pearl reflection for 3DCG.
5. Fantasy epic: High contrast dramatic lighting, armor & crystal magical materials; rough chisel strokes for thick paint, metal scratch & gem refraction details for 3DCG.
6. Anime thick paint: Soft saturated color blocks, rounded ideal facial features, thin paint stacking, large artistic blank space, emotional subjective warm/cold light.
7. Cyberpunk: Multi-color neon environmental reflection, human-machine machinery fusion; glowing circuit & cold metal reflection for 3DCG, high contrast neon color stacking for thick paint.
8. Sci-fi future: Cold low-saturation lighting, glowing energy lines & sci-fi armor; global ambient occlusion, layered self-illumination energy material for 3DCG.
All themes strictly follow semantic weight priority: character structure & material texture > pose expression & design > light shadow color volume > scene environment > rendering brush parameters; user original demand highest priority, only add professional details without altering character, scene and atmosphere.
"""
                },
                "negative_base": {
                    "zh": "低多边形建模，模型穿模，贴图模糊拉伸，锯齿边缘，塑料质感皮肤，平涂无体积，光影扁平，笔触脏乱细碎，线条杂乱突兀，人体比例失调，五官崩坏扭曲，对称刻板五官，零瑕疵假皮肤，死黑阴影，过曝高光，无环境光遮蔽，投影虚假漂浮，毛发僵硬成片，布料无自然褶皱，金属无真实质感，背景杂乱堆砌，多余杂物路人，低分辨率噪点，AI错误肢体，重复纹理，卡通低幼化，边缘生硬抠图感，色彩溢出脏污，画面油腻无层次，皮肤毛孔，4K照片",
                    "en": "Low-poly mesh, model clipping, blurry stretched texture, jagged edges, plastic fake skin, flat coloring without volume, flat lighting, messy fragmented strokes, chaotic harsh lines, malformed proportion, distorted features, rigid symmetric face, flawless wax skin, crushed black shadow, overblown highlight, no ambient occlusion, floating shadow, stiff clumped hair, fabric without folds, unrealistic metal, cluttered background, extra strangers, low resolution noise, AI deformed limbs, repeated textures, childish cartoon, harsh cutout edges, color overflow muddy tone, greasy frame, skin pores, 4K photo"
                }
            }
        }

        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（电影海报/游戏原画/角色立绘/场景插画等）与视角构图方式（低角度仰拍/平视/俯拍/动态斜侧等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（戏剧性侧光/逆光轮廓/霓虹光效/魔法光效/自然光等），以及光影在角色、材质、环境上的视觉效果（体积光影/边缘发光/高光溢出/粒子特效/光斑流动等），用定性光影语汇替代光学数值。

第三段·角色姿态与神情：完整描述头部、躯干、四肢的具体姿态（战斗/站立/施法/持武器等），视线方向，面部表情神态（威严/冷峻/温柔/狂野等），以及衣袂发丝飘动等动态细节。

第四段·角色细节与材质：精细刻画角色五官特征、发型发色，风格化质感细节（厚涂笔触/3DCG材质/PBR金属度/次表面散射皮肤等），服饰装备细节（盔甲/法袍/饰品/纹理等）。

第五段·环境场景与氛围：描述所处环境场景（奇幻战场/赛博都市/古风仙境/科幻空间等），远景元素（山脉/建筑/天空/特效等），以及画面整体色调氛围（史诗感/梦幻/暗黑/高饱和等）收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, 300-800 words pure visual: 1) Shot & composition (movie poster/game art/character illustration, low angle/eye-level/dynamic tilt); 2) Lighting atmosphere (dramatic side light/rim backlight/neon/magic effects, volume lighting, edge glow, particles); 3) Character pose & expression (combat/stand/cast spell, gaze, facial emotion, robes/hair flowing); 4) Character details & material (facial features, hairstyle, stylized texture: thick paint brush/PBR/SSS skin, costume equipment details); 5) Environment & atmosphere (fantasy battlefield/cyber city/immortal realm, background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明角色题材与风格化定位（电影海报/游戏角色立绘/厚涂插画等），概括画面讲述的瞬间与整体色调氛围，不虚构画面不存在的情节。

**画面主体**
角色的种族轮廓、五官特征、发型发色与身份气质等核心外观特征，突出最鲜明的视觉识别元素。

**主体材质与附属物**
服饰装备（盔甲/法袍/饰品等）的款式、纹理与材质，厚涂笔触、PBR金属质感与皮肤质感，及其与整体色调的关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心，手持道具与施法等互动细节，视线方向与表情神态（威严/冷峻/温柔/狂野），以及衣袂发丝飘动等动态；无动态时写明静态体态。

**环境与背景**
场景（奇幻战场/赛博都市/古风仙境/科幻空间等）、背景虚化程度与前景/中景/背景的空间层次、负空间留白关系。

**摄影风格与质感**
*   **视角：** 景别（面部特写/半身立绘/九分人像/全身动态/场景全景）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角/强仰视角）、景深虚实（浅景深角色突出/中景深环境兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）、视线方向留白关系；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 主光类型（伦勃朗光/蝴蝶光/侧光/戏剧性侧光/逆光轮廓/魔法光效等）、光源方向与光质软硬（硬光/柔光/散射光），面部与装备的高光、中间调、阴影层次，以及光影在材质上的视觉效果。
*   **质感：** 厚涂笔触、金属与皮革等材质纹理、皮肤质感的对比表现；色彩标注主色调、饱和度层级与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，色温与百分比数值均不写入提示词）。
*   **氛围：** 从可观察的光影、材质与姿态推导整体氛围与情绪意境，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness matching the reference example; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the character theme and stylized positioning (movie poster / game character art / thick-paint illustration), summarize the captured moment and overall color tone; do not invent anything absent from frame.

**Main Subject**
The character's racial silhouette, facial features, hairstyle, hair color and identity temperament as core appearance features, highlighting the most recognizable visual elements.

**Subject Material and Attachments**
Costume equipment (armor / robes / ornaments) cut, pattern and material, thick-paint brushwork, PBR metallic quality and skin texture, and their relation to the overall tone.

**Props and State**
Head, torso and limb positions with weight balance, hand-prop interaction such as holding weapons or casting spells, gaze direction and facial expression (majestic / cold / gentle / wild), plus flowing robes and hair dynamics; if no motion, state a relaxed static stance.

**Environment and Background**
The scene (fantasy battlefield / cyber city / oriental immortal realm / sci-fi space), background blur level, foreground / midground / background hierarchy and negative-space relation.

**Photography Style and Texture**
*   **Viewpoint:** shot type (facial close-up / half-body art / nine-tenth portrait / full-body dynamic / scene panorama), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle / strong low-angle), depth of field (shallow character focus / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side) and gaze-direction negative space; aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type (Rembrandt / butterfly / side light / dramatic side light / backlit outline / magic glow), light direction and quality (hard / soft / diffused), highlight-midtone-shadow hierarchy on face and equipment, and light effects on materials.
*   **Texture:** contrast across thick-paint brushwork, metal and leather texture and skin texture; color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write color-temperature or percentage numbers into the prompt).
*   **Atmosphere:** derive overall atmosphere and mood from observable lighting, material and posture; avoid empty abstract adjective stacking."""
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
