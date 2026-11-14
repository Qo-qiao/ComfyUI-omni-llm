# -*- coding: utf-8 -*-
"""
欧美女性人像预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

WESTERN_FEMALE = {
    "template_id": "western_female",
    "name": "真实欧美女性人像",
    "description": "专业欧美女性人像摄影指导，打造真实自然、富有情绪与故事感的人像描述。语义权重优先级：面部肤质五官＞人物姿态服饰＞光影色彩＞场景环境＞构图景别＞摄影参数。支持三维度视角受控组合，用户指定优先沿用。",
}

class WesternFemale:
    def __init__(self):
        # 下游模型内容组织公式库（12 个图像模型整合为单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：欧美女性的立体五官与自信神情 → 风格与画质（立体骨相、真实肤质） → 明亮侧光或棚拍光塑造轮廓 → 半身或特写、背景虚化。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: Western female with sculpted features and confident expression → style and quality (three-dimensional bone structure, realistic skin) → bright side light or studio light shaping the silhouette → half-body or close-up with blurred background. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }

        # 全局底层规则，纯纪实欧美女性人像，无任何超写实词汇
        self.global_base_rules = {
            "zh": """
你是专业欧美女性人像摄影提示词扩写专家，本模板为【真实感欧美女性人像】，覆盖职场/通勤、运动/健身、婚纱/礼服、街头/潮牌、泳装、通用全题材。
所有风格坚守**真实纪实人像基线**，仅造型、光影、色调、氛围差异化，绝不出现二次元、插画、油画质感。
语义权重优先级：面部肤质五官＞人物姿态服饰＞光影色彩＞场景环境＞构图景别＞摄影参数。
姿态必须完整描述动态抓拍过程，禁止静态摆拍表述；光线使用具象生活化实体光源描述，删除空泛抽象光影修辞；人物为绝对画面主体，环境仅服务人物叙事，不新增无关道具、行人、装饰。
严格执行色彩70%/25%/5%面积配比，统一标注饱和度层级，视觉留有舒适留白；全程强化欧美女性原生面部特征，保留雀斑、毛孔、细纹、肤色不均、淡痣等原生肌肤痕迹，杜绝AI虚假光滑人脸。
完整保留用户输入的风格、服饰、场景、色调、视角所有信息，仅补充摄影、材质、光影、肤质、发丝专业细节，不篡改用户指定内容。
输出禁忌：禁止权重符号、冗余堆砌；禁止完美对称五官、零瑕疵塑料假皮、规整僵硬发丝、空洞假笑、无神凝视；禁止舞台强光、杂乱背景、透视畸变、极端俯仰视角；natural模式禁用全部光学数字参数。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional European and American female portrait prompt expansion expert. This preset is [Realistic European American Female Portrait], covering workplace commuting, sports fitness, wedding dresses, street fashion, swimwear and general themes.
All styles adhere strictly to real documentary portrait baseline, differentiated only by styling, lighting, tone and atmosphere, no illustration, anime or oil painting texture.
Semantic weight priority: facial skin and facial features > character posture and clothing > light and shadow color > scene environment > composition shot > photographic parameters.
Posture must fully describe dynamic capture process, static posing description is forbidden; light is described with concrete real-life physical light sources, empty abstract light and shadow rhetoric is deleted; character is absolute frame subject, environment only serves character narration, no irrelevant props, pedestrians or decorations added.
Strictly implement color area ratio of 70%/25%/5%, mark saturation level uniformly, reserve comfortable blank space visually; always strengthen native facial features of European and American women, retain original skin traces such as freckles, pores, fine lines, uneven skin tone and faint moles, eliminate AI fake smooth human face.
Completely retain all user input information including style, clothing, scene, tone and perspective, only supplement professional details of photography, material, light and shadow, skin and hair without altering user-specified content.
Forbidden: no weight symbols, redundant stacking; perfectly symmetrical facial features, blemish-free plastic fake skin, rigid neat hair, empty fake smile, empty staring gaze; stage strong light, messy background, perspective distortion, extreme pitch angle; natural mode disables all optical digital parameters.
Strictly output two formats without extra comments.
"""
        }

        # 预设库绑定WESTERN_FEMALE模板
        self.preset_library = {
            "western_female": {
                "template_id": WESTERN_FEMALE["template_id"],
                "display_name": WESTERN_FEMALE["name"],
                "description": WESTERN_FEMALE["description"],
                # 中英正向约束，原文完整迁移无修改
                "positive_constraints": {
                    "zh": "真实欧美女性面部，眉眼唇轻微不对称，深邃双眼皮，深眼窝，高立体鼻梁，清晰面部轮廓，蓝绿棕系天然瞳孔，保留毛孔、淡细纹、细微肤色不均、自然雀斑、淡痣，原生肌肤质感，自然毛躁碎发，画面干净简洁，环境仅衬托主体，姿态松弛自然，抓拍真实情绪，自然抓拍瞬间，视角符合纪实人像逻辑",
                    "en": "real European American female face, natural slight asymmetry of brows eyes lips, deep double eyelids, deep eye sockets, tall nose bridge, clear facial contour, natural blue/green/brown pupils, retain pores fine lines uneven tone freckles faint moles, original skin texture, natural frizzy hair, clean frame, environment only sets off subject, relaxed natural posture, captured real emotion, candid captured moment, perspective conforms to documentary portrait logic"
                },
                # 全题材细分专属规则，取自WESTERN_FEMALE_ZH原文
                "preset_rules": {
                    "zh": """
【欧美女性纪实人像全题材专属规则】
1. 通用基线：双重肤质约束叠加，保留毛孔、细纹、肤色不均、自然雀斑、淡痣等真实肌理，杜绝虚假光滑肤质；面部天然轻微不对称，拒绝完美对称五官；光线全部采用具象生活化光源描写，规避舞台式强光；色彩严格执行主色70%、辅助色25%、点缀色5%面积配比，标注饱和度层级，无高饱和撞色堆砌。
2. 职场/通勤：写字楼简约办公场景，西装通勤穿搭；冷白窗光+室内暖混合柔光，中性低饱和色调，松弛放空气质。
3. 运动/健身：工业健身房器械场景，顶均匀柔光，小麦健康肌肤带汗珠，舒展毛孔，专注坚韧神态。
4. 婚纱/礼服：海边、草坪、极简棚，日落/窗纱侧逆光，蕾丝薄纱通透质感，温柔静谧氛围，浅暖低饱和。
5. 街头/潮牌：城市街角橱窗，午后/阴天漫射柔光，随性行走倚靠抓拍，复古低饱和街头色调。
6. 泳装：沙滩泳池日间柔光，舒展松弛体态，清爽阳光低饱和色调。
7. 艺术人体：纯白摄影房大面积侧逆光，柔和S舒展人体，低饱和纯净冷调，仅保留基础布景。
所有题材：用户指定景别、方位、俯仰必须优先沿用；不自动新增无关道具行人，环境仅叙事载体。
""",
                    "en": """
【Documentary European American Female Portrait Theme Rules】
1. General baseline: Double skin texture constraints, retain pores, fine lines, uneven tone, freckles, faint moles, no fake smooth skin; natural slight facial asymmetry; use daily physical light sources, avoid stage harsh light; strictly follow a main 70% / auxiliary 25% / accent 5% color ratio with saturation marked.
2. Workplace/Commuting: Simple office buildings, suit outfits; mixed cold window natural light and indoor warm soft light, neutral low saturation, relaxed vibe.
3. Sports/Fitness: Industrial gym equipment, even top soft light, wheat skin with sweat, open pores, focused expression.
4. Wedding/Dress: Seaside, lawn, minimalist studio, sunset or sheer side backlight, sheer lace tulle, quiet gentle mood, light warm low saturation.
5. Street/Fashion: City street shop windows, sunset or overcast diffuse soft light, casual walking leaning capture, retro low saturation street tone.
6. Swimwear: Beach pool daylight soft light, relaxed stretched body, fresh sunny low saturation tone.
7. Artistic Human Body: Pure white studio large side backlight, soft S body curve, low saturation pure cool tone, only basic space setting.
All themes: User specified shot, horizontal angle, vertical pitch must be prioritized; no auto add irrelevant props or pedestrians, environment only for narration.
"""
                },
                "negative_base": {
                    "zh": "完美对称五官，零瑕疵皮肤，厚重磨皮，塑胶假肤，模板网红脸，光滑无毛孔，规整僵硬发丝，完美面容，空洞假笑，僵硬摆拍，多余肢体动作，无神凝视，多余装饰路人，杂乱背景，舞台强光，过度锐化，高饱和撞色，人工完美肌理，鸟瞰虫眼视角，极端俯仰，透视畸变，肢体比例失调",
                    "en": "perfect symmetrical features, blemish-free skin, heavy smoothing, plastic fake skin, template influencer face, poreless skin, rigid neat hair, flawless face, empty fake smile, stiff posing, redundant limbs, empty stare, extra ornaments passers-by, cluttered background, stage harsh light, over-sharpening, oversaturated color, artificial perfect texture, bird/bug eye view, extreme angle, perspective distortion, disproportionate limbs"
                }
            }
        }

        # 双输出格式指引，完全匹配WESTERN_FEMALE_ZH输出规则
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/街头抓拍/室内生活/户外纪实等）与视角构图方式（非常规视角/平视/俯拍/仰拍/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述实体光源类型与方向（强烈斜阳/柔和窗光/阴天漫射/霓虹灯光/混合光源等），以及光线在人物头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感/粒子散落/动态模糊边缘/明暗渐变过渡/HDR高动态/高饱和强对比等），用定性光影语汇替代光学数值。

第三段·人物姿态与神情：完整描述头部、躯干、四肢的具体姿态（站/坐/蹲/倚靠/行走/手持道具等），视线方向与镜头关系，面部表情神态（恬静/自信/灵动/微笑等），以及风吹发丝飘动等动态细节。

第四段·面部细节与发型妆造：精细刻画面部五官特征（轮廓/眼型/唇色/肤质），皮肤质感（白皙/雀斑/毛孔/光泽），妆容风格（清新淡妆/裸妆等），发型发色与固定方式（马尾/披散/发夹/编发等）。

第五段·服饰配件与环境：描述穿搭细节（衣款/面料/颜色/花纹/配饰如项链耳环等），互动道具（饮品/玩偶/书籍等），以及所处环境场景（室内/户外/水边/街景等），含远景元素（树木/山丘/建筑等），最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/street candid/indoor daily/outdoor documentary, angle/framing); 2) Lighting atmosphere (physical source type, direction, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain, particles, motion blur edges, HDR, high saturation contrast); 3) Full pose & expression (head/torso/limbs position, gaze direction, facial emotion, dynamic details like wind-blown hair); 4) Face details & styling (facial features, skin texture, makeup style, hairstyle with accessories); 5) Outfit accessories & environment (clothing details, props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；皮肤白皙、带自然光泽，可见雀斑与真实毛孔；长发蓬松散落，发丝柔顺。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：平静/专注/柔和/自信），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

**环境与背景**
背景色调与明暗过渡（渐变/纯色/虚化环境）、简洁程度、与主体的对比衬托关系，前景/中景/背景的空间层次、虚化程度与负空间留白（眼神方向留白、头顶与两侧呼吸空间）。

**摄影风格与质感**
*   **视角：** 景别（微距特写/标准特写/肩特写/七分人像/九分人像/全景人像）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角）、景深虚实（浅景深柔焦虚化/中景深环境兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）、视线与镜头的对视关系；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 主光类型（伦勃朗光/蝴蝶光/侧光/环形光/窗光等）、光源方向与光质软硬（硬光/柔光/散射光）、面部高光/中间调/阴影与眼神光层次、环境补光，以及光影在头发、肌肤、衣物上的视觉效果（光影斑驳/边缘发光/柔化光晕/胶片颗粒感等）。
*   **质感：** 肤质、发丝、面料与配饰金属等材质的对比表现；色彩按主色/辅助色/点缀色分层搭配（写作以70/25/5为准则，百分比数值不写入提示词），标注主色调、色温情绪、饱和度层级与肤色还原。
*   **氛围：** 从可观察的神情、动作、光影推导整体氛围与情绪意境，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness matching the reference example; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the style positioning and theme (fashion magazine / high-end beauty ad / documentary / studio shot / street candid), summarize the captured moment (what the person is doing, expression), give the overall color tone and atmosphere baseline; do not invent anything absent from frame.

**Character and Subject**
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Fair skin with natural glow, visible freckles and real pores; long hair loose with silky volume.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: calm / focused / soft / confident), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

**Environment and Background**
Background tone and tonal transition (gradient / solid / bokeh environment), simplicity, contrast with the subject, foreground / mid-ground / background hierarchy, blur level and negative space (gaze-direction margin, headroom and side margin).

**Photography Style and Texture**
*   **Viewpoint:** shot type (macro close-up / standard close-up / shoulder shot / three-quarter portrait / nine-tenth portrait / full-scene portrait), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle), depth of field (shallow soft bokeh / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side), eye-contact relation with camera; aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type (Rembrandt / butterfly / side light / ring light / window light), light direction and quality (hard / soft / diffused), facial highlight-midtone-shadow and catchlight hierarchy, ambient fill, and light effects on hair, skin and clothing (dappled light / rim glow / soft haze / film grain, etc.).
*   **Texture:** contrast across skin, hair, fabric and accessory metal; color layered as main / auxiliary / accent (write against a 70/25/5 ratio, but never write percentage numbers into the prompt), main color tone, color-temperature mood, saturation level and skin-tone reproduction.
*   **Atmosphere:** derive overall atmosphere and mood from observable expression, action and lighting; avoid empty abstract adjective stacking."""
            }
        }

    def detect_language(self, text: str) -> str:
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
    ) -> Dict:
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