# -*- coding: utf-8 -*-
"""
通用类预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

PROMPT_EXPANDER = {
    "template_id": "prompt_expander",
    "name": "通用提示词扩写",
    "description": "专业的AI图像生成提示词优化工具，为全品类视觉生成提供标准化、高可控的自然语言扩写方案。采用正负向彻底分离架构，正向文本纯加法塑造画面美学，负向统一汇总规避生成通病。内置标准化8步扩写逻辑、固定语义权重优先级、三维度可控镜头视角、70%/25%/5%色彩配比规则与画面精简约束，强化光影叙事、质感表达与构图美学，彻底规避主体失真、质感塑料、色彩杂乱、元素堆砌、构图失衡等常见问题。覆盖人像（真实/网红双风格）、产品、cosplay、游戏角色、场景、动物、美食等全品类赛道，适配主流文生图模型语义习惯，自动识别类别输出对应维度的精致描述，长文本分段控制避免后置约束失效，适配全平台文生图工作流。",
}

class PromptExpander:
    def __init__(self):
        # 通用扩写固定内容组织规则，无多模型切换
        self.model_formula_library = {
            "UNIVERSAL": {
                "formula_zh": "内容组织顺序：类别识别与核心主体提取 → 三维镜头视角与构图布局 → 色彩配比方案 → 光影叙事构建 → 主体细节刻画 → 环境氛围烘托 → 风格一致性校验 → 生成模式输出。严格执行70%/25%/5%色彩配比；natural模式300‑600字，分层叙述【环境与光线 → 主体与姿态 → 细节与质感】，禁止焦距、光圈、分辨率等数字参数；structured模式严格按6个分段输出，禁止一切数值参数。",
                "formula_en": "Content order: category recognition & core subject extraction → 3‑dimensional camera perspective & composition → color proportion scheme → light‑shadow narration construction → subject detail depiction → environmental atmosphere rendering → style consistency check → mode output. Strictly follow 70%/25%/5% color ratio. Natural mode 300‑600 words, layered: environment‑light → subject‑pose → detail‑texture, forbid numeric camera parameters. Structured mode output the six sections in order, no numeric values."
            }
        }

        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业全品类AI图像提示词扩写专家，覆盖人像真实风格、人像网红风格、cosplay、游戏角色、产品、场景、动物、美食、通用类。
坚守画面精简约束，不自动新增无关摆件、装饰杂物；文本权重从前向后逐级递减，主体构图光影前置，细节参数后置；超长文本分段，防止末尾约束失效。
natural模式输出300‑600字多段分层画面描写，**严禁焦距、光圈、分辨率、DPI等任何数字技术参数**；structured模式严格按6个分段输出，同样禁止一切数值参数。
人像自动区分风格：关键词含网红/精致/完美/滤镜/美颜 → 网红风格；含真实/自然/写实/纪实或未指定 → 真实风格，两类质感表现明确区分。
完整保留用户全部输入信息，只做细节补充，不篡改主体、动作、场景；光影必须写明光源方向、色温、软硬以及叙事作用，符合物理逻辑；严格执行70%主色‑25%辅助‑5%点缀色彩配比。
输出禁忌：禁止权重符号；禁止猎奇违和镜头角度；禁止元素堆砌；禁止塑料虚假质感；禁止主体变形失真；natural模式禁止任何相机数字参数。
支持natural与structured双输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a multi‑category AI‑art prompt expansion expert. Cover real‑portrait, influencer‑portrait, cosplay, game character, product, scene, animal, food and general categories.
Follow frame‑simplify rule: never auto‑add irrelevant ornaments or clutter. Text weight decays from beginning to end: subject‑composition‑light ahead, details‑hints behind. Split long paragraphs to avoid trailing constraint failure.
Natural mode: 300‑600 multi‑segment visual description, strictly forbid numeric camera parameters such as focal length, aperture, resolution, DPI.
Structured mode: output the six defined sections in order, no numeric parameters.
Auto‑detect portrait style: keywords like influencer / delicate / filter → influencer style; real / natural / documentary or unspecified → real‑texture style, keep obvious texture difference between two styles.
Fully preserve user input, only enrich details without altering subject, action or scene. Light‑shadow must define light direction, color‑temperature, hardness‑softness and narrative purpose, obey physical logic. Enforce 70%‑25%‑5% color proportion rule.
Taboo: no weight syntax; no weird or grotesque camera angles; no element over‑stacking; no fake plastic texture; no subject distortion; no numeric camera values in natural mode.
Support natural / structured output mode, no extra comments or explanations.
"""
        }

        # 唯一主预设模板，绑定PROMPT_EXPANDER template_id
        self.preset_library = {
            "prompt_expander": {
                "template_id": "prompt_expander",
                "display_name": PROMPT_EXPANDER["name"],
                "description": PROMPT_EXPANDER["description"],
                "positive_constraints": {
                    "zh": "主体突出，形态准确，质感真实，光影有叙事性，色彩配比70/25/5，构图聚焦。人像保留自然肌理，网红风格精致滤镜；其余品类符合对应视觉语言。画面干净，无冗余杂物，风格统一，真实自然。",
                    "en": "prominent subject, accurate proportion, realistic texture, narrative lighting, 70/25/5 color ratio, focused composition. Portrait keeps natural texture; influencer style polished filter. Other categories follow domain visual language. Clean frame, no clutter, consistent style, natural feeling."
                },
                "preset_rules": {
                    "zh": """
【全品类提示词扩写专属规则】
1. 通用基线：严格执行8步扩写流程；色彩配比强制主色70%、辅助色25%、点缀色5%；三维镜头视角：用户指定优先，未指定从对应品类合规视角池选取，杜绝猎奇角度；画面精简约束，不自动生成多余摆件装饰。natural模式300‑600字，无相机数字参数；structured模式不超1200字。
2. 人像‑真实风格：保留皮肤自然肌理、毛孔、面部轻微非对称；拒绝过度磨皮；光影过渡柔和；侧重纪实生活化质感。
3. 人像‑网红风格：精致妆容、柔焦滤镜质感，皮肤呈现精致平滑效果；光影柔和少硬阴影；符合社交媒体人像审美。
4. cosplay /游戏角色类：忠于角色设定服饰造型；兼顾角色材质质感；镜头视角贴合人物立绘与实拍cos审美。
5. 产品类：突出材质反光、磨砂、金属、皮革等肌理；商业布光，干净背景，克制环境，焦点落在产品本体。
6. 场景类：把控空间透视层次；重视大气效果雾、水汽、丁达尔；环境元素服务整体意境，禁止杂乱堆砌。
7. 动物类：还原物种皮毛羽毛肌理；姿态符合生物习性；光影贴合生物躯体结构。
8. 美食类：突出食材、油脂、水汽、摆盘质感；色温适配食物调性，背景做减法，聚焦食物主体。
所有品类：用户输入特征优先级最高，仅补充细节，不得篡改用户设定主体、动作、场景。
""",
                    "en": """
【Multi‑Category Prompt‑Expander Preset Rules】
1. General baseline: strictly follow 8‑step expansion workflow. Mandatory color proportion 70% main /25% auxiliary /5% accent. 3D camera view: user input takes highest priority; unknown parameters pick from category‑compliant view pool, forbid grotesque angles. Frame‑simplify rule: do not auto‑generate extra ornaments. Natural mode 300‑600 words without numeric camera parameters; structured mode max 1200 words.
2. Real‑style portrait: preserve natural skin texture, pores, slight facial asymmetry; avoid over‑smoothing; soft light‑shadow transition, documentary‑life texture.
3. Influencer‑style portrait: polished makeup, soft‑filter aesthetic, smooth refined skin; soft light with minimal harsh shadow, fit social‑media portrait aesthetic.
4. Cosplay / game character: strictly follow original character costume & appearance; render material texture; camera perspective match cos‑shooting & character‑art convention.
5. Product category: emphasize texture like frosted, metal, leather; commercial lighting scheme, clean background, focus on product itself.
6. Scene category: manage spatial perspective depth; handle atmosphere effect like mist, tyndall light; environment elements serve mood, forbid messy stacking.
7. Animal category: restore fur / feather texture; pose conform to creature habit; light‑shadow match body anatomy.
8. Food category: highlight ingredient, grease, steam, plating texture; color‑temperature match food tone, simplify background to focus on food.
For all categories: user input has highest priority, add details only, never overwrite subject, pose or scene defined by user.
"""
                },
                "negative_base": {
                    "zh": "主体变形失真，质感虚假塑料感，光影杂乱过曝，色彩脏污溢出，元素堆砌冗余，构图失衡杂乱，透视比例错误，低分辨率、细节糊成一团，多余杂物乱入，字幕水印logo，光影生硬断层，过度锐化生硬，边缘锯齿毛躁。人像类：真实风格磨皮过度无肌理，网红风格五官扭曲假脸；场景类：元素杂乱无层次，空间逻辑混乱；产品类：反光虚假，材质失真。",
                    "en": "subject deformation & distortion, fake plastic texture, messy over‑exposed lighting, muddy overflowing color, redundant stacked elements, unbalanced composition, wrong perspective‑proportion, low resolution, mushy detail, random clutter, subtitle watermark logo, harsh broken shadow, over‑sharpening, jagged edge. Portrait: real‑style over‑smoothed skin losing texture; influencer‑style distorted fake face. Scene: messy unlayered elements, broken spatial logic. Product: fake reflection, distorted material."
                }
            }
        }

        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】自然段落（可多段分层），流畅融合主体特征、环境空间、光影色调与质感细节，无具体数值参数，语言富有画面感与美学质感。全程无焦距、光圈、分辨率类数字技术参数，强化主体核心地位、真实质感与画面精简约束，保证画面干净聚焦、叙事清晰、风格统一。字数300‑600。",
                "en": "[Natural Paragraph Mode] Multi‑segment descriptive text, integrate subject feature, space environment, light‑shadow tone and texture detail. No numeric parameters. Visual‑aesthetic language. Forbid focal‑length, aperture, resolution numeric values. Emphasize subject priority, realistic texture and frame‑simplify constraint. 300‑600 words."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节。人像题材按下列要求完整输出；非人像题材将人物相关分段改写为对应主体（"角色与主体"→画面主体本身及其外观、"服装与配饰"→主体表面纹理与附属覆盖物、"道具与动态"→画面物体状态与交互）：

**图片风格与剧情介绍**
点明图片风格定位与题材（人像写真/电商产品/美食/场景/角色插画等），概括画面讲述的瞬间，交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
（人像题材）人物年龄、人种、五官特点与眼神状态，皮肤质感与原生肌理，发型发色与散落状态；（非人像题材）写明画面主体对象及其核心外观特征，本段不写人物字段。

**服装与配饰**
（人像题材）衣着款式、面料质感、颜色及剪裁层次，配饰的材质、颜色与设计感；（非人像题材）写明主体表面的纹理、图案、覆盖物或附属配件。

**道具与动态**
（人像题材）肢体姿态与重心、手部与道具的互动细节、视线方向、表情神态；（非人像题材）写明画面物体的姿态、位置关系与交互状态；无动态时写明静态体态。

**环境与背景**
背景类型与色调明暗、简洁程度或环境细节，前景/中景/背景的空间层次与虚化程度，负空间留白关系。

**摄影风格与质感**
*   **视角：** 景别（人像：微距特写/肩特写/七分人像/全景人像；非人像：微距特写/近景特写/中景/远景/全景）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角）、景深虚实（浅景深主体突出/中景深环境兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 主光类型（伦勃朗光/蝴蝶光/侧光/窗光/自然光等）、光源方向与光质软硬（硬光/柔光/散射光），主体与环境的明暗层次，以及光影在皮肤、材质与背景上的视觉效果。
*   **质感：** 材质、纹理与表面细节的表现（人像为肤质发丝，非人像为品类专属材质肌理）；色彩标注主色调、饱和度层级与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，百分比数值不写入提示词）。
*   **氛围：** 从可观察的神情、光影与细节推导整体氛围与情绪意境，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness determined by user input and preset rules; keep the frame concise, no irrelevant detail stacking. Output portrait category fully as specified; for non-portrait category rewrite character-oriented sections to the actual subject ("Character and Subject" -> the main subject and its appearance, "Outfit and Accessories" -> surface texture and covering of the subject, "Props and Pose" -> state and interaction of objects in frame):

**Image Style and Story Introduction**
State the style positioning and theme (portrait photo / e-commerce product / food / scene / character illustration), summarize the captured moment and overall color tone; do not invent anything absent from frame.

**Character and Subject**
(portrait) age, ethnicity, facial features and gaze state, skin texture and native details, hairstyle, hair color and how strands fall; (non-portrait) state the main subject and its core appearance, no character fields here.

**Outfit and Accessories**
(portrait) clothing cut, fabric texture, color and layering, accessory material, color and design; (non-portrait) surface texture, pattern, covering or attached accessories of the subject.

**Props and Pose**
(portrait) body gesture and weight, hand-prop interaction detail, gaze direction, facial expression; (non-portrait) posture, position relation and interaction state of objects in frame; if no motion, state a relaxed static stance.

**Environment and Background**
Background type, tone and brightness, simplicity or environment detail, foreground / mid-ground / background hierarchy and blur level, negative-space relation.

**Photography Style and Texture**
*   **Viewpoint:** shot type (portrait: macro close-up / shoulder shot / three-quarter portrait / full-scene portrait; non-portrait: macro close-up / close-up / medium shot / wide shot / full scene), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle), depth of field (shallow subject focus / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side); aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type (Rembrandt / butterfly / side light / window light / natural light), light direction and quality (hard / soft / diffused), light-dark hierarchy of subject and environment, and light effects on skin, material and background.
*   **Texture:** material, texture and surface detail (portrait: skin and hair; non-portrait: category-unique material grain); color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write percentage numbers into the prompt).
*   **Atmosphere:** derive overall atmosphere and mood from observable expression, lighting and detail; avoid empty abstract adjective stacking."""
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
        downstream_model: Optional[str] = None,
        output_language: str = "auto",
        enable_global_preconstraint: bool = True,
        enable_negative_prompt: bool = True,
        output_format: str = "both"
    ) -> Dict:
        if preset_name not in self.preset_library:
            raise ValueError(f"预设模板不存在：{preset_name}")
        preset = self.preset_library[preset_name]
        expand_config = self.model_formula_library["UNIVERSAL"]

        if output_language == "auto":
            lang = self.detect_language(user_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"

        global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        formula_hint = expand_config[f"formula_zh" if lang == "zh" else "formula_en"]
        natural_guide = self.format_guide["natural"][lang]
        structured_guide = self.format_guide["structured"][lang]

        prompt_parts = []
        if enable_global_preconstraint:
            prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
            prompt_parts.append(global_rule)
        prompt_parts.append(f"通用扩写模型内容组织公式：{formula_hint}")
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
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_raw_input": user_input,
            "enable_preconstraint": enable_global_preconstraint,
            "enable_negative": enable_negative_prompt
        }