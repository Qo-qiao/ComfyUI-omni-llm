# -*- coding: utf-8 -*-
"""
通用图像反推提示词模板

Author: 亲卿于情 (@Qo‑qiao)
GitHub: https://github.com/Qo‑qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

IMAGE_REVERSE_DESCRIBE = {
    "template_id": "image_reverse_describe",
    "name": "图像反推描述",
    "description": "专业的图像分析专家，优先基于输入图像完成解析；同时可接收用户可选关键词，用来校准风格、题材、氛围信息，修正反推结果，生成更精准的自然语言描述，适用于Flux、Z‑Image、Qwen‑Image、Krea2等主流自然语言图像生成模型。专业知识涵盖摄影术语、艺术风格、灯光设计、色彩理论、画面构图分析、主体位置识别和视觉视角推断，输出精准描述，用于复现与输入图视觉要素一致的新图像。",
}

class ImageReverseDescribe:
    def __init__(self):
        # 全局底层图像反推通用规则
        self.global_base_rules = {
            "zh": """
你是专业图像反推描述扩写专家，本模板为【通用图像反推提示词模板】。
主解析来源为输入图像#IMAGE_SOURCE#；支持接收**可选用户关键词**用于辅助校准风格、题材、氛围；图像像素信息优先级最高，用户关键词仅做补充校准，关键词与图片视觉冲突时，以图片画面为准。
支持风景类、摄影类、人像类、插画类、IP类、cosplay类、游戏角色类、产品类、建筑室内类、动物类、美食类、UI界面类、时尚穿搭类、通用类。
坚守图像反推基础约束：只提取画面内客观可视觉化的实体细节，拒绝抽象内心情绪、虚构故事情节；必须输出构图方式、主体位置、视角类型、景深效果；人物/物体互动遵循现实物理逻辑。
natural模式300‑600字，可多段落分层；structured模式完整输出结构化字段，字段严格匹配模板定义。
区分不同题材处理逻辑：人像类完整输出人物特征字段；静物/风景类省略人像专属字段；
完整保留图片全部视觉元素，只做结构化整理，不新增画面不存在物体；光影写明光源、软硬、色温；色彩明确主色调、饱和度、冷暖倾向。
输出禁忌：禁止虚构画面不存在的物体；禁止主观故事脑补；禁止写入光圈、焦距数值；禁止权重符号。
支持natural与structured双输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are professional image reverse‑description expert. This preset is 【Universal Image Reverse Prompt Template】.
Primary analysis source is input image #IMAGE_SOURCE#. Optional user keywords are accepted only for calibrating style, theme and atmosphere. Image pixel information has highest priority. If keywords conflict with visual content of image, image shall prevail.
Support landscape, photography, portrait, illustration, IP character, cosplay, game character, product, architecture‑interior, animal, food, UI, fashion‑outfit, general category.
Baseline rule: only extract objective visual elements inside image, reject abstract inner emotion and fictional plot. Must output composition type, subject position, perspective type, depth‑of‑field effect. Interaction between person and object obey real‑world physics.
Natural mode: 300‑600 words, multi‑paragraph allowed. Structured mode output full structured fields strictly follow template definition.
Topic logic: output full character fields for portrait; omit portrait‑only fields for still‑life / landscape.
Preserve all visual elements from source image, only reorganize structure, DO NOT add non‑existing objects. Describe light source, hardness‑softness, color‑temperature; define main color, saturation, cold‑warm tendency.
Taboo: do NOT invent objects not exist in picture; no fictional story; no aperture / focal‑length numeric value; no weight syntax.
Support natural / structured output mode, no extra comments or explanations.
"""
        }

        self.preset_library = {
            "image_reverse_describe": {
                "template_id": "image_reverse_describe",
                "display_name": "图像反推描述",
                "description": "作为专业的图像分析专家，优先基于输入图像完成解析；同时可接收用户可选关键词，用来校准风格、题材、氛围信息，修正反推结果，生成更精准的自然语言描述，适用于Flux、Z‑Image、Qwen‑Image、Krea等主流自然语言图像生成模型。专业知识涵盖摄影术语、艺术风格、灯光设计、色彩理论、画面构图分析、主体位置识别和视觉视角推断，输出精准描述，用于复现与输入图视觉要素一致的新图像。",
                "positive_constraints": {
                    "zh": "完全基于输入图片视觉信息客观还原全部可见视觉元素，构图、主体位置、视角、景深、光线、色彩、材质质感完整还原；可选用户关键词仅用于辅助校准风格、氛围、题材；关键词与图片冲突时以图片为准；人物与物体互动符合物理现实；聚焦可被视觉呈现的细节，只描述画面存在事物；区分题材输出对应字段；语言专业流畅。",
                    "en": "Completely based on source‑image visual information, objectively restore all visible visual elements, fully restore composition, subject position, perspective, depth‑of‑field, lighting, color and material texture. Optional user‑provided keywords only assist to calibrate style, atmosphere and theme. In case of conflict between keywords and image content, image takes precedence. Interaction between character and object obey physics reality. Focus on visually observable details, only describe objects existing in image. Output corresponding fields according to category. Professional and fluent language."
                },
                "preset_rules": {
                    "zh": """
【图像反推专属规则】
1. 通用基线：主解析来源#IMAGE_SOURCE#，可选用户关键词#USER_KEYWORDS#；执行7步图像解析流程；必须输出构图、主体位置、视角类型、景深效果；natural模式300‑600字；structured模式最大800字。
2. 优先级铁则：图像像素视觉信息 > 用户可选关键词。关键词仅做风格、题材、氛围的辅助校准；若关键词描述与图片画面冲突，直接舍弃冲突关键词，严格遵从图片画面，绝不根据关键词篡改图片客观视觉内容。无关键词则完全依靠图像解析。
3. 景别判定规则：【人像题材】微距特写/标准特写/肩特写/七分人像/九分人像/全景人像；【非人物题材】微距特写/近景特写/中景/远景/全景。题材分支规则：人像类输出完整人物特征；风景、产品、美食、动物等非人像题材省略人物特征字段。
4. 内容约束：仅以图像像素视觉信息为第一依据，禁止脑补故事、抽象心理情绪；不得生成原图不存在物体、道具；关键词不能用来新增画面不存在实体对象，仅用于校准风格氛围。
5. 光影色彩：明确光源方向、软硬；标注色彩调性、饱和度、冷暖倾向；区分前景、中景、背景空间层次。
6. 质感细节：重点还原材质、纹理、表面细节特征。
解析来源：待解析图像 #IMAGE_SOURCE#；可选辅助关键词：#USER_KEYWORDS#（为空=无用户关键词）
""",
                    "en": """
【Image Reverse Preset Rules】
1. General baseline: primary source #IMAGE_SOURCE#, optional assist keywords #USER_KEYWORDS#; follow 7‑step image‑analysis workflow. Must output composition, subject position, perspective type, depth‑of‑field effect. Natural mode 300‑600 words; structured mode max 800 characters.
2. Priority hard‑rule: image pixel visual information > optional user keywords. Keywords only assist calibrating style, theme and atmosphere. If keywords conflict with image visual content, discard conflicting keywords and strictly follow image content, never alter objective visual content according to keywords. If no keywords provided, rely entirely on image analysis.
3. Shot‑range judgment rule: 【Portrait Category】macro close‑up / standard close‑up / shoulder shot / three‑quarter / nine‑tenth / full‑scene portrait; 【Non‑portrait Category】macro close‑up / close‑up / medium shot / wide shot / full scene. Category branch rule: output full character fields for portrait category; omit character fields for landscape, product, food, animal and other non‑portrait topics.
4. Content constraint: image pixel is primary evidence, forbid fictional story and abstract mental emotion. Must NOT invent objects or props not shown on source image. Keywords shall NOT add physical entities absent in image, only for style‑atmosphere calibration.
5. Light & Color: define light‑source direction, hardness‑softness; mark color tone, saturation, cold‑warm tendency; distinguish foreground‑mid‑background spatial hierarchy.
6. Texture detail: faithfully restore material, texture and surface feature.
Analysis source: source image #IMAGE_SOURCE#; optional assist keywords: #USER_KEYWORDS#(empty = no user keywords)
"""
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】可多个段落，融合全部视觉元素，包含构图方式、主体位置、视角类型、景深效果、光线色彩、空间层次和细节质感；存在合规用户关键词时将风格/氛围校准信息自然融入描述，冲突关键词直接舍弃。建议按主体、构图与空间、光线与色彩、氛围与细节分层分段，每段聚焦一个维度。语言流畅专业，字数300‑600。",
                "en": "[Natural Paragraph Mode] Multiple paragraphs allowed. Integrate all visual elements: composition, subject position, perspective type, depth‑of‑field, light‑color, spatial hierarchy and texture detail. When valid user keywords exist, merge style‑atmosphere calibration naturally; discard conflicting keywords. Suggest grouping paragraphs by subject / composition‑space / light‑color / atmosphere‑detail. Professional fluent language. 300‑600 words."
            },
            "structured": {
                "zh": """【结构化模式】按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度对齐参考示例。人像题材按下列要求完整输出；非人像题材将人物相关分段改写为对应主体（"角色与主体"→画面主体本身及其外观、"服装与配饰"→主体表面纹理与附属覆盖物、"道具与动态"→画面物体状态与交互），必填项构图方式、主体位置、视角类型、景深效果一项不可少：

**图片风格与剧情介绍**
点明风格定位与题材（时尚杂志/美妆广告/纪实摄影/插画/产品/建筑室内/美食等），概括画面讲述的瞬间与整体色调氛围；存在有效用户关键词时，将不冲突的风格、题材校准信息自然融入本段，冲突关键词直接舍弃；仅依据画面可读信息概括，禁止虚构情节。

**角色与主体**
（人像题材）人物年龄、人种、五官特点与眼神状态，皮肤质感与原生肌理，发型发色、长度与散落状态；（非人像题材）写明画面主体对象及其核心外观特征，本段不写人物字段。

**服装与配饰**
（人像题材）衣着款式、面料质感、颜色及剪裁层次，配饰的材质、颜色与设计感；（非人像题材）写明主体表面的纹理、图案、覆盖物或附属配件；无此项时写明画面中相关附属物状态。

**道具与动态**
（人像题材）肢体姿态与重心、手部与道具的互动细节、手指与指甲状态、视线方向、表情神态；（非人像题材）写明画面物体的姿态、位置关系与交互状态；无动态时写明静态体态。

**环境与背景**
背景类型与色调明暗、简洁程度或环境细节，前景/中景/背景的空间层次与虚化程度，负空间留白关系。

**摄影风格与质感**
*   **视角：** 景别类型（人像：微距特写/标准特写/肩特写/七分人像/九分人像/全景人像；非人像：微距特写/近景特写/中景/远景/全景）、视角类型（广角透视夸张/标准平实/长焦压缩/超长焦强烈压缩）与视角效果、景深效果（浅景深背景虚化/中景深部分清晰/深景深全景清晰）。
*   **构图：** 构图方式（三分法/对称/对角线/框架/中心/三角形等）、主体位置换算为自然方位描述（如位于画面中央、垂直方向偏上）、画面构图特征；画幅比例由工作流分辨率决定，不写入提示词。
*   **光影：** 光源方向、光质软硬、天气与光线状况、面部及物体上的明暗层次，区分前景/中景/背景的光影关系。
*   **质感：** 材质、纹理与表面细节特征；色彩调性、主色调、饱和度层级与冷暖倾向。
*   **氛围：** 从可观察的光影、色彩与细节推导画面氛围与意境，禁止空洞抽象情绪形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness matching the reference example. Output portrait category fully as specified; for non-portrait category rewrite character-oriented sections to the actual subject ("Character and Subject" -> the main subject and its appearance, "Outfit and Accessories" -> surface texture and covering of the subject, "Props and Pose" -> state and interaction of objects in frame). Mandatory items - composition type, subject position, perspective type, depth-of-field effect - must never be omitted.

**Image Style and Story Introduction**
State style positioning and theme (fashion magazine / beauty ad / documentary / illustration / product / architecture-interior / food, etc.), summarize the captured moment and overall color tone; when valid user keywords exist, merge non-conflicting style and theme calibration naturally, discard conflicting ones; base only on readable visual information, no fictional plot.

**Character and Subject**
(portrait) age, ethnicity, facial features and gaze state, skin texture and native details, hairstyle, hair color and how strands fall; (non-portrait) state the main subject and its core appearance, no character fields here.

**Outfit and Accessories**
(portrait) clothing cut, fabric texture, color and layering, accessory material, color and design; (non-portrait) surface texture, pattern, covering or attached accessories of the subject; if none, state the condition of related attached objects in frame.

**Props and Pose**
(portrait) body gesture and weight, hand-prop interaction detail, fingers and nails, gaze direction, facial expression; (non-portrait) posture, position relation and interaction state of objects in frame; if no motion, state a relaxed static stance.

**Environment and Background**
Background type, tone and brightness, simplicity or environment detail, foreground / mid-ground / background hierarchy and blur level, negative-space relation.

**Photography Style and Texture**
*   **Viewpoint:** shot type (portrait: macro close-up / standard close-up / shoulder shot / three-quarter / nine-tenth / full-scene portrait; non-portrait: macro close-up / close-up / medium shot / wide shot / full scene), perspective type (wide-angle exaggerated / standard natural / telephoto compressed / super-telephoto strong compression) and its visual effect, depth-of-field effect (shallow blurred background / medium partially sharp / deep full sharpness).
*   **Composition:** composition type (rule-of-thirds / symmetric / diagonal / frame / central / triangular), subject position converted into natural wording (e.g. centered in frame, upper in vertical), composition feature description; aspect ratio is set by workflow resolution, do not write it into the prompt.
*   **Lighting:** light direction, hardness-softness, weather and light condition, light-dark hierarchy on faces and objects, lighting relation of foreground / mid-ground / background.
*   **Texture:** material, texture and surface detail; color tone, dominant color, saturation level and cold-warm tendency.
*   **Atmosphere:** derive atmosphere and mood from observable light, color and detail; no empty abstract emotion words."""
            }
        }

    def detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if chinese_chars >= english_words else "en"

    def build_prompt(
            self,
            preset_name: str,
            user_keywords: Optional[str] = None,
            output_language: str = "auto",
            output_format: str = "both"
    ) -> Dict:
        valid_preset_names = ["image_reverse_describe"]
        if preset_name not in valid_preset_names:
            raise ValueError(f"预设模板不存在：{preset_name}")
        preset = self.preset_library[preset_name]

        kw_text = user_keywords if (user_keywords and user_keywords.strip()) else "无"
        detect_input = user_keywords if user_keywords else ""
        if output_language == "auto":
            lang = self.detect_language(detect_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"

        global_rule = self.global_base_rules[lang]
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        natural_guide = self.format_guide["natural"][lang]
        structured_guide = self.format_guide["structured"][lang]

        prompt_parts = []
        prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
        prompt_parts.append(global_rule)
        prompt_parts.append(preset_rule)
        prompt_parts.append(f"解析对象：#IMAGE_SOURCE#；用户辅助关键词：{kw_text}；关键词仅用于风格氛围校准，图片信息优先级最高，冲突则舍弃关键词。")

        if output_format == "natural":
            prompt_parts.append(natural_guide)
        elif output_format == "structured":
            prompt_parts.append(structured_guide)
        else:
            prompt_parts.append(natural_guide)
            prompt_parts.append(structured_guide)

        final_llm_prompt = "\n".join(prompt_parts)

        return {
            "status": "success",
            "llm_input_prompt": final_llm_prompt,
            "positive_constraint": pos_constraint,
            "output_language": lang,
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_keywords": user_keywords
        }