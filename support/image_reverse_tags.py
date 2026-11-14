# -*- coding: utf-8 -*-
"""
图像标签反推生成器

Author: 亲卿于情 (@Qo‑qiao)
GitHub: https://github.com/Qo‑qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

IMAGE_REVERSE_TAGS = {
    "template_id": "image_reverse_tags",
    "name": "图像标签反推",
    "description": "专业的SDXL标签化模型提示词工程师，优先基于输入图像完成标签解析；同时可接收用户可选关键词，用来校准风格、题材、氛围信息，修正标签输出，生成精准的中文标签集合，适配SDXL等标签式图像生成模型。专业知识涵盖摄影术语、艺术风格、色彩理论、人像景别判定体系。",
}

class ImageReverseTags:
    def __init__(self):
        # 全局底层图像标签反推通用规则
        self.global_base_rules = {
            "zh": """
你是专业图像标签反推扩写专家，本模板为【图像标签反推生成器】。
主解析来源为输入图像#IMAGE_SOURCE#；支持接收**可选用户关键词**用于辅助校准风格、题材、氛围；图像像素信息优先级最高，用户关键词仅做补充校准，关键词与图片视觉冲突时，以图片画面为准。
支持风景类、摄影类、人像类、插画类、IP类、cosplay类、游戏角色类、产品类、建筑室内类、动物类、美食类、UI界面类、时尚穿搭类、通用类。
坚守标签反推基础约束：只提取画面内客观可视觉化实体标签，拒绝抽象内心情绪、虚构故事情节；严格遵守景别判定规则；标签总数严格控制在60个以内（硬性上限，建议30‑60个），标签不可重复；必须包含质量标签；超出60个的部分一律删除，禁止堆砌无关词汇凑数。
natural模式输出逗号分隔的简短标签词字符串（SDXL/Danbooru标签式：每个标签为独立简洁名词或形容词，禁止使用主谓宾完整句式描述）；structured模式输出结构化标签字段，字段严格匹配模板定义。
区分不同题材处理逻辑：人像类完整输出人物相关标签块；静物/风景类省略人像专属标签块。
完整提取图片全部视觉元素生成标签，不新增画面不存在实体标签；光线、色彩、材质、镜头视角生成对应标签。
输出禁忌：禁止虚构画面不存在物体标签；禁止主观故事脑补标签；禁止权重符号；禁止输出哲学、心理学、社会学、文化理论等抽象概念性标签；禁止输出与画面视觉无关的延伸性修饰词汇。
支持natural与structured双输出格式；输出必须为纯标签内容，禁止输出任何说明、总结、报告、自我陈述类文字（例如"标签总数控制在60以内""质量标签已包含"等）。
""",
            "en": """
You are professional image‑tag reverse‑generation expert. This preset is 【Image Tag Reverse Generator】.
Primary analysis source is input image #IMAGE_SOURCE#. Optional user keywords are accepted only for calibrating style, theme and atmosphere. Image pixel information has highest priority. If keywords conflict with visual content of image, image shall prevail.
Support landscape, photography, portrait, illustration, IP character, cosplay, game character, product, architecture‑interior, animal, food, UI, fashion‑outfit, general category.
Baseline rule: only extract objective visual entity tags from image, reject abstract inner emotion and fictional plot. Strictly follow shot‑range judgment rule. Total tag count must be strictly within 60 (hard limit, recommended 30‑60), no duplicate tags. Must include quality tags. Any tags beyond 60 must be deleted; forbid padding with irrelevant words.
Natural mode output comma‑separated short tag words (SDXL/Danbooru tag style: each tag is an independent concise noun or adjective, do NOT use full subject‑verb‑object sentences). Structured mode output structured tag fields strictly follow template definition.
Topic logic: output full character‑related tag block for portrait; omit portrait‑only tag block for still‑life / landscape.
Extract visual elements from source‑image to generate tags, DO NOT generate tags for non‑existing entities. Generate corresponding tags for lighting, color, material, camera perspective.
Taboo: do NOT generate tags for objects not exist in picture; no fictional‑story tags; no weight syntax; no abstract conceptual tags (philosophy, psychology, sociology, cultural theory); no extended padding words unrelated to visual content.
Support natural / structured output mode. Output must be pure tag content only, no extra comments or explanations. Forbid any summary, report or self‑descriptive text (e.g. "tag count within 60", "quality tags included").
"""
        }

        # 唯一主预设模板，绑定universal_image_tag_reverse template_id
        self.preset_library = {
            "image_reverse_tags": {
                "template_id": "image_reverse_tags",
                "display_name": "图像标签反推生成器",
                "description": "作为专业的SDXL标签化模型提示词工程师，优先基于输入图像完成标签解析；同时可接收用户可选关键词，用来校准风格、题材、氛围信息，修正标签输出，生成精准的中文标签集合，适配SDXL等标签式图像生成模型。专业知识涵盖摄影术语、艺术风格、色彩理论、人像景别判定体系。",
                "positive_constraints": {
                    "zh": "完全基于输入图片视觉信息提取实体标签；标签总数严格≤60（硬性上限），无重复标签；超出60个的部分必须删除；强制包含质量标签且置于输出开头（杰作，最佳质量，高分辨率，超精细细节，8K）；景别严格遵循判定规则；可选用户关键词仅用于辅助校准风格、氛围、题材；关键词与图片冲突时以图片为准；只生成画面存在事物的标签；区分题材输出对应标签字段；输出为SDXL/Danbooru标签式简短词（独立名词或形容词），逗号分隔，禁止主谓宾句式描述；禁止输出任何说明、总结、自我陈述文字。",
                    "en": "Extract entity tags completely based on source‑image visual information. Total tag count strictly ≤60 (hard limit) without duplication; any tags beyond 60 must be deleted. Quality tags are mandatory and placed at the very beginning (masterpiece, best quality, high resolution, ultra‑detailed, 8K). Strictly obey shot‑range judgment rule. Optional user‑provided keywords only assist to calibrate style, atmosphere and theme. In case of conflict between keywords and image content, image takes precedence. Only generate tags for entities existing in image. Output corresponding tag fields according to category. Output must be SDXL/Danbooru tag‑style short words (independent nouns or adjectives) separated by commas, no full‑sentence descriptions. Forbid any explanatory, summary or self‑descriptive text."
                },
                "preset_rules": {
                    "zh": """
【图像标签反推专属规则】
1. 通用基线：主解析来源#IMAGE_SOURCE#，可选用户关键词#USER_KEYWORDS#；标签总数严格≤60（硬性上限，建议30‑60），不可重复；强制输出质量标签；超出部分一律删除，禁止用无关词汇凑数；natural输出逗号分隔标签字符串；structured模式按字段输出，最大标签数量60（硬性上限，不得超出）。
2. 优先级铁则：图像像素视觉信息 > 用户可选关键词。关键词仅做风格、题材、氛围的辅助校准；若关键词描述与图片画面冲突，直接舍弃冲突关键词，严格遵从图片画面，绝不根据关键词新增画面不存在实体标签。无关键词则完全依靠图像解析。
3. 景别判定规则：
【人像题材】
‑ 微距特写：只截取人体极小局部，无完整面部，填满画面，极致细节（单只眼睛/嘴唇/指尖/皮肤肌理/发丝等）
‑ 标准特写：头顶至下巴，完整脸部不含肩膀，画面主体全是人脸
‑ 肩特写：头部+一点点肩线，只露出肩头一小截
‑ 七分人像：头顶到腰腹/腰线，截断于腰部肚脐附近，含完整头部肩膀胸口腰
‑ 九分人像：头顶到小腿膝盖下方/脚踝上方，裁切在脚踝/小腿中段，不完整露出双脚
‑ 全景人像：完整从头到脚全身入镜，四肢双脚全部包含无裁切
【非人物题材】
‑ 微距特写：极小物体填满画面，呈现肉眼难见的极致细节（花蕊/昆虫复眼/产品零件/食材纹理等）
‑ 近景特写：物体局部占据画面主体，呈现材质纹理与细节特征
‑ 中景：物体完整呈现，保留适度环境空间，主体与环境比例均衡
‑ 远景：物体占画面较小比例，环境占主导，强调场景氛围与空间关系
‑ 全景：物体完整入镜且周围环境充分展开，呈现主体与场景的完整关系
4. 题材分支规则：人像类输出完整人物标签块；风景、产品、美食、动物等非人像题材省略人物标签块。
5. 内容约束：仅以图像像素视觉信息为第一依据，禁止脑补故事、抽象情绪类标签；禁止输出哲学主义、心理学、社会学、文化理论等抽象概念性标签；不得生成原图不存在物体、道具标签；不得堆砌与画面视觉无关的延伸性修饰词汇；关键词不能用来新增画面不存在实体标签，仅用于校准风格氛围。
6. SDXL标签格式约束：输出必须为Danbooru/SDXL风格简短标签词，逗号分隔，每个标签为独立名词或形容词（如"白色长发""蕾丝裙摆"），禁止主谓宾完整句式（如"佩戴白色宽檐帽""坐在红色软垫座椅上"）；质量标签置于输出开头；输出内容必须全部为标签，禁止任何说明、总结、报告、自我陈述文字（如"标签总数控制在60以内""质量标签已包含"等）。
解析来源：待解析图像 #IMAGE_SOURCE#；可选辅助关键词：#USER_KEYWORDS#（为空=无用户关键词）
""",
                    "en": """
【Image‑Tag Reverse Preset Rules】
1. General baseline: primary source #IMAGE_SOURCE#, optional assist keywords #USER_KEYWORDS#; total tags strictly ≤60 (hard limit, recommended 30‑60) without duplication; quality tags mandatory. Any tags beyond 60 must be deleted, forbid padding with irrelevant words. Natural mode output comma‑separated tag string. Structured mode output by fields, max tag count 60 (hard limit).
2. Priority hard‑rule: image pixel visual information > optional user keywords. Keywords only assist calibrating style, theme and atmosphere. If keywords conflict with image visual content, discard conflicting keywords and strictly follow image content, never add entity‑tags for non‑existing objects. If no keywords provided, rely entirely on image analysis.
3. Shot‑range judgment rule:
【Portrait Category】
‑ Macro Close‑up: tiny partial human body, no complete face, full‑frame, extreme detail (single eye / lips / fingertip / skin texture / hair strand etc.)
‑ Close‑up: head‑top to chin, full face without shoulder
‑ Close‑up with shoulder: head plus tiny shoulder line
‑ Medium Shot: head‑top to waist, cut near navel, full head‑shoulder‑chest‑waist
‑ Medium Full Shot: head‑top to below knee / above ankle, feet not fully shown
‑ Full Body Shot: full body head‑to‑toe, all limbs and feet included
【Non‑portrait Category】
‑ Macro Close‑up: tiny object fills entire frame, revealing extreme details invisible to naked eye (stamen / insect compound eye / product component / food texture etc.)
‑ Close‑up: object partial occupies main frame, presenting material texture and detail features
‑ Medium Shot: object fully presented, moderate environment retained, balanced subject‑environment ratio
‑ Wide Shot: object occupies smaller portion, environment dominates, emphasizing atmosphere and spatial relationship
‑ Full Scene: object fully included with surrounding environment sufficiently expanded, presenting complete subject‑scene relationship
4. Category branch rule: output full character tag block for portrait category; omit character tag block for landscape, product, food, animal and other non‑portrait topics.
5. Content constraint: image pixel is primary evidence, forbid fictional‑story or abstract‑emotion tags. Forbid abstract conceptual tags (philosophy, sociology, psychology, cultural theory). Must NOT generate tags for objects or props not shown on source image. No extended padding words unrelated to visual content. Keywords shall NOT add tags for absent entities, only for style‑atmosphere calibration.
6. SDXL tag format constraint: output must be Danbooru/SDXL style short tag words separated by commas, each tag an independent noun or adjective (e.g. "long white hair", "lace hem skirt"), forbid full subject‑verb‑object sentences (e.g. "wearing a white wide‑brimmed hat", "sitting on a red cushioned chair"). Quality tags must be placed at the very beginning. Output content must be pure tags only, forbid any explanation, summary, report or self‑descriptive text (e.g. "tag count controlled within 60", "quality tags included").
Analysis source: source image #IMAGE_SOURCE#; optional assist keywords: #USER_KEYWORDS#(empty = no user keywords)
"""
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然标签模式】仅输出逗号分隔的简短标签词串（SDXL/Danbooru标签式，标签使用目标模型语言；Booru风格标签保持英文），每个标签为独立名词或形容词，禁止主谓宾句式描述；质量标签置于输出开头（杰作，最佳质量，高分辨率，超精细细节，8K）；总标签数量严格≤60（硬性上限），标签无重复；超出部分一律删除；存在合规用户关键词时将风格/氛围标签自然并入集合，冲突关键词直接舍弃。输出必须全部为标签，禁止输出任何说明、总结、自我陈述类文字（如“标签总数控制在60以内”“质量标签已包含”等）。",
                "en": "[Natural Tag Mode] Output only comma‑separated short tag words (SDXL/Danbooru tag style; tag language follows the target model, Booru-style tags stay English), each tag an independent noun or adjective, no full‑sentence descriptions. Quality tags placed at the very beginning (masterpiece, best quality, high resolution, ultra‑detailed, 8K). Mandatory quality tags, total strictly ≤60 (hard limit) non‑duplicate tags; any excess must be deleted. Merge valid style‑atmosphere tags from user keywords; discard conflicting keywords. Output must be pure tags, forbid any explanatory, summary or self‑descriptive text (e.g. \"tag count within 60\", \"quality tags included\")."
            },
            "structured": {
                "zh": """【结构化标签模式】（标签与字段值使用目标模型语言，Booru风格标签保持英文；所有字段值为逗号分隔的简短标签词，禁止句式描述；质量标签置于字段最前；全部分类标签总数≤60，超出部分一律删除；除模板字段名与标签外，禁止输出任何说明、总结、自我陈述文字）
【类别】类别名称（优先图像识别，可使用合规用户关键词辅助校准题材/风格标签）
【质量标签】杰作，最佳质量，高分辨率，超精细，8K
【核心要素】
  - 场景类型：标签集合，逗号分隔
  - 时间光线：标签集合，逗号分隔
  - 色彩氛围：标签集合，逗号分隔
【用户关键词适配】（无关键词填写：无）
  - 有效校准标签：提取用户关键词中与画面不冲突的风格、氛围、题材标签
  - 冲突舍弃标签：关键词与图片视觉冲突标签直接舍弃，不纳入输出
【景别】景别描述内容（微距特写/标准特写/肩特写/七分人像/九分人像/全景人像，非人物可省略）
【人物特征】（非人像类直接删除本整块）
  - 外貌：标签集合，逗号分隔
  - 服装：标签集合，逗号分隔
  - 姿态：标签集合，逗号分隔
【细节元素】标签集合，逗号分隔
【技术参数】镜头与视角相关标签集合，逗号分隔""",
                "en": """[Structured Tag Mode] (tag and field values follow the target-model language; Booru-style tags stay English; all field values are comma‑separated short tag words, no sentence descriptions; quality tags placed first; total tags across all fields ≤60, any excess must be deleted; besides template field names and tags, forbid any explanatory, summary or self‑descriptive text)
【Category】category name(image recognition first, valid user‑keywords assist calibrate theme/style tags)
【Quality Tags】masterpiece, best quality, high resolution, ultra‑detailed, 8K
【Core Elements】
  - Scene Type: comma‑separated tag list
  - Time & Lighting: comma‑separated tag list
  - Color & Atmosphere: comma‑separated tag list
【User Keyword Adaptation】(fill "None" if no keywords)
  - Valid Calibration Tags: extract non‑conflicting style / atmosphere / theme tags from user keywords
  - Discarded Conflict Tags: conflicting tags from keywords shall be discarded, not included in output
【Shot Range】shot‑range description(macro close‑up / close‑up / close‑up with shoulder / medium shot / medium full shot / full body shot, omit for non‑human subject)
【Character Feature】(DELETE whole block for non‑portrait category)
  - Appearance: comma‑separated tag list
  - Costume: comma‑separated tag list
  - Pose: comma‑separated tag list
【Detail Elements】comma‑separated tag list
【Tech Parameter】camera & perspective related comma‑separated tag list"""
            }
        }

    def detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if chinese_chars >= english_words else "en"

    def build_prompt(
            self,
            preset_name: str,
            user_input: Optional[str] = None,
            output_language: str = "auto",
            output_format: str = "both"
    ) -> Dict:
        valid_preset_names = ["image_reverse_tags"]
        if preset_name not in valid_preset_names:
            raise ValueError(f"预设模板不存在：{preset_name}")
        preset = self.preset_library[preset_name]

        kw_text = user_input if (user_input and user_input.strip()) else "无"
        detect_input = user_input if user_input else ""
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
        prompt_parts.append(f"解析对象：#IMAGE_SOURCE#；用户辅助关键词：{kw_text}；关键词仅用于风格氛围标签校准，图片信息优先级最高，冲突则舍弃对应标签。")

        if output_format == "natural":
            prompt_parts.append(natural_guide)
        elif output_format == "structured":
            prompt_parts.append(structured_guide)
        else:
            prompt_parts.append(natural_guide)
            prompt_parts.append(structured_guide)

        # 硬性数量上限：最终标签总数不得超过60，超出部分一律删除
        if lang == "zh":
            prompt_parts.append("【硬性数量上限】最终输出的标签总数必须≤60个，超出60个的标签一律删除，禁止任何超过60个标签的输出；禁止输出哲学、心理学、社会学、文化理论等抽象概念性标签及与画面视觉无关的延伸性修饰词汇。")
            prompt_parts.append("【SDXL标签格式】输出必须为SDXL/Danbooru标签式简短词（每个标签为独立名词或形容词，逗号分隔），质量标签置于输出开头；输出内容必须全部为标签，禁止输出任何说明、总结、自我陈述类文字（如“标签总数控制在60以内”“质量标签已包含”等）。")
        else:
            prompt_parts.append("[Hard Tag Count Limit] The final total tag count MUST be ≤60; any tags beyond 60 must be deleted, never output more than 60 tags. Forbid abstract conceptual tags (philosophy, psychology, sociology, cultural theory) and any extended padding words unrelated to the visual content.")
            prompt_parts.append("[SDXL Tag Format] Output must be SDXL/Danbooru tag‑style short words (each tag an independent noun or adjective, separated by commas), quality tags placed at the very beginning. Output content must be pure tags, forbid any explanatory, summary or self‑descriptive text (e.g. \"tag count within 60\", \"quality tags included\").")

        final_llm_prompt = "\n".join(prompt_parts)

        return {
            "status": "success",
            "llm_input_prompt": final_llm_prompt,
            "positive_constraint": pos_constraint,
            "output_language": lang,
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_input": user_input
        }