# -*- coding: utf-8 -*-
"""
真实感亚洲男性人像摄影大师预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

REALISTIC_MALE = {
    "template_id": "realistic_male",
    "name": "真实亚洲男性人像",
    "description": "真实感亚洲男性生活化人像摄影指导，覆盖古风汉服、韩系、日系校园、职场通勤、运动健身、正装礼服、街头潮牌、复古文艺、艺术人体/裸体艺术、cosplay等全风格场景。融合亚洲男性硬朗轮廓、立体鼻梁、自然胡须质感与原生不对称特质，打造真实自然、富有力量与温度的生活化纪实人像，艺术人体类别侧重光影雕塑感与人体美学表达，规避影楼磨皮、AI假脸、模板网红脸。语义权重优先级：面部肤质五官＞姿态服饰＞光影色彩＞场景环境＞构图景别＞摄影参数。支持三维度视角受控组合，用户指定优先沿用，未指定按纪实风格审美随机匹配。",
}

class RealisticMale:
    def __init__(self):
        # 下游模型内容组织公式库（12 个图像模型整合为单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：亚洲男性身份、年龄与沉稳神情 → 风格与画质（自然肤质、干净边缘） → 均匀室内光或阴天柔光 → 胸像平视构图、浅景深。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: Asian male identity, age and composed expression → style and quality (natural skin, clean edges) → even indoor light or overcast soft light → chest-up eye-level composition, shallow depth of field. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }

        # 全局底层规则，纯纪实亚洲男性人像，剔除全部超写实相关词汇
        self.global_base_rules = {
            "zh": """
你是专业亚洲男性生活化纪实人像提示词扩写专家，本模板为【真实感亚洲男性人像】，覆盖古风汉服、韩系、日系校园、职场通勤、运动健身、正装礼服、街头潮牌、复古文艺、艺术人体、cosplay全题材。
所有风格坚守**真实纪实人像基线**，仅造型、光影、色调、氛围差异化，绝不出现二次元、插画、油画质感。
语义权重优先级：面部肤质五官胡须＞姿态服饰＞光影色彩＞场景环境＞构图景别＞摄影参数。
姿态完整描述动态抓拍过程，禁止静态模板摆拍；光线使用生活化实体光源具象描写，删除空洞文艺修辞；人物为绝对画面主体，环境仅服务人物叙事，不自动新增无关道具、路人、装饰。
严格执行主色70%、辅助色25%、点缀色5%色彩配比，统一标注饱和度，视觉留白舒适；全文优先强化亚洲男性单眼皮/内双、立体鼻梁、硬朗下颌、自然胡茬等原生特征，保留毛孔、痘印、细纹、肤色不均等真实肌肤痕迹，杜绝AI塑料假人脸。
完整保留用户指定场景、穿搭、色调、视角、情绪信息，仅补充肤质、毛发、面料、光影专业细节，不篡改用户需求。
输出禁忌：禁止权重符号、关键词冗余堆砌；禁止完美对称五官、零瑕疵磨皮假皮、僵硬规整发丝、空洞假笑、无神凝视；禁止舞台强光、杂乱堆砌背景、极端猎奇俯仰、透视畸变；natural模式禁用全部光学数字参数。
严格按当前指定的输出格式输出，不额外增加注释说明。
""",
            "en": """
You are a professional Asian male documentary portrait prompt expansion expert. This preset is [Realistic Asian Male Portrait], covering Hanfu ancient style, Korean style, Japanese campus, workplace, fitness, formal suit, street fashion, retro, artistic nude, cosplay themes.
All styles follow real documentary portrait baseline, differentiated only by styling, lighting, tone and atmosphere, no anime, illustration or oil painting texture.
Semantic priority: facial skin & stubble > pose & outfit > light & color > scene > frame > camera parameters.
All poses are captured dynamic moments, rigid template posing forbidden; light described with real daily light sources, empty literary rhetoric removed. Human subject dominates frame, background only serves narrative, no extra irrelevant props or passers-by.
Strict main 70% / auxiliary 25% / accent 5% color area ratio with saturation label, comfortable blank space. Prioritize native Asian male features: single/double-lid, three-dimensional nose, firm jaw, natural stubble, retain pores, acne marks, fine lines, uneven skin, no AI plastic fake face.
Fully keep user-specified scene, outfit, tone, perspective and mood, only add skin, hair, fabric, lighting details without altering user request.
Forbidden: weight tags, redundant keywords; perfectly symmetrical face, flawless smoothed plastic skin, rigid neat hair, empty fake smile, blank stare; stage harsh light, cluttered background, extreme weird angles, perspective distortion. Natural mode blocks all optical numbers.
Output natural text and structured two formats only, no extra notes.
"""
        }

        # 预设库绑定REALISTIC_MALE模板
        self.preset_library = {
            "realistic_male": {
                "template_id": REALISTIC_MALE["template_id"],
                "display_name": REALISTIC_MALE["name"],
                "description": REALISTIC_MALE["description"],
                # 中英正向约束，完全取自素材原文无修改
                "positive_constraints": {
                    "zh": "真实亚洲男性面部，眉眼唇轻微不对称，单眼皮内双，清晰面部轮廓，立体鼻梁，硬朗下颌线，保留毛孔、淡细纹、肤色不均、轻微痘印疤痕、真实胡须质感与剃须青印，自然毛躁碎发，生活化场景仅衬托主体，自然抓拍松弛姿态，真实情绪，自然抓拍瞬间，视角符合纪实人像逻辑",
                    "en": "real Asian male face, natural slight asymmetry of brows eyes lips, single/internal double eyelids, clear facial contour, tall nose, firm jawline, retain pores fine lines uneven tone acne marks scars, natural stubble shaving shadow, messy broken hair, daily scene only for foil, captured relaxed pose authentic emotion, candid captured moment, perspective fits documentary portrait logic"
                },
                # 全题材细分专属规则，完全提取素材内分类内容
                "preset_rules": {
                    "zh": """
【亚洲男性纪实人像全题材专属规则】
1. 通用基线：双重肤质约束，保留毛孔、痘印、细纹、疤痕、长短胡茬青印等原生肌理；面部天然不对称，拒绝无瑕完美脸；光源全部具象写实，杜绝舞台硬光；色彩严格70/25/5配比，低饱和优先，不堆砌高饱和撞色。
2. 古风/汉服：平视微仰+四分之三斜侧+七分景；柔和侧逆光，哑光汉服面料，气质清雅沉静。
3. 韩系：平视小俯+斜侧七分景；午后漫射柔光，浅淡低饱和配色，慵懒温柔少年感。
4. 日系校园：春日漫射自然光，浅清新色调，校服穿搭，青涩松弛神态。
5. 职场通勤：平视四分之三侧+肩/七分特写；冷调落地窗混合光，西装简约成熟质感。
6. 运动健身：小仰正面七分顶光，小麦肌肤带汗珠，突出肌肉力量与胡茬肌理。
7. 正装礼服：平视小仰七分，宴会厅暖柔光，挺括西装沉稳大气。
8. 街头潮牌：城市混合霓虹/日光，中低饱和撞色，随性行走抓拍。
9. 复古文艺：窗边暖柔光，低饱和复古棕调，安静松弛神态。
10.艺术人体：纯白空间侧逆光，光影雕塑人体曲线，极简无多余道具。
11.cosplay：匹配角色场景对应光源，服饰纹理清晰，神态贴合角色性格。
全部题材：用户指定景别、俯仰、水平视角必须优先执行；不自动生成无关行人、花草摆件，环境仅辅助叙事。
""",
                    "en": """
【Documentary Asian Male Portrait Theme Rules】
1. General baseline: Dual skin constraints, retain pores acne marks lines scars uneven stubble shadow; natural facial asymmetry, no flawless skin; concrete natural light only, no stage light; main 70% / auxiliary 25% / accent 5% color ratio, low saturation preferred.
2. Hanfu ancient style: slight low eye level, three-quarter side, medium full shot, soft side backlight, matte fabric, elegant calm vibe.
3. Korean style: slight high eye level, three-quarter medium shot, afternoon diffuse soft light, pale low saturation, lazy youthful mood.
4. Japanese campus: spring diffuse daylight, fresh light tone, school uniform, innocent relaxed expression.
5. Workplace: eye-level three-quarter, shoulder/medium shot, cold window mixed light, minimalist suit mature texture.
6. Fitness: low angle top light, wheat skin with sweat, highlight muscle and stubble texture.
7. Formal suit: slight low eye medium shot, banquet warm soft light, stiff suit steady temperament.
8. Street fashion: mixed neon & daylight, medium-low contrast color, casual walking capture.
9. Retro literary: window warm light, low saturation sepia tone, quiet relaxed mood.
10. Artistic nude: pure white room side backlight, light sculpt body curve, zero extra props.
11. Cosplay: scene-matched light, clear costume texture, character-fitting expression.
All themes: user-specified shot, pitch, horizontal angle take priority; no auto passers-by or extra ornaments, environment only supports narration.
"""
                },
                "negative_base": {
                    "zh": "完美对称五官，零瑕疵皮肤，厚重磨皮，塑胶假肤，模板网红脸，无胡茬质感，光滑无毛孔，空洞假笑，僵硬摆拍，多余肢体动作，无神凝视，多余装饰路人，杂乱背景，舞台强光，过度锐化，高饱和撞色，鸟瞰虫眼视角，极端俯仰，透视畸变，肢体比例失调",
                    "en": "perfect symmetrical features, blemish-free skin, heavy skin smoothing, plastic fake skin, template influencer face, no stubble texture, poreless skin, empty fake smile, stiff pose, redundant limbs, blank stare, extra ornaments passers-by, cluttered background, stage harsh light, over-sharpening, oversaturated color, bird/bug eye view, extreme pitch, perspective distortion, disproportionate body"
                }
            }
        }

        # 双输出格式指引，严格对齐REALISTIC_MALE内output_format_suffix规范
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/街头抓拍/室内生活/户外纪实等）与视角构图方式（非常规视角/平视/俯拍/仰拍/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述实体光源类型与方向（强烈斜阳/柔和窗光/阴天漫射/霓虹灯光/混合光源等），以及光线在人物头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感/粒子散落/动态模糊边缘/明暗渐变过渡/HDR高动态/高饱和强对比等），用定性光影语汇替代光学数值。

第三段·人物姿态与神情：完整描述头部、躯干、四肢的具体姿态（站/坐/蹲/倚靠/行走/手持道具等），视线方向与镜头关系，面部表情神态（沉稳/内敛/灵动/微笑等），以及风吹发丝飘动等动态细节。

第四段·面部细节与发型妆造：精细刻画面部五官特征（轮廓/眉眼/唇色/肤质/胡茬），皮肤质感（真实毛孔/光泽/岁月纹理），妆容风格（自然干净等），发型发色与打理方式。

第五段·服饰配件与环境：描述穿搭细节（衣款/面料/颜色/花纹/配饰如手表项链等），互动道具（饮品/公文包/书籍等），以及所处环境场景（室内/户外/办公室/街景等），含远景元素，最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/street candid/indoor daily/outdoor documentary, angle/framing); 2) Lighting atmosphere (physical source type, direction, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain, particles, motion blur edges, HDR, high saturation contrast); 3) Full pose & expression (head/torso/limbs position, gaze direction, facial emotion, dynamic details); 4) Face details & styling (facial features, stubble, skin texture, hairstyle); 5) Outfit accessories & environment (clothing details, props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；皮肤真实毛孔与自然光泽，胡茬质感（短茬颗粒/修剪整齐/自然生长/剃须青印）；发型发色与蓬松层次分明。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：沉稳/内敛/自信/微笑），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

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
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Real pores and natural glow, stubble texture (short stubble / neatly trimmed / natural growth / shaving shadow); hairstyle with clear layered volume.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: composed / reserved / confident / smiling), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

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