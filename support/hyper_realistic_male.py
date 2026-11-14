# -*- coding: utf-8 -*-
"""
超写实男性人像预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

HYPER_REALISTIC_MALE = {
    "template_id": "hyper_realistic_male",
    "name": "超写实男性人像",
    "description": "全能超写实真人复刻商业男士人像摄影指导，全覆盖古风国风、现代都市、复古胶片、暗黑轻奢、时尚杂志、高端职场、极简棚拍、科幻男士人像等全题材。兼容亚洲/欧美男性五官骨骼、胡茬体态特征，面部干净精致，无痘印、无明显瑕疵，保留原生毛孔与自然皮肤肌理，杜绝塑料假肤、AI模板脸、网红过度磨皮感。语义权重优先级：面部骨骼肤质胡茬＞体态姿态服饰＞光影色调氛围＞场景构图＞摄影参数。所有风格坚守真人写实基线，仅氛围与造型差异化，古风、职场、胶片、科幻均为题材分支，不脱离超写实核心，所有男士模特姿态沉稳克制，无夸张僵硬表现。",
}

class HyperRealisticMale:
    def __init__(self):
        # 下游模型内容组织公式库（12 个图像模型整合为单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：男性身份、年龄、穿搭与硬朗神情 → 超写实肌理、胡须与皮肤细节 → 偏冷侧光或伦勃朗光塑造立体骨相 → 胸像或特写、浅景深。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: male identity, age, outfit and firm expression → photorealistic texture with beard and skin details → cool side light or Rembrandt light shaping the bone structure → chest-up or close-up, shallow depth of field. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }

        # 全局底层通用规则，中英双语，适配男性人像
        self.global_base_rules = {
            "zh": """
你是专业高端全风格超写实男士人像摄影提示词扩写专家，本模板为【通用超写实男性人像】，全覆盖：古风国风、现代都市、复古胶片、暗黑轻奢、时尚杂志、高端职场、极简棚拍、科幻男士人像所有题材。
所有风格坚守**真人超写实基线**，仅造型、光影、色调、氛围差异化，绝不出现二次元、插画、油画质感。
男士面部干净精致，无痘印、无明显瑕疵，保留皮肤原生毛孔、自然肌理、浅细纹与自然肤色层次，拒绝过度磨皮导致的塑胶假肤。
姿态必须使用具体肢体结构描述，全部采用专业男模标准摆姿，禁止模糊形容词；光线方向明确，光影过渡柔和通透；男士绝对画面主体，环境仅衬托氛围。
完整保留用户输入的风格、服饰、场景、色调、姿态、视角所有信息，仅补充摄影、材质、光影、肤质、胡茬、发丝专业细节，不新增无关物体、多余元素。
所有男士穿搭、国风锦袍、职场西装、时尚大衣均作为高端男士人像题材，姿态沉稳克制、硬朗高级，禁止夸张畸形体态、过度柔弱造型。
输出禁忌：禁止权重符号、多余相机参数、冗余堆砌；禁止卡通二次元、畸形肢体、坏手烂指、网红假脸、磨皮蜡皮；禁止杂乱背景、空洞假笑、抓拍自拍、透视畸变。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional universal photorealistic male portrait prompt expert. This preset covers all styles: ancient chinese style, modern urban, retro film, dark luxury, fashion magazine, business elite, minimalist studio, sci-fi male portrait.
All styles adhere strictly to photorealistic human baseline, differentiated only by styling, lighting, tone and atmosphere, no illustration, anime or oil painting texture.
Male face is clean and exquisite, no acne marks, no obvious blemishes, retain original skin pores, natural texture, shallow fine lines and natural skin tone layers, reject plastic fake skin caused by excessive skin smoothing.
All male outfits, hanfu, business suits and fashion coats belong to high-end male portrait themes with steady and restrained poses, no exaggerated deformed body or overly soft figure.
Pose described with concrete body structure, no vague words. Clear light direction and soft shadow transition. Male subject dominates the frame, background only for atmosphere.
Completely retain user input style, clothing, scene, tone, pose and perspective. Only supplement professional photography, texture, lighting, skin, stubble and hair details without irrelevant elements.
Forbidden: no weight symbols, no redundant camera parameters, no anime/cartoon/illustration, no deformed anatomy, no bad hands, no over-retouched skin, no messy background, no fake smile, no snapshot selfie, no perspective distortion.
Strictly output two formats without extra comments.
"""
        }

        # 唯一主预设库，绑定超写实男性人像专属模板
        self.preset_library = {
            "hyper_realistic_male": {
                "template_id": "hyper_realistic_male",
                "display_name": HYPER_REALISTIC_MALE["name"],
                "description": HYPER_REALISTIC_MALE["description"],
                "positive_constraints": {
                    "zh": "超写实真人质感，硬朗男性面部骨骼，清晰下颌线条，眉眼唇轻微不对称，区分亚洲/欧美男性特征；面部干净，保留毛孔、自然肌理与浅细纹，皮肤质感真实有层次；原生蓬松发丝与自然胡茬，干净布景，专业男模沉稳姿态，真实情绪；所有风格分支均保持真人写实基线，姿态挺拔硬朗有力量感",
                    "en": "photorealistic real human texture, tough male facial bone structure, clear jawline, natural slight asymmetry of eyes, eyebrows and lips, distinguish Asian/European male features; clean face, retain pores, natural texture and shallow fine lines with real layered skin quality; layered fluffy hair and natural stubble, restrained clean scene, steady professional male model pose, calm expression; all styles maintain photorealistic baseline with upright powerful posture"
                },
                "preset_rules": {
                    "zh": """
【男士全风格超写实专属规则】
1. 通用基线：男士面部干净精致，无痘印、无明显瑕疵，保留皮肤原生毛孔、浅细纹、自然肤色层次；保留面部轻微不对称，杜绝完美蜡像脸、网红过度磨皮帅哥脸；肤色过渡自然均匀，高光不过曝，暗部不死黑，光影层次通透，强化自然胡茬生长肌理；色彩严格执行主色70%、辅助色25%、点缀色5%配比，低饱和优先。
2. 古风国风风格：强化东方男士利落下颌、内敛清俊骨相、温润沉稳气质，适配锦袍、素色汉服、玉饰古风造型；优先柔和侧逆柔光、漫射棚布光，低饱和素雅清冷色调。
3. 现代职场风格：适配西装、轻奢通勤穿搭、极简灰度影棚；光影干净立体塑型，色调低饱和冷调高级，体态挺拔规整，适配轻奢室内、纯色棚拍场景。
4. 复古胶片风格：保留真实胶片颗粒、复古暖调褪色质感、柔和漫射光；肤质保留原生毛孔、浅细纹、自然肤色层次，不重度精修，氛围怀旧儒雅。
5. 暗黑轻奢风格：高对比伦勃朗侧光、低饱和暗调质感、硬朗明暗分割；气质冷冽禁欲贵气，极简深色布景，突出男士立体骨骼与疏离气场。
6. 男士时尚杂志风格：硬光柔光组合立体修容光影、高清通透写实质感；体态利落舒展、力量感线条，适配纯色影棚、高端轻奢置景。
7. 极简男士肖像风格：纯色纯白/灰度影棚布景，正面蝴蝶柔光，弱化多余装饰；聚焦面部骨骼、胡茬、毛孔肌理，气质清冷纯粹。
8. 科幻艺术人像风格：冷调硬光、金属低饱和配色、未来极简布景；男士体态硬朗富有张力，服饰金属面料肌理清晰。
所有风格：用户指定内容优先，仅补充专业光影、肤质、胡茬、面料细节，不篡改用户题材与氛围。
""",
                    "en": """
【Universal Photorealistic Preset Rules for Male Portrait】
1. General baseline: Male face is clean and exquisite, no acne marks, no obvious blemishes, retain original skin pores, shallow fine lines and natural skin tone layers; keep slight facial asymmetry, no perfect wax face or over-retouched internet celebrity male face. Natural uniform skin tone, no overexposed highlight or crushed shadow, transparent light and shadow layers, emphasize natural stubble texture. Color strictly follows a main 70% / auxiliary 25% / accent 5% ratio.
2. Ancient Chinese style: Sharp jawline and gentle oriental bone structure for asian men, suitable for hanfu and jade accessories; soft side backlight, low saturation elegant tone.
3. Modern business style: Suits and commute outfits, minimalist gray studio; clean three-dimensional lighting, low saturation cold tone, upright body in light luxury indoor or solid color studio.
4. Retro film style: Authentic film grain, warm faded tone, soft diffused light, retain skin pores, shallow fine lines and natural skin tone layers without heavy retouching, retro elegant atmosphere.
5. Dark luxury style: High contrast chiaroscuro side light, low saturation dark tone, cold and restrained temperament, minimalist dark background to highlight male bone and alienated aura.
6. Men fashion magazine style: Mix hard & soft light for three-dimensional shadow, neat powerful body line, solid color studio and light luxury scene.
7. Minimalist male portrait: Pure white / gray studio, front butterfly soft light, no redundant decoration, focus on facial bone, stubble and pore texture, pure cold temperament.
8. Sci-fi art portrait: Cold hard light, metal low saturation color scheme, futuristic minimalist set, tough male body and clear metal fabric texture.
All styles: user-specified content takes priority, only supplement professional light, skin, stubble and fabric details, without altering user's theme and atmosphere.
"""
                },
                "negative_base": {
                    "zh": "完美对称五官，过度磨皮，塑胶假肤，AI模板脸，无胡茬光滑面部，僵硬摆拍，空洞假笑，肢体畸形，柔弱纤细体态，坏手多手指，画面杂乱，透视畸变，高饱和艳色，二次元卡通质感，强光曝光异常",
                    "en": "perfect symmetrical face, excessive skin smoothing, plastic wax skin, AI template face, smooth face without stubble, stiff pose, empty fake smile, deformed limbs, weak slender figure, bad hands extra fingers, cluttered frame, perspective distortion, oversaturated color, anime cartoon style, harsh light abnormal exposure"
                }
            }
        }

        # 输出格式指引
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/棚拍硬照/环境人像/抓拍/摆拍等）与视角构图方式（非常规视角/平视/俯拍/仰拍/斜侧/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（强烈阳光/柔和窗光/逆光/侧光/顶光/霓虹灯/混合光源等），以及光线在人物头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感/粒子散落/动态模糊边缘/明暗渐变过渡/HDR高动态/高饱和强对比等），用定性光影语汇替代光学数值。

第三段·人物姿态与神情：完整描述头部、躯干、四肢的具体姿态（站/坐/蹲/倚靠/手持道具等），视线方向与镜头关系，面部表情神态（沉稳/果敢/沉思/微笑等），以及风吹发丝飘动等动态细节。

第四段·面部细节与发型妆造：精细刻画面部五官特征（轮廓/眉眼/唇色/肤质/胡茬），皮肤质感（真实毛孔/光泽/岁月纹理），妆容风格（自然裸妆/干净清爽等），发型发色与打理方式。

第五段·服饰配件与环境：描述穿搭细节（衣款/面料/颜色/花纹/配饰如手表项链等），互动道具（饮品/公文包/书籍等），以及所处环境场景（室内/户外/办公室/街景等），含远景元素，最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/studio hard light/environmental/candid, angle/framing); 2) Lighting atmosphere (source type, direction, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain, particles, motion blur edges, HDR, high saturation contrast); 3) Full pose & expression (head/torso/limbs position, gaze direction, facial emotion, dynamic details); 4) Face details & styling (facial features, stubble, skin texture, hairstyle); 5) Outfit accessories & environment (clothing details, props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；照片级细腻肤质、毛孔纹理与通透光泽清晰可见，胡茬颗粒感分明；发丝根根分明、蓬松自然。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：沉稳/果敢/自信/柔和），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

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
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Photo-level fine skin texture with visible pore detail and translucent glow, defined stubble grain; strand-defined fluffy hair.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: composed / resolute / confident / soft), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

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
