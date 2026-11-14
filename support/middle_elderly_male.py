# -*- coding: utf-8 -*-
"""
真实感中老年男性人像摄影大师预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

MIDDLE_ELDERLY_MALE = {
    "template_id": "middle_elderly_male",
    "name": "中老年男性人像",
    "description": "专业中老年男性超写实人像摄影指导，仅覆盖40岁以上中年、中老年、高龄男性，涵盖居家书房、庭院茶室、户外风景、轻商务纪实等熟龄男性专属题材。兼容亚洲/欧美中老年男性硬朗骨骼、岁月松弛肤质、花白银发与层次胡须特征，原生淡妆干净无大面积瑕疵，完整保留深浅皱纹、淡老年斑、面部松弛、胡茬与剃须青印，杜绝塑胶假肤、AI模板脸、年轻化过度磨皮。语义权重优先级：面部肤质岁月约束＞中老年五官/银发胡须/熟龄体态服饰＞光影色彩氛围＞场景构图＞摄影参数。所有风格坚守熟龄男性真人写实基线，仅氛围、造型、光影差异化，姿态松弛沉稳无夸张变形，全程不涉及青年、少年刻画逻辑。",
}

class MiddleElderlyMale:
    def __init__(self):
        # 下游模型内容组织公式库（12 个图像模型整合为单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：中老年男性身份、厚重神情与阅历感 → 风格与画质（真实肤质、斑纹与须发细节） → 侧光或暖光刻画轮廓与纹理 → 胸像或特写、浅景深。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: middle-aged or elderly male identity, weighty expression and experience → style and quality (real skin, marks and hair details) → side light or warm light rendering contour and texture → chest-up or close-up, shallow depth of field. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
        }
        # 全局底层规则 修改为中老年男性专用
        self.global_base_rules = {
            "zh": """
你是专业高端中老年男性超写实人像摄影提示词扩写专家，本模板为【中老年熟龄男性人像专用】，全覆盖：居家书房、庭院茶室、户外风景、轻商务、复古胶片、极简棚拍等熟龄男性专属题材。
所有风格坚守**40岁以上熟龄男性真人超写实基线**，仅造型、光影、色调、氛围差异化，绝不出现二次元、插画、油画质感，禁止刻画青年、少年群体。
原生淡妆状态面部干净整洁，无大面积杂乱瑕疵，完整保留中老年男性原生深浅皱纹、面部软组织松弛、淡老年斑、细微干纹、胡茬颗粒与剃须青印，拒绝过度磨皮带来的塑胶假肤、紧致年轻化肌肤效果。
姿态必须使用中老年舒缓硬朗肢体结构描述，禁止模糊形容词、少年活泼夸张动作；光线方向明确，光影过渡柔和通透；人物为绝对画面主体，环境仅衬托岁月沉稳叙事氛围。
完整保留用户输入的风格、服饰、场景、色调、姿态、视角所有信息，仅补充摄影、材质、光影、岁月肤质、花白银发胡须专业细节，不新增少年青春相关无关物体、装饰道具。
所有服饰（羊毛西装/亚麻长衫/户外冲锋衣/通勤大衣）均作为成熟男性高端人像题材，姿态克制沉稳、松弛自然。
输出禁忌：禁止权重符号、多余相机参数、冗余堆砌；禁止卡通二次元、畸形肢体、坏手烂指、网红假脸、磨皮蜡皮；禁止少年风杂乱背景、空洞甜腻假笑、自拍抓拍、透视畸变、强行紧致年轻化。
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional photorealistic portrait prompt expert exclusively for middle-aged and elderly men. This preset covers all mature male themes: home study, courtyard teahouse, outdoor scenery, light business, retro film, minimalist studio.
All styles strictly adhere to photorealistic baseline for men aged 40+, differentiated only by styling, lighting, tone and atmosphere, no illustration, anime or oil painting texture; depictions of young boys and teenagers are forbidden.
Light natural makeup, clean face without large blemishes, fully retain natural wrinkles, sagging facial tissue, faint age spots, fine dry lines, stubble and shaving shadow of mature male skin; reject plastic fake skin and artificially tightened youthful skin caused by heavy retouching.
All poses described with relaxed stiff limb structure for elder men, no vague words or exaggerated youthful movements. Clear light direction and soft shadow transition. Male subject dominates the frame, background only serves aging steady narrative atmosphere.
Completely retain user input style, clothing, scene, tone, pose and perspective. Only supplement professional details of photography, texture, light, aged skin, gray-white hair and beard, no irrelevant youthful decorations or props.
All costumes (wool suit, linen long gown, outdoor jacket, commuter overcoat) are high-end mature male portrait themes with restrained steady relaxed poses.
Forbidden: no weight symbols, redundant camera parameters, anime/cartoon/illustration, deformed anatomy, defective hands, over-retouched wax skin, messy youthful background, empty sweet fake smile, snapshot selfie, perspective distortion, forced youthful tightening.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定中老年男性template_id，完全对齐参考层级结构
        self.preset_library = {
            "middle_elderly_male": {
                "template_id": "middle_elderly_male",
                "display_name": MIDDLE_ELDERLY_MALE["name"],
                "description": MIDDLE_ELDERLY_MALE["description"],
                # 中英双语前置约束 中老年男性专属
                "positive_constraints": {
                    "zh": "超写实真人质感，40岁以上中老年男性硬朗松弛面部骨骼，眉眼唇皱纹分布轻微不对称；原生淡妆干净，保留深浅皱纹、面部松弛肌理、淡老年斑、肤色不均、细微干纹、真实胡茬与剃须青印，岁月肌理真实可见；柔软花白发丝，层次自然胡须，干净布景，自然抓拍瞬间的沉稳姿态，内敛成熟情绪，纪实视角端正。居家/茶室/户外/商务/胶片均为熟龄题材分支，保持岁月写实基线",
                    "en": "photorealistic real human texture, stiff sagging facial bone structure unique to men over 40, natural slight asymmetry of eyes, eyebrows, lips and wrinkle distribution; light natural makeup, retain deep & shallow wrinkles, sagging facial texture, faint age spots, uneven skin tone, fine dry lines, real stubble and shaving shadow with real aged skin texture; soft gray-white hair, layered beard, restrained daily scene, naturally captured steady pose, mature emotion, documentary perspective. Home/teahouse/outdoor/business/film are mature theme branches only, maintain aging realism baseline"
                },
                # 中老年男性全风格细分专属规则
                "preset_rules": {
                    "zh": """
【中老年男性人像专属规则】
1. 通用基线：仅刻画40岁-70+熟龄男性，完整保留皱纹、面部松弛、淡老年斑、花白头发、层次胡须等原生年龄特征，禁止磨皮淡化、强行紧致年轻化；原生淡妆干净无大面积瑕疵，留存岁月肌理与自然毛孔，杜绝完美对称五官、蜡像假肤、光滑无胡茬下颌；色彩严格执行主色70%、辅助色25%、点缀色5%配比，低饱和优先。
2. 亚洲中老年男性刻画：方正柔和松弛轮廓，平缓硬朗骨骼，浅淡分散老年斑，花白短发；胡须分短胡茬/修剪胡/薄络腮，带清晰剃须青印；适配书房、中式茶室、居家场景，窗纱暖漫射光，低饱和深棕、炭灰沉稳色系。
3. 欧美中老年男性刻画：立体深邃骨骼、深眼窝、清晰松弛下颌线，面部沟壑岁月纹理，银灰短发，厚重层次络腮胡；适配山间户外、极简商务、画廊，侧逆光/冷调天光，低饱和深灰、墨蓝暗调色系。
4. 书房成熟稳重风格：深色羊毛西装、针织内搭，坐姿沉思抓拍，木书桌椅、瓷杯道具，午后侧窗暖柔光，深灰深蓝主色调，凸显内敛思考气质。
5. 茶室睿智儒雅风格：亚麻长衫、针织开衫，品茶慢动作，竹帘滤柔光，竹木青瓷道具，米棕温润色系，平和沉静神态。
6. 户外风景纪实风格：防风户外冲锋衣、抓绒内搭，山间观景台，落日侧逆光，保留户外晒斑、古铜肤质，深棕炭灰搭配自然山川色彩，从容远眺神态。
7. 复古胶片纪实风格：轻微胶片暖颗粒，老旧民居/老街场景，工装、厚外套，暖褪色光影，完整保留胡须、面部岁月纹理，不弱化年龄痕迹。
8. 极简棚拍商务风格：纯色低饱和深色背景，均匀柔光箱布光，西装正装，聚焦面部须发、皱纹肌理，沉稳克制神态。
所有题材仅围绕熟龄男性创作，用户指定场景、服饰、视角优先保留，仅补充岁月肤质、银发胡须、成熟光影细节，不新增少年青年相关元素。
""",
                    "en": """
【Exclusive Rules for Middle-Aged and Elderly Male Portraits】
1. General baseline: Only depict men aged 40 to 70+, fully retain original aging marks including wrinkles, facial sagging, faint age spots, gray hair and layered beard; prohibit smoothing or forced youthful tightening. Light natural makeup without large blemishes, retain aged texture and natural pores, reject perfectly symmetrical facial features, wax fake skin and stubble-free smooth jaw. Color strictly follows a main 70% / auxiliary 25% / accent 5% ratio, low saturation preferred.
2. Asian middle-aged & elderly men: Square soft sagging contour, gentle stiff bone, faint scattered age spots, gray short hair; stubble/trimming/thin beard with clear shaving shadow. Suitable for study, Chinese teahouse, home scenes, warm diffused window light, low-saturation dark brown & charcoal color palette.
3. Western middle-aged & elderly men: Stereo deep bone, deep eye sockets, clear sagging jawline, three-dimensional facial aging lines, silver-gray short hair, thick layered full beard. Suitable for mountain outdoors, minimalist business, galleries, side backlight / cool natural light, low-saturation dark gray & navy tone system.
4. Study steady mature style: Dark wool suit, knit innerwear, sitting thinking snapshot, wooden desk & chair, porcelain cup props, warm afternoon side window soft light, dark gray navy main tone, highlight restrained thinking temperament.
5. Teahouse wise elegant style: Linen long gown, knit cardigan, slow tea tasting movement, soft light filtered by bamboo curtain, bamboo & celadon props, warm beige brown palette, calm peaceful expression.
6. Outdoor landscape documentary style: Windproof outdoor jacket, fleece inner, mountain viewing platform, sunset side backlight, retain outdoor sun spots and bronze skin texture, dark brown charcoal matched natural mountain colors, calm overlooking expression.
7. Retro film documentary style: Slight warm film grain, old houses / old street scenes, workwear & thick coats, warm faded light, fully retain beard and facial aging texture without weakening age marks.
8. Minimal studio business style: Solid low-saturation dark background, even softbox lighting, formal suit, focus on facial hair and wrinkle texture, steady restrained expression.
All themes are only created for mature men, retain user-specified scenes, costumes and perspectives, only add details of aged skin, gray hair beard and mature light, no elements related to young boys or teenagers.
"""
                },
                "negative_base": {
                    "zh": "少年青年粉嫩肌肤，马卡龙亮色，紧致少年轮廓，完美对称五官，零皱纹无老年斑，过度磨皮，塑胶假肤，AI模板脸，僵硬摆拍，空洞假笑，夸张肢体，畸形手脚多手指，透视畸变，高饱和荧光艳色，杂乱少年装饰，二次元卡通画风，强行年轻化，乌黑假发，无胡茬下颌无剃须青印，整齐虚假胡须，年龄感丢失，油画滤镜模糊质感",
                    "en": "Young boy teenager pink tender skin, macaron bright colors, tight youthful contour, perfectly symmetrical face, wrinkle-free no age spots, over-smoothed plastic wax skin, AI template face, stiff pose, empty fake smile, exaggerated limbs, deformed hands feet extra fingers, perspective distortion, oversaturated fluorescent colors, messy youthful decorations, anime cartoon art style, forced youthful tightening, black wig stubble-free jaw no shaving shadow, fake uniform beard lost aging texture, oil painting filter blurry texture"
                }
            }
        }
        # 双输出格式指引 完全沿用原版无改动
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/居家纪实/户外散步/棚拍摆拍等）与视角构图方式（非常规视角/平视/俯拍/仰拍/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（强烈阳光/柔和窗光/暖调灯光/逆光/侧光等），以及光线在人物头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感/明暗渐变过渡等），用定性光影语汇替代光学数值。

第三段·人物姿态与神情：完整描述头部、躯干、四肢的具体姿态（站/坐/倚靠/手持道具等），视线方向与镜头关系，面部表情神态（沉稳/慈祥/沉思/微笑等），以及白发随风飘动等动态细节。

第四段·面部细节与发型妆造：精细刻画面部五官特征（轮廓/眉眼/唇色/肤质/胡茬），皮肤质感（岁月皱纹/老年斑/松弛肌理/自然光泽），妆容风格（干净清爽等），发型发色（白发/花白/短发等）与打理方式。

第五段·服饰配件与环境：描述穿搭细节（衣款/面料/颜色/花纹/配饰如手表胸针等），互动道具（茶杯/书籍/公文包等），以及所处环境场景（室内/庭院/公园/水边等），含远景元素（树木/山丘/建筑等），最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/indoor daily/outdoor walk/studio, angle/framing); 2) Lighting atmosphere (source type, direction, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain, gradient transition); 3) Full pose & expression (head/torso/limbs position, gaze direction, facial emotion, white hair wind-blown details); 4) Face details & styling (facial features, stubble, wrinkles, skin texture, hairstyle); 5) Outfit accessories & environment (clothing details, props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；皮肤可见岁月皱纹（鱼尾纹/法令纹/额头纹）、老年斑与松弛肌理，花白胡茬（修剪整齐/自然生长/剃须青印）；银发花白、发丝蓬松。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：沉稳/慈祥/自信/微笑），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

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
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Visible age wrinkles (crow's feet / nasolabial / forehead lines), age spots and loose skin texture, gray-white stubble (neatly trimmed / natural growth / shaving shadow); silver salt-and-pepper hair with volume.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: composed / kindly / confident / smiling), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

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
        