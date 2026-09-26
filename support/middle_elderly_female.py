# -*- coding: utf-8 -*-
"""
真实老年女性人像预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

MIDDLE_ELDERLY_FEMALE = {
    "template_id": "middle_elderly_female",
    "name": "中老年女性人像",
    "description": "专业中老年女性超写实人像摄影指导，仅覆盖40岁以上中年、中老年、高龄女性，涵盖居家纪实、国风旗袍、复古胶片、轻商务、极简棚拍、艺术人像等熟龄专属题材。兼容亚洲/欧美中老年女性松弛五官、岁月肤质、花白银发特征，原生淡妆干净无大面积瑕疵，完整保留深浅皱纹、淡老年斑、面部松弛肌理，杜绝塑胶假肤、AI模板脸、年轻化过度磨皮。语义权重优先级：面部肤质岁月约束＞中老年五官/银发/熟龄体态服饰＞光影色彩氛围＞场景构图＞摄影参数。所有风格坚守熟龄真人写实基线，仅氛围、造型、光影差异化，姿态松弛舒缓无夸张变形，全程不涉及青年、少女刻画逻辑。超写实人像摄影提示词扩写",
}

class MiddleElderlyFemale:
    def __init__(self):
        # 下游生图模型内容组织公式库（完全沿用参考原版无改动）
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
"formula_zh": "内容组织顺序：整体画面氛围光影 → 人物气质姿态 → 肌肤发丝质感 → 背景留白。侧重氛围叙事，弱化细碎关键词堆砌，画面柔和高级，写实肤质发丝高度细致，纹理清晰可见。",
                 "formula_en": "Content order: overall atmosphere lighting → character pose → skin hair texture → negative space. Focus on atmospheric narration, realistic skin and hair highly detailed with clear texture."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：一位超写实女性（年龄、发型、妆容、服饰与神态）→ 照片级写实与皮肤质感 → 自然光或棚拍布光、清新氛围 → 半身特写、浅景深，逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: a photorealistic female (age, hairstyle, makeup, outfit and expression) → photo-level realism and skin texture → natural or studio lighting, fresh atmosphere → half-body close-up, shallow depth of field, realistic, highly detailed skin and hair texture"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：一位超写实女性主体（年龄、发型、妆容、服饰）→ 照片级写实与皮肤质感 → 柔和自然光或棚拍布光、清新氛围 → 半身或特写、浅景深；建议描述她的神态与目光（需渲染文字直接写入，支持中英双语），逼真，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: a photorealistic female subject (age, hairstyle, makeup, outfit) → photo-level realism with skin texture → soft natural or studio lighting, fresh atmosphere → half-body or close-up, shallow depth of field; describe her expression and gaze (write any rendered text directly, supports Chinese and English), realistic, highly detailed skin and hair texture."
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：女性身份、年龄与神态表情、超写实与电影级皮肤质感 → 风格与画质（真实肤质、发丝细节、柔焦景深） → 自然光或窗光勾勒轮廓与氛围 → 半身或特写构图、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: female identity, age and expression, photorealistic cinematic skin texture → style and quality (real skin, hair strand details, soft-focus depth of field) → natural or window light outlining silhouette and atmosphere → half-body or close-up composition, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture"
            },
            "Qwen_Image2.1": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：女性身份、年龄与神态表情、超写实与电影级皮肤质感 → 风格与画质（真实肤质、发丝细节、柔焦景深） → 自然段落混排冒号标签块（姿势动作：/表情：/发型发色：/穿搭：/场景：）与光影效果关键词列表（光影斑驳/边缘发光/高光溢出/胶片颗粒等） → 自然光或窗光勾勒轮廓与氛围 → 半身或特写构图、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致；支持负向提示词通道，肢体畸变、解剖错误、坏手等缺陷写入负向提示词，正向只做纯加法",
                "formula_en": "Content order: female identity, age and expression, photorealistic cinematic skin texture → style and quality (real skin, hair strand details, soft-focus depth of field) → natural paragraph mixed with colon label blocks (pose:/expression:/hairstyle:/outfit:/scene:) and lighting effect keyword list (dappled light, rim glow, highlight bloom, film grain, etc.) → natural or window light outlining silhouette and atmosphere → half-body or close-up composition, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture; supports a negative prompt channel, write limb distortion, anatomical errors and bad hands into the negative prompt, keep the positive prompt purely additive"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：开篇（风格画质定调，或人物主体直接切入）→ 人物主体（发型发色、五官神态、妆容、肤质纹理、姿态动作与手部细节）→ 服饰配件（款式剪裁、面料材质纹理、鞋履配饰）→ 背景环境（空间层次、材质细节、远景虚化）→ 光影（光源方向、明暗过渡、高光位置）→ 色彩基调（主色与点缀色）→ 画质与构图（焦点位置、背景虚化、微细节清晰度、杂志封面质感）→ 整体氛围收尾。以连贯自然语言分段散文为主，细节密集但不堆砌标签；亦可用密集关键词加权重、质量词置头尾的中英混排写法；画面含文字时直接写出文字内容，无文字时以“画面中无文字”声明收尾；可选附相机镜头技术参数段（焦距、光圈、快门、ISO）。",
                "formula_en": "Content order: opening (style and quality statement, or straight to the subject) → subject (hairstyle, facial features, makeup, skin texture, pose and hand details) → outfit and accessories (cut, fabric texture, shoes, jewelry) → background (spatial layers, material details, distant blur) → lighting (light direction, light-to-shadow transitions, highlight placement) → color palette (main and accent colors) → quality and composition (focus point, background bokeh, micro-detail sharpness, magazine-cover feel) → closing atmosphere. Write as flowing natural-language paragraphs with dense detail but no tag stacking; a dense weighted-tag style with quality tags at head and tail (Chinese-English mixed) also works; if the image contains text, write it out directly, otherwise end with a 'No text present in image' statement; optionally append a camera and lens technical block (focal length, aperture, shutter speed, ISO)."
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
"formula_zh": "内容组织顺序：整体画面基调（温暖柔和氛围）→ 人物松弛姿态与神情 → 自然肌肤质感与细节 → 简约留白环境，写实风格，肤质发丝高度细致，纹理清晰。",
                 "formula_en": "Content order: overall image tone (warm and soft atmosphere) → relaxed pose with expression → natural skin texture and details → simple negative-space environment, realistic style, highly detailed skin and hair with clear texture."
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：面部五官肤质、年龄气质 → 体态姿态、松弛神情 → 光影层次、柔光窗光 → 服饰细节、面料质感 → 轻量环境、minimal background（密集关键词，中英术语并列），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: facial features and skin, age and temperament → body pose, relaxed expression → lighting layers, soft window light → clothing details, fabric texture → lightweight environment, minimal background (dense keywords, Chinese-English terms in parallel), realistic, highly detailed skin and hair texture"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：女性身份、年龄与神态、超写实电影质感 → 风格与画质（真实肤质、发丝与微表情） → 自然光或窗光勾勒轮廓氛围 → 半身或特写、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: female identity, age and expression, photorealistic cinematic texture → style and quality (real skin, hair strands and micro-expressions) → natural or window light outlining silhouette and atmosphere → half-body or close-up, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：超写实女性面部与半身肖像 → 高清写实摄影风格、皮肤肌理与发丝质感 → 柔和窗光与暖调氛围 → 浅景深特写、眼神平视 → 强调真实无磨皮、避免卡通与畸变。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词，肤质发丝高度细致。",
                "formula_en": "Content order: photorealistic female face and half-body portrait → high-definition realistic photography style, skin texture and hair details → soft window light and warm tone → shallow depth close-up, eye-level gaze → emphasize real un-retouched skin, avoid cartoon and distortion. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt, highly detailed skin and hair texture."
            },
            "LongCat_Image": {
                "keyword_dense": False,
                "mix_lang": False,
"formula_zh": "内容组织顺序：主体衣着与特质描写 → 神态与动作刻画 → 环境与背景交代 → 光线与氛围渲染 → 景别与构图说明。纯中文长自然语言描述效果最佳，需渲染文字用引号包裹，肤质发丝高度细致，纹理清晰可见。",
                 "formula_en": "Content order: subject clothing & traits → expression & action → environment & background → light & atmosphere → shot & composition. Long Chinese natural language describes best; wrap any rendered text in quotation marks, highly detailed skin and hair with clear texture."
            },
            "HiDream-O1-Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：超写实女性主体与神态表情 → 场景与构图（浅景深特写）→ 光影与氛围（柔和窗光暖调）→ 画种/摄影风格（高清写实摄影）→ 需渲染文字用引号包裹，肤质发丝高度细致。",
                "formula_en": "Content order: photorealistic female subject & expression → scene & composition (shallow depth close-up) → light & atmosphere (soft window light, warm tone) → art/photography style (high-definition realistic photography) → wrap rendered text in quotes, highly detailed skin and hair texture."
            }
        }
        # 全局底层规则，修改为中老年女性专用
        self.global_base_rules = {
            "zh": """
你是专业高端中老年女性超写实人像摄影提示词扩写专家，本模板为【中老年熟龄女性人像专用】，全覆盖：居家纪实、国风旗袍、现代轻商务、复古胶片、极简棚拍、艺术人体等熟龄专属题材。
所有风格坚守**40岁以上熟龄真人超写实基线**，仅造型、光影、色调、氛围差异化，绝不出现二次元、插画、油画质感，禁止刻画青年、少女、学生群体。
原生淡妆状态面部干净整洁，无大面积杂乱瑕疵，完整保留中老年原生深浅皱纹、面部软组织松弛、淡老年斑、细微干纹与肤色不均，拒绝过度磨皮带来的塑胶假肤、紧致年轻化肌肤效果。
姿态必须使用具体中老年舒缓肢体结构描述，禁止模糊形容词、少女活泼夸张动作；光线方向明确，光影过渡柔和通透；人物为绝对画面主体，环境仅衬托岁月叙事氛围。
完整保留用户输入的风格、服饰、场景、色调、姿态、视角所有信息，仅补充摄影、材质、光影、岁月肤质、花白银发专业细节，不新增少女、青春相关无关物体、装饰道具。
所有服饰（旗袍/羊绒大衣/宽松居家棉麻/艺术人体）均作为成熟女性高端人像题材，姿态克制温婉、松弛自然。
输出禁忌：禁止权重符号、多余相机参数、冗余堆砌；禁止卡通二次元、畸形肢体、坏手烂指、网红假脸、磨皮蜡皮；禁止杂乱少女风背景、空洞甜腻假笑、自拍抓拍、透视畸变、强行紧致年轻化。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional photorealistic portrait prompt expert exclusively for middle-aged and elderly women. This preset covers all mature themes: daily home documentary, Chinese cheongsam, light business, retro film, minimalist studio, artistic nude portrait.
All styles strictly adhere to photorealistic baseline for women aged 40+, differentiated only by styling, lighting, tone and atmosphere, no illustration, anime or oil painting texture; depictions of young girls and teenagers are forbidden.
Light natural makeup, clean face without large blemishes, fully retain natural wrinkles, sagging facial tissue, faint age spots, fine dry lines and uneven skin tone of mature skin; reject plastic fake skin and artificially tightened youthful skin caused by heavy retouching.
All poses described with specific relaxed limb structure for elders, no vague words or exaggerated youthful movements. Clear light direction and soft shadow transition. Elderly female subject dominates the frame, background only serves aging narrative atmosphere.
Completely retain user input style, clothing, scene, tone, pose and perspective. Only supplement professional details of photography, texture, light, aged skin and gray-white hair, no irrelevant youthful decorations or props.
All costumes (cheongsam, cashmere coat, loose linen home wear, artistic nude) are high-end mature portrait themes with restrained gentle relaxed poses.
Forbidden: no weight symbols, redundant camera parameters, anime/cartoon/illustration, deformed anatomy, defective hands, over-retouched wax skin, messy youthful background, empty sweet fake smile, snapshot selfie, perspective distortion, forced youthful facial tightening.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定中老年专用template_id，完整复刻参考内preset_library结构
        self.preset_library = {
            "middle_elderly_female": {
                "template_id": "middle_elderly_female",
                "display_name": MIDDLE_ELDERLY_FEMALE["name"],
                "description": MIDDLE_ELDERLY_FEMALE["description"],
                # 中英双语前置约束，中老年专属
                "positive_constraints": {
                    "zh": "超写实真人质感，40岁以上中老年女性松弛面部骨骼，眉眼唇皱纹分布轻微不对称；原生淡妆干净，保留深浅皱纹、面部松弛肌理、淡老年斑、肤色不均、细微干纹，无过度磨皮与塑胶假肤、无AI模板脸；柔软花白发丝，干净布景，松弛舒缓抓拍姿态，内敛沉静成熟情绪，无透视畸变。居家/旗袍/胶片/棚拍/艺术人体均为熟龄题材分支，保持岁月写实基线，姿态温婉自然",
                    "en": "photorealistic real human texture, sagging facial bone structure unique to women over 40, natural slight asymmetry of eyes, eyebrows, lips and wrinkle distribution; light natural makeup, retain deep & shallow wrinkles, sagging facial texture, faint age spots, uneven skin tone, fine dry lines, no over-smoothing or plastic wax skin or AI template face; soft gray-white hair, restrained daily scene, relaxed snapshot pose, calm mature emotion, no perspective distortion. Daily/cheongsam/film/studio/artistic nude are mature theme branches only, maintain aging realism baseline with gentle posture"
                },
                # 中老年全风格细分专属规则
                "preset_rules": {
                    "zh": """
【中老年女性人像专属规则】
1. 通用基线：仅刻画40岁-70+熟龄女性，完整保留皱纹、面部松弛、淡老年斑、花白头发等原生年龄特征，禁止磨皮淡化、强行紧致年轻化；原生淡妆干净无大面积瑕疵，留存岁月肌理与自然毛孔，杜绝完美对称五官、蜡像假肤。
2. 亚洲中老年女性刻画：柔和圆润松弛面部轮廓，平缓骨骼线条，浅淡分散老年斑，花白发丝层次柔软蓬松；适配居家纪实、新中式旗袍、茶室场景，主用光为窗纱漫射柔光、室内暖调灯光，低饱和大地沉稳色系。
3. 欧美中老年女性刻画：立体骨骼、深眼窝、松弛清晰下颌线，立体沟壑岁月纹理，银灰分层短发；适配画廊、轻商务极简场景，多用侧方轮廓柔光、冷调漫射天光，低饱和灰调色系。
4. 国风旗袍熟龄风格：选用棉麻、重磅真丝宽松成熟旗袍版型，体态舒缓端庄，无紧身夸张剪裁，配色酒红、藏蓝、驼色沉稳色系，庭院、茶室柔光纪实拍摄。
5. 复古胶片纪实风格：带有轻微自然胶片颗粒，暖调褪色柔光，不刻意精修淡化皱纹，场景选用老式民居、老街，还原生活化松弛抓拍质感。
6. 极简棚拍熟龄风格：低饱和纯色简约背景，均匀柔光铺光，重点突出面部岁月肌理与银发层次；服饰以羊绒、针织、宽松通勤外套为主。
7. 居家日常纪实风格：宽松棉麻家居服饰，松弛坐卧体态，午后自然漫射阳光，搭配木家具、针线、旧书本等中老年专属生活道具。
8. 艺术人体熟龄风格：纯白极简摄影空间，侧逆光勾勒成熟松弛身体曲线，完整保留全身岁月肌肤纹理，姿态沉静内敛、优雅克制。
所有题材仅围绕熟龄女性创作，用户指定场景、服饰、视角优先保留，仅补充岁月肤质、银发、成熟光影细节，不新增少女、青年相关元素。
""",
                    "en": """
【Exclusive Rules for Middle-Aged and Elderly Female Portraits】
1. General baseline: Only depict women aged 40 to 70+, fully retain original aging marks including wrinkles, facial sagging, faint age spots, gray hair; prohibit smoothing or forced youthful tightening. Light natural makeup without large blemishes, retain aged texture and natural pores, reject perfectly symmetrical facial features and wax fake skin.
2. Asian middle-aged & elderly women: Soft round sagging facial contour, gentle bone lines, faint scattered age spots, soft layered gray hair. Suitable for daily home records, new Chinese cheongsam, teahouse scenes; light source: diffused window soft light, warm indoor lamp, low-saturation earth tone palette.
3. Western middle-aged & elderly women: Stereoscopic bone structure, deep eye sockets, clear sagging jawline, three-dimensional facial aging lines, layered silver-gray short hair. Suitable for galleries, minimalist light business scenes, side contour soft light, cool diffuse natural light, low-saturation gray color system.
4. Chinese cheongsam style for elder women: Loose mature cheongsam made of linen and heavy silk, dignified relaxed posture without tight exaggerated cuts, stable color matching including wine red, navy and camel, soft light shooting in courtyards and teahouses.
5. Retro film documentary style: Slight natural film grain, warm faded soft light, no retouching to erase wrinkles, old houses and old streets as shooting scenes, relaxed daily snapshot texture.
6. Minimal studio style for elder women: Low-saturation solid simple background, even soft box lighting, focus on facial aging texture and silver hair layers; costumes are mainly cashmere, knitwear and loose commuter coats.
7. Daily home documentary style: Loose linen home wear, relaxed sitting and lying posture, afternoon diffuse natural sunlight, daily props for elders such as wooden furniture, needlework and old books.
8. Artistic nude mature style: Pure white minimalist photo space, side backlight outlines mature relaxed body curves, fully retain aged skin texture all over body, calm restrained elegant posture.
All themes are only created for mature women, retain user-specified scenes, costumes and perspectives, only add details of aged skin, silver hair and mature light, no elements related to young girls or teenagers.
"""
                },
                "negative_base": {
                    "zh": "少女青年粉嫩肌肤，马卡龙亮色，紧致少女轮廓，完美对称五官，零皱纹无老年斑，过度磨皮，塑胶假肤，AI模板脸，僵硬摆拍，空洞假笑，夸张肢体，畸形手脚多手指，透视畸变，高饱和荧光艳色，杂乱少女装饰，二次元卡通画风，强行年轻化，乌黑假发，年龄感丢失",
                    "en": "Young girl teenager pink tender skin, macaron bright colors, tight youthful contour, perfectly symmetrical face, wrinkle-free no age spots, over-smoothed plastic wax skin, AI template face, stiff pose, empty fake smile, exaggerated limbs, deformed hands feet extra fingers, perspective distortion, oversaturated fluorescent colors, messy girlish decorations, anime cartoon art style, forced youthful tightening, black wig lost aging texture"
                }
            }
        }
        # 双输出格式指引（完全沿用参考原版无改动）
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/居家纪实/户外散步/棚拍摆拍等）与视角构图方式（非常规视角/平视/俯拍/仰拍/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（强烈阳光/柔和窗光/暖调灯光/逆光/侧光等），以及光线在人物头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感/明暗渐变过渡等），用定性光影语汇替代光学数值。

第三段·人物姿态与神情：完整描述头部、躯干、四肢的具体姿态（站/坐/倚靠/手持道具等），视线方向与镜头关系，面部表情神态（从容/慈祥/沉思/微笑等），以及银发随风飘动等动态细节。

第四段·面部细节与发型妆造：精细刻画面部五官特征（轮廓/眼型/唇色/肤质），皮肤质感（岁月皱纹/老年斑/松弛肌理/自然光泽），妆容风格（淡雅自然/素颜等），发型发色（银发/花白/盘发/短发等）与打理方式。

第五段·服饰配件与环境：描述穿搭细节（衣款/面料/颜色/花纹/配饰如胸针手镯等），互动道具（茶杯/书籍/花束等），以及所处环境场景（室内/庭院/公园/水边等），含远景元素（树木/山丘/建筑等），最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/indoor daily/outdoor walk/studio, angle/framing); 2) Lighting atmosphere (source type, direction, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain, gradient transition); 3) Full pose & expression (head/torso/limbs position, gaze direction, facial emotion, silver hair wind-blown details); 4) Face details & styling (facial features, wrinkles, age spots, skin texture, hairstyle); 5) Outfit accessories & environment (clothing details, props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度对齐参考示例，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；皮肤可见岁月皱纹（鱼尾纹/法令纹/额头纹）、老年斑与松弛肌理，保留自然光泽；银发或花白发色，发丝蓬松空气感。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：从容/慈祥/沉思/微笑），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

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
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Visible age wrinkles (crow's feet / nasolabial / forehead lines), age spots and loose skin texture with natural glow; silver or salt-and-pepper hair with airy volume.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: unhurried / kindly / thoughtful / smiling), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

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


