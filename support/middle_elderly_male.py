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
        # 下游生图模型内容组织公式库 完全沿用参考原版无改动
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
                "formula_zh": "内容组织顺序：一位超写实男性（年龄、发型、妆容、服饰与神态）→ 照片级写实与皮肤质感 → 自然光或棚拍布光、清新氛围 → 半身特写、浅景深，逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: a photorealistic male (age, hairstyle, makeup, outfit and expression) → photo-level realism and skin texture → natural or studio lighting, fresh atmosphere → half-body close-up, shallow depth of field, realistic, highly detailed skin and hair texture"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：一位超写实男性主体（年龄、发型、妆容、服饰）→ 照片级写实与皮肤质感 → 柔和自然光或棚拍布光、清新氛围 → 半身或特写、浅景深；建议描述他的神态与目光（需渲染文字直接写入，支持中英双语），逼真，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: a photorealistic male subject (age, hairstyle, makeup, outfit) → photo-level realism with skin texture → soft natural or studio lighting, fresh atmosphere → half-body or close-up, shallow depth of field; describe his expression and gaze (write any rendered text directly, supports Chinese and English), realistic, highly detailed skin and hair texture."
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：男性身份、年龄与神态表情、超写实与电影级皮肤质感 → 风格与画质（真实肤质、发丝细节、柔焦景深） → 自然光或窗光勾勒轮廓与氛围 → 半身或特写构图、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: male identity, age and expression, photorealistic cinematic skin texture → style and quality (real skin, hair strand details, soft-focus depth of field) → natural or window light outlining silhouette and atmosphere → half-body or close-up composition, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：景别构图（特写/半身/全身）→ 人物主体（发型发色、五官特征、妆容风格、神情表情、姿态动作）→ 服饰配件（衣款面料、配饰道具）→ 光影氛围（光源类型、光影视觉效果）→ 环境场景（所处空间、远景元素）→ 风格（画面风格定位）→ 画质（浅景深虚化/柔焦/锐化）→ 极简布景（密集关键词，中英术语并列），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: shot composition (close-up/half-body/full-body) → subject (hairstyle, facial features, makeup, expression, pose) → outfit & accessories (clothing, props) → lighting atmosphere (source type, visual effects) → environment (space, background elements) → style (image style) → quality (shallow DOF/soft focus/sharpness) → minimalist set (dense keywords, Chinese-English terms in parallel), realistic, highly detailed skin and hair texture"
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
                "formula_zh": "内容组织顺序：男性身份、年龄与神态、超写实电影质感 → 风格与画质（真实肤质、发丝与微表情） → 自然光或窗光勾勒轮廓氛围 → 半身或特写、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: male identity, age and expression, photorealistic cinematic texture → style and quality (real skin, hair strands and micro-expressions) → natural or window light outlining silhouette and atmosphere → half-body or close-up, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：超写实男性面部与半身肖像 → 高清写实摄影风格、皮肤肌理与发丝质感 → 柔和窗光与暖调氛围 → 浅景深特写、眼神平视 → 强调真实无磨皮、避免卡通与畸变。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词，肤质发丝高度细致。",
                "formula_en": "Content order: photorealistic male face and half-body portrait → high-definition realistic photography style, skin texture and hair details → soft window light and warm tone → shallow depth close-up, eye-level gaze → emphasize real un-retouched skin, avoid cartoon and distortion. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt, highly detailed skin and hair texture."
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
                "formula_zh": "内容组织顺序：超写实男性主体与神态表情 → 场景与构图（浅景深特写）→ 光影与氛围（柔和窗光暖调）→ 画种/摄影风格（高清写实摄影）→ 需渲染文字用引号包裹，肤质发丝高度细致。",
                "formula_en": "Content order: photorealistic male subject & expression → scene & composition (shallow depth close-up) → light & atmosphere (soft window light, warm tone) → art/photography style (high-definition realistic photography) → wrap rendered text in quotes, highly detailed skin and hair texture."
            }
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
严格输出两种格式，不添加额外注释、说明、解释。
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
                    "zh": "超写实真人质感，40岁以上中老年男性硬朗松弛面部骨骼，眉眼唇皱纹分布轻微不对称；原生淡妆干净，保留深浅皱纹、面部松弛肌理、淡老年斑、肤色不均、细微干纹、真实胡茬与剃须青印，无过度磨皮与塑胶假肤、无AI模板脸；柔软花白发丝，层次自然胡须，干净布景，松弛沉稳抓拍姿态，内敛成熟情绪，无透视畸变。居家/茶室/户外/商务/胶片均为熟龄题材分支，保持岁月写实基线",
                    "en": "photorealistic real human texture, stiff sagging facial bone structure unique to men over 40, natural slight asymmetry of eyes, eyebrows, lips and wrinkle distribution; light natural makeup, retain deep & shallow wrinkles, sagging facial texture, faint age spots, uneven skin tone, fine dry lines, real stubble and shaving shadow, no over-smoothing or plastic wax skin or AI template face; soft gray-white hair, layered beard, restrained daily scene, relaxed steady snapshot pose, mature emotion, no perspective distortion. Home/teahouse/outdoor/business/film are mature theme branches only, maintain aging realism baseline"
                },
                # 中老年男性全风格细分专属规则
                "preset_rules": {
                    "zh": """
【中老年男性人像专属规则】
1. 通用基线：仅刻画40岁-70+熟龄男性，完整保留皱纹、面部松弛、淡老年斑、花白头发、层次胡须等原生年龄特征，禁止磨皮淡化、强行紧致年轻化；原生淡妆干净无大面积瑕疵，留存岁月肌理与自然毛孔，杜绝完美对称五官、蜡像假肤、光滑无胡茬下颌。
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
1. General baseline: Only depict men aged 40 to 70+, fully retain original aging marks including wrinkles, facial sagging, faint age spots, gray hair and layered beard; prohibit smoothing or forced youthful tightening. Light natural makeup without large blemishes, retain aged texture and natural pores, reject perfectly symmetrical facial features, wax fake skin and stubble-free smooth jaw.
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
                "zh": """【结构化模式】严格顺序输出：
1.人种五官轮廓特征
2.风格与服饰造型定位
3.肤质与毛发原生细节
4.三维度镜头视角与构图
   - 画面比例：竖版人像（4:5/3:4）/ 横版环境人像（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：微距特写 / 标准特写 / 肩特写 / 七分人像 / 九分人像 / 全景人像，对应叙事重心与细节展现层级
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注主体展现效果与叙事特点
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 强仰视角（脚部前景延伸），对应心理感受与画面张力
   - 景深氛围：浅景深柔焦虚化 / 中景深环境兼顾 / 深景深全景清晰，标注虚实层次对应的主次关系
5.姿态体态与表情神态
   - 头部姿态：微侧/仰头/低头/回眸，颈部线条与视线方向
   - 躯干姿态：挺直/放松/前倾/后仰，肩线角度与身体重心
   - 上肢姿态：手臂弯曲角度、手部摆放位置（叉腰/托腮/自然下垂/手持道具）
   - 下肢姿态：站姿重心分配、坐姿腿部交叠、躺卧腿部伸展/蜷缩、动态迈步/静止支撑
   - 表情神态：眼神聚焦方向、嘴角弧度、眉宇情绪（沉稳/慈祥/自信/微笑）
5.1 人像专属细节（仅人像类使用）
   - 眼神光：环形眼神光（眼下圆形光斑）/ 方形眼神光（窗光反射）/ 自然窗光（柔和反射）
   - 肤质表现：岁月皱纹（鱼尾纹/法令纹/额头纹）/ 老年斑（色素沉积自然分布）/ 松弛肌理（皮肤自然下垂质感）/ 自然光泽（健康肤色）
   - 发丝质感：银发飘逸（银色光泽流动）/ 花白质感（黑白发丝交织）/ 蓬松空气感（发量充盈）
   - 胡茬质感：花白胡茬/修剪整齐/自然生长/剃须青印
   - 面部光影：高光区（额头/鼻梁/颧骨提亮）/ 中间调（面颊/下巴自然过渡）/ 阴影区（鼻翼侧/脸颊侧立体）
6.色彩配比与整体调性
   - 主色调：占比70%，奠定整体基调（暖调/冷调/中性）
   - 辅助色：占比25%，丰富层次与环境过渡
   - 点缀色：占比5%，制造视觉焦点与细节提亮
   - 色温情绪：暖调（3200K-4500K）=温馨/复古/亲切；冷调（5500K-7000K）=清冷/高级/疏离；中性（5000K-5500K）=自然/真实/平和
   - 饱和度：低饱和=高级/文艺/复古；中饱和=自然/真实；高饱和=活力/时尚/冲击
   - 肤色还原：偏黄调（亚洲肤色自然）/ 偏粉调（欧美肤色白皙）/ 自然通透（健康血色）
7.专业布光方式与光影层次
   - 主光类型：伦勃朗光（鼻翼三角光影）/蝴蝶光（鼻下对称阴影）/侧光（明暗分割）/环形光（面部均匀立体）
   - 光源方向：正侧光45°/90°侧光/逆光轮廓/顶光戏剧/底光诡异/窗光网格投影
   - 光质软硬：硬光（清晰边缘阴影）/柔光（渐变过渡阴影）/散射光（均匀无影）
   - 环境光：补光比例、反光板效果、环境反射色调
   - 光影特效：光影斑驳/动态光斑/边缘发光/柔化光晕/胶片颗粒感/明暗渐变过渡/HDR高动态/高饱和强对比
8.背景与环境
   - 虚化程度：奶油般化开（f/1.4-1.8极致虚化）/ 柔美光斑（f/2.8光斑）/ 环境可辨（f/4-5.6）
   - 环境呼应：色彩呼应（背景与服装色调统一）/ 光影呼应（环境光与主光协调）
   - 负空间：眼神方向留白（看向处留空间）/ 呼吸空间（头顶/两侧留白）
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许完整相机参数描述（焦距、光圈、快门速度、ISO、白平衡），附带空间效果释义：
- 特写面部肤质毛孔：85mm-100mm中长焦，f/1.4-f/2.8大光圈，1/200s-1/500s快门，ISO100-400，背景虚化柔和，突出面部细节
- 半身杂志感人像：85mm中长焦，f/2.8-f/4光圈，1/125s-1/250s快门，ISO200-800，压缩空间突出人物主体
- 全身环境人像：50mm标准中焦，f/4-f/5.6光圈，1/60s-1/125s快门，ISO100-200，与主体保持常规距离，背景与主体比例协调
- 动态抓拍：200mm长焦，f/2.8-f/4光圈，1/1000s-1/4000s高速快门，ISO400-1600，冻结高速运动瞬间
- 蓝调时刻/夜景：35mm-50mm，f/1.4-f/2大光圈，1/30s-1/60s慢速快门，ISO800-3200，捕捉低光环境氛围""",
                "en": """[Structured Mode] Output strictly in this order:
1. Ethnic facial features
2. Style and clothing positioning
3. Skin, hair and stubble natural details
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical portrait (4:5/3:4) / horizontal environmental (16:9/3:2) / square (1:1)
   - Distance (shot type): macro close-up / standard close-up / shoulder shot / three-quarter portrait / nine-tenth portrait / full-scene portrait, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe display effect & narrative feature
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / strong low-angle (soles foreground extension), describe mental feeling & frame tension
   - Depth of field: shallow DOF soft bokeh / medium DOF environment balanced / deep DOF full sharpness
5. Pose body and expression
   - Head pose: slight tilt/up/down/turn back, neck line & gaze direction
   - Torso pose: upright/relaxed/lean forward/back, shoulder angle & body weight
   - Upper limb: arm bend angle, hand placement (on waist/under chin/hanging/holding props)
   - Lower limb: standing weight distribution/leg cross sitting/lying legs extended/curled/dynamic stepping/static support
   - Expression: eye focus direction, mouth curve, brow emotion (calm/focused/soft/confident)
5.1 Portrait-specific details (portrait only)
   - Catchlight: ring catchlight (circular under-eye) / square catchlight (window reflection) / natural window light (soft reflection)
   - Skin texture: wrinkles (crow's feet/nasolabial/forehead lines) / age spots (natural pigment distribution) / loose skin texture (natural sagging) / natural glow (healthy complexion)
   - Hair texture: silver hair flowing (silver sheen) / salt-and-pepper (black-white interwoven) / fluffy airy (voluminous)
   - Stubble texture: gray-white stubble / neatly trimmed / natural growth / shaving shadow
   - Facial lighting: highlight zone (forehead/nose bridge/cheekbone brightening) / midtone (cheek/chin natural transition) / shadow zone (nose side/cheek side dimension)
6. Color ratio and overall tone
   - Main Color: 70%, set overall tone (warm/cool/neutral)
   - Auxiliary Color: 25%, enrich hierarchy & environment transition
   - Accent Color: 5%, create visual focal point & detail highlight
   - Color temperature mood: warm (3200K-4500K) = cozy/retro/intimate; cool (5500K-7000K) = cold/high-end/detached; neutral (5000K-5500K) = natural/true/peaceful
   - Saturation: low saturation = high-end/artistic/retro; medium = natural/true; high = vibrant/fashion/impact
   - Skin tone: yellowish (Asian natural) / pinkish (Western fair) / natural translucent (healthy blood color)
7. Professional lighting method
   - Key light type: Rembrandt (triangle under nose) / butterfly (symmetric shadow under nose) / side light (light-dark split) / ring light (even facial dimension)
   - Light direction: 45° side / 90° side / backlit outline / top dramatic / bottom eerie / window grid projection
   - Light quality: hard (clear edge shadow) / soft (gradual transition) / diffused (even shadowless)
   - Ambient light: fill light ratio, reflector effect, environmental reflection tone
   - Lighting effects: dappled light / dynamic light spots / rim glow / soft haze / highlight bloom / film grain / particles / motion blur edges / gradient transition / HDR / high saturation contrast
8. Background & Environment
   - Bokeh: creamy smooth (f/1.4-1.8 extreme blur) / beautiful light orbs (f/2.8 bokeh) / environment discernible (f/4-5.6)
   - Environment echo: color echo (background-clothing tone unity) / lighting echo (ambient light-key light coordination)
   - Negative space: gaze direction留白 (space where looking) / breathing room (headroom/sides margin)
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative focal length/aperture description with spatial effect explanation, shutter/ISO/white balance numerical parameters forbidden:
- Close-up facial skin pores: 85mm-100mm medium telephoto, camera away from subject, soft background blur, highlighting facial details
- Half-body magazine portrait: 85mm medium telephoto, compressed space highlighting subject
- Full-body environmental portrait: 50mm standard mid-range, natural distance from subject, balanced subject-background ratio"""
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
        