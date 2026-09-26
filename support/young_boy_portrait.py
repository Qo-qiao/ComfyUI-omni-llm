# -*- coding: utf-8 -*-
"""
真实感少年儿童人像预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

YOUNG_BOY_PORTRAIT = {
    "template_id": "young_boy_portrait",
    "name": "少年儿童人像",
    "description": "专业少年儿童超写实人像摄影指导，仅覆盖婴幼儿、学前、小学、13-17岁少年全孩童年龄段，区分亚洲/欧美孩童五官、稚嫩肤质、原生毛发特质。语义权重优先级：面部肤质五官孩童特质＞自然动态姿态服饰＞自然光影色彩＞生活化场景＞构图景别＞摄影参数。所有风格坚守孩童真人写实基线，仅氛围、穿搭、场景差异化，全程无成人化造型、成熟神态刻画，姿态灵动松弛无僵硬摆拍。",
}

class YoungBoyPortrait:
    def __init__(self):
        # 下游生图模型内容组织公式库 完全沿用参考原版无改动
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
"formula_zh": "内容组织顺序：整体画面氛围光影 → 孩童气质灵动姿态 → 肌肤发丝质感 → 背景留白。侧重童真氛围叙事，弱化细碎关键词堆砌，画面柔和治愈，写实肤质发丝高度细致，纹理清晰可见。",
                 "formula_en": "Content order: overall atmosphere lighting → kid lively pose → skin hair texture → negative space. Focus on innocent atmosphere narration, realistic skin and hair highly detailed with clear texture."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：一位超写实孩童（年龄、发型、服饰与表情）→ 照片级写实与皮肤质感 → 自然光或柔和布光、活泼氛围 → 半身特写、浅景深，逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: a photorealistic child (age, hairstyle, outfit and expression) → photo-level realism and skin texture → natural or soft lighting, lively atmosphere → half-body close-up, shallow depth of field, realistic, highly detailed skin and hair texture"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：一位超写实孩童主体（年龄、发型、服饰）→ 照片级写实与皮肤质感 → 柔和自然光、活泼氛围 → 半身或特写、浅景深；建议描述孩子的神态与目光（需渲染文字直接写入，支持中英双语），逼真，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: a photorealistic child subject (age, hairstyle, outfit) → photo-level realism with skin texture → soft natural light, lively atmosphere → half-body or close-up, shallow depth of field; describe the child's expression and gaze (write any rendered text directly, supports Chinese and English), realistic, highly detailed skin and hair texture."
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：孩童身份、年龄与天真表情、超写实与电影级皮肤质感 → 风格与画质（真实肤质、发丝细节、柔焦景深） → 自然光或柔和光勾勒轮廓与氛围 → 半身或特写构图、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: child identity, age and innocent expression, photorealistic cinematic skin texture → style and quality (real skin, hair strand details, soft-focus depth of field) → natural or soft light outlining silhouette and atmosphere → half-body or close-up composition, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture"
            },
            "Qwen_Image2.1": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：孩童身份、年龄与天真表情、超写实与电影级皮肤质感 → 风格与画质（真实肤质、发丝细节、柔焦景深） → 自然段落混排冒号标签块（姿势动作：/表情：/发型发色：/穿搭：/场景：）与光影效果关键词列表（光影斑驳/边缘发光/高光溢出/胶片颗粒等） → 自然光或柔和光勾勒轮廓与氛围 → 半身或特写构图、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致；支持负向提示词通道，肢体畸变、解剖错误、坏手等缺陷写入负向提示词，正向只做纯加法",
                "formula_en": "Content order: child identity, age and innocent expression, photorealistic cinematic skin texture → style and quality (real skin, hair strand details, soft-focus depth of field) → natural paragraph mixed with colon label blocks (pose:/expression:/hairstyle:/outfit:/scene:) and lighting effect keyword list (dappled light, rim glow, highlight bloom, film grain, etc.) → natural or soft light outlining silhouette and atmosphere → half-body or close-up composition, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture; supports a negative prompt channel, write limb distortion, anatomical errors and bad hands into the negative prompt, keep the positive prompt purely additive"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：开篇（风格画质定调，或孩童主体直接切入）→ 孩童主体（发型、五官神态、稚嫩肤质、活泼姿态与手部细节）→ 服饰配件（衣款面料、童趣道具、鞋履）→ 背景环境（空间层次、材质细节、远景虚化）→ 光影（治愈柔光、自然光、明暗过渡）→ 色彩基调（主色与点缀色）→ 画质与构图（焦点位置、背景虚化、微细节清晰度、杂志封面质感）→ 整体氛围收尾。以连贯自然语言分段散文为主，细节密集但不堆砌标签；亦可用密集关键词加权重、质量词置头尾的中英混排写法；画面含文字时直接写出文字内容，无文字时以“画面中无文字”声明收尾；可选附相机镜头技术参数段（焦距、光圈、快门、ISO）。",
                "formula_en": "Content order: opening (style and quality statement, or straight to the child subject) → child subject (hairstyle, facial features, youthful skin, lively pose and hand details) → outfit and accessories (clothing fabric, playful props, shoes) → background (spatial layers, material details, distant blur) → lighting (healing soft light, natural light, light-to-shadow transitions) → color palette (main and accent colors) → quality and composition (focus point, background bokeh, micro-detail sharpness, magazine-cover feel) → closing atmosphere. Write as flowing natural-language paragraphs with dense detail but no tag stacking; a dense weighted-tag style with quality tags at head and tail (Chinese-English mixed) also works; if the image contains text, write it out directly, otherwise end with a 'No text present in image' statement; optionally append a camera and lens technical block (focal length, aperture, shutter speed, ISO)."
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
"formula_zh": "内容组织顺序：整体画面基调（治愈柔和氛围）→ 孩童松弛灵动姿态 → 自然肌肤质感与细节 → 简约童趣留白环境，写实风格，肤质发丝高度细致，纹理清晰。",
                 "formula_en": "Content order: overall image tone (healing soft atmosphere) → kid relaxed lively pose → natural skin texture and details → simple playful negative-space environment, realistic style, highly detailed skin and hair with clear texture."
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：孩童五官肤质、年龄气质 → 灵动体态姿态 → 光影层次、柔和自然光 → 服饰细节、面料质感 → 轻量环境、minimal background（密集关键词，中英术语并列），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: kid facial features and skin, age and temperament → lively body pose → lighting layers, soft natural light → clothing details, fabric texture → lightweight environment, minimal background (dense keywords, Chinese-English terms in parallel), realistic, highly detailed skin and hair texture"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：孩童身份、年龄与天真表情、超写实电影质感 → 风格与画质（真实肤质、发丝与微表情） → 自然光或柔和光勾勒轮廓氛围 → 半身或特写、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: child identity, age and innocent expression, photorealistic cinematic texture → style and quality (real skin, hair strands and micro-expressions) → natural or soft light outlining silhouette and atmosphere → half-body or close-up, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：超写实孩童面部与半身肖像 → 高清写实摄影风格、皮肤肌理与发丝质感 → 柔和自然光与温暖氛围 → 浅景深特写、眼神平视 → 强调真实无磨皮、避免畸变与成人化。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词，肤质发丝高度细致。",
                "formula_en": "Content order: photorealistic child face and half-body portrait → high-definition realistic photography style, skin texture and hair details → soft natural light and warm tone → shallow depth close-up, eye-level gaze → emphasize real un-retouched skin, avoid distortion and adultification. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt, highly detailed skin and hair texture."
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
                "formula_zh": "内容组织顺序：超写实孩童主体与天真表情 → 场景与构图（浅景深特写）→ 光影与氛围（柔和自然光温暖）→ 画种/摄影风格（高清写实摄影）→ 需渲染文字用引号包裹，肤质发丝高度细致。",
                "formula_en": "Content order: photorealistic child subject & innocent expression → scene & composition (shallow depth close-up) → light & atmosphere (soft natural light, warm) → art/photography style (high-definition realistic photography) → wrap rendered text in quotes, highly detailed skin and hair texture."
            }
        }
        # 全局底层规则 改为少年儿童专用
        self.global_base_rules = {
            "zh": """
你是专业高端少年儿童超写实人像摄影提示词扩写专家，本模板为【孩童少年人像专用】，全覆盖婴幼儿、学前、小学、13-17岁少年；题材包含春日户外、森林探险、校园日常、居家童真、复古胶片、极简棚拍。
所有风格坚守**孩童真人超写实基线**，仅穿搭、场景、光影差异化，禁止成人化五官、成熟体态、虚假精致网红童颜，无二次元、插画、油画质感。
原生无厚重妆造，保留孩童细嫩毛孔、轻微运动泛红、浅浅晒痕、面部细小绒毛，拒绝塑胶假肤、极致磨皮零瑕疵皮肤。
姿态必须使用孩童灵动松弛肢体描述，禁用成人僵硬摆姿、成熟稳重动作；光线仅采用户外/居家自然柔光，禁用影楼硬舞台强光。
完整保留用户输入年龄、性别、场景、服饰、色调、姿态，仅补充孩童肤质、胎发、棉质面料、童真光影细节，不添加成人道具、成熟装饰。
所有孩童穿搭、户外造型、校园校服均为童真纪实人像，体态柔软灵动，禁止畸形肢体、夸张动作。
输出禁忌：权重符号、冗余相机参数、卡通二次元、畸形手脚、多手指、空洞假笑、成人沉稳神态、高饱和刺眼撞色、杂乱网红布景、透视畸变。
严格输出两种格式，不额外增加注释说明。
""",
            "en": """
You are a professional photorealistic portrait expert exclusively for kids and teenagers. This preset covers infants, preschool, primary school, teens aged 13-17, including outdoor spring, forest adventure, campus, home daily, retro film, minimalist studio.
All styles follow kid photorealistic baseline, no adult facial features, mature body, artificial perfect kid face; no anime, illustration, oil painting texture.
No heavy makeup, retain tender pores, natural flush from activity, faint sun marks, fine facial fuzz; reject plastic fake skin and fully smoothed flawless skin.
All poses described as lively relaxed kid movements, no stiff adult posing or mature gestures; only natural outdoor/soft indoor light, no harsh studio stage light.
Fully retain user input age, gender, scene, outfit, tone and pose; only add kid skin, baby hair, cotton fabric and innocent light details, no adult props or mature decorations.
All kid clothes, school uniform, outdoor wear are innocent documentary portraits with soft lively bodies, no deformed limbs or exaggerated movements.
Forbidden: weight symbols, redundant camera parameters, anime, malformed hands, extra fingers, empty fake smile, mature calm expression, oversaturated clashing color, messy internet background, perspective distortion.
Strictly output two formats without extra notes.
"""
        }
        # 唯一主预设模板，绑定孩童专属template_id，层级完全对齐参考范例
        self.preset_library = {
            "young_boy_portrait": {
                "template_id": "young_boy_portrait",
                "display_name": YOUNG_BOY_PORTRAIT["name"],
                "description": YOUNG_BOY_PORTRAIT["description"],
                "positive_constraints": {
                    "zh": "影视级超写实真人质感，亚洲/欧美孩童原生圆润稚嫩面部骨骼，眉眼唇天然轻微不对称；无厚重妆造，完整保留细嫩毛孔、运动泛红、浅晒痕、细小绒毛，零过度磨皮、无塑胶假肤、无AI网红精致童颜；蓬松胎发细碎毛躁有通透感，生活化简约童趣布景，原生灵动抓拍姿态，纯粹治愈童真情绪，无透视畸变。户外/校园/居家/胶片/棚拍均为孩童题材分支，全程杜绝成人成熟体态与神态",
                    "en": "Cinematic photorealistic texture, round tender facial bone for Asian/Western kids, natural slight asymmetry of eyes, eyebrows and lips; no heavy makeup, fully retain tender pores, activity flush, faint sun marks, fine fuzz, no over-smoothing, no plastic fake skin, no AI perfect kid face; fluffy messy translucent baby hair, simple daily child scene, natural lively snapshot pose, pure healing innocent mood, no perspective distortion. Outdoor/campus/home/film/studio are kid-only themes, no mature adult body or expression."
                },
                "preset_rules": {
                    "zh": """
【少年儿童人像专属规则】
1. 通用基线：仅刻画1-17岁孩童少年，完整保留细嫩肌肤、胎碎发、孩童圆润脸型；禁止成人紧致轮廓、厚重磨皮、零瑕疵完美脸蛋、成熟沉稳神态。
2. 亚洲孩童刻画：柔和圆润娃娃脸，内双浅眼，乌黑透亮瞳孔，细腻薄嫩肌肤，浅淡晒红；适配公园、教室、居家卧室，午后漫射柔光，马卡龙低饱和配色。
3. 欧美孩童刻画立体柔和五官，多彩浅瞳色，蓬松浅棕/金胎发，通透白皙嫩皮；适配林间、郊外草坪，阴天漫射天光，多巴胺清新色系。
4. 春日户外童真风格：纯棉浅色系童装，气球/小花道具，草坪树荫斑驳柔光，动态追逐抓拍，鲜活治愈笑容。
5. 森林探险纪实风格：耐磨户外小外套，放大镜、甲虫、落叶道具，树冠细碎逆光，专注好奇孩童神态，保留户外晒痕薄汗。
6. 校园日常风格：宽松校服，课桌黑板场景，午后斜窗暖光，展示画作、手工等真实校园瞬间。
7. 复古胶片孩童风格：轻微暖胶片颗粒，老旧街巷、小院场景，宽松旧童装，不弱化肌肤细小绒毛与运动泛红。
8. 极简棚拍孩童风格：低饱和马卡龙纯色背景，均匀柔光箱，聚焦圆润面部与蓬松胎发，神态纯粹无刻意假笑。
所有孩童题材仅保留用户指定场景穿搭，只补充稚嫩肤质、胎发、童真光影细节，不加入成人相关元素。
""",
                    "en": """
【Exclusive Rules for Kid & Teen Portraits】
1. General baseline: Only depict kids 1-17, retain tender skin, fluffy baby hair, round kid face; no tight adult facial shape, heavy smoothing, flawless perfect face, mature calm expression.
2. Asian kids: Soft round baby face, shallow inner double eyelids, black clear pupils, thin tender skin, faint sun flush; suitable for park, classroom, bedroom, afternoon soft light, low-saturation macaron palette.
3. Western kids: Soft stereo facial features, light colorful pupils, fluffy light baby hair, translucent fair tender skin; suitable for forest, lawn, overcast natural light, fresh dopamine colors.
4. Spring outdoor innocent style: Cotton light kid clothes, balloon/flower props, dappled tree shade light, lively chasing snapshot, healing genuine smile.
5. Forest adventure documentary: Durable kid outdoor jacket, magnifier/beetle/leaf props, fragmented backlight from tree canopy, curious focused expression, retain outdoor sweat and sun marks.
6. Campus daily style: Loose school uniform, desk & blackboard scene, warm afternoon side window light, real moments of showing drawing and handcrafts.
7. Retro film kid style: Slight warm film grain, old alley/courtyard scene, loose vintage kid clothes, never erase fine skin fuzz and activity flush.
8. Minimal studio kid style: Low-saturation macaron solid background, even softbox light, focus on round face and fluffy baby hair, pure expression without forced fake smile.
All kid themes keep user specified scene & outfit, only add tender skin, baby hair and innocent light details, no adult related elements.
"""
                },
                "negative_base": {
                    "zh": "完美对称五官，零瑕疵肌肤，过度磨皮无毛孔，塑料蜡皮，AI网红精致童颜，成人僵硬摆姿，空洞程式假笑，成熟沉稳神态，畸形手脚，多手指，透视畸变，高饱和刺眼撞色，网红梦幻布景，二次元插画，卡通画风，成人服饰，紧致成熟面部，乌黑规整假发，肌肤无绒毛无晒痕，肌理完全丢失，多余路人杂乱装饰",
                    "en": "perfect symmetrical facial features, flawless skin, over-smoothed poreless plastic wax skin, AI perfect kid face, stiff adult pose, empty fake smile, mature calm expression, deformed hands, extra fingers, perspective distortion, oversaturated harsh clashing color, internet dreamy background, anime illustration, cartoon style, adult clothes, tight mature face, neat black wig, no skin fuzz or sun marks, lost texture, messy extra strangers & decorations"
                }
            }
        }
        # 双输出格式指引 完全沿用参考原版无修改
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/户外玩耍/居家纪实/棚拍摆拍等）与视角构图方式（蹲平视角/平视/俯拍/仰拍/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（明亮自然光/柔和窗光/户外散射光等），以及光线在孩童头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感等），用定性光影语汇替代光学数值。

第三段·孩童姿态与神情：完整描述头部、躯干、四肢的具体姿态（奔跑/蹲坐/站立/手持玩具等），视线方向与镜头关系，面部表情神态（天真/活泼/好奇/专注/笑容等），以及发丝随风飘动等动态细节。

第四段·面部细节与发型：精细刻画孩童面部五官特征（圆润脸型/大眼睛/稚嫩唇色/嫩滑肤质），皮肤质感（白皙嫩滑/婴儿肥/自然光泽），发型发色与打理方式（短发/马尾/刘海等）。

第五段·服饰配件与环境：描述穿搭细节（童装款式/面料/颜色/花纹/可爱配饰等），互动道具（玩具/零食/气球等），以及所处环境场景（公园/室内/游乐场/花园等），含远景元素（树木/草地/建筑等），最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/outdoor play/indoor daily/studio, crouching eye-level/angle/framing); 2) Lighting atmosphere (bright natural light/soft window light, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain); 3) Full pose & expression (running/squatting/standing/holding toys, gaze direction, facial emotion, hair wind-blown details); 4) Face details & styling (round face, big eyes, tender lips, smooth skin, hairstyle); 5) Outfit accessories & environment (kids clothing details, toys/props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度对齐参考示例，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；皮肤白皙嫩滑、带婴儿肥与自然红润光泽；童趣发型（短发/马尾/刘海），发丝柔顺蓬松。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等，另有奔跑/蹲坐/持玩具等孩童动态），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：天真/活泼/好奇/专注/笑容），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

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
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Fair smooth baby skin with chubby cheeks and healthy rosy glow; playful hairstyle (short / ponytail / bangs), soft fluffy hair.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement, plus running / squatting / holding toys), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: innocent / lively / curious / focused / smiling), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

**Environment and Background**
Background tone and tonal transition (gradient / solid / bokeh environment), simplicity, contrast with the subject, foreground / mid-ground / background hierarchy, blur level and negative space (gaze-direction margin, headroom and side margin).

**Photography Style and Texture**
*   **Viewpoint:** shot type (macro close-up / standard close-up / shoulder shot / three-quarter portrait / nine-tenth portrait / full-scene portrait), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle), depth of field (shallow soft bokeh / medium environment balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side), eye-contact relation with camera; aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type (Rembrandt / butterfly / side light / ring light / window light), light direction and quality (hard / soft / diffused), facial highlight-midtone-shadow and catchlight hierarchy, ambient fill, and light effects on hair, skin and clothing (dappled light / rim glow / soft haze / film grain, etc.).
*   **Texture:** contrast across skin, hair, fabric and accessory metal; color layered as main / auxiliary / accent (write against a 70/25/5 ratio, but never write percentage numbers into the prompt), main color tone, color-temperature mood, saturation level and skin-tone reproduction.
*   **Atmosphere:** derive overall atmosphere and mood from observable expression, action and lighting; avoid empty abstract adjective stacking."""
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
