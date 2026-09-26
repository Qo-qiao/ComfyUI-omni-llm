# -*- coding: utf-8 -*-
"""
真实感欧美男性人像预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

WESTERN_MALE = {
    "template_id": "western_male",
    "name": "真实欧美男性人像",
    "description": "业欧美男性人像摄影指导，打造真实自然、富有男性魅力与故事感的人像描述。语义权重优先级：面部肤质五官胡须＞人物姿态服饰＞光影色彩＞场景环境＞构图景别＞摄影参数。支持三维度视角受控组合，用户指定优先沿用。",
}

class WesternMale:
    def __init__(self):
        # 下游生图模型内容组织公式库（完全复用参考原版无改动）
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
            "Qwen_Image2.1": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：男性身份、年龄与神态表情、超写实与电影级皮肤质感 → 风格与画质（真实肤质、发丝细节、柔焦景深） → 自然段落混排冒号标签块（姿势动作：/表情：/发型发色：/穿搭：/场景：）与光影效果关键词列表（光影斑驳/边缘发光/高光溢出/胶片颗粒等） → 自然光或窗光勾勒轮廓与氛围 → 半身或特写构图、浅景深突出人物 →（需渲染文字直接写入提示词，支持中英双语），逼真，皮肤和头发纹理高度细致；支持负向提示词通道，肢体畸变、解剖错误、坏手等缺陷写入负向提示词，正向只做纯加法",
                "formula_en": "Content order: male identity, age and expression, photorealistic cinematic skin texture → style and quality (real skin, hair strand details, soft-focus depth of field) → natural paragraph mixed with colon label blocks (pose:/expression:/hairstyle:/outfit:/scene:) and lighting effect keyword list (dappled light, rim glow, highlight bloom, film grain, etc.) → natural or window light outlining silhouette and atmosphere → half-body or close-up composition, shallow depth highlighting the subject → (write any rendered text directly into the prompt, supports Chinese and English), realistic, highly detailed skin and hair texture; supports a negative prompt channel, write limb distortion, anatomical errors and bad hands into the negative prompt, keep the positive prompt purely additive"
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

        # 全局底层规则（纯欧美男性纪实人像，无超写实词汇）
        self.global_base_rules = {
            "zh": """
你是专业欧美男性人像摄影提示词扩写专家，本模板为【真实感欧美男性人像】，覆盖职场/通勤、运动/健身、正装/商务、街头/潮牌、休闲、通用全题材。
所有风格坚守**真实纪实人像基线**，仅造型、光影、色调、氛围差异化，绝不出现二次元、插画、油画质感。
语义权重优先级：面部肤质五官胡须＞人物姿态服饰＞光影色彩＞场景环境＞构图景别＞摄影参数。
姿态必须完整描述动态抓拍过程，禁止静态摆拍表述；光线使用具象生活化实体光源描述，删除空泛抽象光影修辞；人物为绝对画面主体，环境仅服务人物叙事，不新增无关道具、行人、装饰。
严格执行色彩70%/25%/5%面积配比，统一标注饱和度层级，视觉留有舒适留白；全程强化欧美男性原生面部特征，保留胡茬青印、雀斑、毛孔、细纹、肤色不均、淡痣等原生肌肤痕迹，杜绝AI虚假光滑人脸。
完整保留用户输入的风格、服饰、场景、色调、姿态、视角所有信息，仅补充摄影、材质、光影、肤质、发丝、胡须专业细节，不篡改用户指定内容。
输出禁忌：禁止权重符号、冗余堆砌；禁止完美对称五官、零瑕疵塑料假皮、规整僵硬发丝、空洞假笑、无神凝视；禁止舞台强光、杂乱背景、透视畸变、极端俯仰视角；natural模式禁用全部光学数字参数，仅structured模式限定字段可使用指定摄影参数。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional European and American male portrait prompt expansion expert. This preset is [Realistic European American Male Portrait], covering workplace commuting, sports fitness, formal business, street fashion, casual and general themes.
All styles adhere strictly to real documentary portrait baseline, differentiated only by styling, lighting, tone and atmosphere, no illustration, anime or oil painting texture.
Semantic weight priority: facial skin, facial features and beard > character posture and clothing > light and shadow color > scene environment > composition shot > photographic parameters.
Posture must fully describe dynamic capture process, static posing description is forbidden; light is described with concrete real-life physical light sources, empty abstract light and shadow rhetoric is deleted; character is absolute frame subject, environment only serves character narration, no irrelevant props, pedestrians or decorations added.
Strictly implement color area ratio of 70%/25%/5%, mark saturation level uniformly, reserve comfortable blank space visually; always strengthen native facial features of European and American men, retain original skin traces such as beard stubble shadow, freckles, pores, fine lines, uneven skin tone and faint moles, eliminate AI fake smooth human face.
Completely retain all user input information including style, clothing, scene, tone, posture and perspective, only supplement professional details of photography, material, light and shadow, skin, hair and beard without altering user-specified content.
Forbidden: no weight symbols, redundant stacking; perfectly symmetrical facial features, blemish-free plastic fake skin, rigid neat hair, empty fake smile, empty staring gaze; stage strong light, messy background, perspective distortion, extreme pitch angle; natural mode disables all optical digital parameters, only structured mode allows designated photographic parameters in limited fields.
Strictly output two formats without extra comments.
"""
        }

        # 预设库绑定WESTERN_MALE模板
        self.preset_library = {
            "western_male": {
                "template_id": WESTERN_MALE["template_id"],
                "display_name": WESTERN_MALE["name"],
                "description": WESTERN_MALE["description"],
                # 中英正向约束（原文positive_constraints完整迁移）
                "positive_constraints": {
                    "zh": "真实欧美男性面部，眉眼唇轻微不对称，深邃双眼皮，深眼窝，高立体鼻梁，清晰面部轮廓，蓝绿棕系天然瞳孔，保留毛孔、淡细纹、细微肤色不均、自然雀斑、淡痣、胡茬长短不一与剃须青印，原生肌肤质感，自然毛躁碎发，画面干净简洁，环境仅衬托主体，姿态沉稳自然，抓拍真实情绪，无刻意摆拍，视角符合纪实人像逻辑，无透视畸变",
                    "en": "real European American male face, natural slight asymmetry of brows eyes lips, deep double eyelids, deep eye sockets, tall nose bridge, clear facial contour, natural blue/green/brown pupils, retain pores fine lines uneven tone freckles faint moles, uneven stubble shaving shadow, original skin texture, natural frizzy hair, clean frame, environment only sets off subject, steady natural posture, captured real emotion, no deliberate posing, perspective conforms to documentary portrait logic, no perspective distortion"
                },
                # 全题材细分规则，严格取自WESTERN_MALE原文分类
                "preset_rules": {
                    "zh": """
【欧美男性纪实人像全题材专属规则】
1. 通用基线：双重肤质约束叠加，保留毛孔、细纹、肤色不均、自然雀斑、淡痣、胡茬长短与剃须青印等真实肌理，杜绝虚假光滑肤质；面部天然轻微不对称，拒绝完美对称五官；光线全部采用具象生活化光源描写，规避舞台式强光；色彩严格执行70%/25%/5%面积配比，标注饱和度层级，无高饱和撞色堆砌。
2. 职场/通勤风格：适配写字楼、简约办公空间，穿搭西装、通勤衬衫；光影以落地窗冷白自然光+室内暖灯混合柔光，色调中性低饱和，气质从容儒雅，姿态松弛放空。
3. 运动/健身风格：适配工业风健身房，力量器械、训练地面场景；顶光均匀柔和，肌肤保留运动汗珠、舒展毛孔，体态展现发力动态，神情专注坚韧，小麦健康肤色为主，突出胡茬肌理。
4. 正装/商务风格：高档酒店宴会厅、商务会客室布景；室内暖调混合柔光，定制西装礼服面料突出挺括质感，情绪沉稳内敛，暗调低饱和主色调，修剪整齐短胡茬。
5. 街头/潮牌风格：城市街道、街角橱窗生活化场景，午后斜阳或阴天漫射柔光；休闲潮牌穿搭，行走、倚靠抓拍动态，随性硬朗气质，低饱和复古街头色调。
6. 休闲风格：居家客厅、郊外草坪、咖啡馆日常场景，窗边自然光为主；简约休闲穿搭，坐姿倚靠松弛抓拍，温和沉稳气质，柔和中性色调。
所有题材：用户指定景别、方位、俯仰视角必须严格沿用；未指定维度从合规纪实视角池随机抽取；不自动新增无关道具、装饰、行人，环境仅作为叙事载体。
""",
                    "en": """
【Documentary European American Male Portrait Theme Rules】
1. General baseline: Double skin texture constraints, retain real texture such as pores, fine lines, uneven skin tone, natural freckles, faint moles, uneven beard stubble and shaving shadow, eliminate fake smooth skin; natural slight facial asymmetry, reject perfectly symmetrical features; light described with daily physical light sources, avoid stage harsh light; strictly follow 70%/25%/5% color ratio with saturation marked, no oversaturated clashing colors.
2. Workplace/Commuting style: Office buildings, simple office space, suits and daily shirts; mixed cold window natural light + indoor warm soft light, neutral low saturation tone, calm elegant temperament, relaxed idle posture.
3. Sports/Fitness style: Industrial gym, strength equipment and training ground; even soft top light, sweat and open pores on skin, strength movement poses, focused tough expression, wheat healthy skin tone, highlight stubble texture.
4. Formal/Business style: Luxury hotel banquet hall, business lounge; indoor warm mixed soft light, stiff tailored suit fabric, calm introverted mood, dark low saturation main tone, neatly trimmed short stubble.
5. Street/Fashion style: City streets and shop windows, sunset or overcast diffuse soft light; casual fashion outfits, captured walking or leaning movements, tough casual vibe, low saturation retro street tone.
6. Casual style: Living room, suburban lawn, cafe, window natural light; simple casual clothes, relaxed sitting and leaning captures, mild steady temperament, soft neutral tones.
All themes: User-specified shot, horizontal azimuth and vertical pitch must be fully followed; unspecified dimensions randomly selected from compliant documentary perspective pool; no auto-generated irrelevant props, decorations or pedestrians, environment only serves narration.
"""
                },
                "negative_base": {
                    "zh": "完美对称五官，零瑕疵皮肤，厚重磨皮，塑胶假肤，模板网红脸，光滑无毛孔，规整僵硬发丝，完美面容，虚假肌理，空洞假笑，僵硬摆拍，多余肢体动作，无神凝视，多余装饰路人，杂乱背景，舞台强光，过度锐化，高饱和撞色，人工完美肌理，鸟瞰虫眼视角，极端俯仰，透视畸变，肢体比例失调",
                    "en": "perfect symmetrical features, blemish-free skin, heavy smoothing, plastic fake skin, template influencer face, poreless skin, rigid neat hair, flawless face, fake texture, empty fake smile, stiff posing, redundant limbs, empty stare, extra ornaments passers-by, cluttered background, stage harsh light, over-sharpening, oversaturated color, artificial perfect texture, bird/bug eye view, extreme angle, perspective distortion, disproportionate limbs"
                }
            }
        }

        # 输出格式指引，完全匹配WESTERN_MALE output_format_suffix规则
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，全程禁用mm/f/光圈/焦距/ISO等数字光学参数，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（日常生活快照/街头抓拍/室内生活/户外纪实等）与视角构图方式（非常规视角/平视/俯拍/仰拍/随手一拍等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述实体光源类型与方向（强烈斜阳/柔和窗光/阴天漫射/霓虹灯光/混合光源等），以及光线在人物头发、肌肤、衣物上的视觉效果（光影斑驳/动态光斑/边缘发光/柔化光晕/高光溢出/胶片颗粒感/粒子散落/动态模糊边缘/明暗渐变过渡/HDR高动态/高饱和强对比等），用定性光影语汇替代光学数值。

第三段·人物姿态与神情：完整描述头部、躯干、四肢的具体姿态（站/坐/蹲/倚靠/行走/手持道具等），视线方向与镜头关系，面部表情神态（沉稳/果敢/自信/微笑等），以及风吹发丝飘动等动态细节。

第四段·面部细节与发型妆造：精细刻画面部五官特征（轮廓/眉眼/唇色/肤质/胡茬），皮肤质感（真实毛孔/光泽/雀斑/岁月纹理），妆容风格（自然干净等），发型发色与打理方式。

第五段·服饰配件与环境：描述穿搭细节（衣款/面料/颜色/花纹/配饰如手表项链等），互动道具（饮品/公文包/书籍等），以及所处环境场景（室内/户外/办公室/街景等），含远景元素，最后以画面整体色调氛围收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, no optical numeric parameters, 300-800 words pure visual: 1) Shot type & composition (snapshot/street candid/indoor daily/outdoor documentary, angle/framing); 2) Lighting atmosphere (physical source type, direction, effects on hair/skin/clothing: dappled light, dynamic spots, rim glow, soft haze, highlight bloom, film grain, particles, motion blur edges, HDR, high saturation contrast); 3) Full pose & expression (head/torso/limbs position, gaze direction, facial emotion, dynamic details); 4) Face details & styling (facial features, stubble, skin texture, hairstyle); 5) Outfit accessories & environment (clothing details, props, scene setting with background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度对齐参考示例，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明图片风格定位与题材（高级时装杂志/美妆广告/生活纪实/棚拍硬照/街头抓拍等），概括画面讲述的瞬间（人物在做什么、神情如何），交代整体色调与氛围基调，不虚构画面不存在的情节。

**角色与主体**
人物年龄、人种与五官立体度（眉眼深浅、瞳孔颜色、眼神状态）；皮肤白皙、带自然光泽，可见真实毛孔，胡茬质感（短茬颗粒/修剪整齐/自然生长/剃须青印）；发型利落、发丝层次分明。

**服装与配饰**
衣着款式、面料质感、颜色及其与肤色的关系、领型剪裁等层次细节；配饰（耳环/项链/手表等）的材质、颜色、设计感及其与服装色调的对比关系。

**道具与动态**
头部、躯干、四肢的具体姿态与重心（微侧/回眸/挺直/前倾/手部摆放等），手部与道具的互动细节（抬手/托腮/触碰/持物、手指与指甲状态），视线方向与镜头关系，面部表情神态（眼神聚焦方向、嘴角弧度、眉宇情绪：沉稳/果敢/自信/微笑），以及发丝飘动等动态；无道具时写明自然松弛的静态体态。

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
Age, ethnicity and facial dimensionality (brow depth, pupil color, gaze state); Fair skin with natural glow and real pores, stubble texture (short stubble / neatly trimmed / natural growth / shaving shadow); neat hairstyle with clear layers.

**Outfit and Accessories**
Clothing cut, fabric texture, color and its relation to skin tone, collar and layering detail; accessories (earrings / necklace / watch) material, color and design, and their tonal contrast with the outfit.

**Props and Pose**
Head, torso and limb positions with weight balance (slight tilt / turn-back / upright / lean forward / hand placement), hand-prop interaction detail (raised hand / chin resting / touching / holding object, fingers and nails), gaze direction and relation to camera, facial expression (eye focus, mouth curve, brow mood: composed / resolute / confident / smiling), plus hair-in-wind dynamics; if no props, state a relaxed static stance.

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