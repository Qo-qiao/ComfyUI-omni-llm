# -*- coding: utf-8 -*-
"""
图像反推预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

IMAGE_REVERSE_TAGS_ZH = {
    "name": "图像标签反推生成器（智能分类版）",
    "description": "作为专业的SDXL标签化模型提示词工程师，我将自动识别图像类别并生成精确的中文标签列表。我的专业知识涵盖摄影术语、艺术风格、色彩理论等领域。",
    "input_template": "逐步思考：1）分析图像内容，识别主导视觉元素；2）从以下类别中选择最合适的：风景类、摄影类、人像类、插画类、IP类、cosplay类、游戏角色类、产品类、建筑室内类、动物类、美食类、UI界面类、时尚穿搭类、通用类；3）如未匹配，根据图像内容创建合适的类别名称；4）按照类别特定字段结构生成{mode}标签列表；5）验证：30-60个标签，无重复，包含质量标签。\n\n用户输入：#\n\n规则：1.仅输出JSON，无额外文字；2.标签总数30-60，不重复；3.必须包含质量标签（杰作、最佳质量、高分辨率、超精细）。4.严格按照下方字段结构输出。",
    "output_format_suffix": {
        "natural": "逗号分隔的标签字符串。",
        "structured": "【类别】类别名称\n【质量标签】杰作，最佳质量，高分辨率，超精细，8K\n【核心要素】\n  - 场景类型：场景描述内容\n  - 时间光线：时间与光线描述内容\n  - 色彩氛围：色彩与氛围描述内容\n【景别】景别描述内容（微距特写/标准特写/肩特写/七分人像/九分人像/全景人像）\n【细节元素】细节元素描述内容\n【技术参数】镜头与视角描述内容"
    },
    "task_requirements": [
        "先识别图像类别，再按对应字段输出标签",
        "必须包含杰作、最佳质量、高分辨率、超精细等质量标签",
        "标签简洁，每个字段内逗号分隔，总标签数30-60",
        "各类别字段顺序可调整，但字段名必须完全一致",
        "natural模式：输出逗号分隔的标签字符串；structured模式：输出结构化提示词",
        "景别判断规则：\n  - 微距特写（Macro Close-up）：只截取人体极小局部，无完整面部，填满画面，极致细节（单只眼睛/嘴唇/指尖/皮肤肌理/发丝等）\n  - 标准特写（Close-up）：头顶至下巴，完整脸部不含肩膀，画面主体全是人脸\n  - 肩特写（Close-up with shoulder）：头部+一点点肩线，只露出肩头一小截\n  - 七分人像（Medium Shot）：头顶到腰腹/腰线，截断于腰部肚脐附近，含完整头部肩膀胸口腰\n  - 九分人像（Medium Full Shot）：头顶到小腿膝盖下方/脚踝上方，裁切在脚踝/小腿中段，不完整露出双脚\n  - 全景人像（Full Body Shot）：完整从头到脚全身入镜，四肢双脚全部包含无裁切"
    ],
    "constraints": {
        "max_tags": 60,
        "min_tags": 30,
        "no_duplicates": True,
        "exclude_abstract": True
    },
    "examples": [
        {
            "category": "风景类",
            "natural": "杰作，最佳质量，高分辨率，8K，风景照，山景，雪山，云海，峡谷，日出，金色时刻，硬光，侧逆光，冷色调，蓝白灰，宁静，空旷感，航拍视角，远景，长焦镜头，16:9宽幅，松树，薄雾，层峦叠嶂，阴影对比，蓝天白云，天空渐变，大气透视，近深远淡，空气湿度，能见度极高，晨雾缭绕，晨曦穿透，层次分明，透视感强，氛围感强",
            "structured": "【类别】风景类\n【质量标签】杰作，最佳质量，高分辨率，8K\n【核心要素】\n  - 场景类型：山景，雪山，云海，峡谷\n  - 时间光线：日出，金色时刻，硬光，侧逆光\n  - 色彩氛围：冷色调，蓝白灰，宁静，空旷感\n【天空大气】\n  - 天空状态：蓝天白云，天空渐变，日出晨曦\n  - 大气条件：大气透视，空气湿度适中，能见度极高，晨雾缭绕\n  - 光影层次：近深远淡，晨曦穿透云层，光线层次分明\n【细节元素】松树，薄雾，层峦叠嶂，阴影对比\n【技术参数】航拍视角，远景，长焦镜头，16:9宽幅"
        },
        {
            "category": "人像类",
            "natural": "杰作，最佳质量，高分辨率，8K，人像摄影，亚洲女性，25岁，黑色长发，披肩，杏仁眼，温柔表情，微笑，自然肤色，细腻皮肤，浅米色针织衫，袖口蕾丝，咖啡馆窗边，午后阳光，柔和侧光，低饱和度，温暖色调，平视视角，中景，35mm定焦，f/1.8，浅景深，温馨氛围，安静阅读，书籍，咖啡杯，木质桌面，薄纱窗帘",
            "structured": "【类别】人像类\n【质量标签】杰作，最佳质量，高分辨率，8K\n【核心要素】\n  - 场景类型：咖啡馆室内\n  - 时间光线：午后自然光，柔和侧光\n  - 色彩氛围：温暖色调，低饱和度，温馨舒适\n【人物特征】\n  - 外貌：亚洲女性，25岁，黑色长发披肩，杏仁眼，温柔微笑\n  - 服装：浅米色针织衫，袖口蕾丝装饰\n  - 姿态：安静阅读，右手翻页\n【细节元素】书籍，咖啡杯，木质桌面，薄纱窗帘，窗光尘埃\n【景别】七分人像（Medium Shot）：头顶到腰部，完整头部肩膀胸口腰，截断于腰部肚脐附近\n【技术参数】平视视角，中景，35mm定焦，f/1.8，浅景深"
        },
        {
            "category": "动物类",
            "natural": "杰作，最佳质量，高分辨率，8K，动物摄影，猫，橙色虎斑猫，绿色眼睛，坐姿，蓬松毛发，室内窗台，阳光洒落，温暖光影，柔和光线，高对比度，特写镜头，50mm微距，浅景深，毛绒质感，胡须清晰，眼神灵动，窗台绿植，木质纹理，自然光，温馨氛围，可爱表情，警觉姿态",
            "structured": "【类别】动物类\n【质量标签】杰作，最佳质量，高分辨率，8K\n【核心要素】\n  - 场景类型：室内窗台\n  - 时间光线：自然窗光，柔和侧光\n  - 色彩氛围：温暖色调，高对比度\n【动物特征】\n  - 物种：橙色虎斑猫\n  - 外貌：绿色眼睛，蓬松毛发，胡须清晰\n  - 姿态：坐姿，警觉，眼神灵动\n【细节元素】窗台绿植，木质纹理，阳光光斑\n【技术参数】特写镜头，50mm微距，浅景深"
        },
        {
            "category": "美食类",
            "natural": "杰作，最佳质量，高分辨率，8K，美食摄影，草莓蛋糕，奶油装饰，新鲜草莓，木质托盘，白色餐盘，午后阳光，柔和逆光，暖色调，高饱和度，45度俯视，浅景深，焦外虚化，蛋糕纹理，草莓水珠，奶油细腻，温馨氛围，诱人食欲，烘焙质感，甜点，下午茶，咖啡搭配，桌面纹理",
            "structured": "【类别】美食类\n【质量标签】杰作，最佳质量，高分辨率，8K\n【核心要素】\n  - 场景类型：餐桌静物\n  - 时间光线：午后阳光，柔和逆光\n  - 色彩氛围：暖色调，高饱和度，诱人食欲\n【美食特征】\n  - 主体：草莓蛋糕，奶油装饰，新鲜草莓\n  - 质感：蛋糕纹理，奶油细腻，草莓水珠\n【细节元素】木质托盘，白色餐盘，咖啡杯，桌面纹理\n【技术参数】45度俯视，浅景深，焦外虚化"
        },
        {
            "category": "建筑室内类",
            "natural": "杰作，最佳质量，高分辨率，8K，室内设计，现代简约，客厅，落地窗，自然光，白色墙面，原木家具，灰色沙发，绿植装饰，开放式布局，空间通透，光影层次，柔和光线，冷色调，简洁干净，广角镜头，对称构图，透视感强，北欧风格，舒适氛围，几何线条，室内植物，装饰品",
            "structured": "【类别】建筑室内类\n【质量标签】杰作，最佳质量，高分辨率，8K\n【核心要素】\n  - 场景类型：现代简约客厅\n  - 时间光线：自然窗光，均匀柔和\n  - 色彩氛围：冷色调，简洁干净\n【空间特征】\n  - 布局：开放式布局，空间通透\n  - 材质：白色墙面，原木家具，灰色沙发\n  - 装饰：落地窗，绿植，装饰品\n【细节元素】几何线条，光影层次，室内植物\n【技术参数】广角镜头，对称构图，透视感强"
        },
        {
            "category": "产品类",
            "natural": "杰作，最佳质量，高分辨率，8K，产品摄影，无线耳机，黑色磨砂，金属质感，悬浮摆放，镜面反射，深蓝色背景，三点布光，主光左前，补光右侧，背光轮廓，45度俯视，高对比度，科技感，精致细节，耳罩纹理，海绵透气孔，质感细腻，专业灯光，商业摄影，高端质感",
            "structured": "【类别】产品类\n【质量标签】杰作，最佳质量，高分辨率，8K\n【核心要素】\n  - 场景类型：商业摄影棚\n  - 光线方案：三点布光，主光左前，补光右侧，背光轮廓\n  - 色彩氛围：深蓝色背景，高对比度，科技感\n【产品特征】\n  - 主体：无线降噪耳机\n  - 材质：黑色磨砂塑料，金属拉丝\n  - 细节：耳罩纹理，海绵透气孔\n【细节元素】镜面反射倒影，悬浮摆放\n【技术参数】45度俯视，专业灯光，商业摄影"
        }
    ]
}

IMAGE_REVERSE_TAGS_EN = {
    "name": "Image Tag Reverse Generator (Smart Classification)",
    "description": "As a professional SDXL tagging model prompt engineer, I automatically recognize image categories and generate precise English tag lists. My expertise covers photography terminology, art styles, color theory, and more.",
    "input_template": "Step-by-step thinking: 1) Analyze the image content, identify dominant visual elements; 2) Choose the most appropriate category from: Landscape, Photography, Portrait, Illustration, IP, Cosplay, Game Character, Product, Architecture/Interior, Animal, Food, UI Interface, Fashion/Style, General; 3) If no match, create a suitable category name based on image content; 4) Generate {mode} tag list according to the category-specific field structure; 5) Verify: 30-60 tags, no duplicates, include quality tags.\n\nUser input: #\n\nRules: 1. Output only JSON, no extra text; 2. Total tags 30-60, no duplicates; 3. Must include quality tags (masterpiece, best quality, high resolution, ultra-detailed). 4. Strictly follow the field structure below.",
    "output_format_suffix": {
        "natural": "Comma-separated tag string.",
        "structured": "【Category】Category Name\n【Quality Tags】masterpiece, best quality, high resolution, ultra-detailed, 8K\n【Core Elements】\n  - Scene Type: scene description\n  - Time/Lighting: time and lighting description\n  - Color/Mood: color and atmosphere description\n【Detail Elements】detail element description\n【Technical Parameters】lens and perspective description",
    },
    "task_requirements": [
        "First identify image category, then output tags according to corresponding field group",
        "Must include masterpiece, best quality, high resolution, ultra-detailed and other quality tags",
        "Tags concise, comma separated within each field, total 30-60",
        "Field order can be adjusted, but field names must be exactly consistent",
        "natural mode: output comma-separated tag string; structured mode: output structured prompt",
        "Shot type judgment rules:\n  - Macro Close-up: captures only a tiny human body part, no complete face, fills the frame, extreme detail (single eye/lips/fingertips/skin texture/hair strands etc.)\n  - Close-up: top of head to chin, complete face without shoulders\n  - Close-up with Shoulder: head + a small portion of shoulders, only a hint of shoulder line\n  - Medium Shot: top of head to waist/abdomen, cut off at waist/navel area, includes complete head shoulders chest waist\n  - Medium Full Shot: top of head to below knee/above ankle, cut off at ankle/calf mid-section, feet not fully visible\n  - Full Body Shot: complete head to toe full body, all limbs and feet included without cropping"
    ],
    "constraints": {
        "max_tags": 60,
        "min_tags": 30,
        "no_duplicates": True,
        "exclude_abstract": True
    },
    "examples": [
        {
            "category": "Landscape",
            "natural": "masterpiece, best quality, high resolution, 8K, landscape photography, mountain scenery, snow-capped mountains, sea of clouds, canyon, sunrise, golden hour, hard light, side backlight, cool tone, blue-white-gray palette, tranquil atmosphere, vast openness, aerial perspective, wide angle shot, telephoto lens, 16:9 aspect ratio, pine trees, morning mist, layered mountain peaks, shadow contrast, blue sky with white clouds, sky gradient, atmospheric perspective, foreground sharp background soft, moderate humidity, high visibility, morning fog, dawn light breaking through clouds, distinct layering, strong sense of depth, cinematic atmosphere",
            "structured": "【Category】Landscape\n【Quality Tags】masterpiece, best quality, high resolution, 8K\n【Core Elements】\n  - Scene Type: mountain scenery, snow-capped mountains, sea of clouds, canyon\n  - Time/Lighting: sunrise, golden hour, hard light, side backlight\n  - Color/Mood: cool tone, blue-white-gray palette, tranquil atmosphere, vast openness\n【Sky/Atmosphere】\n  - Sky State: blue sky, white clouds, sky gradient, dawn\n  - Atmospheric Conditions: atmospheric perspective, moderate humidity, high visibility, morning mist\n  - Light Layers: foreground sharp background soft, dawn light breaking through clouds, clear light layers\n【Detail Elements】pine trees, mist, layered mountain peaks, shadow contrast\n【Technical Parameters】aerial view, wide angle shot, telephoto lens, 16:9 aspect ratio"
        },
        {
            "category": "Portrait",
            "natural": "masterpiece, best quality, high resolution, 8K, portrait photography, Asian female, age 25, long black hair, shoulder-length, almond-shaped eyes, gentle expression, soft smile, natural skin tone, smooth skin texture, light beige knit sweater, lace-trimmed cuffs, cafe window seat, afternoon sunlight, soft side lighting, low saturation, warm tone palette, eye-level perspective, half body (waist up, navel position), 35mm prime lens, f/1.8 aperture, shallow depth of field, cozy atmosphere, peaceful reading, open book, coffee cup, wooden table surface, sheer curtain",
            "structured": "【Category】Portrait\n【Quality Tags】masterpiece, best quality, high resolution, 8K\n【Core Elements】\n  - Scene Type: cafe interior\n  - Time/Lighting: afternoon natural light, soft side lighting\n  - Color/Mood: warm tone, low saturation, cozy and comfortable\n【Character Features】\n  - Appearance: Asian female, age 25, long black hair, shoulder-length, almond-shaped eyes, gentle smile\n  - Clothing: light beige knit sweater, lace-trimmed cuffs\n  - Pose: quietly reading, turning pages with right hand\n【Detail Elements】open book, coffee cup, wooden table surface, sheer curtain, dust particles in window light\n【Shot Type】Medium Shot: top of head to waist, complete head shoulders chest waist, cut off at waist/navel area, subject is upper body\n【Technical Parameters】eye-level perspective, 35mm prime lens, f/1.8 aperture, shallow depth of field"
        },
        {
            "category": "Animal",
            "natural": "masterpiece, best quality, high resolution, 8K, animal photography, cat, orange tabby cat, emerald green eyes, sitting pose, fluffy fur texture, indoor windowsill, natural sunlight, warm lighting, soft diffused light, high contrast, close-up shot, 50mm macro lens, shallow depth of field, detailed fur texture, sharp whiskers, expressive eyes, windowsill potted plant, wooden texture, natural light source, cozy atmosphere, cute expression, alert posture",
            "structured": "【Category】Animal\n【Quality Tags】masterpiece, best quality, high resolution, 8K\n【Core Elements】\n  - Scene Type: indoor windowsill\n  - Time/Lighting: natural window light, soft side lighting\n  - Color/Mood: warm tone, high contrast\n【Animal Features】\n  - Species: orange tabby cat\n  - Appearance: emerald green eyes, fluffy fur, sharp whiskers\n  - Pose: sitting, alert posture, expressive eyes\n【Detail Elements】windowsill potted plant, wooden texture, sunlight spots\n【Technical Parameters】close-up shot, 50mm macro lens, shallow depth of field"
        },
        {
            "category": "Food",
            "natural": "masterpiece, best quality, high resolution, 8K, food photography, strawberry shortcake, whipped cream decoration, fresh ripe strawberries, wooden serving tray, white ceramic plate, afternoon sunlight, soft backlighting, warm tone palette, high saturation, 45-degree overhead view, shallow depth of field, beautiful bokeh, cake crumb texture, strawberry water droplets, creamy smoothness, warm inviting atmosphere, appetizing presentation, baked texture, dessert, afternoon tea setting, coffee pairing, wooden table texture",
            "structured": "【Category】Food\n【Quality Tags】masterpiece, best quality, high resolution, 8K\n【Core Elements】\n  - Scene Type: table still life\n  - Time/Lighting: afternoon sunlight, soft backlighting\n  - Color/Mood: warm tone, high saturation, appetizing\n【Food Features】\n  - Subject: strawberry shortcake, whipped cream decoration, fresh ripe strawberries\n  - Texture: cake crumb texture, creamy smoothness, strawberry water droplets\n【Detail Elements】wooden serving tray, white ceramic plate, coffee cup, wooden table texture\n【Technical Parameters】45-degree overhead view, shallow depth of field, beautiful bokeh"
        },
        {
            "category": "Architecture/Interior",
            "natural": "masterpiece, best quality, high resolution, 8K, interior design photography, modern minimalist style, living room space, floor-to-ceiling windows, abundant natural light, white walls, wooden furniture pieces, gray fabric sofa, green plants, open floor plan, spacious room, light and shadow play, soft diffused lighting, cool tone palette, clean aesthetic, wide-angle lens, symmetrical composition, strong sense of perspective, Nordic design style, cozy atmosphere, geometric lines, indoor potted plants, decorative ornaments",
            "structured": "【Category】Architecture/Interior\n【Quality Tags】masterpiece, best quality, high resolution, 8K\n【Core Elements】\n  - Scene Type: modern minimalist living room\n  - Time/Lighting: natural window light, even and soft\n  - Color/Mood: cool tone, clean and fresh\n【Spatial Features】\n  - Layout: open floor plan, spacious\n  - Materials: white walls, wooden furniture pieces, gray fabric sofa\n  - Decor: floor-to-ceiling windows, green plants, decorative ornaments\n【Detail Elements】geometric lines, light layers, indoor potted plants\n【Technical Parameters】wide-angle lens, symmetrical composition, strong sense of perspective"
        },
        {
            "category": "Product",
            "natural": "masterpiece, best quality, high resolution, 8K, product photography, wireless noise-canceling headphones, black matte finish, brushed metal texture, floating display presentation, mirror surface reflection, deep blue background, three-point lighting setup, key light front left, fill light right, rim light back, 45-degree overhead view, high contrast, modern tech feel, fine details, earcup texture, cushion vents, intricate texture details, professional studio lighting, commercial product photography, premium quality",
            "structured": "【Category】Product\n【Quality Tags】masterpiece, best quality, high resolution, 8K\n【Core Elements】\n  - Scene Type: commercial photo studio\n  - Lighting Scheme: three-point lighting, key light front left, fill light right, rim light back\n  - Color/Mood: deep blue background, high contrast, modern tech feel\n【Product Features】\n  - Subject: wireless noise-canceling headphones\n  - Material: black matte plastic, brushed metal finish\n  - Details: earcup texture, cushion vents\n【Detail Elements】mirror surface reflection, floating display presentation\n【Technical Parameters】45-degree overhead view, professional studio lighting, commercial product photography"
        }
    ]
}

IMAGE_REVERSE_DESCRIBE_ZH = {
    "name": "图像反推描述器（自然语言版）",
    "description": "作为专业的图像分析专家，我将自动识别图像类别并生成专业、流畅的自然语言描述，适用于Flux、Z-Image、Qwen-Image、Krea等主流自然语言图像生成模型。我的专业知识涵盖摄影术语、艺术风格、灯光设计、色彩理论、画面构图分析、主体位置识别和视觉视角推断，以精准控制生成与输入图一致的新图像。",
    "input_template": "逐步思考：1）分析图像内容，识别关键视觉元素和艺术风格；2）从以下类别中选择最合适的：风景类、摄影类、人像类、插画类、IP类、cosplay类、游戏角色类、产品类、建筑室内类、动物类、美食类、UI界面类、时尚穿搭类、通用类；3）分析画面构图（三分法/对称/对角线/框架/中心/三角形等）和主体位置（用水平+垂直百分比表示，如水平50%居中，垂直40%偏上）；4）推断视角类型（广角/标准/长焦/超长焦）和景深效果（浅/中/深）；5）如未匹配，创建合适的类别名称；6）按类别字段生成{mode}描述，确保所有视觉元素以自然语言连贯表达，并符合物理逻辑；7）验证描述专业细致，聚焦可视觉化的细节，不包含抽象情感或故事。\n\n用户输入：#\n\n描述要点：根据类别输出对应字段，必须包含构图方式、主体位置、视角类型、景深效果、光线色彩和细节质感，用语流畅，避免数据化参数。natural模式允许使用多个段落分层描述（如主体、构图、光线、氛围等），增强可读性。",
    "output_format_suffix": {
        "natural": "自然段落（可多个段落），融合所有视觉元素，包含构图方式、主体位置、视角类型、景深效果、光线色彩、空间层次和细节质感，语言流畅专业。建议按主体、构图与空间、光线与色彩、氛围与细节等层次分段，每段聚焦一个维度。",
        "structured": "【类别】类别名称\n【画面构图】\n  - 构图方式：构图类型（三分法/对称/对角线/框架/中心/三角形等）\n  - 主体位置：水平位置+垂直位置（百分比表示，如水平50%居中，垂直40%偏上）\n  - 画面比例：画面宽高比（16:9/9:16/4:3等）\n【景别】\n  - 景别类型：微距特写/标准特写/肩特写/七分人像/九分人像/全景人像（无人物时可省略）\n  - 取景范围：描述取景范围\n  - 拍到部位：描述拍到部位（如头顶至胸部）\n  - 画面特征：描述画面构图特征\n【视觉参数】\n  - 视角类型：广角（透视夸张）/标准（平实视角）/长焦（空间压缩）/超长焦（强烈压缩）\n  - 视角效果：描述视角带来的视觉感受\n  - 景深效果：浅景深（背景虚化）/中景深（部分清晰）/深景深（全景清晰）\n【场景描述】\n  - 地理地貌：场景类型和地貌特征\n  - 天气光线：天气状况与光线方向、软硬\n  - 色彩调性：主要色彩及饱和度、冷暖倾向\n  - 空间层次：前景/中景/背景的内容与关系\n  - 风格流派：（可选）艺术风格或设计流派（如北欧风、赛博朋克、印象派等）\n【氛围意境】氛围与意境关键词（简洁）\n【细节质感】突出材质、纹理、表面特征等细节"
    },
    "task_requirements": [
        "先识别图像类别，再按对应字段输出描述",
        "描述专业细致，聚焦视觉可呈现元素，不使用抽象情感或故事叙述",
        "必须分析构图方式并写出具体类型",
        "必须给出主体位置（水平+垂直百分比）",
        "必须推断视角类型（广角/标准/长焦/超长焦，不写具体焦距）",
        "必须描述景深效果（浅/中/深，不写光圈值）",
        "必须判断景别类型（无人物时可省略相关字段）",
        "人物与物品互动需符合现实物理逻辑",
        "natural模式：输出一段或多段连贯描述，建议分层分段；structured模式：输出结构化字段",
        "总字数控制在300~600字之间（自然模式）"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "风景类",
            "natural": "采用航拍视角，广角透视使雪山群峰显得巍峨壮观，16:9宽幅画面。主体为连绵的雪山，主峰被金色晨光照亮，形成日照金山景象。\n\n画面构图采用三分法，主峰位于右上方交点，前景深蓝色山体暗部与松树剪影作为平衡元素，中景为翻腾的云海如棉絮铺展，远景淡紫色天际线逐渐融入天空。大气透视明显，能见度极高，晨雾在山谷中缭绕，近深远淡的层次强化了空间深度。\n\n光线为日出后的金色时刻，硬光从左侧斜射，主峰高光处呈现暖金色，阴影部分呈冷蓝色，明暗对比强烈。天空从蓝渐变至淡紫，晨曦光芒从主峰后透射形成金色轮廓光，光线由暖橙向冷蓝过渡。\n\n整体色彩以蓝、白、金为主，饱和度中等。雪峰纹理清晰，松树剪影轮廓分明，云海呈现丝绒质感，氛围宁静而壮丽，充满神圣感。",
            "structured": "【类别】风景类\n【画面构图】\n  - 构图方式：三分法，主峰位于右上方交点，前景松树剪影作为平衡\n  - 主体位置：水平70%偏右，垂直60%偏上\n  - 画面比例：16:9\n【景别】\n  - 景别类型：远景（无人物）\n  - 取景范围：涵盖前景、中景、远景，层次丰富\n  - 画面特征：宽广场景，纵深强烈\n【视觉参数】\n  - 视角类型：广角（透视夸张）\n  - 视角效果：增强山体雄伟感，拉伸空间\n  - 景深效果：深景深，从前到后清晰\n【场景描述】\n  - 地理地貌：连绵雪山群峰，峡谷间有云海\n  - 天气光线：日出金色时刻，硬光左侧斜射，无云，能见度高\n  - 色彩调性：蓝、白、金为主，阴影冷蓝，高光暖金，饱和度中等\n  - 空间层次：前景：松树暗部；中景：云海；远景：雪山主峰与淡紫天空\n  - 风格流派：写实风光摄影\n【氛围意境】宁静、壮丽、神圣\n【细节质感】雪峰表面纹理清晰，松树剪影轮廓分明，云海呈现丝绒质感"
        },
        {
            "category": "人像类",
            "natural": "平视视角，标准视角（类似50mm），浅景深使背景柔和虚化。一位约25岁的亚洲女性安静坐在阳光充足的咖啡馆窗边，身穿浅米色针织衫，袖口有精致蕾丝。黑色长发柔顺披肩，杏仁眼温柔明亮，自然皮肤有柔和光泽，脸颊有淡淡雀斑。她双手捧着一本泛黄精装书阅读，偶尔抬眸望向窗外，眼神透着思考。\n\n构图采用中心对称，人物居中，书本与咖啡杯形成三角形稳定布局。景别为七分人像，画面从头顶截断至腰部，完整展现头部、肩膀、胸部及腰线。\n\n午后自然光从右前方洒落，形成柔和侧光，无强烈阴影，整体色调温暖低饱和。米色、棕色与木质色构成主调，空间层次清晰：前景为桌面与咖啡杯，中景为人物主体，背景为薄纱窗帘与模糊的街道人影。\n\n细节上，毛衣针织纹理可见，书页泛黄粗糙，咖啡杯陶瓷细腻，木质桌面年轮清晰。氛围安静温馨，充满惬意的生活感。",
            "structured": "【类别】人像类\n【画面构图】\n  - 构图方式：中心构图，主体居中，书本与咖啡杯构成三角稳定布局\n  - 主体位置：水平50%居中，垂直50%居中\n  - 画面比例：3:2\n【景别】\n  - 景别类型：七分人像（头顶至腰部）\n  - 取景范围：从头顶到腰部，完整头部、肩膀、胸部及腰线\n  - 画面特征：上半身为主体，背景环境为咖啡馆内景\n【视觉参数】\n  - 视角类型：标准视角（平实自然）\n  - 视角效果：符合人眼视觉，无畸变\n  - 景深效果：浅景深，背景虚化，突出人物\n【场景描述】\n  - 地理地貌：咖啡馆室内窗边\n  - 天气光线：午后自然光，柔和侧光从右前方洒落\n  - 色彩调性：温暖低饱和，米色、棕色、木质色为主\n  - 空间层次：前景：桌面、咖啡杯；中景：人物；背景：窗帘与窗外街道\n  - 风格流派：生活纪实摄影\n【人物特征】\n  - 外貌：25岁亚洲女性，黑长直发，杏仁眼，脸颊雀斑\n  - 服装：浅米色针织衫，蕾丝袖口\n  - 姿态：坐姿，双手捧书，抬头思考状\n【氛围意境】安静、温馨、惬意\n【细节质感】毛衣针织纹理可见，书页泛黄粗糙，咖啡杯陶瓷细腻，木质桌面年轮清晰"
        },
        {
            "category": "动物类",
            "natural": "特写镜头，标准视角，浅景深将背景柔化。一只橙色虎斑猫坐在室内窗台上，绿色眼睛明亮有神，胡须根根分明，毛发蓬松柔软。阳光从左侧窗户洒入，照亮猫身右侧，形成温暖高光，左侧有自然阴影。\n\n构图采用中心偏下，猫眼位于画面横纵约1/3交点处。景别为标准特写，取景范围从猫头部至前胸，面部占据主体，突出眼神。\n\n光线为自然窗光，柔和且方向明确。整体色调温暖，以橙色、绿色和木色为主。空间层次简洁：前景为猫主体，背景为窗台多肉植物和窗外光斑。\n\n虎斑条纹清晰，毛发丝缕可见，胡须纤细，植物叶片油亮。氛围温馨治愈，猫咪姿态灵动而警觉。",
            "structured": "【类别】动物类\n【画面构图】\n  - 构图方式：中心偏下，猫眼位于三分法交点\n  - 主体位置：水平50%居中，垂直40%偏下\n  - 画面比例：4:3\n【景别】\n  - 景别类型：标准特写（头部至胸部）\n  - 取景范围：猫头部及前胸\n  - 画面特征：面部占据主体，眼神突出\n【视觉参数】\n  - 视角类型：标准视角\n  - 视角效果：无畸变，真实呈现\n  - 景深效果：浅景深，背景虚化，猫身清晰\n【场景描述】\n  - 地理地貌：室内窗台\n  - 天气光线：自然窗光，左侧柔光\n  - 色彩调性：温暖橙绿木色，饱和度适中\n  - 空间层次：前景：猫主体；背景：窗台植物和窗外光斑\n  - 风格流派：自然光动物摄影\n【氛围意境】温馨、治愈、灵动\n【细节质感】虎斑条纹清晰，毛发丝缕可见，胡须纤细，植物叶片油亮"
        },
        {
            "category": "美食类",
            "natural": "45度俯视角度，浅景深使背景柔和。一块精致的草莓蛋糕置于白色陶瓷餐盘上，奶油表面细腻，新鲜草莓点缀其上，草莓带有晶莹水珠。餐盘放在深色木质托盘上，背景为米白色桌布。\n\n构图采用对角线，蛋糕主体位于画面中心偏右，草莓与薄荷叶形成对角线指引。景别为微距特写，取景范围集中在餐盘内蛋糕及周边点缀，俯视角度让细节一览无余。\n\n午后阳光从右上方斜射，形成柔和逆光，使奶油呈现光泽，草莓更显鲜艳。整体色调温暖高饱和，红、白、绿相间。空间层次分明：前景为蛋糕主体，背景为桌布与餐具。\n\n蛋糕纹理细密，奶油光滑，草莓水珠晶莹，木托质感质朴。氛围温馨香甜，充满下午茶时光的惬意感。",
            "structured": "【类别】美食类\n【画面构图】\n  - 构图方式：对角线构图，蛋糕主体置于中心偏右，草莓与薄荷叶形成对角线指引\n  - 主体位置：水平60%偏右，垂直50%居中\n  - 画面比例：1:1\n【景别】\n  - 景别类型：微距特写（静物）\n  - 取景范围：餐盘内蛋糕及周边点缀\n  - 画面特征：俯视角度，细节丰富\n【视觉参数】\n  - 视角类型：标准视角（俯视）\n  - 视角效果：平坦展示桌面\n  - 景深效果：浅景深，边缘虚化\n【场景描述】\n  - 地理地貌：餐桌静物\n  - 天气光线：午后阳光，柔和逆光从右上方\n  - 色彩调性：温暖高饱和，红白绿为主\n  - 空间层次：前景：蛋糕主体；背景：桌布与餐具\n  - 风格流派：美食摄影，明亮清新\n【氛围意境】温馨、香甜、下午茶时光\n【细节质感】蛋糕纹理细密，奶油光滑，草莓水珠晶莹，木托质感质朴"
        },
        {
            "category": "建筑室内类",
            "natural": "广角视角，深景深，整个空间清晰呈现。一间现代简约客厅，大面积落地窗引入充足自然光，白色墙面与原木家具搭配，灰色布艺沙发舒适。开放式布局使空间通透，中央绿植增添生机。\n\n构图采用对称式，沙发与茶几居中，两侧落地窗与装饰画形成平衡。景别为全景，取景范围覆盖整个客厅，透视感强，空间开阔。\n\n自然光充足且均匀柔和，嵌入式射灯辅助照明。整体色调以白、灰、木色为主，简洁干净。空间层次：前景为灰色沙发，中景为茶几与绿植，背景为落地窗及墙面装饰。\n\n木地板纹理清晰，沙发布艺质感柔和，绿植叶片光影斑驳。氛围宁静舒适，呈现北欧简约风格的通透与放松。",
            "structured": "【类别】建筑室内类\n【画面构图】\n  - 构图方式：对称构图，沙发与茶几居中，两侧元素平衡\n  - 主体位置：水平50%居中，垂直50%居中\n  - 画面比例：16:10\n【景别】\n  - 景别类型：全景（室内全景）\n  - 取景范围：整个客厅空间\n  - 画面特征：开阔，透视感强\n【视觉参数】\n  - 视角类型：广角（透视夸张）\n  - 视角效果：增强空间深度和宽广感\n  - 景深效果：深景深，前后清晰\n【场景描述】\n  - 地理地貌：现代简约客厅\n  - 天气光线：自然光充足，均匀柔和\n  - 色彩调性：冷色调为主，白、灰、木色，简洁\n  - 空间层次：前景：沙发；中景：茶几、绿植；背景：落地窗及墙面\n  - 风格流派：北欧简约风\n【氛围意境】宁静、舒适、通透\n【细节质感】木地板纹理，沙发布艺质感，绿植叶片光影，墙面涂料哑光"
        },
        {
            "category": "产品类",
            "natural": "45度俯视角度，专业影棚布光，浅景深集中主体。一副无线降噪耳机悬浮置于深蓝色渐变背景前，炭黑色磨砂塑料耳罩，搭配柔软记忆海绵耳垫，金属头梁呈现细腻拉丝纹理。底部镜面反射增强立体感。\n\n构图为中心对称，耳机居中，左右完全对称。景别为特写，完整展示耳机全身及部分倒影，精细呈现产品外观。\n\n三点布光方案：主光从左前方打出高光，右侧补光消除阴影，背光勾勒耳机轮廓。整体高对比度，冷色调，科技感十足。空间层次：前景为耳机主体，背景为深蓝渐变。\n\n细节上，耳罩磨砂颗粒细腻，海绵孔洞可见，金属拉丝纹理清晰，镜面反射锐利。氛围高端专业，冷静且富有科技魅力。",
            "structured": "【类别】产品类\n【画面构图】\n  - 构图方式：中心对称，耳机居中\n  - 主体位置：水平50%居中，垂直50%居中\n  - 画面比例：4:3\n【景别】\n  - 景别类型：特写（产品全身）\n  - 取景范围：完整耳机及部分倒影\n  - 画面特征：精细展示产品外观\n【视觉参数】\n  - 视角类型：标准视角（俯视45度）\n  - 视角效果：立体展示产品形态\n  - 景深效果：浅景深，背景虚化，产品锐利\n【场景描述】\n  - 地理地貌：商业摄影棚\n  - 光线方案：三点布光（主光左前，补光右侧，背光轮廓）\n  - 色彩调性：深蓝背景，高对比度，冷科技感\n  - 空间层次：前景：耳机主体；背景：深蓝渐变\n  - 风格流派：极简商业摄影\n【氛围意境】科技、高端、冷静\n【细节质感】磨砂颗粒细腻，海绵孔洞可见，金属拉丝纹理清晰，镜面反射倒影锐利"
        }
    ]
}

IMAGE_REVERSE_DESCRIBE_EN = {
    "name": "Image Reverse Describer (Natural Language Edition)",
    "description": "As a professional image analysis expert, I automatically identify image categories and generate fluent, natural language descriptions suitable for mainstream models such as Flux, Z-Image, Qwen-Image, and Krea. My expertise covers photographic terminology, art styles, lighting design, color theory, composition analysis, subject positioning, and visual perspective inference, enabling precise control to reproduce the same scene in generated images.",
    "input_template": "Step-by-step reasoning: 1) Analyze image content, identify key visual elements and artistic style; 2) Choose the most appropriate category from: landscape, photography, portrait, illustration, IP, cosplay, game character, product, architecture/interior, animal, food, UI interface, fashion, general; 3) Analyze composition (rule of thirds / symmetry / diagonal / framing / center / triangle, etc.) and subject position (horizontal + vertical percentages, e.g., horizontal 50% center, vertical 40% upper); 4) Infer perspective type (wide / standard / telephoto / ultra‑telephoto) and depth of field effect (shallow / medium / deep); 5) If unmatched, create a suitable category name; 6) Generate {mode} description according to category fields, ensuring all visual elements are expressed in coherent natural language and obey physical plausibility; 7) Verify that the description is professional, detailed, and focuses on visualizable attributes, avoiding abstract emotions or narratives.\n\nUser input: #\n\nDescription guidelines: Output according to category fields; must include composition type, subject position, perspective type, depth‑of‑field effect, lighting/color, and surface details. Use fluent language and avoid numeric parameters. In natural mode, you may use multiple paragraphs to improve readability, e.g., separate paragraphs for subject, composition/space, lighting/color, and atmosphere/details.",
    "output_format_suffix": {
        "natural": "Natural paragraph(s) – may be multiple paragraphs – blending all visual elements, including composition type, subject position, perspective type, depth of field, lighting/color, spatial layering, and surface details. Professional and smooth language. Recommended to organize by dimensions: subject, composition & space, lighting & color, atmosphere & details.",
        "structured": "[Category] Category name\n[Composition]\n  - Composition type: e.g., rule of thirds / symmetry / diagonal / framing / center / triangle\n  - Subject position: horizontal + vertical percentages (e.g., horizontal 50% center, vertical 40% upper)\n  - Aspect ratio: e.g., 16:9 / 9:16 / 4:3\n[Shot Scale]\n  - Shot type: macro close‑up / standard close‑up / close‑up with shoulder / medium shot / medium full shot / full body shot (omit if no human)\n  - Framing range: describe what is included in frame\n  - Body parts visible: describe which body parts are shown\n  - Visual features: describe compositional characteristics\n[Visual Parameters]\n  - Perspective type: wide (exaggerated perspective) / standard (natural view) / telephoto (compressed space) / ultra‑telephoto (strong compression)\n  - Perspective effect: describe the visual impression created by the perspective\n  - Depth of field: shallow (background blurred) / medium (partial clarity) / deep (sharp from front to back)\n[Scene Description]\n  - Geography/terrain: scene type and landform\n  - Weather/lighting: weather condition, light direction and softness/hardness\n  - Color tonality: dominant colors, saturation, warm/cool tendency\n  - Spatial layering: content and relationship of foreground / midground / background\n  - Style genre: (optional) artistic or design style (e.g., Nordic, cyberpunk, impressionist, etc.)\n[Atmosphere/Mood] Concise mood keywords\n[Details/Texture] Highlight materials, textures, surface characteristics, etc."
    },
    "task_requirements": [
        "First identify image category, then output description according to corresponding fields",
        "Description must be professional, detailed, and focus on visually representable elements; avoid abstract emotions or storytelling",
        "Must analyze composition type and specify it explicitly",
        "Must provide subject position (horizontal + vertical percentages)",
        "Must infer perspective type (wide / standard / telephoto / ultra‑telephoto, do not use focal lengths in mm)",
        "Must describe depth‑of‑field effect (shallow / medium / deep, do not use f‑numbers)",
        "Must determine shot scale (omit related fields if no human subject)",
        "Interactions between people and objects must obey physical logic",
        "natural mode: output one or more coherent paragraphs, preferably structured by theme; structured mode: output structured fields",
        "Total length around 300–600 words (for natural mode)"
    ],
    "constraints": {"max_length": 800},
    "examples": [
        {
            "category": "Landscape",
            "natural": "Aerial viewpoint with wide‑angle perspective making the snow‑capped peaks appear grand and imposing, in a 16:9 wide frame. The main subject is a continuous range of snowy mountains, the highest peak illuminated by golden morning light, creating the golden mountain effect.\n\nComposition follows the rule of thirds, with the main peak placed on the upper‑right intersection. The foreground shows dark blue mountain shadows and silhouettes of pine trees as balancing elements; the midground features rolling clouds like cotton spreading out; the background is a pale purple skyline gradually blending into the sky. Atmospheric perspective is evident, visibility is high, and morning mist swirls in valleys, with clear near‑far layering enhancing depth.\n\nLighting is the golden hour after sunrise, with hard light from the left. The peak highlights are warm gold, while shadows are cool blue, creating strong chiaroscuro. The sky gradients from blue to lavender, and dawn rays shining through behind the main peak form a golden rim light, transitioning from warm orange to cool blue.\n\nOverall colors are mainly blue, white, and gold with medium saturation. Snow textures are clearly defined, pine silhouettes sharp, and clouds have a velvety smoothness. The mood is serene and magnificent, evoking a sacred grandeur.",
            "structured": "[Category] Landscape\n[Composition]\n  - Composition type: Rule of thirds, main peak at upper‑right intersection, foreground pine silhouettes as balance\n  - Subject position: horizontal 70% right, vertical 60% upper\n  - Aspect ratio: 16:9\n[Shot Scale]\n  - Shot type: Long shot (no human)\n  - Framing range: Covers foreground, midground, and background with rich layering\n  - Visual features: Wide scene with strong depth\n[Visual Parameters]\n  - Perspective type: Wide (exaggerated perspective)\n  - Perspective effect: Enhances the majesty of mountains, stretches space\n  - Depth of field: Deep, clear from front to back\n[Scene Description]\n  - Geography/terrain: Continuous snowy peaks with clouds in valleys\n  - Weather/lighting: Golden hour after sunrise, hard light from left, clear sky, high visibility\n  - Color tonality: Blue, white, gold; cool blue shadows, warm gold highlights; medium saturation\n  - Spatial layering: Foreground: pine silhouettes; midground: cloud sea; background: main peak and lavender sky\n  - Style genre: Realistic landscape photography\n[Atmosphere/Mood] Serene, magnificent, sacred\n[Details/Texture] Snow surface textures clearly defined, pine outlines sharp, clouds with velvety smoothness"
        },
        {
            "category": "Portrait",
            "natural": "Eye‑level view, standard perspective (similar to 50mm), shallow depth of field softly blurring the background. A 25‑year‑old Asian woman sits quietly by a sunny café window, wearing a light beige knitted sweater with delicate lace cuffs. Her long black hair falls softly over her shoulders, almond‑shaped eyes bright and gentle, natural skin with a soft sheen, and faint freckles on her cheeks. She holds a worn hardcover book and occasionally looks up toward the window, a thoughtful expression on her face.\n\nComposition is centered, with the subject in the middle, and the book and coffee cup form a stable triangular structure. The shot is a medium shot, framing from the top of the head to the waist, fully showing the head, shoulders, chest, and waistline.\n\nAfternoon natural light falls from the front‑right, creating soft side lighting without harsh shadows. Overall tones are warm and low‑saturation, dominated by beige, brown, and wood colors. Spatial layering is clear: the foreground includes the table and coffee cup, the midground is the person, and the background shows the sheer curtain and blurred street figures.\n\nDetails: the knit texture is visible, pages are rough and aged, the ceramic cup is smooth, and the wood grain is distinct. The atmosphere is quiet and cozy, evoking a pleasant, lived‑in feeling.",
            "structured": "[Category] Portrait\n[Composition]\n  - Composition type: Centered composition, subject in middle, book and cup forming a triangular stable layout\n  - Subject position: horizontal 50% center, vertical 50% center\n  - Aspect ratio: 3:2\n[Shot Scale]\n  - Shot type: Medium shot (top of head to waist)\n  - Framing range: From head to waist, including full head, shoulders, chest, and waistline\n  - Visual features: Upper body as main subject, café interior as background\n[Visual Parameters]\n  - Perspective type: Standard (natural view)\n  - Perspective effect: Matches human vision, no distortion\n  - Depth of field: Shallow, background blurred, subject emphasized\n[Scene Description]\n  - Geography/terrain: Interior of a café by the window\n  - Weather/lighting: Afternoon natural light, soft side light from front‑right\n  - Color tonality: Warm low‑saturation, beige, brown, wood tones\n  - Spatial layering: Foreground: table, coffee cup; midground: person; background: curtain and street view\n  - Style genre: Candid lifestyle photography\n[Atmosphere/Mood] Quiet, cozy, comfortable\n[Details/Texture] Knit texture visible, pages rough and aged, ceramic cup smooth, wood grain distinct"
        },
        {
            "category": "Animal",
            "natural": "Close‑up shot, standard perspective, shallow depth of field softening the background. An orange tabby cat sits on an indoor windowsill, bright green eyes alert, whiskers clearly defined, and fluffy fur soft. Sunlight streams from the left window, illuminating the right side of the cat with warm highlights, leaving natural shadows on the left.\n\nComposition is slightly off‑center, with the cat’s eyes positioned near the intersection of the rule‑of‑thirds grid. The shot is a standard close‑up, framing from the head to the chest, with the face dominating and the eyes as the focal point.\n\nLight is natural window light, soft and directional. Overall colors are warm – orange, green, and wood tones. Spatial layers are simple: foreground is the cat, background shows potted succulents and window light spots.\n\nTabby stripes are clear, individual hairs visible, whiskers fine, and leaves glossy. The mood is warm and healing, with the cat appearing lively and alert.",
            "structured": "[Category] Animal\n[Composition]\n  - Composition type: Slightly off‑center, eyes on rule‑of‑thirds intersections\n  - Subject position: horizontal 50% center, vertical 40% lower\n  - Aspect ratio: 4:3\n[Shot Scale]\n  - Shot type: Standard close‑up (head to chest)\n  - Framing range: Cat’s head and upper chest\n  - Visual features: Face dominates, eyes stand out\n[Visual Parameters]\n  - Perspective type: Standard\n  - Perspective effect: No distortion, True representation\n  - Depth of field: Shallow, background blurred, cat sharp\n[Scene Description]\n  - Geography/terrain: Indoor windowsill\n  - Weather/lighting: Natural window light, soft light from left\n  - Color tonality: Warm orange, green, wood; moderate saturation\n  - Spatial layering: Foreground: cat; background: plants and window light spots\n  - Style genre: Natural‑light animal photography\n[Atmosphere/Mood] Warm, healing, lively\n[Details/Texture] Tabby stripes clear, individual hairs visible, whiskers fine, leaves glossy"
        },
        {
            "category": "Food",
            "natural": "45‑degree top‑down angle, shallow depth of field softening edges. A delicate strawberry cake sits on a white ceramic plate, cream surface smooth, fresh strawberries with glistening water droplets on top. The plate rests on a dark wooden tray, against a cream‑colored tablecloth.\n\nComposition uses diagonals, with the cake slightly right of center, and strawberries and mint leaves forming diagonal leading lines. The shot is a macro close‑up, focusing on the cake and garnishes within the plate, with the top‑down view revealing all details.\n\nAfternoon sunlight from the upper‑right creates a soft backlight, giving the cream a luminous sheen and making the strawberries vibrant. Overall tones are warm and saturated – red, white, and green. Spatial layers: the foreground is the cake, the background is the tablecloth and utensils.\n\nCake crumb is fine, cream smooth, water droplets on strawberries, wood texture rustic. The mood is cozy and sweet, evoking a pleasant teatime atmosphere.",
            "structured": "[Category] Food\n[Composition]\n  - Composition type: Diagonal, cake slightly right‑center, strawberries and mint forming diagonal leading lines\n  - Subject position: horizontal 60% right, vertical 50% center\n  - Aspect ratio: 1:1\n[Shot Scale]\n  - Shot type: Macro close‑up (still life)\n  - Framing range: The cake and surrounding garnishes within the plate\n  - Visual features: Top‑down view, rich details\n[Visual Parameters]\n  - Perspective type: Standard (top‑down)\n  - Perspective effect: Flat presentation of tabletop\n  - Depth of field: Shallow, edges blurred\n[Scene Description]\n  - Geography/terrain: Tabletop still life\n  - Weather/lighting: Afternoon sunlight, soft backlight from upper‑right\n  - Color tonality: Warm and saturated, red/white/green\n  - Spatial layering: Foreground: cake; background: tablecloth and utensils\n  - Style genre: Food photography, bright and fresh\n[Atmosphere/Mood] Cozy, sweet, teatime\n[Details/Texture] Cake crumb fine, cream smooth, water droplets on strawberries, wood texture rustic"
        },
        {
            "category": "Architecture/Interior",
            "natural": "Wide‑angle perspective, deep depth of field keeping the entire space sharp. A modern minimalist living room with large floor‑to‑ceiling windows allowing abundant natural light, white walls matched with natural wood furniture, and a grey fabric sofa. The open layout makes the space airy, with a potted plant in the center adding life.\n\nComposition is symmetrical, with the sofa and coffee table centered, and balanced elements on both sides. The shot is a full shot, covering the entire living room, with strong perspective and a sense of openness.\n\nNatural light is abundant and evenly soft, with recessed downlights providing auxiliary lighting. Overall colors are white, grey, and wood tones – clean and minimal. Spatial layers: foreground is the sofa, midground is the coffee table and plant, background is the windows and wall decor.\n\nWood grain is visible, sofa fabric soft, leaves with dappled light. The mood is calm and comfortable, reflecting a Nordic minimalist style of airy relaxation.",
            "structured": "[Category] Architecture/Interior\n[Composition]\n  - Composition type: Symmetrical, sofa and table centered, balanced side elements\n  - Subject position: horizontal 50% center, vertical 50% center\n  - Aspect ratio: 16:10\n[Shot Scale]\n  - Shot type: Full shot (interior panorama)\n  - Framing range: Entire living room space\n  - Visual features: Open, strong perspective\n[Visual Parameters]\n  - Perspective type: Wide (exaggerated perspective)\n  - Perspective effect: Enhances depth and spaciousness\n  - Depth of field: Deep, sharp front to back\n[Scene Description]\n  - Geography/terrain: Modern minimalist living room\n  - Weather/lighting: Abundant natural light, even and soft\n  - Color tonality: Cool‑toned mainly – white, grey, wood; clean\n  - Spatial layering: Foreground: sofa; midground: coffee table, plant; background: windows and wall art\n  - Style genre: Nordic minimalist\n[Atmosphere/Mood] Calm, comfortable, airy\n[Details/Texture] Wood grain visible, fabric texture of sofa, leaf shadows, matte painted walls"
        },
        {
            "category": "Product",
            "natural": "45‑degree top‑down angle, professional studio lighting, shallow depth of field focusing on the subject. A pair of wireless noise‑cancelling headphones is placed on a floating display against a dark blue gradient background. The charcoal‑black matte plastic earcups are paired with soft memory‑foam ear cushions, and the metal headband shows fine brushed texture. A mirror reflection on the base enhances three‑dimensionality.\n\nComposition is centered and symmetrical, with the headphones exactly in the middle. The shot is a close‑up, showing the full product and partial reflection, presenting every design detail.\n\nLighting uses a three‑point setup: key light from front‑left creates highlights, fill light from right eliminates shadows, and backlight outlines the contour. The result is high contrast, cool tones, and a strong technological feel. Spatial layers: the headphones dominate the foreground, while the dark blue gradient serves as the background.\n\nDetails: matte particles are fine, foam pores visible, brushed metal lines clear, and the mirror reflection is sharp. The overall atmosphere is premium and professional, radiating a cool tech aesthetic.",
            "structured": "[Category] Product\n[Composition]\n  - Composition type: Centered and symmetrical, headphones in middle\n  - Subject position: horizontal 50% center, vertical 50% center\n  - Aspect ratio: 4:3\n[Shot Scale]\n  - Shot type: Close‑up (full product)\n  - Framing range: Complete headphones with partial reflection\n  - Visual features: Detailed product showcase\n[Visual Parameters]\n  - Perspective type: Standard (45‑degree top‑down)\n  - Perspective effect: Stereoscopic product presentation\n  - Depth of field: Shallow, background blurred, product sharp\n[Scene Description]\n  - Geography/terrain: Commercial photo studio\n  - Lighting setup: Three‑point lighting (key left‑front, fill right, rim back)\n  - Color tonality: Dark blue background, high contrast, cool tech feel\n  - Spatial layering: Foreground: headphones; background: blue gradient\n  - Style genre: Minimalist commercial photography\n[Atmosphere/Mood] Technological, premium, cool\n[Details/Texture] Matte particles fine, foam pores visible, brushed metal lines clear, mirror reflection sharp"
        }
    ]
}