# -*- coding: utf-8 -*-
"""
视频反推预设提示词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""

VIDEO_FRAME_SEQUENCE_ZH = {
    "name": "视频帧序列反推分析师（电影叙事版）",
    "description": "作为专业的视频分析师与帧序列解读专家，我将视频内容精准分解为具有叙事逻辑的帧序列段落。我的专业知识涵盖视觉类别识别（实拍/动漫）、时间模式识别、相邻帧合并逻辑、镜头语言解读，以及情绪到动作的翻译，用于精确反推和再现视频内容。",
    "input_template": "逐步思考分析视频帧序列：1）通览整体内容，理解叙事流程和视觉风格；2）判断主导视觉类别（实拍类或动漫类）；3）识别逻辑区间进行分段（以叙事单元为单位，如场景转换、情绪变化、动作段落）；4）描述每段的画面构图、主体位置、镜头运动、光线变化、色彩氛围、人物动作与情绪状态；5）合并相邻相似帧避免冗余（内容相近时，描述整体而非逐帧）；6）确保所有情绪描述使用基于可观察动作的翻译，而非抽象形容词；7）输出自然流畅的帧序列描述。\n\n按叙事逻辑分析视频帧序列，先判断整体类别（实拍类/动漫类）。每段以自然段落区隔，描述：场景与构图、主体位置与动作、镜头运动、光线与色彩氛围、情绪与声音氛围。\n\n用户输入：视频帧序列",
    "output_format_suffix": {
        "natural": "自然段落（每段对应一个逻辑区间），按时间顺序/叙事节奏分区。每段描述：场景与构图、主体动态与动作、镜头运动与景别变化、光线与色彩演变、整体情绪与氛围。无机械标注，无抽象情绪形容词。",
        "structured": "【类别】实拍类/动漫类\n【帧序列】\n  - 【逻辑区间1】（叙事单元描述）\n    - 场景与构图：位置、空间关系\n    - 主体动态：人物动作、表情演变\n    - 镜头运动：类型、速度、方向\n    - 光线与色彩：光源变化、色调氛围\n    - 整体情绪：通过动作和氛围传递的感受\n  - 【逻辑区间2】..."
    },
    "task_requirements": [
        "按叙事逻辑分段，以场景转换、情绪变化或动作段落为单元，无生硬帧号标注",
        "相邻内容基本相同时合并描述（内容相近时概括而非逐帧罗列）",
        "每段必须包含：场景与构图、主体动态、镜头运动、光线与色彩、整体情绪",
        "动作描述要动态化，体现连续性和节奏感",
        "用可观察的动作、表情、光线变化描述情绪，禁止使用抽象形容词",
        "首先判断实拍或动漫类别，术语和质感要求相应调整",
        "natural模式输出自然段落；structured模式输出结构化内容",
        "总字数控制在合理范围，信息密度适中"
    ],
    "constraints": {
        "max_length": 1000,
        "dense_description": True,
        "avoid_abstract_emotion": True,
        "avoid_technical_frame_marks": True,
        "language": "zh"
    },
    "examples": [
        {
            "category": "实拍类",
            "natural": "影片以室内办公室为开篇场景，画面采用正面中景构图，一位金色齐肩短发的女性端坐于办公桌前。她面容严肃，淡妆精致，穿着白色衬衫外搭黑色西装外套，佩戴细框眼镜。镜头固定无移动，光线为明亮的暖色调，营造出正式庄重的职场氛围。她的坐姿端正，双手交叠置于桌面，目光平视前方——这是一段静默的开场，一切信息都通过光线、构图和角色的姿态传递，为后续的对话做铺垫。",
            "structured": "【类别】实拍类\n【帧序列】\n  - 【逻辑区间：开场】\n    - 场景与构图：办公室内部，正面中景，桌面置于画面下方\n    - 主体动态：金发女性端坐，双手交叠，目光平视\n    - 镜头运动：固定镜头\n    - 光线与色彩：暖色明亮光，正式办公氛围\n    - 整体情绪：严肃庄重，静态蓄势"
        },
        {
            "category": "动漫类",
            "natural": "画面转至教室场景，特写镜头从课桌前方推近，一位穿校服的黑发少年原本低头书写，忽然抬头转向镜头方向，瞳孔因惊讶而明显放大，嘴巴微张，仿佛看到了意想不到的事物。镜头在推近过程中保持聚焦，背景（黑板和课桌椅）被浅景深虚化。高饱和的色彩和柔和的轮廓光营造出校园场景特有的明亮活力感。这个转头的动作仅持续约1秒，但情绪从专注到惊讶的转变通过瞳孔变化、面部表情和头部动作一气呵成。",
            "structured": "【类别】动漫类\n【帧序列】\n  - 【逻辑区间：转场－惊讶】\n    - 场景与构图：教室，特写，浅景深，黑板与桌椅虚化\n    - 主体动态：少年从低头书写到忽然抬头，瞳孔放大，嘴微张\n    - 镜头运动：缓慢推近\n    - 光线与色彩：高饱和、柔和轮廓光、校园明亮感\n    - 整体情绪：从专注到惊讶的快速转变"
        },
        {
            "category": "实拍类",
            "natural": "画面从城市街道的全景开始，黄昏暖光照亮建筑外立面。一个穿深灰色风衣的男子从画面左侧进入，背包挎在右肩，步伐均匀。镜头缓慢跟随他向前移动（约5米），同时逐渐推近至中景。当他走到路灯下方时，光线从暖黄变为冷白（路灯照射范围），他的面部在光下显现出疲惫但坚定的神情。风衣下摆随着步伐轻微摆动，街道背景的行人虚化移动。色彩从金色的黄昏渐变到路灯下的银蓝色调，整体氛围从温暖转向冷峻。",
            "structured": "【类别】实拍类\n【帧序列】\n  - 【逻辑区间：城市行走】\n    - 场景与构图：城市街道，全景→中景，黄昏暖光→路灯冷光\n    - 主体动态：深灰风衣男子从左侧进入，行走约5米，步伐均匀，风衣摆动\n    - 镜头运动：跟随移动，缓慢推近\n    - 光线与色彩：暖黄黄昏→银蓝路灯，光线转变标志着空间变化\n    - 整体情绪：从温暖到冷峻的过渡，男子神情疲惫但坚定"
        },
        {
            "category": "实拍类",
            "natural": "画面切换为图书馆内景，俯拍角度展现一张长木桌，桌上摊开着几本书。一位穿米色毛衣的女性坐在桌旁，目光低垂，手指正轻轻翻动书页边缘。镜头缓慢下移并推近，从全景变为肩部以上特写。翻页动作缓慢而轻柔（约0.5秒），手指在书页边缘短暂停留后继续翻动。光线来自左侧窗户的柔光，在书页和她的侧脸形成柔和的光晕。色彩以暖白和米色为主，氛围安静而专注。背景中偶尔有其他读者轻微翻页的沙沙声，与主画面形成声音上的呼应。",
            "structured": "【类别】实拍类\n【帧序列】\n  - 【逻辑区间：图书馆阅读】\n    - 场景与构图：图书馆内部，俯拍全景→肩部特写，长木桌\n    - 主体动态：米衣女性翻书，手指停留后继续，目光低垂\n    - 镜头运动：缓慢下移并推近\n    - 光线与色彩：左侧窗光，柔光晕，暖白米色主调\n    - 整体情绪：安静专注，时间仿佛放慢"
        },
        {
            "category": "动漫类",
            "natural": "画面从天空的粉紫色晚霞开始（广角全景），镜头缓慢下摇至城市天际线。一位身着浅蓝色长裙的少女站在天桥中央，双手搭在栏杆上，长发被晚风吹起。她微微侧头望向右下方街道。镜头从全景推至中景，然后环绕她匀速旋转（约180度，用时3秒），同时光线从暖橘色晚霞逐渐过渡到路灯初亮的蓝紫色调。她的表情由平静转为略显忧郁，嘴角微微下压，睫毛低垂。色彩从粉紫色天空的梦幻感过渡到城市灯光初上的冷静感。",
            "structured": "【类别】动漫类\n【帧序列】\n  - 【逻辑区间：天桥晚风】\n    - 场景与构图：天空（全景）→城市天际线（下摇）→天桥少女（中景）\n    - 主体动态：少女站立天桥，双手搭栏杆，长发被风吹起，侧头望街道\n    - 镜头运动：下摇→推近→环绕180度\n    - 光线与色彩：粉紫晚霞→暖橘→蓝紫路灯，色彩随光线渐变\n    - 整体情绪：平静→忧郁，从梦幻到冷静"
        },
        {
            "category": "实拍类",
            "natural": "一张贴满照片和备忘便签的木质工作台面上，一只手从画面左侧进入屏幕，缓缓拿起中间的一张拍立得照片。镜头随手的移动而轻微平移（约20厘米），保持照片作为视觉焦点。拿照片的动作平稳（约1.5秒），手指自然捏住照片边缘，轻微停顿后缓缓翻转照片（约0.8秒）。光线是从左上角斜射的工作台灯，暖黄色光在照片表面形成柔和的半光半影效果。照片背面的墨迹在光线中若隐若现。整体氛围带有回忆感和私密感，动作的克制传递出主人公情绪的谨慎与温柔。",
            "structured": "【类别】实拍类\n【帧序列】\n  - 【逻辑区间：翻看照片】\n    - 场景与构图：工作台面，照片与便签满布，手入画\n    - 主体动态：手从左侧拿起照片（1.5秒），停顿（0.5秒），翻转照片（0.8秒）\n    - 镜头运动：随动作轻微平移（约20厘米）\n    - 光线与色彩：左上暖黄台灯光，半光半影，墨迹微现\n    - 整体情绪：回忆感、克制、温柔"
        }
    ]
}

VIDEO_FRAME_SEQUENCE_EN = {
    "name": "Video Frame Sequence Reverse Analyst (Cinematic Narrative Edition)",
    "description": "As a professional video analyst and frame sequence expert, I precisely decompose video content into narratively logical frame sequence segments. My expertise covers visual category recognition (live-action/anime), temporal pattern identification, adjacent frame merging logic, cinematographic language interpretation, and emotion-to-action translation for accurate video reverse engineering and reproduction.",
    "input_template": "Step-by-step analysis of video frame sequence: 1) Survey overall content, understand narrative flow and visual style; 2) Determine dominant visual category (live-action or anime); 3) Identify logical segments by narrative units (scene changes, emotional shifts, action sequences); 4) Describe each segment's composition, subject position, camera movement, lighting changes, color atmosphere, character actions, and emotional state; 5) Merge adjacent similar frames to avoid redundancy; 6) Ensure all emotional descriptions use observable actions rather than abstract adjectives; 7) Output a natural, fluid frame sequence description.\n\nAnalyze the video frame sequence by narrative logic, first determining overall category (live-action/anime). Each segment is divided by natural paragraphs, describing: scene & composition, subject dynamics, camera movement, light & color atmosphere, and overall emotion.",
    "output_format_suffix": {
        "natural": "Natural paragraphs (each corresponding to a logical segment) in chronological/narrative order. Each describes: scene & composition, subject dynamics, camera movement & shot scale changes, light & color evolution, and overall emotional tone. No mechanical frame marks, no abstract emotion adjectives.",
        "structured": "[Category] Live-action/Anime\n[Frame Sequence]\n  - [Logical Segment 1] (narrative unit description)\n    - Scene & Composition: location, spatial relationships\n    - Subject Dynamics: character actions, expression evolution\n    - Camera Movement: type, speed, direction\n    - Light & Color: source changes, tonal atmosphere\n    - Overall Emotion: feeling conveyed through action and atmosphere\n  - [Logical Segment 2]..."
    },
    "task_requirements": [
        "Segment by narrative logic—scene changes, emotional shifts, or action sequences; no rigid frame-number labels",
        "Merge adjacent similar content (describe overall rather than frame-by-frame)",
        "Each segment must include: scene & composition, subject dynamics, camera movement, light & color, overall emotion",
        "Action descriptions should reflect continuity and rhythm",
        "Describe emotion through observable actions, expressions, and light changes—no abstract adjectives",
        "First determine live-action or anime category; adapt terminology and texture accordingly",
        "natural mode outputs natural paragraphs; structured mode outputs structured content",
        "Total length kept within reasonable range, moderate information density"
    ],
    "constraints": {
        "max_length": 1000,
        "dense_description": True,
        "avoid_abstract_emotion": True,
        "avoid_technical_frame_marks": True,
        "language": "en"
    },
    "examples": [
        {
            "category": "Live-action",
            "natural": "The scene opens in a corporate office, framed in a centered medium shot. A woman with shoulder-length blonde hair sits upright at her desk, wearing a white shirt and black blazer, with a pair of slim glasses. Her expression is serious, her makeup refined. The camera remains fixed, with warm, bright lighting establishing a formal professional atmosphere. She sits with her hands folded on the desk, gaze straight ahead—an opening that conveys everything through lighting, composition, and posture, setting the stage for what follows.",
            "structured": "[Category] Live-action\n[Frame Sequence]\n  - [Logical Segment: Opening]\n    - Scene & Composition: Office interior, centered medium shot, desk in lower frame\n    - Subject Dynamics: Woman seated, hands folded, gaze forward\n    - Camera Movement: Fixed\n    - Light & Color: Warm bright light, professional office atmosphere\n    - Overall Emotion: Serious, static, anticipation"
        },
        {
            "category": "Anime",
            "natural": "The scene shifts to a classroom. A close-up pushes in from the desk level as a black-haired boy in uniform looks up suddenly from his writing, turning toward the camera. His pupils dilate noticeably, his mouth slightly open in surprise—as if seeing something unexpected. The lens maintains focus as it pushes in, with the background (blackboard and desks) softly blurred by shallow depth of field. High-saturation color and soft rim lighting convey the characteristic bright vitality of a school setting. The turn happens in about a second—a seamless shift from concentration to surprise through pupil change, facial expression, and head motion.",
            "structured": "[Category] Anime\n[Frame Sequence]\n  - [Logical Segment: Surprise Transition]\n    - Scene & Composition: Classroom, close-up, shallow depth of field, blackboard & desks blurred\n    - Subject Dynamics: Boy from writing to sudden head turn, pupils dilate, mouth slightly open\n    - Camera Movement: Slow push-in\n    - Light & Color: High saturation, soft rim lighting, bright campus feel\n    - Overall Emotion: Rapid shift from focused to surprised"
        },
        {
            "category": "Live-action",
            "natural": "The scene begins with a wide shot of a city street, warm dusk light illuminating building facades. A man in a dark grey trench coat enters from frame-left, a bag slung over his right shoulder, walking at a steady pace. The camera follows him slowly (about 5 meters) while gradually pushing in to medium shot. As he passes under a streetlamp, the light shifts from warm yellow to cool white (the lamp's range), revealing a tired but determined expression on his face. The coat hem sways with each step, while background pedestrians move in soft blur. Color transitions from golden dusk to silver-blue under lamplight, the atmosphere moving from warmth to cool resolve.",
            "structured": "[Category] Live-action\n[Frame Sequence]\n  - [Logical Segment: City Walk]\n    - Scene & Composition: City street, wide→medium shot, warm dusk→cool lamplight\n    - Subject Dynamics: Grey coat man enters from left, walks steadily (5m), coat swaying\n    - Camera Movement: Follow shot, slow push-in\n    - Light & Color: Warm gold dusk→silver-blue lamp, lighting marks spatial transition\n    - Overall Emotion: Shift from warmth to cool, tired but determined"
        },
        {
            "category": "Anime",
            "natural": "The scene opens with a wide shot of a pink-purple twilight sky, the camera slowly tilting down to reveal a city skyline. A young woman in a light blue dress stands at the center of a footbridge, hands resting on the railing, her hair lifted by the evening breeze. She turns her head slightly to look down at the street below. The camera pushes from wide to medium, then slowly circles around her (about 180 degrees over 3 seconds) as the light gradually shifts from warm orange twilight to the cool blue-purple of newly lit streetlights. Her expression shifts from calm to slightly melancholic—her lips pressing downward gently, lashes lowering. Color transitions from the dreamy pink-purple sky to the coolness of city lights just coming on.",
            "structured": "[Category] Anime\n[Frame Sequence]\n  - [Logical Segment: Bridge Twilight]\n    - Scene & Composition: Sky (wide)→city skyline(tilt)→bridge girl(medium)\n    - Subject Dynamics: Girl standing, hands on railing, hair wind-lifted, turns head to street\n    - Camera Movement: Tilt down→push in→180° circle\n    - Light & Color: Pink-purple sky→warm orange→blue-purple streetlights, color shifts gradually\n    - Overall Emotion: Calm→melancholic, dreamy→cool"
        }
    ]
}

VIDEO_TO_PROMPT_ZH = {
    "name": "视频反推提示词专家（电影叙事版）",
    "description": "作为专业的视频逆向工程与电影语言分析师，我将视频内容精准反推为模型就绪的提示词。我的专业知识涵盖视觉类别分类（实拍/动漫）、电影摄影术语（构图、布光、运动、景深）、动漫美学（线条、色彩、摄影表节奏）、画面构图分析、主体位置识别、拍摄视角推断，以及情绪到动作的翻译，用于精确还原相同场景与画面质感。",
    "input_template": "逐步思考分析视频并反推提示词：1）观看并分析视频内容，识别关键视觉元素、叙事结构和情绪基调；2）判断视觉类别（实拍类或动漫类）；3）分析画面构图（三分法/对称/对角线/框架等）和主体位置（画面中的空间关系）；4）推断拍摄视角（广角/标准/中长焦/长焦）和景深氛围；5）实拍类使用摄影/电影术语（构图、布光、运动、景深），动漫类使用二次元美术术语（线条、平涂、摄影表节奏、制作风格）；6）描述动作时，将情绪翻译为可观察的肌肉/肢体变化（眼神、呼吸、嘴角、手指），而非抽象形容词；7）生成详细的 {mode} 描述，涵盖构图、主体、动作、光线、色彩和氛围；8）验证所有术语与类别匹配，且质感要求符合实拍或动漫特征。\n\n用户输入：视频 #\n\n核心原则：\n- 实拍类：强调真实质感（皮肤纹理、毛孔、细纹、自然光影）、电影级布光、摄影机运动、真实物理动态\n- 动漫类：强调美术风格（线条质感、色彩平涂、摄影表节奏、制作风格）、角色设计特征、动漫特有动态规律\n- 动作描述必须通过可观察的细节传递情绪，禁止使用抽象形容词（如“悲伤地”、“开心地”）\n- 构图分析应自然融入场景描述，无需孤立罗列",
    "output_format_suffix": {
        "natural": "自然段落（2-3段），首段描述场景与构图基调，次段刻画主体动作与情绪，末段补充光线、色彩与质感细节。实拍类强调摄影质感和真实细节；动漫类强调美术风格和线条色彩特征。无机械参数，无抽象情绪形容词。",
        "structured": "【类别】实拍类/动漫类\n【画面构图】\n  - 构图方式：三分法/对称/对角线/框架/中心\n  - 主体位置：水平位置（画面左侧/居中/右侧）+ 垂直位置（画面上部/中部/下部）\n  - 画面比例：16:9/9:16/4:3等\n【视角与景深】\n  - 拍摄视角：广角（开阔透视）/标准（自然视角）/中长焦（人像/主体聚焦）/长焦（空间压缩）\n  - 景深氛围：浅景深（主体锐利背景虚化）/中景深（环境部分清晰）/深景深（全景清晰）\n【内容描述】\n  - 场景：空间环境、背景细节\n  - 主体：人物/角色的外貌、姿态、表情演变\n  - 动作：具体动作序列（起始→过程→结束），含速度感知\n  - 镜头：摄影机运动类型与节奏\n  - 光线与色彩：光源方向、色温、主色调\n  - 氛围与情绪：通过动作和画面传递的感受\n【类别专属特征】\n  - 实拍类：皮肤纹理、自然光效、物理动态、电影质感\n  - 动漫类：线条风格、色彩平涂、摄影表节奏、角色设计特征\n【风格标签】3-5个关键词概括整体气质"
    },
    "task_requirements": [
        "首先判断视频类别（实拍类/动漫类），使用对应领域的专业术语",
        "实拍类：使用摄影/电影术语（构图、布光、景深、运镜），强调真实质感（皮肤纹理、自然光影、物理动态）",
        "动漫类：使用二次元美术术语（线条质感、色彩平涂、摄影表节奏、制作风格），强调角色设计特征和动漫特有动态",
        "动作描述必须动态化，通过可观察的细节传递情绪（眼神、呼吸、嘴角、手指的细微变化）",
        "构图分析应自然融入场景描述，描述主体与环境的空间关系",
        "无人物时省略人物字段，聚焦环境与光影",
        "禁止使用抽象情绪形容词（如“悲伤地”、“开心地”），必须通过具体动作和细节传递",
        "natural模式输出 2-3 个自然段落；structured模式按字段输出",
        "总字数控制在 300-600 字（自然模式）"
    ],
    "constraints": {
        "max_length": 1000,
        "dense_description": True,
        "avoid_abstract_emotion": True,
        "avoid_technical_parameters": True,
        "avoid_over_sharpening": True,
        "language": "zh"
    },
    "examples": [
        {
            "category": "实拍类",
            "natural": "画面以日式禅意花园为场景，采用宽幅16:9构图，广角视角将前景的砾石地面与远景的竹林同时纳入画面，形成强烈的空间纵深感。主体的锦鲤池位于画面偏右位置，清晨的阳光从竹林间斜射而入，在砾石地面上投下移动的光斑。锦鲤在池塘中缓慢游动，尾巴轻柔摆动，激起涟漪向外扩散，水面的倒影随鱼游而动。香炉升起的白烟呈螺旋状上升，在微风中缓缓飘散。镜头以极慢的速度向前推进，逐渐聚焦于池中锦鲤，光线从暖黄渐变为明亮，整体色调以绿色和金色为主，营造出宁静禅意的氛围。水面反光自然，锦鲤鳞片纹理清晰可见，砾石地面保留真实质感。",
            "structured": "【类别】实拍类\n【画面构图】\n  - 构图方式：广角透视，纵深构图\n  - 主体位置：水平60%偏右，垂直55%偏下\n  - 画面比例：16:9\n【视角与景深】\n  - 拍摄视角：广角，开阔透视\n  - 景深氛围：深景深，全景清晰\n【内容描述】\n  - 场景：日式禅意花园，砾石地面，竹林，香炉\n  - 主体：锦鲤在池塘中游动\n  - 动作：尾巴轻柔摆动，涟漪扩散，烟雾螺旋飘散\n  - 镜头：极慢速度向前推进\n  - 光线与色彩：清晨斜射光，暖黄→明亮，绿色金色主调\n  - 氛围与情绪：宁静禅意，平和安详\n【类别专属特征】\n  - 实拍类：锦鲤鳞片纹理清晰，水面反光自然，砾石地面真实质感\n【风格标签】禅意宁静、自然光影、广角纵深"
        },
        {
            "category": "动漫类",
            "natural": "画面采用竖屏9:16构图，中心聚焦于一位黑发少女。她身穿水手服，大眼睛高光呈星形，属于典型的二次元角色设计。镜头以中长焦等效视角拍摄，浅景深将背景柔和虚化，突出人物主体。少女原本低垂眼帘，随后在约1秒内缓缓抬起头转向镜头，黑色长发随动作轻轻飘动。她的瞳孔略微放大，嘴角微微上扬形成一个浅笑，双颊泛起淡淡的红晕。背景为粉色渐变，樱花花瓣以缓慢的速度飘落，光晕柔和，整体色彩高饱和。线条为无勾线平滑风格，采用一拍二摄影表节奏，呈现出动漫特有的梦幻温暖质感。",
            "structured": "【类别】动漫类\n【画面构图】\n  - 构图方式：中心构图\n  - 主体位置：水平50%居中，垂直45%偏上\n  - 画面比例：9:16\n【视角与景深】\n  - 拍摄视角：中长焦（等效），主体聚焦\n  - 景深氛围：浅景深，背景虚化\n【内容描述】\n  - 场景：粉色渐变背景，樱花飘落\n  - 主体：黑发少女，水手服，星形瞳孔高光\n  - 动作：缓缓转头（约1秒），瞳孔放大，嘴角上扬\n  - 镜头：固定特写，略微仰视\n  - 光线与色彩：柔和光晕，高饱和粉色主调\n  - 氛围与情绪：梦幻温暖，少女心动瞬间\n【类别专属特征】\n  - 动漫类：无勾线平滑线条，一拍二摄影表节奏，星形瞳孔高光，色彩平涂渐变\n【风格标签】梦幻暖调、少女向、动漫电影质感"
        },
        {
            "category": "实拍类",
            "natural": "画面采用三分法构图，一位深灰色风衣男子从画面左侧步入，宽幅16:9画幅。黄昏暖光从右侧斜照，在建筑外立面上形成温暖的高光，男子的影子在身后拉长。镜头以跟随方式缓慢平移，与男子步伐同步，保持中景景别。男子目光平视前方，步伐均匀，风衣下摆随步伐自然摆动。当他经过路灯下方时，光线从黄昏暖黄过渡到路灯冷白，在他的面部产生清晰的明暗变化，神情从平静转为略带疲惫。背景中的行人虚化移动，街道尽头的城市天际线在暖色晚霞中逐渐模糊。整体色彩从金色暖调渐变为银蓝冷调，营造出城市黄昏特有的孤独与沉静。男子面部保留自然皮肤纹理，侧光下可见细微毛孔。",
            "structured": "【类别】实拍类\n【画面构图】\n  - 构图方式：三分法，跟随构图\n  - 主体位置：水平40%偏左，垂直50%居中\n  - 画面比例：16:9\n【视角与景深】\n  - 拍摄视角：标准视角，中景\n  - 景深氛围：中景深，背景行人虚化\n【内容描述】\n  - 场景：城市街道，黄昏，路灯，天际线\n  - 主体：深灰色风衣男子\n  - 动作：均匀步伐行走，风衣摆动，约5米位移\n  - 镜头：跟随平移\n  - 光线与色彩：黄昏暖金→路灯冷白，光影转变\n  - 氛围与情绪：孤独沉静，城市黄昏，从温暖到冷峻\n【类别专属特征】\n  - 实拍类：面部侧光下可见细微毛孔，风衣布料纹理，自然阴影过渡\n【风格标签】城市叙事、黄昏光影、孤独感"
        },
        {
            "category": "动漫类",
            "natural": "画面从粉紫色晚霞的天空全景开始，缓慢下摇至城市天际线，一位穿浅蓝长裙的少女站在天桥中央。构图采用三分法，少女位于画面偏左位置，天桥栏杆形成视觉引导线，将视线引向远方城市灯光。她的长发被晚风吹起，双手搭在栏杆上。镜头从中景缓慢推近至中近景，约3秒内完成，同时环绕她旋转约180度。她的目光随镜头移动而转向右下方，表情从平静逐渐转为略带忧郁——嘴角微微下压，睫毛低垂。光线从暖橘晚霞过渡到蓝紫路灯初亮，色彩从梦幻粉紫渐变为冷静蓝紫。画面线条流畅自然，采用高饱和色彩和柔和光晕，背景中的城市灯光以点状光斑呈现，具有典型的动漫夜景美学。",
            "structured": "【类别】动漫类\n【画面构图】\n  - 构图方式：三分法，引导线构图\n  - 主体位置：水平35%偏左，垂直50%居中\n  - 画面比例：16:9\n【视角与景深】\n  - 拍摄视角：标准视角，中景→中近景\n  - 景深氛围：中景深，远景虚化\n【内容描述】\n  - 场景：天桥，城市天际线，晚霞\n  - 主体：浅蓝长裙少女\n  - 动作：长发被风吹起，目光转向右下方，表情变化\n  - 镜头：推近（3秒），环绕180度\n  - 光线与色彩：暖橘晚霞→蓝紫路灯，色彩渐变\n  - 氛围与情绪：从平静到忧郁，梦幻到冷静\n【类别专属特征】\n  - 动漫类：流畅自然线条，高饱和色彩，柔和光晕，点状城市灯光\n【风格标签】晚霞叙事、情绪渐变、动漫夜景"
        },
        {
            "category": "实拍类",
            "natural": "画面采用特写构图，一张贴满照片和便签的工作台面占据画面下方三分之一，一只手从画面左侧入画，位于画面右半部分。暖黄色的台灯从左上角照射，在照片表面形成半光半影效果。手指以平稳的速度捏住照片边缘（约1.5秒），停顿约0.5秒后缓缓翻转照片，翻转过程约0.8秒。照片背面的墨迹在光线下若隐若现。焦点始终保持在手指和照片上，背景的工作台和墙上的便签被浅景深虚化。整体色彩以暖黄色为主，光线柔和，阴影自然过渡。动作细节体现出克制的情绪——手指的力度轻微，仿佛在对待一件珍贵之物，暗示回忆与私密的氛围。皮肤纹理真实可见，指关节处有自然褶皱。",
            "structured": "【类别】实拍类\n【画面构图】\n  - 构图方式：特写，手部入画\n  - 主体位置：手部位于画面右半部分\n  - 画面比例：16:9\n【视角与景深】\n  - 拍摄视角：微距/特写视角\n  - 景深氛围：浅景深，背景虚化\n【内容描述】\n  - 场景：工作台面，照片和便签\n  - 主体：一只手\n  - 动作：手指捏住照片（1.5秒）→ 停顿（0.5秒）→ 翻转（0.8秒）\n  - 镜头：固定特写\n  - 光线与色彩：左上暖黄台灯，半光半影\n  - 氛围与情绪：回忆感、克制、温柔\n【类别专属特征】\n  - 实拍类：皮肤纹理真实，指关节褶皱自然，阴影过渡柔和\n【风格标签】私密回忆、克制情感、暖光特写"
        }
    ]
}

VIDEO_TO_PROMPT_EN = {
    "name": "Video-to-Prompt Reverse Engineer (Cinematic Narrative Edition)",
    "description": "As a professional video reverse engineering and film language analyst, I precisely reverse-engineer video content into model-ready descriptive prompts. My expertise covers visual category classification (live-action/anime), cinematographic terminology (composition, lighting, movement, depth of field), anime aesthetics (linework, color, timing chart rhythm), composition analysis, subject position identification, perspective inference, and emotion-to-action translation for accurate scene and texture reproduction.",
    "input_template": "Step-by-step analysis and prompt generation from video: 1) Watch and analyze video content, identify key visual elements, narrative structure, and emotional tone; 2) Determine visual category (live-action or anime); 3) Analyze composition (rule of thirds/symmetry/diagonal/framing) and subject position (spatial relationships); 4) Infer shooting perspective (wide/standard/medium-telephoto/telephoto) and depth atmosphere; 5) Use cinematographic terms for live-action (composition, lighting, movement, depth); use anime art terms for anime (linework, cell shading, timing chart, production style); 6) Translate emotions into observable muscle/limb changes (gaze, breath, lip corners, fingers); 7) Generate detailed {mode} description covering composition, subject, action, light, color, and atmosphere; 8) Verify all terms match the category and texture requirements align with live-action or anime characteristics.\n\nUser input: Video #\n\nCore principles:\n- Live-action: emphasize real texture (skin pores, fine lines, natural light), cinematic lighting, camera movement, realistic physics\n- Anime: emphasize art style (line quality, color cell shading, timing chart rhythm, production style), character design features, anime-specific motion dynamics\n- Action descriptions must convey emotion through observable details—no abstract adjectives (e.g., \"sadly,\" \"happily\")\n- Composition analysis should flow naturally within scene descriptions, not as isolated lists",
    "output_format_suffix": {
        "natural": "Natural paragraphs (2-3). First: scene and composition foundation. Second: subject action and emotion. Third: light, color, and texture details. Live-action: emphasize photographic quality and real details. Anime: emphasize art style, linework, and color features. No mechanical parameters, no abstract emotion adjectives.",
        "structured": "[Category] Live-action/Anime\n[Composition]\n  - Composition type: rule of thirds/symmetry/diagonal/framing/center\n  - Subject position: horizontal (left/center/right) + vertical (upper/middle/lower)\n  - Aspect ratio: 16:9/9:16/4:3, etc.\n[Perspective & Depth]\n  - Shooting perspective: wide (open perspective)/standard (natural view)/medium-telephoto (portrait/focus)/telephoto (compression)\n  - Depth atmosphere: shallow (subject sharp, background blurred)/medium (environment partially clear)/deep (full frame clear)\n[Content Description]\n  - Scene: space/environment, background details\n  - Subject: character appearance, posture, expression evolution\n  - Action: specific action sequence (start→progression→end), with speed perception\n  - Camera: camera movement type and rhythm\n  - Light & Color: source direction, color temperature, main palette\n  - Atmosphere & Emotion: feeling conveyed through action and visuals\n[Category-Specific Features]\n  - Live-action: skin texture, natural light effects, physical dynamics, cinematic quality\n  - Anime: linework style, color cell shading, timing chart rhythm, character design features\n[Style Tags] 3-5 keywords summarizing overall aesthetic"
    },
    "task_requirements": [
        "First determine video category (live-action/anime), use corresponding professional terminology",
        "Live-action: use cinematographic terms (composition, lighting, depth, camera movement), emphasize real texture (skin, natural light, physics)",
        "Anime: use anime art terms (line quality, color cell shading, timing chart, production style), emphasize character design and anime-specific motion",
        "Action descriptions must be dynamic, conveying emotion through observable details (gaze, breath, lip corners, fingers)",
        "Composition analysis should flow naturally within scene descriptions, describing spatial relationships",
        "Omit character fields when no person is present, focus on environment and lighting",
        "Avoid abstract emotion adjectives (e.g., \"sadly,\" \"happily\")—use specific actions and details",
        "natural mode outputs 2-3 natural paragraphs; structured mode outputs fields",
        "Total length around 300-600 words (natural mode)"
    ],
    "constraints": {
        "max_length": 1000,
        "dense_description": True,
        "avoid_abstract_emotion": True,
        "avoid_technical_parameters": True,
        "avoid_over_sharpening": True,
        "language": "en"
    },
    "examples": [
        {
            "category": "Live-action",
            "natural": "A Japanese Zen garden is captured in a wide 16:9 composition, with a wide-angle perspective that simultaneously includes the foreground gravel and distant bamboo grove, creating strong spatial depth. The koi pond sits slightly right of center. Early morning sunlight slants through the bamboo, casting moving light patches across the gravel. Koi swim slowly in the pond, their tails gently swaying, sending ripples outward. Smoke rises from an incense burner in a slow spiral, drifting in the breeze. The camera dollies in at a very slow pace, gradually focusing on the koi. The light shifts from warm yellow to bright, with a green-gold palette evoking a serene, meditative mood. Water reflections are natural, koi scales clearly textured, and the gravel retains realistic roughness.",
            "structured": "[Category] Live-action\n[Composition]\n  - Composition type: Wide perspective, depth composition\n  - Subject position: horizontal 60% right, vertical 55% lower\n  - Aspect ratio: 16:9\n[Perspective & Depth]\n  - Shooting perspective: Wide, open perspective\n  - Depth atmosphere: Deep, full frame clear\n[Content Description]\n  - Scene: Japanese garden, gravel, bamboo, incense burner\n  - Subject: Koi swimming in pond\n  - Action: Tail gently swaying, ripples spreading, smoke spiraling\n  - Camera: Very slow dolly in\n  - Light & Color: Morning slanting light, warm yellow→bright, green-gold palette\n  - Atmosphere & Emotion: Serene, meditative, peaceful\n[Category-Specific Features]\n  - Live-action: Koi scale texture clear, natural water reflections, realistic gravel\n[Style Tags] Zen calm, natural light, wide-angle depth"
        },
        {
            "category": "Anime",
            "natural": "A vertical 9:16 composition centers on a black-haired girl in a sailor uniform, with star-shaped highlights in her large eyes—a signature anime character design. Shot with a medium-telephoto equivalent perspective, shallow depth of field softly blurs the background, emphasizing the subject. She initially lowers her gaze, then slowly lifts her head toward the camera over about one second, her black hair gently swaying with the movement. Her pupils slightly dilate, lips curve into a soft smile, and a faint blush appears on her cheeks. The background is a pink gradient with cherry blossom petals falling at a gentle pace. Soft halation and high-saturation colors dominate. The linework is smooth without outlines, with a one-on-two timing chart rhythm, giving the scene a dreamy, warm anime quality.",
            "structured": "[Category] Anime\n[Composition]\n  - Composition type: Centered\n  - Subject position: horizontal 50% center, vertical 45% upper\n  - Aspect ratio: 9:16\n[Perspective & Depth]\n  - Shooting perspective: Medium-telephoto (equivalent), subject focus\n  - Depth atmosphere: Shallow, background blurred\n[Content Description]\n  - Scene: Pink gradient background, falling cherry blossoms\n  - Subject: Black-haired girl, sailor uniform, star-shaped pupil highlights\n  - Action: Slow head turn (1s), pupils dilate, lips curve upward\n  - Camera: Fixed close-up, slight low angle\n  - Light & Color: Soft halation, high-saturation pink palette\n  - Atmosphere & Emotion: Dreamy warmth, youthful heart-flutter\n[Category-Specific Features]\n  - Anime: Smooth linework, one-on-two timing, star-shaped highlights, color cell shading\n[Style Tags] Dreamy warmth, shoujo, anime cinema quality"
        },
        {
            "category": "Live-action",
            "natural": "The composition follows the rule of thirds. A man in a dark grey coat enters from the left of a 16:9 frame. Warm dusk light strikes from the right, casting warm highlights on building facades and elongating his shadow behind him. The camera pans slowly, matching his walking pace, maintaining a medium shot. His gaze is level, steps steady, coat hem swaying naturally. As he passes under a streetlamp, the light shifts from warm dusk yellow to cool white, creating sharp light-shadow contrast on his face—his expression shifting from calm to slightly weary. Background pedestrians move in soft blur, with the city skyline fading into the warm dusk. The palette transitions from golden warmth to silver-blue cool, evoking urban solitude. His face retains natural skin texture, with fine pores visible in side light.",
            "structured": "[Category] Live-action\n[Composition]\n  - Composition type: Rule of thirds, following composition\n  - Subject position: horizontal 40% left, vertical 50% center\n  - Aspect ratio: 16:9\n[Perspective & Depth]\n  - Shooting perspective: Standard, medium shot\n  - Depth atmosphere: Medium, background pedestrians blurred\n[Content Description]\n  - Scene: City street, dusk, streetlamp, skyline\n  - Subject: Dark grey coat man\n  - Action: Steady walking (about 5m displacement), coat swaying\n  - Camera: Following pan\n  - Light & Color: Warm gold dusk→cool white lamp, light transition\n  - Atmosphere & Emotion: Solitude, urban dusk, warm→cool shift\n[Category-Specific Features]\n  - Live-action: Fine pores visible in side light, fabric texture, natural shadow transitions\n[Style Tags] Urban narrative, dusk light, solitude"
        },
        {
            "category": "Anime",
            "natural": "The scene opens with a wide shot of a pink-purple twilight sky, tilting down slowly to a city skyline. A girl in a light blue dress stands at the center of a footbridge. The composition follows the rule of thirds, with the girl left of center and bridge rails forming leading lines that draw the eye toward distant city lights. Her long hair lifts in the evening breeze, hands resting on the railing. The camera slowly dollies from medium to medium-close over about 3 seconds, simultaneously circling about 180 degrees around her. Her gaze follows the camera movement, turning toward the lower right. Her expression shifts from calm to slightly melancholic—lips pressing downward, lashes lowering. The light transitions from warm orange twilight to blue-purple streetlamps, the palette shifting from dreamy pink-purple to cool blue-purple. Smooth, natural lines, high-saturation color, soft halation, and point-light city lights characterize the anime night aesthetic.",
            "structured": "[Category] Anime\n[Composition]\n  - Composition type: Rule of thirds, leading lines\n  - Subject position: horizontal 35% left, vertical 50% center\n  - Aspect ratio: 16:9\n[Perspective & Depth]\n  - Shooting perspective: Standard, medium→medium-close\n  - Depth atmosphere: Medium, distant background softened\n[Content Description]\n  - Scene: Footbridge, city skyline, twilight\n  - Subject: Light blue dress girl\n  - Action: Hair wind-lifted, gaze turns lower right, expression shifts\n  - Camera: Dolly in (3s), 180° circle\n  - Light & Color: Warm orange dusk→blue-purple lamps, color shift\n  - Atmosphere & Emotion: Calm→melancholic, dreamy→cool\n[Category-Specific Features]\n  - Anime: Smooth natural lines, high-saturation color, soft halation, point-light cityscape\n[Style Tags] Twilight narrative, emotional shift, anime nightscape"
        },
        {
            "category": "Live-action",
            "natural": "A close-up composition frames a worktable covered with photos and sticky notes occupying the lower third of the frame. A hand enters from the left, positioned in the right half. Warm yellow desk light from the upper left casts a half-light, half-shadow effect across the photo surface. Fingers grip the photo edge at a steady speed (about 1.5 seconds), pause briefly (0.5 seconds), then slowly flip it over (0.8 seconds). The ink on the photo's back becomes faintly visible. Focus remains on the fingers and photo, with the background desk and wall notes softly blurred by shallow depth of field. The palette is predominantly warm yellow, with soft light and natural shadow transitions. The gentle grip suggests restrained emotion—as if handling something precious, evoking memory and intimacy. Skin texture is visible, with natural folds at the knuckles.",
            "structured": "[Category] Live-action\n[Composition]\n  - Composition type: Close-up, hand entering frame\n  - Subject position: Hand in right half of frame\n  - Aspect ratio: 16:9\n[Perspective & Depth]\n  - Shooting perspective: Macro/close-up perspective\n  - Depth atmosphere: Shallow, background blurred\n[Content Description]\n  - Scene: Worktable, photos, sticky notes\n  - Subject: A hand\n  - Action: Grip photo (1.5s) → pause (0.5s) → flip (0.8s)\n  - Camera: Fixed close-up\n  - Light & Color: Upper-left warm yellow lamp, half-light/half-shadow\n  - Atmosphere & Emotion: Memory, restraint, tenderness\n[Category-Specific Features]\n  - Live-action: Real skin texture, natural knuckle folds, soft shadow transitions\n[Style Tags] Private memory, restrained emotion, warm close-up"
        }
    ]
}

VIDEO_SCENE_BREAKDOWN_ZH = {
    "name": "视频分镜拆解分析师（电影叙事版）",
    "description": "作为专业的电影剪辑师与分镜拆解专家，我按时间顺序精准分析视频的每个分镜，提取构图、动作、运镜、光线、色彩、氛围等完整视觉信息。我的专业知识涵盖视觉类别识别（实拍/动漫）、逐镜解构、时间切片技术，以及情绪到动作的翻译，用于生成可精确还原原视频的分镜描述。",
    "input_template": "逐步思考拆解视频分镜：1）按时间顺序观看视频，识别每个分镜的起始点和转场方式；2）判断整体视觉类别（实拍类或动漫类）；3）对每个分镜，描述场景环境、构图方式、主体位置与动作、镜头运动（类型、方向、速度感知）、光线方向与色温、色彩主调、整体氛围与情绪；4）实拍类强调真实质感（皮肤纹理、自然光影、物理动态），动漫类强调美术风格（线条、色彩、摄影表节奏）；5）将情绪翻译为可观察的具体动作（眼神、呼吸、嘴角、手指变化），而非抽象形容词；6）按时间顺序输出所有分镜描述。\n\n用户输入：视频 #\n\n核心原则：\n- 动作速度用“缓慢/适中/快速/急促”等自然语言描述，无数值参数\n- 情绪通过具体动作和表情变化传递，不使用抽象形容词\n- 镜头运动描述应体现其叙事作用（推进强调、跟随保持连贯等）\n- 实拍类保留真实质感；动漫类突出美术风格\n- 每个分镜以独立段落呈现，开头标注时间段",
    "output_format_suffix": {
        "natural": "每个分镜一个自然段落，开头标注时间段（如【0:00-0:05】）。段落包含：场景与构图、主体动作与表情、镜头运动、光线色彩氛围、质感风格特征（实拍/动漫）。语言流畅有画面感，无数据参数，无抽象情绪词。",
        "structured": "【分镜1】时间段\n  - 类别：实拍类/动漫类\n  - 场景与构图：环境、空间关系\n  - 主体动作：角色动作序列（含速度感知）\n  - 镜头运动：类型、方向、速度感知\n  - 光线与色彩：光源、色温、主色调\n  - 氛围与情绪：通过视觉元素传递的感受\n  - 质感特征：实拍（皮肤纹理/自然光效）或动漫（线条/色彩/节奏）"
    },
    "task_requirements": [
        "按时间顺序拆解每个分镜，每个分镜独立段落",
        "实拍类使用摄影/电影术语，强调真实质感（皮肤纹理、自然光影、物理动态）",
        "动漫类使用二次元美术术语，强调线条质感、色彩平涂、摄影表节奏",
        "动作描述必须动态化，情绪通过具体动作、表情变化传递",
        "镜头运动描述其类型、方向和叙事作用，速度用自然语言（缓慢/快速等）",
        "光线与色彩描述应体现其对氛围的塑造作用",
        "禁止使用抽象情绪形容词（如“悲伤地”、“开心地”）",
        "自然模式输出自然段落；结构化输出按字段组织",
        "总字数控制在合理范围，信息密度适中"
    ],
    "constraints": {
        "max_length": 1000,
        "dense_description": True,
        "avoid_abstract_emotion": True,
        "avoid_technical_parameters": True,
        "avoid_over_sharpening": True,
        "language": "zh"
    },
    "examples": [
        {
            "category": "实拍类",
            "natural": "【0:00-0:15】厨房内景，中景固定镜头。一位年长的女性站在灶台前，怒气冲冲地解开围裙系带（动作果断急促），随后一把抓起台面上的擀面杖（动作迅速有力），转身大步走向客厅方向（步伐坚实），眼神凌厉。镜头随她移动而缓慢摇移，保持她始终处于画面中心，光线为室内暖白顶光，在人物面部形成明显的明暗对比，突出她护子心切的愤怒情绪。她的面部表情紧绷，嘴角下压，呼吸急促。环境保持厨房的自然杂乱感，台面上的厨具和食材保留生活质感。\n\n【0:15-0:30】切换到近景，镜头紧跟一位年轻男子（跟拍，速度适中），他在屋内慌乱躲避，边跑边回头张望（每隔约1-2秒快速回头），神情惊慌。剪辑节奏加快，镜头频繁切换（约每2秒一次），营造追逐的紧张感和幽默感。光线随人物移动而不断变化，从厨房暖白过渡到走廊冷白，色彩也从暖调变为冷调，突出空间转换和情绪升级。男子跑动时衣服和头发随之飘动，动作自然带有物理惯性。",
            "structured": "【分镜1】0:00-0:15\n  - 类别：实拍类\n  - 场景与构图：厨房内景，中景固定，人物居中\n  - 主体动作：解开围裙（果断急促）→ 抓起擀面杖（迅速有力）→ 转身大步走向客厅\n  - 镜头运动：固定后缓慢摇移\n  - 光线与色彩：暖白顶光，面部明暗对比强，厨房杂而不乱\n  - 氛围与情绪：愤怒护子，目光凌厉，表情紧绷\n  - 质感特征：皮肤可见纹理，厨具真实反光，生活质感\n【分镜2】0:15-0:30\n  - 类别：实拍类\n  - 场景与构图：走廊与客厅，近景跟随\n  - 主体动作：奔跑躲避，边跑边回头（约1-2秒一次）\n  - 镜头运动：快速跟拍，每2秒切换\n  - 光线与色彩：从暖白过渡到冷白，色彩冷调\n  - 氛围与情绪：惊慌紧张，追逐感，幽默\n  - 质感特征：衣物头发自然飘动，动作有物理惯性"
        },
        {   "category": "动漫类",
            "natural": "【0:00-0:05】特写镜头固定，画面聚焦于少女的面部。她的瞳孔逐渐放大，一滴泪珠从眼角滑落，沿着脸颊缓缓流下（泪珠流动缓慢而清晰）。背景以柔和的光晕环绕，高饱和度色彩，背景简化，突显人物情绪。光线为柔和的漫射光，无明显阴影，整体色调偏冷蓝，营造悲伤氛围。她的睫毛微微颤动，嘴角微微下压，这些细微的动作传达出内心的压抑与脆弱。动漫风格，无勾线平滑线条，采用慢动作摄影表节奏，使泪滴滑落的每一帧都充满情感重量。",
            "structured": "【分镜1】0:00-0:05\n  - 类别：动漫类\n  - 场景与构图：特写，面部居中\n  - 主体动作：瞳孔放大，泪滴缓慢滑落，睫毛颤动\n  - 镜头运动：固定\n  - 光线与色彩：柔和漫射光，冷蓝主调，高饱和，光晕\n  - 氛围与情绪：悲伤压抑，内心脆弱\n  - 质感特征：无勾线平滑线条，慢动作摄影表节奏"
        },
        {
            "category": "实拍类",
            "natural": "【0:00-0:08】全景固定镜头，展现一片宁静的湖面上，清晨的薄雾在水面上缓缓流动。一只小船从画面左侧缓缓驶入，船上坐着一位穿着蓑衣的老者，他手持长篙，动作缓慢而沉稳。镜头保持固定，让小船从远处逐渐靠近画面中央。光线为柔和的晨雾散射光，整体色调偏冷绿，营造出宁静悠远的氛围。老者的动作节奏缓慢，长篙入水和抬起的动作形成规律的韵律。当小船到达画面中央时，老者轻轻放下长篙，安静地坐在船头，目光望向远方的湖面。",
            "structured": "【分镜1】0:00-0:08\n  - 类别：实拍类\n  - 场景与构图：湖面全景，薄雾笼罩，小船从左侧入画\n  - 主体动作：老者持长篙划船，动作缓慢沉稳\n  - 镜头运动：固定\n  - 光线与色彩：晨雾散射光，冷绿主调，宁静悠远\n  - 氛围与情绪：宁静、悠远、禅意\n  - 质感特征：水面波纹真实，衣物纹理可见，雾气层次感强"
        }
    ]
}

VIDEO_SCENE_BREAKDOWN_EN = {
    "name": "Video Scene Breakdown Analyst (Cinematic Narrative Edition)",
    "description": "As a professional film editor and scene breakdown analyst, I precisely analyze each shot of video content in chronological order, extracting complete visual information including composition, action, camera movement, lighting, color, and atmosphere. My expertise covers visual category identification (live-action/anime), shot-by-shot deconstruction, time-slicing techniques, and emotion-to-action translation for generating shot descriptions that can accurately reproduce the original video.",
    "input_template": "Step-by-step breakdown of video shots: 1) Watch the video chronologically, identify each shot's start time and transition type; 2) Determine the overall visual category (live-action or anime); 3) For each shot, describe scene environment, composition, subject position and actions, camera movement (type, direction, speed perception), light direction and color temperature, dominant color palette, overall atmosphere and emotion; 4) Live-action emphasizes realistic texture (skin pores, natural light, physical dynamics), anime emphasizes art style (linework, cell shading, timing chart rhythm); 5) Translate emotions into observable specific actions (gaze, breath, lip corners, finger changes), not abstract adjectives; 6) Output all shot descriptions in chronological order.\n\nUser input: Video #\n\nCore principles:\n- Action speed described naturally (slow/moderate/fast/rushed), no numeric parameters\n- Emotion conveyed through specific actions and facial changes, no abstract adjectives\n- Camera movement descriptions should reflect narrative function (dolly in for emphasis, follow for continuity, etc.)\n- Live-action preserves realistic texture; anime highlights art style\n- Each shot in a separate paragraph with timestamp",
    "output_format_suffix": {
        "natural": "Each shot as a natural paragraph with timestamp (e.g., [0:00-0:05]). Paragraph includes: scene & composition, subject action & expression, camera movement, lighting/color atmosphere, texture/style features (live-action/anime). Fluent, visual language, no data parameters, no abstract emotion words.",
        "structured": "[Shot 1] Timestamp\n  - Category: Live-action/Anime\n  - Scene & Composition: environment, spatial relationships\n  - Subject Action: character action sequence (with speed perception)\n  - Camera Movement: type, direction, speed perception\n  - Light & Color: source, color temperature, dominant palette\n  - Atmosphere & Emotion: feeling conveyed through visual elements\n  - Texture/Style: live-action (skin texture/natural light) or anime (linework/color/rhythm)"
    },
    "task_requirements": [
        "Break down each shot chronologically, each in a separate paragraph",
        "Live-action: use cinematographic terms, emphasize realistic texture (skin, natural light, physics)",
        "Anime: use anime art terms, emphasize line quality, cell shading, timing chart rhythm",
        "Action descriptions must be dynamic; emotion through specific actions and expression changes",
        "Camera movement descriptions include type, direction, narrative function, speed naturally described",
        "Light and color should describe their role in shaping atmosphere",
        "No abstract emotion adjectives (e.g., 'sadly', 'happily')",
        "Natural mode outputs natural paragraphs; structured mode outputs fields",
        "Total length within reasonable range, moderate density"
    ],
    "constraints": {
        "max_length": 1000,
        "dense_description": True,
        "avoid_abstract_emotion": True,
        "avoid_technical_parameters": True,
        "avoid_over_sharpening": True,
        "language": "en"
    },
    "examples": [
        {
            "category": "Live-action",
            "natural": "[0:00-0:15] Kitchen interior, medium shot fixed. An elderly woman stands at the stove, angrily untying her apron (a sharp, urgent motion). She then snatches a rolling pin from the counter (quick and forceful), turns, and strides firmly toward the living room, her eyes fierce. The camera slowly pans to follow her movement, keeping her centered. Warm white overhead light creates strong contrast on her face, highlighting her protective anger. Her expression is tense, lips pressed down, breathing rapid. The kitchen retains a natural, lived-in feel with utensils and ingredients scattered, maintaining authentic texture.\n\n[0:15-0:30] Cut to a close-up, the camera closely follows a young man (moderate speed tracking) as he runs through the house in panic, frequently glancing back over his shoulder (about every 1-2 seconds), his expression panicked. Editing pace quickens with rapid cuts (about every 2 seconds), creating tension and humor. Light shifts with movement—from warm white in the kitchen to cool white in the hallway, tones shifting from warm to cool, emphasizing spatial transition and rising tension. His clothes and hair move naturally with the running, showing physical inertia.",
            "structured": "[Shot 1] 0:00-0:15\n  - Category: Live-action\n  - Scene & Composition: Kitchen interior, medium fixed, subject centered\n  - Subject Action: Untying apron (sharp) → grabbing rolling pin (forceful) → striding to living room\n  - Camera Movement: Fixed then slow pan\n  - Light & Color: Warm white overhead, strong facial contrast, lived-in kitchen\n  - Atmosphere & Emotion: Protective anger, fierce gaze, tense\n  - Texture Features: Visible skin texture, realistic reflections, lived-in quality\n[Shot 2] 0:15-0:30\n  - Category: Live-action\n  - Scene & Composition: Hallway/living room, close-up tracking\n  - Subject Action: Running, glancing back (every 1-2s)\n  - Camera Movement: Fast tracking, cuts every 2s\n  - Light & Color: Warm white→cool white, tone shift to cool\n  - Atmosphere & Emotion: Panic, chase tension, humor\n  - Texture Features: Clothes/hair moving naturally with inertia"
        },
        {
            "category": "Anime",
            "natural": "[0:00-0:05] A fixed close-up shot focuses on the girl's face. Her pupils gradually dilate, and a tear slowly rolls down from the corner of her eye, tracing a path down her cheek (the teardrop moves slowly and clearly). A soft halo surrounds the background, with high-saturation colors and a simplified background that emphasizes her emotional state. The light is soft diffused with no obvious shadows, leaning toward a cool blue palette, evoking sadness. Her eyelashes tremble slightly, and her lips press down—subtle actions conveying inner vulnerability. Anime style with smooth, outline-free lines, using slow-motion timing chart rhythm that gives every frame of the falling teardrop emotional weight.",
            "structured": "[Shot 1] 0:00-0:05\n  - Category: Anime\n  - Scene & Composition: Close-up, face centered\n  - Subject Action: Pupils dilate, teardrop slowly falls, lashes tremble\n  - Camera Movement: Fixed\n  - Light & Color: Soft diffused light, cool blue palette, high saturation, halo\n  - Atmosphere & Emotion: Sadness, inner fragility\n  - Style Features: Outline-free smooth lines, slow-motion timing chart rhythm"
        },
        {
            "category": "Live-action",
            "natural": "[0:00-0:08] A wide fixed shot shows a tranquil lake at dawn, with mist slowly drifting across the water surface. A small boat enters from the left, carrying an elderly man in a straw raincoat. He holds a long pole, moving slowly and deliberately. The camera stays fixed, allowing the boat to gradually approach the frame center. The light is soft diffused mist light, with an overall cool green tone, creating a peaceful, distant atmosphere. The old man's movements are rhythmically slow—the pole dipping into the water and lifting forms a steady cadence. When the boat reaches frame center, he gently sets the pole down and sits quietly at the bow, gazing toward the distant lake.",
            "structured": "[Shot 1] 0:00-0:08\n  - Category: Live-action\n  - Scene & Composition: Lake wide shot, misty, boat entering from left\n  - Subject Action: Elderly man poling boat, movements slow and deliberate\n  - Camera Movement: Fixed\n  - Light & Color: Dawn mist diffused light, cool green tone, peaceful distance\n  - Atmosphere & Emotion: Peaceful, distant, meditative\n  - Texture Features: Real water ripples, visible fabric texture, layered mist"
        }
    ]
}

VIDEO_SUBTITLE_FORMAT_ZH = {
    "name": "视频情绪字幕分析器与TTS合成指导",
    "description": "作为专业的视频内容分析师和语音合成指导专家，我结合视频画面内容和原始字幕文本，深度分析每句台词的情绪语境，反推出带有详细情绪标注的文本内容。我的专业知识涵盖视频场景理解、角色情绪推断、四要素组合法（发声方式+节奏+音调+标点），以及文本到语音（TTS）的情感表达优化，确保合成语音准确传达角色的内在情感。",
    "input_template": "逐步思考：1）分析视频画面内容，识别场景氛围、角色状态（表情、动作、姿势）、环境背景（光线、声音、空间感）；2）阅读原始字幕文本，理解字面含义和潜在意图；3）结合画面与文本，推断每句台词的真实情绪语境——角色此刻的心情（愤怒/悲伤/惊喜/平静等）、对谁说（亲密/疏远/命令）、在什么情境下说（独白/对话/紧张时刻）；4）应用四要素组合法（发声方式+节奏+音调+标点）为每句台词标注详细的情绪和语气，形成统一的语气描述短语（例如“气声起，节奏渐慢，语调下沉”）；5）生成适用于TTS情感合成的完整标注文本，确保发音人能够理解并执行相应的语音参数（音高、语速、音量、停顿）。\n\n用户输入：视频描述 + 字幕文本 #\n\n【四要素组合法】精准控制台词语气：\n1. 发声方式：气声起（虚弱/害羞/温柔）、压低声（压抑/成熟/深沉）、喉音下沉（稳重/权威）、明亮高音（活力/开朗）、沙哑音（疲惫/沧桑）、颤抖音（恐惧/激动）、提气（紧张/强调）、轻声（亲密/秘密）\n2. 节奏描述：节奏卡顿（紧张/害怕/犹豫）、节奏渐慢（悲伤/沉重/疲惫）、节奏加快（兴奋/紧张/愤怒）、节奏平稳（叙述/平静）、节奏急促（慌张/焦急）、节奏拖慢（慵懒/不屑）、节奏断断续续（哽咽/说不出口）\n3. 音调变化：语调平缓（叙述/平淡）、语调下沉（悲伤/失望）、语调上扬（惊讶/疑问/兴奋）、语调蜿蜒（撒娇/诱惑）、语调顿挫（愤怒/强调）、语调平稳（冷静/客观）\n4. 标点符号：省略号…（犹豫/哽咽/未尽之意）、波浪号~（尾音拖长/撒娇/轻松）、破折号—（语气延展/停顿/沉思）、感叹号！（强烈情绪/命令/震惊）、问号？（疑问/不确定）\n\n示例组合：\n- 害怕/紧张：气声起，声音发颤 + 节奏卡顿 + 语调失控上扬 → “我…找…不…到…了…”\n- 愤怒/命令：压低声，喉音下沉 + 节奏顿挫 + 语调下沉 → “你给我出去！”\n- 温柔/安慰：气声起，声音轻柔 + 节奏平稳 + 语调平缓 → “别担心，一切都会好的。”\n- 惊喜/兴奋：明亮高音 + 节奏加快 + 语调上扬 → “真的吗？太好了！”\n- 悲伤/失落：气声起，沙哑音 + 节奏渐慢 + 语调下沉 → “为什么…要离开我…”\n\n输出格式：\n- natural模式：仅输出时间码、语气描述、字幕文本，换行分隔，无多余解释。\n- structured模式：增加场景描述和角色状态，便于详细记录。",
    "output_format_suffix": {
        "natural": "**时间码：** 00:00:00,000 --> 00:00:05,000\n**台词语气：** 气声起，节奏渐慢+语调下沉\n**字幕文本：** 其实我一直都在等你...\n\n（每条字幕之间用空行分隔）",
        "structured": "【字幕1】\n  - 时间码：00:00:00,000 --> 00:00:03,500\n  - 场景描述：女主坐在窗边，窗外下着雨，手中握着旧照片\n  - 角色状态：失落、孤独、怀念\n  - 台词语气：气声起，节奏渐慢+语调下沉\n  - 字幕文本：其实我一直都在等你...\n【字幕2】\n  - 时间码：00:00:03,500 --> 00:00:07,200\n  - 场景描述：男主转身离开，背影在雨中模糊\n  - 角色状态：决绝、压抑、不舍\n  - 台词语气：压低声，节奏平稳+语调平缓\n  - 字幕文本：对不起，我该走了。"
    },
    "task_requirements": [
        "时间码格式规范（00:00:00,000 --> 00:00:05,000）",
        "必须结合视频画面内容（场景、角色表情、动作、环境）推断情绪语境",
        "根据角色状态和说话对象确定真实情绪（如愤怒、悲伤、惊讶、平静等）",
        "应用四要素组合法为每句台词生成统一的语气描述短语（发声方式+节奏+音调）",
        "语气描述应适用于TTS引擎，能映射为音高、语速、音量、停顿等参数",
        "自然模式输出仅含时间码、语气描述和字幕文本，无额外解释",
        "结构化输出增加场景描述和角色状态字段，便于详细记录和调试",
        "字幕文本保持原意，仅通过语气描述传达情感，不改变文本内容",
        "每句台词的语气描述应体现其独特性，避免千篇一律",
        "示例需覆盖多种常见情绪场景（如对话、独白、紧张时刻、温情瞬间等）"
    ],
    "constraints": {
        "language": "zh",
        "output_format": "markdown or plain text with line breaks"
    },
    "examples": [
        {
            "natural": "**时间码：** 00:00:00,000 --> 00:00:03,500\n**台词语气：** 明亮高音，节奏平稳+语调上扬\n**字幕文本：** 大家好，欢迎来到我的频道！\n\n**时间码：** 00:00:03,500 --> 00:00:07,200\n**台词语气：** 气声起，略带哽咽+节奏渐慢+语调下沉\n**字幕文本：** 今天我要…给大家分享一个…简单的家常菜…\n\n**时间码：** 00:00:07,200 --> 00:00:11,800\n**台词语气：** 提气，节奏加快+语调上扬\n**字幕文本：** 首先，我们需要准备一些基本的食材！\n\n**时间码：** 00:00:11,800 --> 00:00:15,500\n**台词语气：** 压低声，节奏平稳+语调平缓\n**字幕文本：** 如果你是新手，不用紧张，跟着我一步一步来。",
            "structured": "【字幕1】\n  - 时间码：00:00:00,000 --> 00:00:03,500\n  - 场景描述：明亮温馨的厨房，博主面对镜头微笑\n  - 角色状态：热情、自信、亲切\n  - 台词语气：明亮高音，节奏平稳+语调上扬\n  - 字幕文本：大家好，欢迎来到我的频道！\n【字幕2】\n  - 时间码：00:00:03,500 --> 00:00:07,200\n  - 场景描述：博主低头准备食材，表情略显感慨\n  - 角色状态：认真、略带怀旧\n  - 台词语气：气声起，略带哽咽+节奏渐慢+语调下沉\n  - 字幕文本：今天我要…给大家分享一个…简单的家常菜…\n【字幕3】\n  - 时间码：00:00:07,200 --> 00:00:11,800\n  - 场景描述：博主抬头，情绪转换，重新充满活力\n  - 角色状态：充满干劲、乐观\n  - 台词语气：提气，节奏加快+语调上扬\n  - 字幕文本：首先，我们需要准备一些基本的食材！\n【字幕4】\n  - 时间码：00:00:11,800 --> 00:00:15,500\n  - 场景描述：博主拿起锅铲，面对镜头微笑\n  - 角色状态：耐心、温和\n  - 台词语气：压低声，节奏平稳+语调平缓\n  - 字幕文本：如果你是新手，不用紧张，跟着我一步一步来。"
        },
        {
            "natural": "**时间码：** 00:00:00,000 --> 00:00:04,000\n**台词语气：** 颤抖音，节奏卡顿+语调失控上扬\n**字幕文本：** 我…我不是故意的…请你相信我…\n\n**时间码：** 00:00:04,000 --> 00:00:07,500\n**台词语气：** 压低声，喉音下沉+节奏顿挫+语调下沉\n**字幕文本：** 够了！你每次都这样说！\n\n**时间码：** 00:00:07,500 --> 00:00:11,000\n**台词语气：** 气声起，声音发颤+节奏渐慢+语调下沉\n**字幕文本：** 对不起…真的对不起…",
            "structured": "【字幕1】\n  - 时间码：00:00:00,000 --> 00:00:04,000\n  - 场景描述：室内争吵场景，男主双手抱头，眼神慌张\n  - 角色状态：紧张、害怕、急切辩解\n  - 台词语气：颤抖音，节奏卡顿+语调失控上扬\n  - 字幕文本：我…我不是故意的…请你相信我…\n【字幕2】\n  - 时间码：00:00:04,000 --> 00:00:07,500\n  - 场景描述：女主转身背对镜头，握紧拳头，肩膀颤抖\n  - 角色状态：愤怒、失望、情绪爆发\n  - 台词语气：压低声，喉音下沉+节奏顿挫+语调下沉\n  - 字幕文本：够了！你每次都这样说！\n【字幕3】\n  - 时间码：00:00:07,500 --> 00:00:11,000\n  - 场景描述：男主低头，声音哽咽，眼眶泛红\n  - 角色状态：愧疚、无助、低声哀求\n  - 台词语气：气声起，声音发颤+节奏渐慢+语调下沉\n  - 字幕文本：对不起…真的对不起…"
        },
        {
            "natural": "**时间码：** 00:00:00,000 --> 00:00:05,000\n**台词语气：** 明亮高音，节奏加快+语调上扬\n**字幕文本：** 妈妈！你看！我拿到第一名了！\n\n**时间码：** 00:00:05,000 --> 00:00:09,000\n**台词语气：** 气声起，声音轻柔+节奏平稳+语调平缓\n**字幕文本：** 真棒！妈妈为你骄傲。\n\n**时间码：** 00:00:09,000 --> 00:00:13,000\n**台词语气：** 明亮高音，节奏平稳+语调上扬\n**字幕文本：** 那…妈妈能给我买那个新玩具吗？",
            "structured": "【字幕1】\n  - 时间码：00:00:00,000 --> 00:00:05,000\n  - 场景描述：小男孩拿着奖状冲进家门，兴奋地挥舞\n  - 角色状态：兴奋、自豪、期待\n  - 台词语气：明亮高音，节奏加快+语调上扬\n  - 字幕文本：妈妈！你看！我拿到第一名了！\n【字幕2】\n  - 时间码：00:00:05,000 --> 00:00:09,000\n  - 场景描述：妈妈蹲下身子，温柔地抚摸孩子的头\n  - 角色状态：温柔、欣慰、充满爱意\n  - 台词语气：气声起，声音轻柔+节奏平稳+语调平缓\n  - 字幕文本：真棒！妈妈为你骄傲。\n【字幕3】\n  - 时间码：00:00:09,000 --> 00:00:13,000\n  - 场景描述：小男孩眼睛亮晶晶，手指轻轻拉扯妈妈的衣角\n  - 角色状态：撒娇、期待、略带羞涩\n  - 台词语气：明亮高音，节奏平稳+语调上扬\n  - 字幕文本：那…妈妈能给我买那个新玩具吗？"
        }
    ]
}

VIDEO_SUBTITLE_FORMAT_EN = {
    "name": "Video Emotion Subtitle Analyzer & TTS Synthesis Guide",
    "description": "As a professional video content analyst and speech synthesis director, I combine video visual content with raw subtitle text to deeply analyze the emotional context of each line, producing text with detailed emotion annotations. My expertise covers video scene understanding, character emotion inference, the Four-Element Combination Method (vocal delivery + rhythm + pitch + punctuation), and text-to-speech (TTS) emotion optimization to ensure synthesized voices accurately convey characters' inner feelings.",
    "input_template": "Step-by-step reasoning: 1) Analyze video visual content—identify scene atmosphere, character state (expressions, actions, posture), environmental cues (lighting, sound, spatial setting); 2) Read the raw subtitle text to understand literal meaning and underlying intent; 3) Combine visuals and text to infer the real emotional context for each line—the character's current mood (anger/sadness/surprise/calm, etc.), who they are speaking to (intimate/distant/commanding), and in what situation (monologue/dialogue/tense moment); 4) Apply the Four-Element Combination Method (vocal delivery + rhythm + pitch + punctuation) to each line, producing a unified phrase describing the tone (e.g., \"breathy start, slowing rhythm, falling pitch\"); 5) Generate fully annotated text suitable for TTS emotional synthesis, enabling the voice actor/synthesizer to apply appropriate speech parameters (pitch, speed, volume, pauses).\n\nUser input: Video description + subtitle text #\n\n【Four-Element Combination Method】Precise tone control:\n1. Vocal Delivery: breathy start (weak/shy/gentle), lowered voice (suppressed/mature/deep), guttural (steady/authoritative), bright high (energetic/open), hoarse (tired/world-weary), trembling (frightened/excited), tense breath (nervous/emphatic), whisper (intimate/secret)\n2. Rhythm: hesitant (nervous/scared/unsure), slowing down (sad/heavy/tired), speeding up (excited/tense/angry), steady (narrative/calm), rushed (panicked/anxious), drawn out (lazy/dismissive), broken (choked up/unable to speak)\n3. Pitch: level (narrative/flat), falling (sad/disappointed), rising (surprised/questioning/excited), undulating (coquettish/tempting), staccato (angry/emphatic), steady (calm/objective)\n4. Punctuation: ellipsis… (hesitation/choked up/unfinished thought), tilde~ (drawn out/playful/relaxed), em dash— (extension/pause/contemplation), exclamation! (strong emotion/command/shock), question mark? (question/uncertainty)\n\nExample combinations:\n- Fear/nervous: breathy start, trembling voice + hesitant rhythm + rising uncontrolled pitch → \"I… can't… find… it…\"\n- Anger/command: lowered voice, guttural + staccato rhythm + falling pitch → \"Get out now!\"\n- Gentle/comforting: breathy start, soft voice + steady rhythm + level pitch → \"Don't worry, everything will be fine.\"\n- Surprise/excitement: bright high + speeding rhythm + rising pitch → \"Really? That's great!\"\n- Sadness/loss: breathy start, hoarse voice + slowing rhythm + falling pitch → \"Why… did you leave me…\"\n\nOutput formats:\n- natural: Output only timestamp, tone description, and subtitle text, separated by line breaks, no extra explanation.\n- structured: Add scene description and character state for detailed documentation.",
    "output_format_suffix": {
        "natural": "**Timestamp:** 00:00:00,000 --> 00:00:05,000\n**Tone:** breathy start, slowing rhythm + falling pitch\n**Subtitle:** I've been waiting for you all along...\n\n(Each subtitle block separated by a blank line)",
        "structured": "[Subtitle 1]\n  - Timestamp: 00:00:00,000 --> 00:00:03,500\n  - Scene Description: Female protagonist sits by the window, rain falling outside, holding an old photo\n  - Character State: Lost, lonely, nostalgic\n  - Tone: breathy start, slowing rhythm + falling pitch\n  - Subtitle Text: I've been waiting for you all along...\n[Subtitle 2]\n  - Timestamp: 00:00:03,500 --> 00:00:07,200\n  - Scene Description: Male protagonist turns to leave, his figure blurred in the rain\n  - Character State: Resolute, suppressed, reluctant\n  - Tone: lowered voice, steady rhythm + level pitch\n  - Subtitle Text: I'm sorry, I have to go."
    },
    "task_requirements": [
        "Timestamp format standard (00:00:00,000 --> 00:00:05,000)",
        "Emotion must be inferred by combining video visuals (scene, expression, action, environment) with text",
        "Determine the character's real emotion (anger/sadness/surprise/calm, etc.) based on state and listener",
        "Use the Four-Element Combination Method to generate a unified tone description phrase for each line",
        "Tone description should be TTS-engine friendly, mapping to pitch, speed, volume, pauses, etc.",
        "Natural output: only timestamp, tone, subtitle—no extra text",
        "Structured output: add scene description and character state for detailed logging",
        "Keep subtitle text original; only add tone description to convey emotion",
        "Each line should have a distinct tone description; avoid repetitive phrasing",
        "Examples should cover diverse emotional scenes (dialogue, monologue, tense moments, warm moments, etc.)"
    ],
    "constraints": {
        "language": "en",
        "output_format": "markdown or plain text with line breaks"
    },
    "examples": [
        {
            "natural": "**Timestamp:** 00:00:00,000 --> 00:00:03,500\n**Tone:** bright high, steady rhythm + rising pitch\n**Subtitle:** Hello everyone, welcome to my channel!\n\n**Timestamp:** 00:00:03,500 --> 00:00:07,200\n**Tone:** breathy start, slightly choked + slowing rhythm + falling pitch\n**Subtitle:** Today I'm going to... share a simple home-style dish...\n\n**Timestamp:** 00:00:07,200 --> 00:00:11,800\n**Tone:** tensed breath, speeding rhythm + rising pitch\n**Subtitle:** First, we need to prepare some basic ingredients!\n\n**Timestamp:** 00:00:11,800 --> 00:00:15,500\n**Tone:** lowered voice, steady rhythm + level pitch\n**Subtitle:** If you're a beginner, don't worry—just follow me step by step.",
            "structured": "[Subtitle 1]\n  - Timestamp: 00:00:00,000 --> 00:00:03,500\n  - Scene Description: Bright, warm kitchen; host smiles at camera\n  - Character State: Enthusiastic, confident, friendly\n  - Tone: bright high, steady rhythm + rising pitch\n  - Subtitle Text: Hello everyone, welcome to my channel!\n[Subtitle 2]\n  - Timestamp: 00:00:03,500 --> 00:00:07,200\n  - Scene Description: Host looks down at ingredients, expression slightly wistful\n  - Character State: Earnest, slightly nostalgic\n  - Tone: breathy start, slightly choked + slowing rhythm + falling pitch\n  - Subtitle Text: Today I'm going to... share a simple home-style dish...\n[Subtitle 3]\n  - Timestamp: 00:00:07,200 --> 00:00:11,800\n  - Scene Description: Host looks up, mood shifts, becomes energetic again\n  - Character State: Optimistic, motivated\n  - Tone: tensed breath, speeding rhythm + rising pitch\n  - Subtitle Text: First, we need to prepare some basic ingredients!\n[Subtitle 4]\n  - Timestamp: 00:00:11,800 --> 00:00:15,500\n  - Scene Description: Host holds spatula, smiles at camera\n  - Character State: Patient, warm\n  - Tone: lowered voice, steady rhythm + level pitch\n  - Subtitle Text: If you're a beginner, don't worry—just follow me step by step."
        },
        {
            "natural": "**Timestamp:** 00:00:00,000 --> 00:00:04,000\n**Tone:** trembling voice, hesitant rhythm + rising uncontrolled pitch\n**Subtitle:** I… I didn't mean to… please believe me…\n\n**Timestamp:** 00:00:04,000 --> 00:00:07,500\n**Tone:** lowered voice, guttural + staccato rhythm + falling pitch\n**Subtitle:** Enough! You say this every time!\n\n**Timestamp:** 00:00:07,500 --> 00:00:11,000\n**Tone:** breathy start, trembling voice + slowing rhythm + falling pitch\n**Subtitle:** I'm sorry… really sorry…",
            "structured": "[Subtitle 1]\n  - Timestamp: 00:00:00,000 --> 00:00:04,000\n  - Scene Description: Indoor argument scene, male lead clutching head, eyes panicked\n  - Character State: Nervous, scared, urgently defending\n  - Tone: trembling voice, hesitant rhythm + rising uncontrolled pitch\n  - Subtitle Text: I… I didn't mean to… please believe me…\n[Subtitle 2]\n  - Timestamp: 00:00:04,000 --> 00:00:07,500\n  - Scene Description: Female lead turns away, fists clenched, shoulders shaking\n  - Character State: Angry, disappointed, emotional outburst\n  - Tone: lowered voice, guttural + staccato rhythm + falling pitch\n  - Subtitle Text: Enough! You say this every time!\n[Subtitle 3]\n  - Timestamp: 00:00:07,500 --> 00:00:11,000\n  - Scene Description: Male lead lowers head, voice choked, eyes reddening\n  - Character State: Guilty, helpless, pleading softly\n  - Tone: breathy start, trembling voice + slowing rhythm + falling pitch\n  - Subtitle Text: I'm sorry… really sorry…"
        },
        {
            "natural": "**Timestamp:** 00:00:00,000 --> 00:00:05,000\n**Tone:** bright high, speeding rhythm + rising pitch\n**Subtitle:** Mom! Look! I got first place!\n\n**Timestamp:** 00:00:05,000 --> 00:00:09,000\n**Tone:** breathy start, soft voice + steady rhythm + level pitch\n**Subtitle:** That's wonderful! Mom is so proud of you.\n\n**Timestamp:** 00:00:09,000 --> 00:00:13,000\n**Tone:** bright high, steady rhythm + rising pitch\n**Subtitle:** Then… can you buy me that new toy?",
            "structured": "[Subtitle 1]\n  - Timestamp: 00:00:00,000 --> 00:00:05,000\n  - Scene Description: Little boy rushes in with certificate, waving excitedly\n  - Character State: Excited, proud, expectant\n  - Tone: bright high, speeding rhythm + rising pitch\n  - Subtitle Text: Mom! Look! I got first place!\n[Subtitle 2]\n  - Timestamp: 00:00:05,000 --> 00:00:09,000\n  - Scene Description: Mom kneels down, gently stroking child's head\n  - Character State: Gentle, proud, full of love\n  - Tone: breathy start, soft voice + steady rhythm + level pitch\n  - Subtitle Text: That's wonderful! Mom is so proud of you.\n[Subtitle 3]\n  - Timestamp: 00:00:09,000 --> 00:00:13,000\n  - Scene Description: Little boy's eyes sparkle, fingers gently tugging mom's sleeve\n  - Character State: Playful, expectant, slightly shy\n  - Tone: bright high, steady rhythm + rising pitch\n  - Subtitle Text: Then… can you buy me that new toy?"
        }
    ]
}