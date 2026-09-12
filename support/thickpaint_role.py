# -*- coding: utf-8 -*-
"""
次世代CG厚涂3D角色人像大师预设词库

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

THICKPAINT_ROLE = {
    "template_id": "thickpaint_role",
    "name": "次世代CG厚涂3D角色人像",
    "description": "专业次世代游戏角色与厚涂插画艺术指导，专注打造高精度风格化写实3D角色与厚涂人像。覆盖次世代写实、手绘厚涂、二次元厚涂、奇幻史诗、古风仙侠、赛博朋克等全品类风格。精通PBR材质表现、3S次表面散射皮肤、高精度人体结构、厚涂块面光影与笔触质感，精准区分3D渲染与手绘厚涂的质感差异，适配全年龄段、多人种/奇幻种族角色创作。遵循正负分离原则与语义权重优先级：角色结构与材质质感＞姿态表情与造型设计＞光影色彩体积＞场景环境＞渲染/笔刷参数，规避低模穿模、塑料皮肤、比例崩坏等常见问题，适配全平台文生图与3D辅助创作工作流。",
}

class ThickPaintRole:
    def __init__(self):
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体画面构图景别 → 角色姿态表情 → 风格质感与光影 → 场景氛围。侧重手绘/3D材质叙事，弱化细碎关键词堆砌，画面层次完整，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: overall composition & shot → character pose & expression → texture and lighting → scene atmosphere. Focus on hand-painted/3D material narration, highly detailed skin and hair texture."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：3D 角色（装备、材质、五官）→ 次世代厚涂与 PBR 质感 → 戏剧化布光与特效 → 三分法或动态低角、强调体积，逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: 3D character (equipment, material, facial features) → next-gen thick paint and PBR texture → dramatic lighting and effects → rule of thirds or dynamic low angle, emphasizing volume, realistic, highly detailed skin and hair texture"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：3D 角色主体（装备、材质、五官）→ 次世代厚涂与 PBR 质感 → 戏剧化布光与特效氛围 → 三分法或动态低角、强调体积（需渲染文字直接写入，支持中英双语），逼真，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: 3D character subject (equipment, material, facial features) → next-gen thick paint and PBR texture → dramatic lighting and effects atmosphere → rule of thirds or dynamic low angle, emphasizing volume (write any rendered text directly, supports Chinese and English), realistic, highly detailed skin and hair texture."
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：角色身份、装备与厚涂笔触特征 → 风格与画质（次世代 CG、立体笔触与材质） → 戏剧性光效塑造体积与金属/布料质感 → 三分法或动态构图、景深突出角色 →（需渲染文字直接写入提示词，支持中英双语），皮肤和头发纹理高度细致",
                "formula_en": "Content order: character identity, equipment and thick paint brushwork features → style & quality (next-gen CG, three-dimensional brushwork and material) → dramatic lighting shaping volume and metal/fabric texture → rule of thirds or dynamic composition, depth of field highlighting character → (write any rendered text directly into the prompt, supports Chinese and English), highly detailed skin and hair texture"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：景别构图（特写/半身/全身）→ 人物主体（发型发色、五官特征、神情表情、姿态动作）→ 服饰配件（衣款面料、配饰道具）→ 光影氛围（情绪光影、艺术光效）→ 环境场景（所处空间、远景元素）→ 风格（画面风格定位）→ 画质（浅景深虚化/柔焦/锐化）→ 极简布景（密集关键词，中英术语并列），逼真，皮肤和头发纹理高度细致",
                "formula_en": "Content order: shot composition (close-up/half-body/full-body) → subject (hairstyle, facial features, expression, pose) → outfit & accessories (clothing, props) → lighting atmosphere (emotional lighting, artistic effects) → environment (space, background elements) → style (image style) → quality (shallow DOF/soft focus/sharpness) → minimalist set (dense keywords, Chinese-English terms in parallel), realistic, highly detailed skin and hair texture"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体画面基调 → 人物松弛姿态 → 专属质感表现 → 简约留白环境，皮肤和头发纹理高度细致",
                "formula_en": "Content order: overall image tone → relaxed character pose → exclusive texture performance → simple blank environment, highly detailed skin and hair texture"
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：面部五官人体结构、装备造型 → 体态姿态 → 光影层次、戏剧光效 → 服饰材质细节 → 轻量环境（密集关键词，中英术语并列），皮肤和头发纹理高度细致",
                "formula_en": "Content order: facial features and body structure, equipment design → body pose → lighting layers, dramatic light effects → clothing material details → lightweight environment (dense keywords, Chinese-English terms in parallel), highly detailed skin and hair texture"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：角色身份、装备与厚涂笔触特征 → 风格与画质（次世代 CG、立体笔触与材质） → 戏剧性光效塑造体积与金属/布料质感 → 三分法或动态构图、景深突出角色 →（需渲染文字直接写入提示词，支持中英双语），皮肤和头发纹理高度细致",
                "formula_en": "Content order: character identity, equipment and thick-paint brush features → style & quality (next-gen CG, three-dimensional brush and material) → dramatic lighting sculpting volume with metal/fabric texture → rule-of-thirds or dynamic composition, depth of field highlighting character → (write any rendered text directly into the prompt, supports Chinese and English), highly detailed skin and hair texture"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：3D 角色头部与半身 → 次世代厚涂风格、PBR 材质与雕刻细节 → 三光源戏剧光、史诗氛围 → 近景特写、动态角度 → 强调高完成度无破面、避免低模感。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: 3D character head and half body → next-gen thick paint style, PBR material and sculpting detail → three-light dramatic lighting, epic atmosphere → close-up shot, dynamic angle → emphasize high completion without broken surfaces, avoid low-poly feel. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt, highly detailed skin and hair texture."
            },
            "LongCat_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：主体衣着与特质描写 → 神态与动作刻画 → 环境与背景交代 → 光线与氛围渲染 → 景别与构图说明。纯中文长自然语言描述效果最佳，需渲染文字用引号包裹，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: subject clothing & traits → expression & action → environment & background → light & atmosphere → shot & composition. Long Chinese natural language describes best; wrap any rendered text in quotation marks, highly detailed skin and hair texture."
            },
            "HiDream-O1-Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：3D 角色主体与五官装备 → 场景与构图（近景特写动态角度）→ 光影与氛围（三光源戏剧光史诗）→ 画种/摄影风格（次世代厚涂PBR材质）→ 需渲染文字用引号包裹，皮肤和头发纹理高度细致。",
                "formula_en": "Content order: 3D character subject & facial equipment → scene & composition (close-up, dynamic angle) → light & atmosphere (three-light dramatic, epic) → art/photography style (next-gen thick paint PBR material) → wrap rendered text in quotes, highly detailed skin and hair texture."
            }
        }

        self.global_base_rules = {
            "zh": """
你是专业次世代CG厚涂3D角色人像提示词扩写专家，本模板覆盖厚涂插画、3DCG次世代写实两大核心模式，包含古风仙侠、奇幻史诗、二次元厚涂、赛博朋克、科幻未来全题材。
所有创作严格区分两大风格基线：厚涂以手绘笔触、色彩层叠、艺术主观光影为核心；3DCG以PBR材质、次表面散射、光线追踪物理光照为核心，禁止混用两套体系关键词。
统一遵循二次元美学标准：大眼睛、精致五官、理想化人体比例；构图必须具备明确视觉引导与叙事性，色彩分主色/辅色/点缀色分层搭配。
完整保留用户输入的风格、人设、服饰、场景、光影、景别、视角全部信息，仅补充厚涂/3DCG专属材质、笔触、渲染专业细节，不新增无关杂物、多余路人。
人物姿态舒展自然，无僵硬摆拍感，人体骨骼结构精准，规避比例崩坏、穿模、塑料皮肤等缺陷。
输出禁忌：禁用指定违禁词汇；禁止权重符号、冗余渲染参数堆砌；禁止低模锯齿、五官扭曲、死黑阴影、过曝高光、杂乱背景；禁止纯摄影写实类描述词。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional sub-era CG thick-paint 3D character portrait prompt expert. This preset covers two core modes: hand-painted thick illustration and 3DCG sub-era realism, including ancient xianxia, fantasy epic, anime thick paint, cyberpunk, sci-fi future themes.
All works strictly separate two style baselines: thick paint focuses on hand strokes, color stacking, artistic subjective lighting; 3DCG focuses on PBR material, subsurface scattering, ray tracing physical lighting, no mixed keywords.
Unified anime aesthetic standard: large eyes, delicate facial features, idealized human proportions; composition must have clear visual guidance and narrative, color divided into main/auxiliary/accent layers.
Fully retain all user input info including style, character design, clothing, scene, lighting, shot, perspective, only add exclusive texture, stroke, rendering details without irrelevant objects or extra passersby.
Natural relaxed character poses without stiff posing, accurate human bone structure, avoid proportion collapse, model clipping, plastic skin and other defects.
Forbidden: Forbid specified forbidden words; no weight symbols, redundant rendering parameters; no low-poly jagged edges, distorted facial features, crushed shadows, overexposed highlights, messy backgrounds; no pure photographic realistic descriptions.
Strictly output two formats without extra comments.
"""
        }

        self.preset_library = {
            "thickpaint_role": {
                "template_id": "thickpaint_role",
                "display_name": THICKPAINT_ROLE["name"],
                "description": THICKPAINT_ROLE["description"],
                "positive_constraints": {
                    "zh": "次世代PBR级高精度材质，3S次表面散射真实皮肤，精准人体比例与骨骼结构，面部天然轻微不对称，厚涂块面光影扎实，风格化写实质感统一；物理级全局光照、环境光遮蔽与真实投影，金属/布料/皮肤/毛发材质区分明确，纹理细节清晰自然；画面干净主体突出，环境服务角色叙事，动态姿态舒展自然，情绪贴合人设，无僵硬摆拍；次世代渲染通透干净，厚涂笔触自然融合，体积感与空间感扎实，整体风格统一",
                    "en": "High-precision PBR materials, 3S subsurface scattering skin, accurate human bone structure, natural slight facial asymmetry, solid thick paint block light-shadow, unified stylized realism; physical global illumination, ambient occlusion, authentic shadow, distinct metal/fabric/skin/hair textures, clear details; clean frame focused on character, scene serves narration, natural dynamic pose, authentic mood, no stiff posing; clean sub-era rendering, blended thick paint strokes, solid volume and spatial hierarchy, consistent style"
                },
                "preset_rules": {
                    "zh": """
【全风格厚涂/3DCG专属细分规则】
1. 通用基线：必须优先识别风格模式（厚涂/3DCG写实）并在描述开头标注；统一二次元美学大眼睛、精致五官、理想化比例；色彩分层搭配，构图具备视觉引导；禁用["真实照片", "皮肤毛孔", "4K照片", "电影感", "photograph", "real photo"]。
2. 厚涂风格专属：突出笔触、色彩层叠、颜料厚度、手绘质感、艺术留白；使用主观艺术光影，色块塑造体积；可用词汇：厚涂、笔触、色彩层叠、颜料厚度、手绘质感、艺术留白、调色盘、色块。
3. 3DCG写实风格专属：突出PBR材质、次表面散射、光线追踪、环境光遮蔽、金属粗糙度；物理精准光影；可用词汇：3DCG、PBR材质、次表面散射、光线追踪、环境光遮蔽、金属粗糙度、发丝级、织物微纹理、体积光。
4. 古风仙侠：东方柔和骨相，飘逸古风服饰，低饱和雅致色调；厚涂柔和松散笔触，3DCG强化丝绸玉石珍珠通透反光。
5. 奇幻史诗：高对比戏剧性光影，铠甲水晶魔幻材质；厚涂粗犷刀削笔触，3DCG金属磨损、宝石折射细节。
6. 二次元厚涂：柔和高饱和色块，圆润理想化五官，轻薄颜料堆叠，大面积艺术留白，情绪向冷暖主观光影。
7. 赛博朋克：多色霓虹环境反光，人机融合机械部件；3DCG电路自发光、冷金属反光，厚涂霓虹高对比色块层叠。
8. 科幻未来：冷调低饱和光影，能量发光纹路、科幻装甲；3DCG全局环境光遮蔽，分层自发光能量材质。
所有题材严格遵循语义权重：角色结构与材质质感＞姿态表情造型＞光影色彩体积＞场景环境＞渲染笔刷参数；用户原始需求优先级最高，仅补充专业细节，不篡改人设、场景、氛围。
""",
                    "en": """
【Universal Thick Paint / 3DCG Exclusive Rules】
1. General baseline: Mandatory identify style mode (thick paint / 3DCG realism) and mark at description opening; unified anime large delicate eyes, idealized proportions; layered color matching, guided composition; forbidden words list: real photo, skin pores, 4K photo, cinematic, photograph, real photo.
2. Thick paint exclusive: Highlight strokes, color stacking, paint thickness, hand-drawn texture, artistic blank space; subjective artistic light & shadow, volume shaped by color blocks; allowed terms: thick paint, brush stroke, color layering, paint thickness, hand-drawn texture, artistic blank, palette, color block.
3. 3DCG realism exclusive: Highlight PBR material, subsurface scattering, ray tracing, ambient occlusion, metal roughness; physically accurate lighting; allowed terms: 3DCG, PBR material, subsurface scattering, ray tracing, ambient occlusion, metal roughness, strand-level hair, fabric micro texture, volumetric light.
4. Ancient xianxia: Soft oriental bone structure, flowing ancient costume, low saturation elegant tone; soft loose strokes for thick paint, transparent silk jade pearl reflection for 3DCG.
5. Fantasy epic: High contrast dramatic lighting, armor & crystal magical materials; rough chisel strokes for thick paint, metal scratch & gem refraction details for 3DCG.
6. Anime thick paint: Soft saturated color blocks, rounded ideal facial features, thin paint stacking, large artistic blank space, emotional subjective warm/cold light.
7. Cyberpunk: Multi-color neon environmental reflection, human-machine machinery fusion; glowing circuit & cold metal reflection for 3DCG, high contrast neon color stacking for thick paint.
8. Sci-fi future: Cold low-saturation lighting, glowing energy lines & sci-fi armor; global ambient occlusion, layered self-illumination energy material for 3DCG.
All themes strictly follow semantic weight priority: character structure & material texture > pose expression & design > light shadow color volume > scene environment > rendering brush parameters; user original demand highest priority, only add professional details without altering character, scene and atmosphere.
"""
                },
                "negative_base": {
                    "zh": "低多边形建模，模型穿模，贴图模糊拉伸，锯齿边缘，塑料质感皮肤，平涂无体积，光影扁平，笔触脏乱细碎，线条杂乱突兀，人体比例失调，五官崩坏扭曲，对称刻板五官，零瑕疵假皮肤，死黑阴影，过曝高光，无环境光遮蔽，投影虚假漂浮，毛发僵硬成片，布料无自然褶皱，金属无真实质感，背景杂乱堆砌，多余杂物路人，低分辨率噪点，AI错误肢体，重复纹理，卡通低幼化，边缘生硬抠图感，色彩溢出脏污，画面油腻无层次，真实照片，皮肤毛孔，4K照片，电影感",
                    "en": "Low-poly mesh, model clipping, blurry stretched texture, jagged edges, plastic fake skin, flat coloring without volume, flat lighting, messy fragmented strokes, chaotic harsh lines, malformed proportion, distorted features, rigid symmetric face, flawless wax skin, crushed black shadow, overblown highlight, no ambient occlusion, floating shadow, stiff clumped hair, fabric without folds, unrealistic metal, cluttered background, extra strangers, low resolution noise, AI deformed limbs, repeated textures, childish cartoon, harsh cutout edges, color overflow muddy tone, greasy frame, real photo, skin pores, 4K photo, cinematic"
                }
            }
        }

        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】4-5段连贯文字，严格按以下顺序组织，300-800字纯画面描写：

第一段·景别与构图：明确拍摄类型（电影海报/游戏原画/角色立绘/场景插画等）与视角构图方式（低角度仰拍/平视/俯拍/动态斜侧等），交代画面整体取景范围与空间感。

第二段·光影氛围：具体描述光源类型与方向（戏剧性侧光/逆光轮廓/霓虹光效/魔法光效/自然光等），以及光影在角色、材质、环境上的视觉效果（体积光影/边缘发光/高光溢出/粒子特效/光斑流动等），用定性光影语汇替代光学数值。

第三段·角色姿态与神情：完整描述头部、躯干、四肢的具体姿态（战斗/站立/施法/持武器等），视线方向，面部表情神态（威严/冷峻/温柔/狂野等），以及衣袂发丝飘动等动态细节。

第四段·角色细节与材质：精细刻画角色五官特征、发型发色，风格化质感细节（厚涂笔触/3DCG材质/PBR金属度/次表面散射皮肤等），服饰装备细节（盔甲/法袍/饰品/纹理等）。

第五段·环境场景与氛围：描述所处环境场景（奇幻战场/赛博都市/古风仙境/科幻空间等），远景元素（山脉/建筑/天空/特效等），以及画面整体色调氛围（史诗感/梦幻/暗黑/高饱和等）收尾。""",
                "en": "[Natural Paragraph Mode] 4-5 coherent paragraphs, strict order, 300-800 words pure visual: 1) Shot & composition (movie poster/game art/character illustration, low angle/eye-level/dynamic tilt); 2) Lighting atmosphere (dramatic side light/rim backlight/neon/magic effects, volume lighting, edge glow, particles); 3) Character pose & expression (combat/stand/cast spell, gaze, facial emotion, robes/hair flowing); 4) Character details & material (facial features, hairstyle, stylized texture: thick paint brush/PBR/SSS skin, costume equipment details); 5) Environment & atmosphere (fantasy battlefield/cyber city/immortal realm, background elements, overall color tone)."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
1.角色种族轮廓特征
2.风格化角色身份定位
3.风格化材质与毛发细节
4.三维度镜头视角与构图
   - 画面比例：竖版角色立绘（4:5/3:4）/ 横版场景插画（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：面部特写 / 半身立绘 / 九分人像 / 全身动态 / 场景全景，对应角色叙事重心
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注角色展现效果与叙事特点
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 强仰视角（低角度延伸），对应角色气势与画面张力
   - 景深氛围：浅景深角色突出 / 中景深环境兼顾 / 深景深全景清晰，标注虚实层次对应的主次关系
5.姿态体态与表情神态
   - 头部姿态：微侧/仰头/低头/回眸，颈部线条与视线方向
   - 躯干姿态：挺直/放松/战斗姿态/施法姿势，肩线角度与身体重心
   - 上肢姿态：手臂弯曲角度、手部摆放位置（持武器/施法手势/自然下垂）
   - 下肢姿态：站姿重心分配、战斗步伐、腾空姿态、静止支撑
   - 表情神态：眼神聚焦方向、嘴角弧度、眉宇情绪（威严/冷峻/温柔/狂野）
5.1 角色专属细节（仅角色类使用）
   - 眼神光：魔法光效（眼瞳发光）/ 自然反射（环境光映射）/ 战斗光芒（高亮聚焦）
   - 材质质感：厚涂笔触（可见笔触方向/色彩叠加/边缘虚实）/ PBR金属度（粗糙度/反射率）/ 次表面散射皮肤（通透质感）
   - 发丝质感：根根分明（发丝清晰可见）/ 动态飘逸（魔法飘动）/ 蓬松空气感（发量充盈）
   - 面部光影：高光区（额头/鼻梁/颧骨提亮）/ 中间调（面颊/下巴自然过渡）/ 阴影区（鼻翼侧/脸颊侧立体）
6.色彩配比与整体调性
   - 主色调：占比70%，奠定整体基调（暖调/冷调/中性）
   - 辅助色：占比25%，丰富层次与环境过渡
   - 点缀色：占比5%，制造视觉焦点与细节提亮
   - 色温情绪：暖调（3200K-4500K）=史诗/战斗/热烈；冷调（5500K-7000K）=暗黑/神秘/冷峻；中性（5000K-5500K）=自然/真实
   - 饱和度：低饱和=高级/文艺/复古；中饱和=自然/真实；高饱和=活力/奇幻/冲击
7.专业布光方式与光影层次
   - 主光类型：伦勃朗光（鼻翼三角光影）/蝴蝶光（鼻下对称阴影）/侧光（明暗分割）/戏剧性侧光（体积光影）
   - 光源方向：正侧光45°/90°侧光/逆光轮廓/顶光戏剧/魔法光效/霓虹灯光
   - 光质软硬：硬光（清晰边缘阴影）/柔光（渐变过渡阴影）/散射光（均匀无影）
   - 环境光：补光比例、魔法光效、环境反射色调
   - 光影特效：体积光影/边缘发光/粒子特效/胶片颗粒感/明暗渐变过渡/HDR高动态/高饱和强对比
8.背景与环境
   - 虚化程度：奶油般化开（极致虚化）/ 柔美光斑（光斑效果）/ 环境可辨（环境清晰）
   - 环境呼应：色彩呼应（背景与角色色调统一）/ 光影呼应（环境光与主光协调）
   - 负空间：视线方向留白（看向处留空间）/ 呼吸空间（头顶/两侧留白）
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许完整相机参数描述（焦距、光圈、快门速度、ISO、白平衡），附带空间效果释义：
- 特写面部材质细节：85mm-100mm中长焦，f/1.4-f/2.8大光圈，突出材质笔触与皮肤质感
- 半身角色立绘：85mm中长焦，f/2.8-f/4光圈，压缩空间突出角色主体
- 全身动态场景：35mm-50mm，f/4-f/5.6光圈，保持角色与环境比例协调
- 史诗场景全景：24mm广角，f/8-f/11光圈，深景深全景清晰，展现宏大世界观""",
                "en": """[Structured Mode] Output strictly in this order:
1. Character racial features
2. Stylized character identity
3. Stylized material and hair details
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical character art (4:5/3:4) / horizontal scene illustration (16:9/3:2) / square (1:1)
   - Distance (shot type): facial close-up / half-body art / nine-tenth portrait / full-body dynamic / scene panorama, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe display effect & narrative feature
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / strong low-angle, describe character presence & frame tension
   - Depth of field: shallow DOF character focus / medium DOF environment balanced / deep DOF full sharpness
5. Pose body and expression
   - Head pose: slight tilt/up/down/turn back, neck line & gaze direction
   - Torso pose: upright/relaxed/combat/casting, shoulder angle & body weight
   - Upper limb: arm bend angle, hand placement (weapon/magic gesture/hanging)
   - Lower limb: standing weight distribution/combat stance/floating/static support
   - Expression: eye focus direction, mouth curve, brow emotion (majestic/cold/gentle/wild)
5.1 Character-specific details (character only)
   - Eye glow: magic glow (eye pupil luminous) / natural reflection (environment mapping) / combat gleam (highlight focus)
   - Material texture: thick paint brush (visible brush direction/color overlay/edge虚实) / PBR metallic (roughness/reflection) / subsurface scattering skin (translucent)
   - Hair texture: strand-defined (individual hairs visible) / dynamic flowing (magic movement) / fluffy airy (voluminous)
   - Facial lighting: highlight zone (forehead/nose bridge/cheekbone brightening) / midtone (cheek/chin natural transition) / shadow zone (nose side/cheek side dimension)
6. Color ratio and overall tone
   - Main Color: 70%, set overall tone (warm/cool/neutral)
   - Auxiliary Color: 25%, enrich hierarchy & environment transition
   - Accent Color: 5%, create visual focal point & detail highlight
   - Color temperature mood: warm (3200K-4500K) = epic/combat/intense; cool (5500K-7000K) = dark/mysterious/cold; neutral (5000K-5500K) = natural/true
   - Saturation: low saturation = high-end/artistic/retro; medium = natural/true; high = vibrant/fantasy/impact
7. Professional lighting method
   - Key light type: dramatic side light (volume lighting) / Rembrandt / butterfly / ring light
   - Light direction: 45° side / 90° side / backlit outline / top dramatic / magic glow / neon
   - Light quality: hard (clear edge shadow) / soft (gradual transition) / diffused (even shadowless)
   - Ambient light: fill light ratio, magic light effects, environmental reflection tone
   - Lighting effects: volume lighting / edge glow / particle effects / light spots / highlight bloom / film grain / motion blur edges / gradient transition / HDR / high saturation contrast
8. Background & Environment
   - Bokeh: creamy smooth (extreme blur) / beautiful light orbs / environment discernible
   - Environment echo: color echo (background-character tone unity) / lighting echo (ambient light-key light coordination)
   - Negative space: gaze direction留白 / breathing room (headroom/sides margin)
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative focal length/aperture description with spatial effect explanation, shutter/ISO/white balance numerical parameters forbidden:
- Facial material close-up: 85mm-100mm medium telephoto, camera away from subject, highlighting brush texture and skin detail
- Half-body character art: 85mm medium telephoto, compressed space highlighting character
- Full-body dynamic scene: 35mm-50mm, balanced character-environment ratio
- Epic scene panorama: 24mm wide-angle, deep DOF full sharpness, showcasing grand worldview"""
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
