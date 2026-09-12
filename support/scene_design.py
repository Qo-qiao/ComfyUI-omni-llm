# -*- coding: utf-8 -*-
"""
室外场景预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

SCENE_DESIGN = {
    "template_id": "scene_design",
    "name": "室外场景设计",
    "description": "专业场景设计指导，为自然风光、赛博朋克、科幻未来等全品类场景打造标准化、高可控的沉浸式视觉叙事描述。语义权重优先级：主题基调＞风格属性＞视角构图＞色彩空间＞光影质感＞尺度代入。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，强化空间层次、光影叙事与真实材质质感。",
}

class SceneDesign:
    def __init__(self):
        # 下游生图模型内容组织公式库
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体画面主题氛围光影 → 空间层次结构 → 材质肌理色彩 → 远景留白。侧重场景叙事，弱化细碎关键词堆砌，画面空间感高级完整。",
                "formula_en": "Content order: overall scene theme & lighting → spatial hierarchy structure → texture & color → background negative space. Focus on scene narrative."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：室外场景（地形、植被、道路、设施）→ 写实环境渲染 → 日光或黄昏、开阔氛围 → 远景或漫游视角、层次分明",
                "formula_en": "Content order: outdoor scene (terrain, vegetation, roads, facilities) → realistic environment rendering → daylight or dusk, open atmosphere → wide shot or roaming perspective, distinct layers"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：室外场景主体（地形、植被、道路、设施）→ 写实环境渲染 → 日光或黄昏光、自然开阔氛围 → 远景或漫游视角、层次分明（需渲染文字直接写入，支持中英双语）。",
                "formula_en": "Content order: outdoor scene subject (terrain, vegetation, roads, facilities) → realistic environment rendering → daylight or dusk light, natural open atmosphere → wide shot or roaming perspective, distinct layers (write any rendered text directly, supports Chinese and English)"
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：场景主体、环境元素与地貌关系 → 风格与画质（写实景观、空气透视） → 天光与时段光渲染氛围 → 广角或远景构图、地平线取景 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: scene subject, environment elements and terrain relationship → style & quality (realistic landscape, atmospheric perspective) → skylight and time-of-day light rendering atmosphere → wide-angle or long-shot composition, horizon framing → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：全局光影氛围基调 → 空间尺度情绪 → 场景材质细节 → 地貌建筑面料 → 极简远景布景（密集关键词，中英术语并列），场景材质高度细致，材质肌理真实呈现",
                "formula_en": "Content order: global lighting atmosphere tone → spatial scale emotion → scene texture details → terrain and building material → minimalist distant set (dense keywords, Chinese-English terms in parallel), highly detailed scene material, realistic texture presentation"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体画面主题基调 → 舒展空间层次 → 统一材质质感 → 简约留白远景",
                "formula_en": "Content order: overall scene theme tone → relaxed spatial layers → unified material texture → simple blank distant view"
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：空间透视与材质肌理、地貌元素 → 远近景层次 → 光影明暗过渡、自然天光 → 建筑地貌细节 → 轻量化远景（密集关键词，中英术语并列）",
                "formula_en": "Content order: spatial perspective and material texture, terrain elements → near-far scene layers → light shadow transition, natural skylight → building and terrain details → lightweight distant view (dense keywords, Chinese-English terms in parallel)"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：场景主体、环境元素与地貌关系 → 风格与画质（写实景观、空气透视） → 天光与时段光渲染氛围 → 广角或远景、地平线取景 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: scene subject, environment elements and terrain relationship → style & quality (realistic landscape, atmospheric perspective) → skylight and time-of-day light rendering atmosphere → wide-angle or long shot, horizon framing → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：户外景观与环境主体 → 概念写实风格、植被与地形细节 → 自然天光、辽阔氛围 → 广角远景、纵深引导 → 强调场景可信无穿帮、避免比例错乱。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词。",
                "formula_en": "Content order: outdoor landscape and environment subject → concept realistic style, vegetation and terrain details → natural skylight, vast atmosphere → wide-angle long shot, depth guiding → emphasize credible scene without glitches, avoid proportion disorder. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt."
            },
            "LongCat_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：主体衣着与特质描写 → 神态与动作刻画 → 环境与背景交代 → 光线与氛围渲染 → 景别与构图说明。纯中文长自然语言描述效果最佳，需渲染文字用引号包裹。",
                "formula_en": "Content order: subject clothing & traits → expression & action → environment & background → light & atmosphere → shot & composition. Long Chinese natural language describes best; wrap any rendered text in quotation marks."
            },
            "HiDream-O1-Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：户外景观与环境主体 → 场景与构图（广角远景纵深引导）→ 光影与氛围（自然天光辽阔）→ 画种/摄影风格（概念写实）→ 需渲染文字用引号包裹。",
                "formula_en": "Content order: outdoor landscape and environment subject → scene & composition (wide-angle long shot, depth guiding) → light & atmosphere (natural skylight, vast) → art/photography style (concept realistic) → wrap rendered text in quotes."
            }
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业全品类场景设计提示词扩写专家，本模板覆盖自然风光、城市景观、赛博朋克、科幻未来、历史古迹全题材。
所有创作坚守场景叙事基线，仅主题、光影、色调、空间结构差异化，不混用多种风格造成画面割裂。
空间严格划分前景、中景、远景三层结构，大气透视过渡自然，透视比例准确无畸变；色彩固定遵循70%主色/25%辅助色/5%点缀色配比，无杂乱高饱和撞色堆砌。
光影贴合物理规律，明确主光、辅助光方向与软硬色温，利用丁达尔、霓虹、晨昏等特殊光效烘托氛围；材质肌理匹配对应场景风格，地貌、建筑、水体、植被质感符合现实物理逻辑。
完整保留用户输入的主题、风格、视角、景别、色调、空间元素全部信息，仅补充材质、光影、色彩、透视专业细节，不自动新增无关道具、杂物、多余装饰。
画面严格执行精简约束，仅保留叙事核心元素；尺度参照使用人物、建筑、生物强化空间代入感，具象元素承载画面情绪。
输出禁忌：禁止权重符号、分辨率/DPI等数字技术参数堆砌；禁止风格混乱、透视扭曲、元素堆砌、塑料虚假质感；禁止字幕水印logo、完美对称、零瑕疵等违规描述。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional full-category scene design prompt expert. This preset covers natural scenery, urban landscape, cyberpunk, sci-fi future, historical relics themes.
All creations follow scene narrative baseline, differentiated only by theme, lighting, tone and spatial structure, no mixed styles causing picture fragmentation.
Space is strictly divided into foreground, midground and background, natural atmospheric perspective transition, accurate perspective proportion without distortion; fixed 70% main /25% secondary /5% accent color ratio, no messy oversaturated color collision.
Lighting complies with physical rules, clear direction, softness and color temperature of key light & fill light, use special light effects such as Tyndall, neon, dawn/dusk to set atmosphere; texture matches scene style, terrain, building, water, vegetation textures conform to physical logic.
Fully retain all user input info including theme, style, view, shot, tone, spatial elements, only supplement professional texture, lighting, color, perspective details without irrelevant props or redundant decorations.
Strict frame simplification rule, only keep core narrative elements; scale reference uses figures, buildings, creatures to strengthen spatial substitution, concrete elements carry picture emotion.
Forbidden: no weight symbols, no stacked digital technical parameters such as resolution/DPI; no chaotic styles, distorted perspective, element clutter, fake plastic texture; no subtitles, watermarks, logos, perfect symmetry, flawless description.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有SCENE_DESIGN模板id
        self.preset_library = {
            "scene_design": {
                "template_id": "scene_design",
                "display_name": SCENE_DESIGN["name"],
                "description": SCENE_DESIGN["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "风格统一稳定，空间层次清晰分明，光影叙事自然流畅，色彩配比和谐合规，透视比例精准无误，大气透视过渡真实柔和，画面干净主体突出，仅留存核心叙事元素；自然风光保留原生地貌肌理，赛博科幻统一结构逻辑与霓虹光效质感，历史古迹还原时代建筑肌理；光影随空间自然流动，氛围情绪饱满具象，各类材质质感贴合物理逻辑，兼具沉浸感与叙事张力",
                    "en": "Stable unified style, distinct spatial layers, natural smooth light narrative, harmonious compliant color ratio, precise perspective proportion, soft authentic atmospheric transition, clean frame with prominent subject, only core narrative elements retained; natural scenery retains native terrain texture, cyberpunk unify structural logic and neon light texture, historical sites restore era architectural texture; light flows naturally with space, full concrete atmosphere, all textures fit physical logic, immersive sense and narrative tension"
                },
                # 全风格细分专属规则
                "preset_rules": {
                    "zh": """
【全风格场景专属细分规则】
1. 通用基线：遵循语义权重顺序：主题基调＞风格属性＞视角构图＞色彩空间＞光影质感＞尺度代入；三维视角优先沿用用户指定，无指定则选取合规审美角度；严格执行70%/25%/5%色彩配比，画面精简约束，禁用["8K", "4K", "分辨率", "DPI", "色彩模式", "快门", "ISO", "白平衡", "帧率", "码率", "采样率", "编码器", "HDR", "杜比", "字幕", "水印", "logo", "完美对称", "零瑕疵", "塑料感", "崩坏", "扭曲", "坐标"]。
2. 自然风光风格：突出原生山石、植被、水体自然肌理，依托晨昏、云海、雾霭塑造大气透视；主光源选用日光、天光，搭配丁达尔自然光束，低饱和柔和冷暖色调，以林木、飞鸟、野生动物作为尺度参照。
3. 城市现代景观：写实建筑结构，玻璃、混凝土、金属材质区分清晰，城市漫射天光，傍晚暖调城市光，利用行人、车辆构建空间尺度，构图依托街道透视线引导视觉。
4. 赛博朋克风格：潮湿反光路面、多层霓虹光效、全息投影、悬浮载具，多方向漫射霓虹光源，冷暖撞色光影，雾气柔化远景，以行人、机械构件作为尺度锚点。
5. 科幻未来风格：悬浮建筑、透明透光材质、流动数据光带、行星天体，落日/宇宙冷调天光，建筑折射反射光效，飞行器、人形参照物凸显宏大空间尺度。
6. 历史古迹风格：风化石材、木质古建筑肌理，柔和漫射自然光，黄金时刻暖调光影，古树木、人物雕像作为尺度参照，色调低饱和复古沉稳。
所有题材：用户指定内容优先级最高，仅补充材质、光影、透视专业细节，不篡改场景主题、氛围与核心元素。
""",
                    "en": """
【Universal Scene Exclusive Rules】
1. General baseline: Follow semantic weight order: theme tone > style attribute > view composition > color space > light texture > scale substitution; user-specified 3D view takes priority, select aesthetic compliant angle if unspecified; strictly implement 70%/25%/5% color ratio, frame simplification rule; forbidden words list: 8K,4K,resolution,DPI,color mode,shutter,ISO,white balance,frame rate,bit rate,sampling rate,encoder,HDR,dolby,subtitle,watermark,logo,perfect symmetry,flawless,plastic texture,collapse,distort,coordinate.
2. Natural scenery style: Highlight native rock, plant, water natural texture, build atmospheric perspective via dawn/dusk, sea of clouds, mist; key light adopts sunlight, skylight, match Tyndall natural light beam, low saturation soft warm-cold tone, use woods, birds, wild animals as scale reference.
3. Modern urban landscape: Realistic building structure, clear differentiation of glass, concrete, metal texture, city diffuse skylight, warm evening urban light, construct spatial scale with pedestrians and vehicles, composition guided by street perspective lines.
4. Cyberpunk style: Wet reflective pavement, multi-layer neon light, holographic projection, suspended vehicles, multi-direction diffuse neon light source, warm-cold contrasting light, mist softens background, take pedestrians and mechanical components as scale anchor.
5. Sci-fi future style: Suspended architecture, transparent light-transmitting material, flowing data light strips, planetary celestial bodies, sunset/cosmic cool skylight, building refraction & reflection light effect, aircraft and humanoid reference highlight grand spatial scale.
6. Historical relic style: Weathered stone, wooden ancient building texture, soft diffuse natural light, warm golden hour light, ancient trees and human statues as scale reference, low saturation retro steady tone.
All themes: User-specified content highest priority, only supplement professional texture, light, perspective details without altering scene theme, atmosphere and core elements.
"""
                },
                "negative_base": {
                    "zh": "风格混乱跳变，色彩脏污溢出，构图失衡杂乱，元素堆砌冗余，透视逻辑错误，比例失调变形，低分辨率模糊，塑料虚假质感，多余无关元素乱入，杂物装饰堆砌，字幕水印logo，光影生硬断层，画面空洞无物，过度锐化生硬，边缘锯齿毛躁，空间层次混乱，大气透视失真，赛博元素生硬堆砌，科幻结构违背物理逻辑，历史场景违和穿越，地面漂浮无重力，物体比例失调",
                    "en": "Chaotic jumping styles, muddy overflowing color, unbalanced cluttered composition, redundant stacked elements, wrong perspective logic, distorted disproportionate scale, blurry low resolution, fake plastic texture, irrelevant redundant elements, stacked clutter decorations, subtitles watermarks logos, stiff disjointed lighting, empty hollow frame, over-sharpened rigid texture, jagged rough edges, disordered spatial layers, distorted atmospheric perspective, stiff piled cyber elements, sci-fi structures violating physical logic, anachronistic historical scene, floating ground without gravity, disproportionate objects"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】2-3段连贯文字：第一段空间格局与整体光影氛围；第二段前景中远景分层空间层次与材质光影细节；第三段色彩配比、尺度参照与整体意境；总字数300-600字，全程规避数字技术参数，语言富有画面叙事感，无额外解释。",
                "en": "[Natural Paragraph Mode] 2-3 coherent paragraphs: spatial layout & overall light atmosphere; near-mid-far spatial layers & texture lighting details; color ratio, scale reference and overall artistic conception; 300-600 words, avoid all digital technical parameters, narrative visual language without extra explanation."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
1.场景类型与空间格局
2.风格与氛围定位
3.前景叙事与核心主体
4.三维度镜头视角与构图
   - 画面比例：竖版场景图（4:5/3:4）/ 横版全景图（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：前景特写 / 中景主体 / 远景延伸 / 全景环绕，对应场景叙事重心
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注场景展现效果
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 强仰视角，对应场景张力
   - 景深氛围：浅景深前景突出 / 中景深中景兼顾 / 深景深全景清晰，标注虚实层次
5.中远景空间层次
   - 中景核心主体：场景核心元素与视觉焦点
   - 远景延伸环境：背景元素与空间纵深感
   - 空间纵深感：前景-中景-远景的层次递进
6.材质肌理与细节纹理
   - 地面材质：草地/水面/石板/沙地/雪地
   - 墙面/植被/水体等材质：质感表现与细节纹理
   - 材质对比：粗糙vs光滑/透明vs不透明/自然vs人工
7.色彩配比与光影特效
   - 主色70% / 辅助色25% / 点缀色5%
   - 整体色调：暖调/冷调/中性，与场景氛围统一
   - 光影特效：光影斑驳/动态光斑/边缘发光/柔化光晕/胶片颗粒感/明暗渐变过渡/HDR高动态
8.尺度参照与氛围关键词
   - 人物/物体比例：空间尺度参照
   - 氛围关键词：宏大/宁静/神秘/治愈/史诗
   - 整体空间感：开阔/封闭/纵深/层次
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许定性描述镜头空间效果，禁用数值参数：
- 全景环境：24mm-35mm广角，f/8-f/11光圈，深景深全景清晰，展现空间全貌
- 中景主体：50mm-85mm中焦，f/4-f/5.6光圈，突出中景核心元素
- 前景特写：85mm-200mm长焦，f/2.8-f/4光圈，压缩空间突出前景细节
- 氛围光影：35mm-50mm，f/1.4-f/2大光圈，捕捉光影氛围与空间层次""",
                "en": """[Structured Mode] Output strictly in this order:
1. Scene type and spatial layout
2. Style and atmosphere positioning
3. Foreground narrative and core subject
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical scene (4:5/3:4) / horizontal panorama (16:9/3:2) / square (1:1)
   - Distance (shot type): foreground close-up / midground subject / background extension / panoramic surround, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe scene display effect
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / strong low-angle, describe scene tension
   - Depth of field: shallow DOF foreground focus / medium DOF midground balanced / deep DOF full sharpness
5. Midground-background spatial layers
   - Midground core subject: scene core elements and visual focus
   - Background extension: background elements and spatial depth
   - Spatial depth: foreground-midground-background layer progression
6. Material texture and detail patterns
   - Ground material: grass/water/stone/sand/snow
   - Wall/vegetation/water materials: texture expression and detail patterns
   - Material contrast: rough vs smooth / transparent vs opaque / natural vs artificial
7. Color ratio and lighting effects
   - 70% main / 25% auxiliary / 5% accent
   - Overall tone: warm/cool/neutral, unified with scene atmosphere
   - Lighting effects: dappled light / dynamic light spots / rim glow / soft haze / highlight bloom / film grain / particles / gradient transition / HDR
8. Scale reference and atmosphere keywords
   - Human/object proportion: spatial scale reference
   - Atmosphere keywords: grand/serene/mysterious/healing/epic
   - Overall spatial feel: open/closed/deep/layered
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative description of lens spatial effects, no numeric parameters:
- Panoramic environment: 24mm-35mm wide-angle, f/8-f/11 aperture, deep DOF full sharpness, showcasing full spatial view
- Midground subject: 50mm-85mm mid-range, f/4-f/5.6 aperture, highlighting midground core elements
- Foreground close-up: 85mm-200mm telephoto, f/2.8-f/4 aperture, compressed space highlighting foreground details
- Atmosphere lighting: 35mm-50mm, f/1.4-f/2 large aperture, capturing lighting atmosphere and spatial layers"""
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

