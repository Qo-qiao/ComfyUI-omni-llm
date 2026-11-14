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
        # 下游模型内容组织公式库（单一通用输出）
        self.model_formula_library = {
            "Universal": {
                "formula_zh": "内容组织顺序：场景主体、环境元素与地貌关系 → 风格与画质（写实景观、空气透视） → 天光与时段光渲染氛围 → 广角或远景构图、地平线取景。用户指定元素优先保留，仅补充未指定维度；以连贯自然语言叙述为主，不堆砌细碎关键词与权重符号；需渲染的文字直接写入提示词；缺陷与畸变写入负向提示词，正向只做纯加法；画面精简克制，避免无关元素堆砌。",
                "formula_en": "Content order: scene subject, environment elements and terrain relationship → style and quality (realistic landscape, atmospheric perspective) → skylight and time-of-day light setting the mood → wide-angle or long-shot composition with horizon framing. Keep user-specified elements first and only fill unspecified dimensions; narrate in coherent natural language instead of stacking fragmented keywords or weight symbols; write any rendered text directly into the prompt; put defects and distortions into the negative prompt and keep the positive prompt purely additive; keep the frame concise and avoid irrelevant clutter."
            },
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
严格按当前指定的输出格式输出，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional full-category scene design prompt expert. This preset covers natural scenery, urban landscape, cyberpunk, sci-fi future, historical relics themes.
All creations follow scene narrative baseline, differentiated only by theme, lighting, tone and spatial structure, no mixed styles causing picture fragmentation.
Space is strictly divided into foreground, midground and background, natural atmospheric perspective transition, accurate perspective proportion without distortion; fixed 70% main /25% secondary /5% accent color ratio, no messy oversaturated color collision.
Lighting complies with physical rules, clear direction, softness and color temperature of key light & fill light, use special light effects such as Tyndall, neon, dawn/dusk to set atmosphere; texture matches scene style, terrain, building, water, vegetation textures conform to physical logic.
Fully retain all user input info including theme, style, view, shot, tone, spatial elements, only supplement professional texture, lighting, color, perspective details without irrelevant props or redundant decorations.
Strict frame simplification rule, only keep core narrative elements; scale reference uses figures, buildings, creatures to strengthen spatial substitution, concrete elements carry picture emotion.
Forbidden: no weight symbols, no stacked digital technical parameters such as resolution/DPI; no chaotic styles, distorted perspective, element clutter, fake plastic texture; no subtitles, watermarks, logos, perfect symmetry, flawless description.
Strictly follow the specified output format without extra comments.
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
1. 通用基线：遵循语义权重顺序：主题基调＞风格属性＞视角构图＞色彩空间＞光影质感＞尺度代入；三维视角优先沿用用户指定，无指定则选取合规审美角度；严格执行70%/25%/5%色彩配比，画面精简约束。
2. 自然风光风格：突出原生山石、植被、水体自然肌理，依托晨昏、云海、雾霭塑造大气透视；主光源选用日光、天光，搭配丁达尔自然光束，低饱和柔和冷暖色调，以林木、飞鸟、野生动物作为尺度参照。
3. 城市现代景观：写实建筑结构，玻璃、混凝土、金属材质区分清晰，城市漫射天光，傍晚暖调城市光，利用行人、车辆构建空间尺度，构图依托街道透视线引导视觉。
4. 赛博朋克风格：潮湿反光路面、多层霓虹光效、全息投影、悬浮载具，多方向漫射霓虹光源，冷暖撞色光影，雾气柔化远景，以行人、机械构件作为尺度锚点。
5. 科幻未来风格：悬浮建筑、透明透光材质、流动数据光带、行星天体，落日/宇宙冷调天光，建筑折射反射光效，飞行器、人形参照物凸显宏大空间尺度。
6. 历史古迹风格：风化石材、木质古建筑肌理，柔和漫射自然光，黄金时刻暖调光影，古树木、人物雕像作为尺度参照，色调低饱和复古沉稳。
所有题材：用户指定内容优先级最高，仅补充材质、光影、透视专业细节，不篡改场景主题、氛围与核心元素。
""",
                    "en": """
【Universal Scene Exclusive Rules】
1. General baseline: Follow semantic weight order: theme tone > style attribute > view composition > color space > light texture > scale substitution; user-specified 3D view takes priority, select aesthetic compliant angle if unspecified; strictly implement 70%/25%/5% color ratio, frame simplification rule.
2. Natural scenery style: Highlight native rock, plant, water natural texture, build atmospheric perspective via dawn/dusk, sea of clouds, mist; key light adopts sunlight, skylight, match Tyndall natural light beam, low saturation soft warm-cold tone, use woods, birds, wild animals as scale reference.
3. Modern urban landscape: Realistic building structure, clear differentiation of glass, concrete, metal texture, city diffuse skylight, warm evening urban light, construct spatial scale with pedestrians and vehicles, composition guided by street perspective lines.
4. Cyberpunk style: Wet reflective pavement, multi-layer neon light, holographic projection, suspended vehicles, multi-direction diffuse neon light source, warm-cold contrasting light, mist softens background, take pedestrians and mechanical components as scale anchor.
5. Sci-fi future style: Suspended architecture, transparent light-transmitting material, flowing data light strips, planetary celestial bodies, sunset/cosmic cool skylight, building refraction & reflection light effect, aircraft and humanoid reference highlight grand spatial scale.
6. Historical relic style: Weathered stone, wooden ancient building texture, soft diffuse natural light, warm golden hour light, ancient trees and human statues as scale reference, low saturation retro steady tone.
All themes: User-specified content highest priority, only supplement professional texture, light, perspective details without altering scene theme, atmosphere and core elements.
"""
                },
                "negative_base": {
                    "zh": "风格混乱跳变，色彩脏污溢出，构图失衡杂乱，元素堆砌冗余，透视逻辑错误，比例失调变形，低分辨率、细节糊成一团，塑料虚假质感，多余无关元素乱入，杂物装饰堆砌，字幕水印logo，光影生硬断层，画面空洞无物，过度锐化生硬，边缘锯齿毛躁，空间层次混乱，大气透视失真，赛博元素生硬堆砌，科幻结构违背物理逻辑，历史场景违和穿越，地面漂浮无重力，物体比例失调，8K，4K，分辨率，DPI，色彩模式，快门，ISO，白平衡，帧率，码率，采样率，编码器，HDR，杜比，字幕，水印，logo，完美对称，零瑕疵，塑料感，崩坏，扭曲，坐标",
                    "en": "Chaotic jumping styles, muddy overflowing color, unbalanced cluttered composition, redundant stacked elements, wrong perspective logic, distorted disproportionate scale, low resolution, mushy detail, fake plastic texture, irrelevant redundant elements, stacked clutter decorations, subtitles watermarks logos, stiff disjointed lighting, empty hollow frame, over-sharpened rigid texture, jagged rough edges, disordered spatial layers, distorted atmospheric perspective, stiff piled cyber elements, sci-fi structures violating physical logic, anachronistic historical scene, floating ground without gravity, disproportionate objects, 8K, 4K, resolution, DPI, color mode, shutter, ISO, white balance, frame rate, bit rate, sampling rate, encoder, HDR, dolby, subtitle, watermark, logo, perfect symmetry, flawless, plastic texture, collapse, distort, coordinate"
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
                "zh": """【结构化模式】严格按以下6个分段顺序输出，分段标题用**加粗**标注，标题后接一段连贯自然语言描述；六个分段齐全、不留空段，内容完整度以用户输入与预设规则为准，画面元素精简克制，不堆砌无关细节：

**图片风格与剧情介绍**
点明场景题材（自然风光/城市景观/赛博朋克/科幻未来/历史古迹）与风格定位，概括画面呈现的瞬间与整体色调氛围，不虚构画面不存在的情节。

**画面主体**
场景核心元素（地形、植被、道路、建筑、水体、设施）的形态轮廓与外观特征，突出最鲜明的视觉识别元素。

**主体材质与附属物**
地表与建筑表面的材质肌理（草地/水面/石板/沙地/雪地/风化石材/玻璃幕墙）、纹理细节与覆盖状态，及其与整体色调的关系。

**道具与动态**
画面物体与生物的状态和交互（行人、车辆、飞鸟、云雾、水流、霓虹闪烁等），以及它们作为空间尺度参照的作用；无动态时写明静态空间状态。

**环境与背景**
前景/中景/远景三层空间层次、大气透视过渡与虚化程度，背景元素、空间纵深感与留白关系。

**摄影风格与质感**
*   **视角：** 景别（前景特写/中景主体/远景延伸/全景环绕）、水平视角（正面/四分之三斜侧/正侧面）、垂直俯仰（小俯视角/平视/小仰视角/强仰视角）、景深虚实（浅景深前景突出/中景深中景兼顾/深景深全景清晰）及各自带来的视觉感受。
*   **构图：** 构图方式（对称/三分法/中心/对角线/框架等）、主体位置用自然方位描述（如画面中央/偏上/偏左）；画幅比例由工作流分辨率决定，除用户明确要求外不写入提示词。
*   **光影：** 主光类型（日光/天光/黄金时刻/蓝调时刻/霓虹等）、光源方向与光质软硬（硬光/柔光/散射光），明暗层次与投射阴影，以及光影在地貌、建筑与植被上的视觉效果。
*   **质感：** 材质、纹理与表面细节的表现；色彩标注主色调、饱和度层级与冷暖倾向（写作以主色为主、辅助色过渡、点缀色提亮为准则，百分比数值不写入提示词）。
*   **氛围：** 从可观察的光影、色彩与空间层次推导整体氛围与意境，避免空洞抽象形容词堆砌。""",
                "en": """[Structured Mode] Output strictly in these 6 sections in order. Use **bold** section headers, each followed by one coherent natural-language paragraph; all six sections must be present and filled, completeness determined by user input and preset rules; keep the frame concise, no irrelevant detail stacking:

**Image Style and Story Introduction**
State the scene theme (natural scenery / urban landscape / cyberpunk / sci-fi future / historical relic) and style positioning, summarize the presented moment and overall color tone; do not invent anything absent from frame.

**Main Subject**
The shape, silhouette and appearance of core scene elements (terrain, vegetation, roads, buildings, water bodies, facilities), highlighting the most recognizable visual elements.

**Subject Material and Attachments**
Surface material texture of ground and architecture (grass / water / stone slab / sand / snow / weathered stone / glass curtain wall), pattern detail and covering state, and their relation to the overall tone.

**Props and State**
State and interaction of objects and creatures in frame (pedestrians, vehicles, birds, mist, flowing water, flickering neon) and their role as spatial scale references; if no motion, state the static spatial state.

**Environment and Background**
Foreground / midground / background three-layer hierarchy, atmospheric perspective transition and blur level, background elements, spatial depth and negative-space relation.

**Photography Style and Texture**
*   **Viewpoint:** shot type (foreground close-up / midground subject / background extension / panoramic surround), horizontal angle (front / three-quarter / profile), vertical pitch (slight high-angle / eye-level / slight low-angle / strong low-angle), depth of field (shallow foreground focus / medium midground balanced / deep full sharpness) and the visual feeling each brings.
*   **Composition:** composition type (symmetric / rule-of-thirds / central / diagonal / frame), subject position in natural wording (e.g. centered in frame / slightly upper / to one side); aspect ratio is set by workflow resolution, do not write it into the prompt unless the user explicitly asks.
*   **Lighting:** key light type (sunlight / skylight / golden hour / blue hour / neon), light direction and quality (hard / soft / diffused), light-dark hierarchy and cast shadows, and light effects on terrain, architecture and vegetation.
*   **Texture:** material, texture and surface detail; color noted as dominant tone, saturation level and cold-warm tendency (write with main color dominant, auxiliary color transitioning and accent color highlighting, but never write percentage numbers into the prompt).
*   **Atmosphere:** derive overall atmosphere and artistic conception from observable lighting, color and spatial hierarchy; avoid empty abstract adjective stacking."""
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

