# -*- coding: utf-8 -*-
"""
室内设计预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

INTERIOR_DESIGN = {
    "template_id": "interior_design",
    "name": "室内设计",
    "description": "专业室内设计指导，为全品类居住与公共空间打造标准化、高可控的空间叙事描述。语义权重优先级：空间类型与风格基调＞视角构图＞硬装软装＞灯光色彩＞人文细节。内置三维度视角、70%/25%/5%色彩配比、双重质感约束与画面精简约束，强化硬装基底、软装层次、灯光叙事与生活温度。",
}

class InteriorDesign:
    def __init__(self):
        # 下游生图模型内容组织公式库
        self.model_formula_library = {
            "Flux1": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：整体室内空间基调灯光 → 硬装软装整体布局 → 材质肌理软装细节 → 留白过渡区域。侧重居家氛围叙事，弱化细碎关键词堆砌，空间温润高级。",
                "formula_en": "Content order: overall interior tone & lighting → hard & soft decoration layout → texture and furnishing details → blank transition area. Focus on home atmosphere narration."
            },
            "Flux2_klein": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：室内空间（风格、家具、材质、配色）→ 写实空间与质感 → 窗光或暖灯、舒适氛围 → 广角透视、展现动线",
                "formula_en": "Content order: interior space (style, furniture, material, color scheme) → realistic space and texture → window light or warm lamp, comfortable atmosphere → wide-angle perspective, showing circulation"
            },
            "Z_image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：室内空间主体（风格、家具、材质、配色）→ 写实空间与质感 → 自然窗光或暖色灯光、舒适氛围 → 广角透视或角落构图、展现动线（需渲染文字直接写入，支持中英双语）。",
                "formula_en": "Content order: interior space subject (style, furniture, material, color scheme) → realistic space and texture → natural window light or warm lighting, comfortable atmosphere → wide-angle perspective or corner composition, showing circulation (write any rendered text directly, supports Chinese and English)"
            },
            "Qwen_Image2512": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：空间功能、家具与材质配色 → 风格与画质（写实室内、纹理可信） → 室内主灯与环境光平衡氛围 → 广角透视呈现空间纵深、注意画面整洁 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: space function, furniture and material color scheme → style & quality (realistic interior, credible texture) → indoor main light and ambient light balancing atmosphere → wide-angle perspective showing spatial depth, keep frame clean → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "Krea2": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：全屋整体灯光氛围基调 → 空间尺度与人居动线 → 硬装软装材质细节 → 布艺木质面料肌理 → 极简边角布景（密集关键词，中英术语并列），材质肌理高度细致，空间质感真实呈现",
                "formula_en": "Content order: whole-house lighting atmosphere tone → spatial scale and living circulation → hard and soft decoration material details → fabric and wood material texture → minimalist corner set (dense keywords, Chinese-English terms in parallel), highly detailed material texture, realistic spatial quality presentation"
            },
            "Boogu": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：全屋基础风格基调 → 舒展流畅室内动线 → 统一全屋材质质感 → 简约留白边角",
                "formula_en": "Content order: whole-house basic style tone → smooth relaxed interior circulation → unified full-house material texture → simple blank corners"
            },
            "Mage_Flow": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：硬装材质与空间透视、功能分区 → 远近家具层次 → 多层照明明暗过渡 → 软装人文细节 → 轻量化留白（密集关键词，中英术语并列）",
                "formula_en": "Content order: hard decoration material and spatial perspective, functional zoning → near-far furniture layers → multi-layer lighting transition → soft furnishing human details → lightweight blank (dense keywords, Chinese-English terms in parallel)"
            },
            "ERNIE_Image": {
                "keyword_dense": True,
                "mix_lang": True,
                "formula_zh": "内容组织顺序：空间功能、家具与材质配色 → 风格与画质（写实室内、纹理可信） → 主灯与环境光平衡氛围 → 广角透视呈现纵深、画面整洁 →（需渲染文字直接写入提示词，支持中英双语）",
                "formula_en": "Content order: space function, furniture and material color scheme → style & quality (realistic interior, credible texture) → main light and ambient light balancing atmosphere → wide-angle perspective showing depth, clean frame → (write any rendered text directly into the prompt, supports Chinese and English)"
            },
            "GLM_Image": {
                "keyword_dense": False,
                "mix_lang": False,
                "formula_zh": "内容组织顺序：室内空间与家具陈设 → 写实效果图风格、材质纹理与软装 → 室内补光、温馨氛围 → 广角透视、空间层次 → 强调布置协调无杂物、避免畸形透视。中文自然语言描述效果最佳，无负向提示词通道，负面意图正向化写入提示词。",
                "formula_en": "Content order: interior space and furniture layout → realistic render style, material texture and soft furnishing → indoor fill light, warm atmosphere → wide-angle perspective, spatial layers → emphasize coordinated arrangement without clutter, avoid distorted perspective. Best described in Chinese natural language; no negative prompt channel, write negative intent positively into prompt."
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
                "formula_zh": "内容组织顺序：室内空间与家具陈设 → 场景与构图（广角透视空间层次）→ 光影与氛围（室内补光温馨）→ 画种/摄影风格（写实效果图）→ 需渲染文字用引号包裹。",
                "formula_en": "Content order: interior space and furniture → scene & composition (wide-angle perspective, spatial layers) → light & atmosphere (indoor fill light, warm) → art/photography style (realistic render) → wrap rendered text in quotes. "
            }
        }
        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业全品类室内设计提示词扩写专家，覆盖客厅、卧室、茶室、厨房、办公等居住/公共全空间，包含奶油、新中式、日式、现代、轻奢、中古等全部家装风格。
所有创作坚守室内人居叙事基线，仅空间类型、家装风格、灯光色调、家具布局差异化，禁止混搭多种风格造成空间割裂。
空间遵循完整硬装体系（墙/地/顶），家具排布形成流畅人行动线；色彩固定70主/25辅/5点缀配比，无杂乱高饱和撞色堆砌。
灯光分层设计：基础照明、重点照明、氛围照明，色温匹配空间功能，光影过渡符合室内自然光与人造光物理逻辑。
硬装软装材质区分清晰，木质、布艺、石材、微水泥、硅藻泥等肌理贴合对应家装风格，触感与视觉表现统一。
完整保留用户输入的空间类型、家装风格、视角景别、家具、灯光色调全部信息，仅补充材质、照明、透视、人文摆件专业细节，不自动新增多余杂物、无效装饰。
画面执行严格精简约束，仅保留功能与叙事核心摆件，人物动线舒适自然。
输出禁忌：禁止权重符号、尺寸/分辨率/DPI/坐标等数值技术参数堆砌；禁止风格混乱、透视扭曲、塑料虚假材质、摆件堆砌；禁止字幕水印logo、完美对称、零瑕疵等违规描述。
严格输出两种格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional full-category interior design prompt expert, covering living room, bedroom, tea room, kitchen, office and other residential/public spaces, including cream, neo-chinese, japanese, modern, light luxury, vintage home styles.
All creations follow indoor living narrative baseline, differentiated only by space type, home style, lighting tone and furniture layout, no mixed styles causing space fragmentation.
Complete hard decoration system (wall/floor/ceiling), furniture layout forms smooth human circulation; fixed 70 main /25 secondary /5 accent color ratio, no messy oversaturated color collision.
Layered lighting design: ambient, task, accent lighting, color temperature matches space function, light transition complies with indoor natural & artificial light physical logic.
Distinct hard & soft decoration textures, wood, fabric, stone, micro-cement, diatom mud fit matching home styles, unified tactile and visual performance.
Fully retain all user input info including space type, home style, view shot, furniture, lighting tone, only supplement texture, lighting, perspective, human ornament details without redundant clutter.
Strict frame simplification rule, only keep functional & core narrative ornaments, comfortable human circulation.
Forbidden: no weight symbols, stacked numeric technical parameters such as size/resolution/DPI/coordinate; no chaotic styles, distorted perspective, fake plastic texture, piled ornaments; no subtitles watermarks logos, perfect symmetry, flawless description.
Strictly output two formats without extra comments.
"""
        }
        # 唯一主预设模板，绑定原有INTERIOR_DESIGN模板id
        self.preset_library = {
            "interior_design": {
                "template_id": "interior_design",
                "display_name": INTERIOR_DESIGN["name"],
                "description": INTERIOR_DESIGN["description"],
                # 中英双语固定前置正向约束
                "positive_constraints": {
                    "zh": "家装风格统一稳定，室内空间逻辑通顺合理，各类材质肌理真实自然，多层灯光层次分明，色彩配比合规和谐，空间透视比例精准，画面干净克制精简，仅留存核心硬装、功能家具与人文摆件；全屋硬装基底完整扎实，软装搭配协调舒适，灯光色温贴合空间功能、烘托情绪，家具排布动线流畅自然；不同家装风格保留专属材质与设计语言，兼具审美质感与生活化温度，通透自然富有呼吸感",
                    "en": "Stable unified home style, reasonable indoor spatial logic, authentic textures, distinct multi-layer lighting, compliant harmonious color ratio, precise space perspective proportion, clean restrained frame, only core hard decoration functional furniture & human ornaments retained; complete solid whole-house hard decoration, coordinated soft furnishing, lighting color temperature matches space function & sets mood, smooth circulation from furniture layout; each home style retains exclusive texture & design language, aesthetic texture and living warmth, transparent natural breathable sense"
                },
                # 全风格细分专属规则
                "preset_rules": {
                    "zh": """
【全风格室内专属细分规则】
1. 通用基线：遵循语义权重顺序：空间类型与风格基调＞视角构图＞硬装软装＞灯光色彩＞人文细节；三维人视视角优先沿用用户指定，无指定选取室内舒适合规角度；严格执行70%/25%/5%色彩配比，画面精简约束；禁用["8K", "4K", "分辨率", "DPI", "色彩模式", "尺寸", "坐标", "渲染参数", "帧率", "码率", "采样率", "编码器", "HDR", "杜比", "字幕", "水印", "logo", "完美对称", "零瑕疵", "塑料感", "崩坏", "扭曲"]。
2. 现代奶油风：微水泥、艺术涂料、原木亚麻柔和材质，无主灯漫射柔光，低饱和暖米色系，软装圆润柔和，绿植陶土小件点缀，动线开阔松弛。
3. 新中式茶室/居室：胡桃实木、微水泥、和纸材质，对称均衡布局，暖黄纸灯自然光结合，低饱和木灰主色，水墨、枯植、铜器人文摆件，东方禅意氛围。
4. 日式空间：榻榻米、硅藻泥、棉麻原木素净材质，无主灯带漫射光，低饱和米稻草色系，极简软装，书法、青苔、手工陶制小件，朴素安静。
5. 现代极简原木：实木地板、乳胶漆、棉麻布艺，主次灯分层中性自然光，干净低饱和木白配色，少量绿植装饰，家具线条利落，留白充足。
6. 轻奢风格：大理石、金属细框、丝绒软装，重点射灯+主灯高通透照明，低饱和浅灰金配色，玻璃、金属精致摆件，高级精致。
7. 中古复古：做旧实木、丝绒、复古瓷砖，暖黄复古落地台灯，暖棕复古主色调，复古画册、老式陶器、绿植，复古慵懒氛围。
所有题材：用户指定内容优先级最高，仅补充硬装、灯光、材质、人文摆件专业细节，不篡改空间类型、家装风格与核心布局。
""",
                    "en": """
【Universal Interior Exclusive Rules】
1. General baseline: Follow semantic weight order: space type & style tone > view composition > hard & soft decoration > lighting color > human details; user-specified human view takes priority, select comfortable indoor angle if unspecified; strictly implement 70%/25%/5% color ratio, frame simplification rule; forbidden words list: 8K,4K,resolution,DPI,color mode,size,coordinate,render parameter,frame rate,bit rate,sampling rate,encoder,HDR,dolby,subtitle,watermark,logo,perfect symmetry,flawless,plastic texture,collapse,distort.
2. Modern cream style: micro-cement, art coating, soft linen log texture, diffuse no-main lamp soft light, low saturation warm cream tone, round soft furnishing, green plant clay ornaments, open relaxed circulation.
3. Neo-Chinese tea room/living space: walnut solid wood, micro-cement, washi texture, symmetrical layout, warm paper lamp plus natural light, low saturation wood-gray main color, ink painting withered plant bronze ornaments, oriental zen atmosphere.
4. Japanese space: tatami, diatom mud, linen log plain texture, no-main lamp strip diffuse light, low saturation rice straw tone, minimalist furnishing, calligraphy moss handmade pottery, plain quiet vibe.
5. Modern minimalist log: solid wood floor, latex paint, cotton linen fabric, neutral natural light with main & auxiliary lamp layers, clean low saturation wood-white color, few green plants, neat furniture lines, ample blank space.
6. Light luxury style: marble, thin metal frame, velvet furnishing, spot task lamp + main high-transparency lighting, low saturation light gray gold color, glass metal delicate ornaments, advanced exquisite sense.
7. Vintage mid-century: aged solid wood, velvet, retro tile, warm vintage floor lamp, warm brown retro main tone, old albums vintage pottery green plants, lazy retro atmosphere.
All themes: User-specified content highest priority, only supplement hard decoration, lighting, texture, human ornament details without altering space type, home style and core layout.
"""
                },
                "negative_base": {
                    "zh": "家装风格混乱跳变，材质虚假塑料质感，灯光刺眼曝光失衡，色彩脏污溢出，构图失衡杂乱，家具摆件冗余堆砌，室内透视逻辑错误，空间比例变形，画面低分辨率模糊，多余杂物乱入，装饰摆件杂乱堆砌，字幕水印logo，光影生硬断层，空间闭塞压抑，家具动线拥堵混乱，过度锐化生硬，家具边缘锯齿比例失调，墙地拼接错误，装饰毫无章法，廉价网红质感",
                    "en": "Chaotic mixed home styles, fake plastic texture, dazzling overexposed lighting, muddy overflowing color, unbalanced cluttered composition, redundant piled furniture ornaments, wrong indoor perspective logic, distorted space proportion, blurry low-res frame, irrelevant clutter, stacked messy decorations, subtitles watermarks logos, stiff disjointed lighting, cramped closed space, blocked messy furniture circulation, over-sharpened rigid texture, jagged furniture edges disproportionate furniture, wrong wall-floor splicing, disordered decorations, cheap internet celebrity texture"
                }
            }
        }
        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】2-3段连贯文字：第一段全屋空间格局与整体家装风格；第二段硬装墙地顶材质与分层灯光设计；第三段软装家具排布、人文摆件与整体色彩意境；总字数300-600字，全程规避尺寸、坐标等数值参数，语言居家叙事富有画面感，无额外解释。",
                "en": "[Natural Paragraph Mode] 2-3 coherent paragraphs: whole space layout & home style; wall/floor/ceiling hard texture & layered lighting; furniture layout human ornaments & overall color artistic conception; 300-600 words, avoid size/coordinate numeric parameters, home narrative visual language without extra explanation."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
1.空间类型与整体格局
2.风格与家装定位
3.硬装墙地顶材质
4.三维度镜头视角与构图
   - 画面比例：竖版空间图（4:5/3:4）/ 横版全景图（16:9/3:2）/ 方形（1:1）
   - 距离维度（景别）：人视全景 / 局部特写 / 俯瞰 / 细节微距，对应空间叙事重心
   - 水平视角维度：正面 / 四分之三斜侧 / 正侧面，标注空间展现效果
   - 垂直俯仰维度：小俯视角 / 平视 / 小仰视角 / 强仰视角（挑高展示），对应空间张力
   - 景深氛围：浅景深局部突出 / 中景深空间兼顾 / 深景深全景清晰，标注虚实层次
5.软装搭配与人文摆件
   - 家具排布：位置关系/高低层次/疏密节奏
   - 布艺配饰：窗帘/地毯/抱枕/挂画
   - 人文摆件：花瓶/书籍/装饰品/绿植
   - 色彩分层：主色70% / 辅助色25% / 点缀色5%
6.分层灯光设计
   - 主照明：吊灯/筒灯/射灯
   - 辅助照明：灯带/壁灯/落地灯
   - 氛围照明：台灯/蜡烛/装饰灯
7.整体风格与色彩意境
   - 家装风格定位：现代/北欧/中式/日式/法式/美式
   - 色彩意境：冷暖调性/饱和度/色彩搭配逻辑
   - 光影特效：温馨氛围光/层次感光影/光影斑驳/动态光斑/边缘发光/柔化光晕/明暗渐变过渡
8.画面品质与居住感
   - 写实程度：照片级/渲染级/手绘感
   - 质感细节：材质肌理/光影层次/空间纵深
   - 整体居住感：温馨/舒适/高级/治愈
9.画面精简约束
10.【技术参数建议】仅structured模式可输出，natural模式禁用；允许定性描述镜头空间效果，禁用数值参数：
- 人视全景：24mm-35mm广角，f/8-f/11光圈，深景深全景清晰，展现空间格局
- 局部特写：85mm中长焦，f/2.8-f/4光圈，突出软装细节与材质肌理
- 俯瞰全景：鱼眼/超广角，f/8-f/11光圈，展现平面布局与动线走向
- 氛围灯光：35mm-50mm，f/1.4-f/2大光圈，捕捉灯光氛围与光影层次""",
                "en": """[Structured Mode] Output strictly in this order:
1. Space type and overall layout
2. Style and home furnishing positioning
3. Hard material wall/floor/ceiling
4. Three-dimensional camera view and composition
   - Aspect ratio: vertical space (4:5/3:4) / horizontal panorama (16:9/3:2) / square (1:1)
   - Distance (shot type): eye-level panorama /局部特写 / overhead / detail macro, mark narrative focus
   - Horizontal view: front / three-quarter / profile, describe space display effect
   - Vertical pitch: slight high-angle / eye-level / slight low-angle / strong low-angle (ceiling display), describe space tension
   - Depth of field: shallow DOF detail focus / medium DOF space balanced / deep DOF full sharpness
5. Soft furnishing and human ornaments
   - Furniture layout: position relationship/height layers/density rhythm
   - Fabric accessories: curtain/carpet/cushion/painting
   - Human ornaments: vase/books/decorations/greenery
   - Color layering: 70% main / 25% auxiliary / 5% accent
6. Layered lighting design
   - Main lighting: pendant/recessed/spotlight
   - Auxiliary lighting: light strip/wall lamp/floor lamp
   - Ambient lighting: table lamp/candle/decorative light
7. Overall style and color artistic conception
   - Home style positioning: modern/Scandinavian/Chinese/Japanese/French/American
   - Color artistic conception: warm-cool tone/saturation/color matching logic
   - Lighting effects: warm ambient light / layered lighting / dappled light / dynamic light spots / rim glow / soft haze / gradient transition
8. Image quality and living feel
   - Realism level: photo-real / rendering / hand-drawn feel
   - Texture details: material texture / light layers / spatial depth
   - Overall living feel: warm/comfortable/high-end/healing
9. Frame simplification constraint
10. [Tech params] Only structured mode can output, natural mode forbidden; only qualitative description of lens spatial effects, no numeric parameters:
- Eye-level panorama: 24mm-35mm wide-angle, f/8-f/11 aperture, deep DOF full sharpness, showcasing spatial layout
- Detail close-up: 85mm medium telephoto, f/2.8-f/4 aperture, highlighting soft furnishing details and material texture
- Overhead panorama: fisheye/super wide-angle, f/8-f/11 aperture, showcasing floor plan and movement flow
- Ambient lighting: 35mm-50mm, f/1.4-f/2 large aperture, capturing lighting atmosphere and light layers"""
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
