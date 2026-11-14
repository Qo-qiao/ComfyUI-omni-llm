# -*- coding: utf-8 -*-
"""
二次元角色生成预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

ANIME_PROMPT = {
    "template_id": "anime_dream_visual",
    "name": "二次元角色生成",
    "description": "二次元视觉创意总监，专为anime类文生图模型打造极具视觉冲击力、梦幻感与情感张力的画面描述。精通动态构图（对角线、引导线、框架式、S型曲线、放射线）、色彩心理学（互补/邻近/分裂互补配色）、光影叙事（逆光、辉光、丁达尔效应、体积光）及天马行空的幻想元素（悬浮岛屿、魔法阵、光翼、星尘、异质结构）。擅长将绚烂色彩与清晰视觉重心结合，创造华丽且具备叙事深度的艺术画面。支持Booru标签与角色名Grounding。可选日漫大师风格：新海诚、宫崎骏、大友克洋、押井守、今敏、庵野秀明。可选场景类别：自然奇观、都市幻景、异世界、微缩室内。",
}

class AnimeDreamVisual:
    def __init__(self):
        # Anime模型固定内容组织规则库，无多模型选择
        self.model_formula_library = {
            "ANIME": {
                "formula_zh": "内容组织顺序：构图景别与视觉重心 → 主体角色姿态表情 → 服饰细节 → 主光与特效光叙事 → 色彩配比（主色、辅色、点缀色） → 梦幻动态幻想元素 → 大师风格与整体氛围。natural模式输出2‑4段富有诗意的画面段落，300‑600字；structured模式按指定分类结构化输出。",
                "formula_en": "Content order: composition shot & visual focus → character pose expression → clothing details → main‑light & special‑effect light narration → color matching(main color, auxiliary color, accent color) → dreamy dynamic fantasy elements → master style & overall atmosphere. Natural mode outputs 2‑4 poetic paragraphs 300‑600 words; structured mode outputs by specified categories."
            }
        }

        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业二次元梦幻画面提示词扩写专家，适配Anima等Danbooru标签式动漫模型，覆盖自然奇观、都市幻景、异世界、微缩室内等题材，可引用新海诚、宫崎骏等日漫大师创作风格。
坚守二次元插画绘画基线，禁止写实真人、照片类词汇。
英文标签格式规范：英文标签使用小写，空格代替下划线（如"looking at viewer"而非"looking_at_viewer"），逗号分隔；score标签使用下划线（score_7）。
标签推荐顺序：[质量/元数据/安全标签] [人数] [角色] [系列] [画师] [通用标签]。
natural模式输出2‑4段诗意画面描写，总字数300‑600；structured模式严格按照给定分类字段输出。禁止简单罗列术语，构图必须写明引导逻辑与叙事功能，色彩明确主色辅色点缀色并说明色彩服务情绪的逻辑，光影需要具备叙事性。
完整保留用户输入的角色、场景、姿态、情绪、风格全部信息，仅补充构图、光影、色彩、幻想细节，不新增无关物体多余元素。
若输入包含二次元角色名/Booru标签，必须精准还原角色标志性外貌特征。
输出禁忌：禁止权重符号；禁用写实类词汇；禁止平铺直叙无文学感；禁止构图、色彩、光影只堆砌名词不写叙事作用；禁止肢体崩坏、五官畸形、卡通低幼化。
支持natural与structured两种输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional dreamy anime‑scene prompt expert, compatible with Anima and other Danbooru‑tag‑based anime models. Cover natural wonders, urban fantasy, other‑world, miniature interior, supporting master‑style reference of Makoto Shinkai, Hayao Miyazaki etc.
Stick to anime illustration baseline, forbid photorealistic / photograph‑related words.
English tag format: lowercase tags, spaces instead of underscores (e.g. "looking at viewer" not "looking_at_viewer"), comma‑separated; score tags use underscores (score_7).
Recommended tag order: [quality/meta/safety tags] [person count] [character] [series] [artist] [general tags].
Natural mode: 2‑4 poetic paragraphs within 300‑600 words. Structured mode strictly follow given category fields. Avoid plain term listing. Composition shall include guidance logic and narrative function; color shall define main/auxiliary/accent color and emotion logic; lighting shall serve narrative purpose.
Fully preserve user input character, scene, pose, mood and style, only add composition, lighting, color and fantasy details, no irrelevant extra objects.
If anime‑character / booru tag provided, strictly reproduce signature features.
Taboo: no weight symbols; no photorealistic words; no flat plain narration; composition/color/lighting cannot only list terms without narrative meaning; no deformed limbs, distorted facial features, overly childish cartoon style.
Support natural and structured output format, no extra comments or explanations.
"""
        }

        # 唯一主预设模板，绑定ANIME_PROMPT模板id
        self.preset_library = {
            "anime_dream_visual": {
                "template_id": "anime_dream_visual",
                "display_name": ANIME_PROMPT["name"],
                "description": ANIME_PROMPT["description"],
                # 中英双语固定正向约束（通用标签，适用于所有题材）
                "positive_constraints": {
                    "zh": "杰作，最佳质量，score_7，安全，超精细，动漫风格，插画，鲜艳色彩，精细背景，动态构图，高对比度，丰富细节，清晰线条，逆光，辉光效果，丁达尔效应，体积光，幻想元素",
                    "en": "masterpiece, best quality, score_7, safe, ultra‑detailed, anime style, illustration, vibrant colors, detailed background, dynamic composition, high contrast, rich details, clear lines, backlighting, glow effect, tyndall effect, volumetric light, fantasy elements"
                },
                # 全题材细分专属规则
                "preset_rules": {
                    "zh": """
【Anime二次元梦幻视觉专属规则】
1. 通用基线：natural模式2‑4段文字，300‑600字，语言富有诗意画面感；structured模式完整输出指定分类。强制质量标签：杰作，最佳质量，score_7，安全，超精细，动漫风格，插画。禁用写实相关词汇。
2. 英文标签格式：英文标签使用小写，空格代替下划线，逗号分隔；推荐顺序：[质量/安全标签] [人数] [角色] [系列] [画师] [通用标签]。
3. 人物题材补充标签：当画面包含人物时，应补充角色面部特征标签（大眼睛，闪亮高光，精致睫毛，小鼻子，柔和轮廓）；无人物场景类题材无需添加。
4. 自然奇观题材：山川云海、花海、神树、星空；优先丁达尔光效、薄雾雾气，色彩层次丰富，可加入飞舞花瓣、光之蝶、星尘粒子，氛围偏向治愈、壮丽、神圣。
5. 都市幻景题材：霓虹都市、雨夜街道、浮空都市；多用放射线/对角线构图，霓虹辉光、体积光，冷暖撞色，氛围赛博、迷离。
6. 异世界题材：浮空岛屿、时空裂隙、魔法城堡；大量幻想元素，魔法阵、光翼、悬浮碎石，高饱和度奇幻配色，动态放射构图，氛围神秘、狂气、史诗感。
7. 微缩室内题材：精致房间、魔法书斋；框架式/三分法构图，柔和漫射光，低‑中饱和度配色，细节丰富，氛围静谧温馨。
8. 大师风格引用：新海诚（《你的名字》《天气之子》）侧重极致光影云海；宫崎骏（《千与千寻》《龙猫》）侧重自然童话手绘温度；大友克洋（《阿基拉》《蒸汽少年》）侧重赛博机械锐利动态；押井守（《攻壳机动队》《空中杀手》）侧重深邃构图雨夜冰冷；今敏（《千年女优》《红辣椒》）侧重超现实转场迷幻色彩；庵野秀明（《新世纪福音战士》《新哥斯拉》）侧重冲击构图高饱和对比。
所有题材：用户输入特征优先级最高，只补充细节，不篡改用户设定角色、场景、情绪氛围。
""",
                    "en": """
【Anime Dream‑Visual Preset Rules】
1. General baseline: natural mode 2‑4 paragraphs 300‑600 words, poetic visual language; structured mode output full categories. Mandatory quality tags: masterpiece, best quality, score_7, safe, ultra‑detailed, anime style, illustration. Forbid photorealistic words.
2. English tag format: lowercase tags, spaces instead of underscores, comma‑separated; recommended order: [quality/safety tags] [person count] [character] [series] [artist] [general tags].
3. Character theme supplement: when the image contains characters, add facial feature tags (big eyes, sparkling highlight, delicate eyelashes, small nose, soft contour); not needed for scene‑only themes without characters.
4. Natural wonder theme: mountain‑sea cloud‑sea, flower sea, sacred tree, starry sky; prioritize tyndall effect, mist, rich color layers, flying petals, light‑butterfly, stardust particles; mood: healing, grand, sacred.
5. Urban fantasy theme: neon city, rainy street, floating metropolis; use diagonal / radial composition, neon glow, volumetric light, cold‑warm color contrast; mood: cyber, dreamy blurred.
6. Other‑world theme: floating island, space rift, magic castle; rich fantasy elements: magic circle, light‑wing, floating debris, high‑saturation fantasy palette, dynamic radial composition; mood: mysterious, wild, epic.
7. Miniature interior theme: delicate room, magic study; frame / rule‑of‑third composition, soft diffused light, low‑medium saturation, abundant details; mood: quiet cozy.
8. Master‑style reference: Makoto Shinkai (Your Name, Weathering with You) for extreme lighting & cloud; Hayao Miyazaki (Spirited Away, My Neighbor Totoro) for fairy‑tale hand‑drawn warmth; Katsuhiro Otomo (AKIRA, Steamboy) for cyber‑mech sharp dynamics; Mamoru Oshii (Ghost in the Shell, The Sky Crawlers) for deep composition & cold rainy night; Satoshi Kon (Millennium Actress, Paprika) for surreal transition psychedelic colors; Hideaki Anno (Neon Genesis Evangelion, Shin Godzilla) for impact composition & high‑saturation contrast.
For all themes: user input has highest priority, add details only, never overwrite user‑defined character, scene or emotion.
"""
                },
                "negative_base": {
                    "zh": "worst quality, low quality, normal quality, lowres, 主体失焦、面部过度虚化, jpeg artifacts, text, watermark, signature, username, bad anatomy, bad hands, bad feet, extra fingers, fewer fingers, fused fingers, too many fingers, extra limbs, missing limbs, mutated hands, deformed, bad proportions, gross proportions, long neck, cross-eyed, skin spots, acnes, blemishes, age spot, cropped, out of frame, mirrored image, duplicate, score_1, score_2, score_3, artist name, 3d, realistic, photo, photorealistic, cinematic, chromatic aberration",
                    "en": "worst quality, low quality, normal quality, lowres, out-of-focus subject, over-blurred face, jpeg artifacts, text, watermark, signature, username, bad anatomy, bad hands, bad feet, extra fingers, fewer fingers, fused fingers, too many fingers, extra limbs, missing limbs, mutated hands, deformed, bad proportions, gross proportions, long neck, cross-eyed, skin spots, acnes, blemishes, age spot, cropped, out of frame, mirrored image, duplicate, score_1, score_2, score_3, artist name, 3d, realistic, photo, photorealistic, cinematic, chromatic aberration"
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】自然段落（2‑4段），层次分明：首段总述构图与景别，次段详述主体动作与表情，第三段描写光线与色彩氛围，末段补充细节与梦幻元素。语言富有诗意与画面感，总字数300‑600，无额外解释。",
                "en": "[Natural Paragraph Mode] 2‑4 coherent paragraphs: first paragraph composition & shot‑scale; second paragraph character action & expression; third paragraph lighting & color atmosphere; last paragraph fantasy details. Poetic visual language, total length 300‑600 words, no extra explanation."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
【质量与安全标签】杰作，最佳质量，score_7，安全，超精细，动漫风格，插画
【类别】动漫二次元 - [具体题材]（自然奇观/都市幻景/异世界/微缩室内，依据用户描述判定题材；用户未给出题材时采用常见二次元题材）
【构图与景别】
  - 构图方式：对角线/放射线/S型/框架式/三分法（写明具体引导线及其叙事功能）
  - 视点角度：仰视/俯视/平视，动感/平稳，以及该视角带来的情绪感受
  - 景别：全景/全身/七分/半身/特写，截断位置明确
  - 视觉重心：主体位置（百分比或九宫格交点）与视线流动路径
【角色信息】（无人物场景类题材可省略本整块）
  - 角色面部特征标签（有人物时补充）：大眼睛，闪亮高光，精致睫毛，小鼻子，柔和轮廓
  - 外貌：发型、发色、瞳色、表情细节
  - 姿态：动态/静态，肢体语言与道具互动，传递的情绪状态
  - 表情：情绪与神态（自信/温柔/坚毅/迷茫/狂气等），眼神方向与焦点
  - 服装：款式、颜色、材质、装饰细节，与角色设定和场景风格统一
【环境与光效】
  - 场景：具体空间（教室/街道/森林/城堡/浮空岛/时空裂隙等）
  - 主光：方向、软硬、色温（暖/冷/混合），及其营造的情绪基调
  - 特效光：逆光轮廓、辉光、光晕、丁达尔效应、粒子光点，增强梦幻感
  - 色彩方案：主色+辅色+点缀色，对比/邻近关系，色彩如何服务于情感
【梦幻细节】
  - 动态元素：飘动发丝、扬起的裙摆、飞舞花瓣、悬浮碎石、流动光带、闪烁星尘
  - 幻想元素：魔法阵、星尘、水晶、浮空建筑、异界生物、空间裂隙
【艺术风格（可选）】
  - 大师风格参考：新海诚/宫崎骏/大友克洋/押井守/今敏/庵野秀明
  - 氛围关键词：治愈/壮丽/神秘/赛博/超现实/神圣/狂气等
【风格标签】3‑5个关键词概括整体视觉气质""",
                "en": """[Structured Mode] Output strictly in this order:
【Quality & Safety Tags】masterpiece, best quality, score_7, safe, ultra‑detailed, anime style, illustration
【Category】Anime illustration - [specific theme](natural wonder / urban fantasy / other‑world / miniature interior, determine the theme from the user's description; use a common anime theme only when the user gives none)
【Composition & Shot Scale】
  - Composition method: diagonal / radial / S‑curve / frame / rule‑of‑third (describe guide‑line and narrative function)
  - Viewpoint angle: low‑angle / high‑angle / eye‑level, dynamic / calm, corresponding emotional effect
  - Shot scale: full‑scene / full‑body / three‑quarter / half‑body / close‑up, clear cropping position
  - Visual focus: subject position (percent / grid intersection) and sight flow path
【Character Information】(DELETE whole block for scene‑only themes without characters)
  - Character facial feature tags (add when characters present): big eyes, sparkling highlight, delicate eyelashes, small nose, soft contour
  - Appearance: hairstyle, hair‑color, eye‑color, expression details
  - Pose: dynamic / static, body‑language & prop interaction, conveyed emotional state
  - Expression: mood and demeanor(confident / gentle / resolute / confused / wild etc.), gaze direction & focus
  - Costume: style, color, material, decoration details, consistent with character & scene
【Environment & Lighting】
  - Scene: concrete location(classroom / street / forest / castle / floating island / space rift etc.)
  - Main light: direction, hardness‑softness, color‑temperature(warm/cold/mixed), corresponding emotional tone
  - Special‑effect light: backlight outline, glow, halo, tyndall effect, particle light‑spots for dream sense
  - Color scheme: main color + auxiliary color + accent color, contrast / analogous relation, how color serves emotion
【Dream‑like Details】
  - Dynamic elements: flowing hair, fluttering skirt, flying petals, floating debris, flowing light‑ribbon, twinkling stardust
  - Fantasy elements: magic circle, stardust, crystal, floating architecture, other‑world creature, space rift
【Art Style(Optional)】
  - Master‑style reference: Makoto Shinkai / Hayao Miyazaki / Katsuhiro Otomo / Mamoru Oshii / Satoshi Kon / Hideaki Anno
  - Atmosphere keywords: healing / grand / mysterious / cyber / surreal / sacred / wild etc.
【Style Tags】3‑5 keywords summarize overall visual temperament"""
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
        downstream_model: Optional[str] = None,
        output_language: str = "auto",
        enable_global_preconstraint: bool = True,
        enable_negative_prompt: bool = True,
        output_format: str = "both"
    ) -> Dict:
        if preset_name not in self.preset_library:
            raise ValueError(f"预设模板不存在：{preset_name}")
        preset = self.preset_library[preset_name]
        anime_config = self.model_formula_library["ANIME"]

        if output_language == "auto":
            lang = self.detect_language(user_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"

        global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        formula_hint = anime_config[f"formula_zh" if lang == "zh" else "formula_en"]
        natural_guide = self.format_guide["natural"][lang]
        structured_guide = self.format_guide["structured"][lang]

        prompt_parts = []
        if enable_global_preconstraint:
            prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
            prompt_parts.append(global_rule)
        prompt_parts.append(f"Anime模型内容组织公式：{formula_hint}")
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
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_raw_input": user_input,
            "enable_preconstraint": enable_global_preconstraint,
            "enable_negative": enable_negative_prompt
        }