# -*- coding: utf-8 -*-
"""
二次元角色标签生成预设模块


Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

# 模板元数据常量（绑定预设库，避免类实例化时未定义引用）
ILLUSTRIOUS = {
    "template_id": "illustrious",
    "name": "二次元Danbooru标签扩写",
    "description": "专业的二次元角色设计总监，专为SDXL模型生成高密度、精确的Danbooru风格标签。精通日系动漫角色塑造，涵盖发型、瞳色、五官细节、动态姿势、丰富表情、类型化服饰以及环境氛围。严格遵循标签规范，确保无重复、无冗余，充分发挥SDXL模型对二次元特征的精确还原能力。",
}

class Illustrious:
    def __init__(self):
        # SDXL固定内容组织规则库（仅SDXL，无多模型选择）
        self.model_formula_library = {
            "SDXL": {
                "formula_zh": "内容组织顺序：角色基础属性 → 发型瞳色面部细节 → 姿势肢体动作 → 表情神态眼部嘴部细节 → 服饰配件道具 → 场景环境光线氛围 → 画风质量标签。输出Danbooru风格英文短标签（标签为英文Booru词汇，中文仅用于解释），逗号分隔；natural模式30-40个英文标签，structured模式不适用标签数量约束；拒绝长句描述。",
                "formula_en": "Content order: character basic attributes → hair, eye and facial details → pose and limb movement → expression, eye and mouth details → clothing accessories props → scene environment lighting atmosphere → art style quality tags. Output Danbooru‑style short English tags separated by commas; natural mode 30-40 English tags, structured mode not subject to a tag-count limit; no long sentences."
            }
        }

        # 全局底层规则
        self.global_base_rules = {
            "zh": """
你是专业二次元Danbooru风格标签扩写专家，适配日系动漫全题材，包含校园、奇幻魔法、和风、都市日常、战斗、海滩夏日等题材。
坚守二次元动漫绘画基线，禁止写实、照片、真人相关词汇输出。
标签为短单元词汇，一个标签仅代表一个视觉语义，禁止长句、自然语言段落描述；禁止空泛形容词，全部使用具象视觉标签。
完整保留用户输入的角色性别、发色、瞳色、发型、服饰、道具、场景、姿态、表情、视角全部信息，仅补充符合Danbooru规范的细节标签，不新增无关物体与多余元素。
若输入包含知名二次元角色名，必须严格匹配该角色标志性外貌、饰品特征。
输出禁忌：禁止权重符号，禁止写实类词汇，禁止重复标签、同义标签堆砌，禁止模式化“XX风格”前缀大量重复；禁止肢体崩坏、五官畸形、低幼卡通化。
支持natural与structured两种输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are a professional Danbooru‑style anime tag expansion expert. Cover campus, fantasy magic, japanese style, urban daily, battle, summer beach and other themes.
Stick to anime drawing baseline, prohibit real‑photo, photorealistic and human‑related words.
Each tag is short unit word, one visual meaning per tag, no long sentences or natural‑language paragraphs; avoid vague adjectives, use concrete visual tags only.
Fully keep user input including gender, hair color, eye color, hairstyle, clothing, props, scene, pose, expression, perspective. Only supplement Danbooru‑compliant detail tags, do not add irrelevant objects.
If famous anime character name is given, strictly follow its signature appearance and accessories.
Taboo: no weight symbols, no photorealistic words, no duplicate or synonym tags, no mass repetitive "XX style" prefixes; no deformed limbs or facial features, no overly childish cartoon style.
Support natural and structured output format, no extra comments or explanations.
"""
        }

        # 唯一主预设模板，绑定原有ILLUSTRIOUS模板id
        self.preset_library = {
            "illustrious": {
                "template_id": "illustrious",
                "display_name": ILLUSTRIOUS["name"],
                "description": ILLUSTRIOUS["description"],
                # 中英双语固定正向约束
                "positive_constraints": {
                    "zh": "杰作，最佳质量，高分辨率，超精细，动漫风格，干净线稿，高清，8K，漂亮细节眼睛，详细面部，动态姿势，生动表情，详细服饰，详细背景，动态光照，鲜艳色彩",
                    "en": "masterpiece, best quality, high resolution, ultra‑detailed, anime style, clean line art, high‑definition, 8K, beautifully detailed eyes, detailed face, dynamic pose, vivid expression, detailed clothing, detailed background, dynamic lighting, vibrant colors"
                },
                # 全题材细分专属规则
                "preset_rules": {
                    "zh": """
【SDXL二次元标签专属规则】
1. 标签基线：natural模式30-40个英文标签，structured模式不适用标签数量约束；全部为英文Booru短标签（中文仅用于解释），逗号分隔；禁止重复标签（完全相同的标签）与同义标签堆砌（如"happy"与"joyful"不得同时出现）；每个标签仅单一语义单元，使用Danbooru高频视觉短词。强制必带画风标签：动漫风格，干净线稿。
2. 校园题材：适配校服、水手服、百褶裙、教室校园场景；多用柔和窗光、室内自然光，色调温暖，突出青春角色神态。
3. 奇幻魔法题材：适配魔法少女、战斗服饰、法杖宝石、披风；增加魔法光晕、粒子光效、动态线条，可搭配废墟、奇幻城堡场景。
4. 和风题材：适配和服、发簪发饰、油纸伞；庭院、樱花、灯笼布景，可搭配花瓣飘落，柔和暖调光线。
5. 都市日常题材：适配风衣、卫衣、休闲便服；街道、城市建筑、霓虹光影，可使用阴天冷色调，刻画清冷或松弛神态。
6. 夏日海滩题材：适配泳装、草帽；大海沙滩、蓝天阳光，水花飞溅，高饱和暖色调，活力表情。
7. 战斗题材：战斗姿态、飘动衣料、武器道具，动态角度，动感光影，强化张力。
所有题材：用户输入特征优先级最高，只补充细节标签，不得篡改用户设定的角色特征、场景、氛围。
""",
                    "en": """
【SDXL Anime Tag Preset Rules】
1. Tag baseline: natural mode 30-40 English tags, structured mode not subject to a tag‑count limit, all short English tags separated by commas; no duplicate tags (identical tags) or synonym‑tag padding (e.g. do NOT output both "happy" and "joyful"); each tag carries only one visual meaning, use high‑frequency Danbooru short words. Mandatory style tags: anime style, clean line art.
2. Campus theme: school uniform, sailor suit, pleated skirt, classroom campus scene; soft window light, indoor natural light, warm tone, youthful expression.
3. Fantasy magic theme: magical girl, battle outfit, staff and gem, cloak; magic glow, particle effects, dynamic lines, ruins or fantasy castle background.
4. Japanese style theme: kimono, hairpin, oil‑paper umbrella; courtyard, cherry blossom, lantern, falling petals, soft warm light.
5. Urban daily theme: windbreaker, hoodie, casual wear; street, city building, neon light, overcast cool tone, cold or relaxed expression.
6. Summer beach theme: swimsuit, straw hat; sea beach, blue sky, sunlight, splashing water, high‑saturation warm tone, energetic facial expression.
7. Battle theme: combat pose, fluttering fabric, weapon props, dynamic camera angle, dynamic light‑shadow, high visual tension.
For all themes: user input features have highest priority, only add detail tags, never overwrite user‑defined character, scene or atmosphere.
"""
                },
                "negative_base": {
                    "zh": "worst quality, low quality, normal quality, lowres, blurry, jpeg artifacts, text, watermark, signature, username, bad anatomy, bad hands, bad feet, extra fingers, fewer fingers, fused fingers, too many fingers, extra limbs, missing limbs, mutated hands, deformed, bad proportions, gross proportions, long neck, cross-eyed, skin spots, acnes, blemishes, age spot, cropped, out of frame, mirrored image, duplicate",
                    "en": "worst quality, low quality, normal quality, lowres, blurry, jpeg artifacts, text, watermark, signature, username, bad anatomy, bad hands, bad feet, extra fingers, fewer fingers, fused fingers, too many fingers, extra limbs, missing limbs, mutated hands, deformed, bad proportions, gross proportions, long neck, cross-eyed, skin spots, acnes, blemishes, age spot, cropped, out of frame, mirrored image, duplicate"
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然标签模式】仅输出逗号分隔英文标签串，30-40个英文标签，只输出标签本体，不包含任何解释、标题、填充文字。",
                "en": "[Natural Tag Mode] Output only comma‑separated English tag string, 30-40 English tags, tags only, no explanation, title or extra text."
            },
            "structured": {
                "zh": """【结构化模式】严格顺序输出：
【基础属性】
  - 视角：视点描述（平视/俯视/仰视，正面/侧视图/背后）
  - 主体数量与性别：1个女孩/2个男孩/多人等，单独或成组
【角色特征】
  - 发型发色：长发/短发/双马尾/单马尾/包子头/卷发，黑发/金发/银发/粉发/蓝发/紫发等
  - 瞳色：蓝色/红色/绿色/紫色/金色/琥珀色/异色瞳等
  - 面部细节：特定特征（如雀斑、虎牙、泪痣、伤疤）
【姿势与动作】
  - 站姿/坐姿/蹲姿/跪坐/侧躺/趴姿等
  - 手臂：自然下垂/双手叉腰/抱臂/背手/手托腮/抬手/指向等
  - 腿部：双腿并拢/分开/交叉/前伸/弯曲等
  - 道具互动：手持物品/触碰到物体等
【表情与神态】
  - 基础表情：微笑/大笑/无表情/严肃/悲伤/哭泣/惊讶/张嘴/闭嘴/抿嘴
  - 情绪增强：生气/害羞/困惑/轻蔑/自信/慵懒
  - 眼部细节：闭眼/眯眼/瞪眼/半睁眼/星星眼/瞳孔高光等
  - 嘴部：微张嘴/张大嘴/露齿笑/吐舌/嘟嘴
【服饰与配件】
  - 上装：水手服/衬衫/夹克/毛衣/背心/T恤等，颜色与花纹（格纹/条纹）
  - 下装：百褶裙/短裤/长裤/紧身裤/运动裤/连衣裙/和服/旗袍等
  - 外套/罩衫：大衣/开衫/斗篷/披风等
  - 鞋袜：及膝袜/过膝袜/船袜/丝袜/皮鞋/运动鞋/靴子/木屐等
  - 头饰：发带/发箍/发夹/蝴蝶结/头纱/皇冠/帽子等
  - 首饰：项链/耳环/手镯/戒指等
  - 道具：武器/乐器/书/伞/手机/食物等
【环境与氛围】
  - 场景：室内/室外，具体地点（教室/卧室/街道/森林/海滩/神社/城堡等）
  - 天色：白天/夜晚/黄昏/黎明
  - 气象：晴天/阴天/雨天/雪天/雾天
  - 光线：自然光/人工光/逆光/侧光/顶光/点光源/柔和/硬朗
  - 色彩基调：暖色/冷色/高饱和/低饱和度/单色/渐变
  - 特效元素：花瓣/樱花/雪花/落叶/萤火虫/光晕/魔法光效/粒子
【艺术风格】
  - 固定：动漫风格，干净线稿""",
                "en": """[Structured Mode] Output strictly in this order:
【Basic Attributes】
  - Viewpoint: eye‑level / high‑angle / low‑angle, front view / side view / back view
  - Subject count & gender: 1girl / 2boys / multiple characters, solo or group
【Character Features】
  - Hair style & color: long hair / short hair / twin tails / single ponytail / bun / curly hair, black / blonde / silver / pink / blue / purple hair etc.
  - Eye color: blue / red / green / purple / gold / amber / heterochromia etc.
  - Facial detail: freckles, fang, tear mole, scar etc.
【Pose & Movement】
  - Standing / sitting / squatting / kneeling / lying on side / prone pose etc.
  - Arm: arms down / hands on hips / crossed arms / hands behind back / hand on cheek / raise hand / point etc.
  - Leg: legs together / apart / crossed / stretch forward / bent etc.
  - Prop interaction: hold item / touch object etc.
【Expression & Demeanor】
  - Basic expression: smile / big laugh / expressionless / serious / sad / cry / surprised / open mouth / closed mouth / pressed lips
  - Mood enhancement: angry / shy / confused / scornful / confident / lazy
  - Eye detail: closed eyes / squint / glare / half‑closed eyes / star eyes / pupil highlight etc.
  - Mouth: slightly open / wide open / grin / tongue out / pout
【Clothing & Accessories】
  - Top: sailor uniform / shirt / jacket / sweater / vest / T‑shirt etc., color & pattern (plaid / stripe)
  - Bottom: pleated skirt / shorts / long pants / leggings / sweatpants / dress / kimono / cheongsam etc.
  - Outerwear: coat / cardigan / cloak / cape etc.
  - Socks & shoes: knee‑high socks / thigh‑high stockings / no‑show socks / silk stockings / leather shoes / sneakers / boots / geta etc.
  - Headwear: hairband / headband / hair clip / bow / veil / crown / hat etc.
  - Jewelry: necklace / earring / bracelet / ring etc.
  - Props: weapon / instrument / book / umbrella / phone / food etc.
【Environment & Atmosphere】
  - Scene: indoor / outdoor, exact location(classroom / bedroom / street / forest / beach / shrine / castle etc.)
  - Sky condition: day / night / dusk / dawn
  - Weather: sunny / overcast / rainy / snowy / foggy
  - Lighting: natural light / artificial light / backlight / side light / top light / point light source / soft / harsh
  - Color tone: warm / cold / high saturation / low saturation / monochrome / gradient
  - Special effect: petal / cherry blossom / snowflake / falling leaf / firefly / glow / magic effect / particle
【Art Style】
  - Mandatory: anime style, clean line art"""
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
        sdxl_config = self.model_formula_library["SDXL"]
        if output_language == "auto":
            lang = self.detect_language(user_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"
        global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        formula_hint = sdxl_config[f"formula_zh" if lang == "zh" else "formula_en"]
        natural_guide = self.format_guide["natural"][lang]
        structured_guide = self.format_guide["structured"][lang]

        prompt_parts = []
        if enable_global_preconstraint:
            prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
            prompt_parts.append(global_rule)
        prompt_parts.append(f"SDXL模型内容组织公式：{formula_hint}")
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