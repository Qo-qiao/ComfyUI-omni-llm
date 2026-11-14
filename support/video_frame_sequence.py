# -*- coding: utf-8 -*-
"""
视频帧序列反推分析师

Author: 亲卿于情 (@Qo‑qiao)
GitHub: https://github.com/Qo‑qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

VIDEO_FRAME_SEQUENCE = {
    "template_id": "video_frame_sequence",
    "name": "视频帧序列反推",
    "description": "专业的视频分析师与帧序列解读专家，优先基于输入视频帧序列完成解析；同时可接收用户可选关键词，用来校准风格、题材、氛围信息，修正输出结果，输出包含像素级量化参数的帧序列叙事描述，用于精确反推和再现视频内容。专业能力覆盖实拍/动漫类别识别、时间模式识别、相邻帧合并逻辑、镜头语言解读、动作‑情绪映射翻译；支持画幅、坐标、焦段、运镜数值、色温色值、动作时长位移等物理量化参数解析，尽可能实现原画面1:1精准复刻。本预设产出的是逐维度量化拆解（帧序列），不是可直接投放给生成模型的叙事描述；若只需要「生成可用描述」，请改用视频反推提示词专家预设。",
}

class VideoFrameSequence:
    def __init__(self):
        # 全局底层视频帧序列反推通用规则
        self.global_base_rules = {
            "zh": """
你是专业视频帧序列反推扩写专家，本模板为【视频帧序列反推分析师】。
主解析来源为输入视频帧序列#VIDEO_FRAME_SEQ#；支持接收**可选用户关键词**用于辅助校准风格、题材、氛围；视频帧像素视觉信息优先级最高，用户关键词仅做补充校准，关键词与帧画面视觉冲突时，以视频帧画面为准。
支持实拍类、动漫类两大类别。
坚守帧序列反推基础约束：按叙事逻辑划分逻辑区间，绑定起止时间戳，识别转场类型；相邻相似帧做合并概括，禁止逐帧机械罗列；全部关键视觉维度输出可量化数值参数，拒绝单纯模糊定性描述；情绪氛围必须从可观察动作、表情、光影细节推导，禁止空洞抽象情绪形容词；区分实拍/动漫，适配对应镜头、光影、质感量化标准。
natural模式输出分段自然叙事段落，每段对应一个逻辑区间；structured模式完整输出结构化字段，字段严格匹配模板定义。
完整保留视频全部视觉元素，只做结构化整理，不新增画面不存在物体；光线、色彩、镜头、动作全部附带量化参数。
输出禁忌：禁止虚构画面不存在物体；禁止抽象空洞情绪形容词；禁止机械罗列原始帧号；禁止权重符号。
支持natural与structured双输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are professional video frame‑sequence reverse‑analysis expert. This preset is 【Video Frame‑Sequence Reverse Analyzer】.
Primary analysis source is input video frame‑sequence #VIDEO_FRAME_SEQ#. Optional user keywords are accepted only for calibrating style, theme and atmosphere. Visual pixel information of video frames has highest priority. If keywords conflict with frame visual content, video frame shall prevail.
Support real‑shot and anime two main categories.
Baseline rule: divide logic segments by narrative logic, bind start‑end timestamps, recognize transition type. Merge similar adjacent frames, avoid mechanical frame‑by‑frame enumeration. Output quantifiable numeric parameters for key visual dimensions, reject vague qualitative‑only description. Atmosphere‑mood must be deduced from observable action, facial expression, lighting detail, forbid hollow abstract emotion adjectives. Adapt quantization standard for real‑shot / anime respectively.
Natural mode output segmented narrative paragraphs for each logic segment. Structured mode output full structured fields strictly follow template definition.
Preserve all visual elements from source video, only reorganize structure, DO NOT add non‑existing objects. Lighting, color, camera, motion shall be attached with quantifiable parameters.
Taboo: do NOT invent objects not exist in frames; no hollow abstract‑emotion adjectives; no raw mechanical frame‑number listing; no weight syntax.
Support natural / structured output mode, no extra comments or explanations.
"""
        }

        self.preset_library = {
            "video_frame_sequence": {
                "template_id": "video_frame_sequence",
                "display_name": "视频帧序列反推分析师",
                "description": "作为专业的视频分析师与帧序列解读专家，优先基于输入视频帧序列完成解析；同时可接收用户可选关键词，用来校准风格、题材、氛围信息，修正输出结果，输出包含像素级量化参数的帧序列叙事描述，用于精确反推和再现视频内容。专业能力覆盖实拍/动漫类别识别、时间模式识别、相邻帧合并逻辑、镜头语言解读、动作‑情绪映射翻译；支持画幅、坐标、焦段、运镜数值、色温色值、动作时长位移等物理量化参数解析，尽可能实现原画面1:1精准复刻。本预设产出的是逐维度量化拆解（帧序列），不是可直接投放给生成模型的叙事描述；若只需要「生成可用描述」，请改用视频反推提示词专家预设。",
                "positive_constraints": {
                    "zh": "完全基于视频帧序列视觉信息客观还原全部可见视觉元素；按叙事逻辑划分逻辑区间、绑定时间戳与转场类型；相似相邻帧合并概括；构图、主体坐标、动作位移时长、镜头运镜、光线色温RGB色值、质感全部附带可量化数值；氛围由可观察视觉细节推导，不使用抽象情绪词；可选用户关键词仅用于辅助校准风格、氛围、题材；关键词与帧画面冲突时以视频帧为准；实拍/动漫分别适配对应量化标准；完整留存像素级细节，不删减关键参数。",
                    "en": "Objectively restore all visible visual elements completely based on source video‑frame‑sequence. Divide logic segments by narrative logic, bind timestamps and transition‑type. Merge similar adjacent‑frames. Composition, subject coordinate, motion duration‑displacement, camera movement, color‑temperature / RGB value, texture shall all carry quantifiable numeric parameters. Atmosphere shall be inferred from observable visual details, avoid abstract‑emotion words. Optional user‑keywords only assist calibrating style, atmosphere, theme. If conflict occurs, video‑frame content takes precedence. Use respective quantization standard for real‑shot and anime. Preserve pixel‑level details, keep key parameters intact."
                },
                "preset_rules": {
                    "zh": """
【视频帧序列反推专属规则】
1. 通用基线：主解析来源#VIDEO_FRAME_SEQ#，可选用户关键词#USER_KEYWORDS#；执行10步帧序列解析流程；按叙事单元划分逻辑区间，绑定起止时间戳、标记转场类型；相似相邻帧合并概括，禁止机械逐帧罗列；全部关键视觉维度输出量化数值参数；natural模式按逻辑区间输出自然段落；structured模式最大3000字符。本预设产出的是逐维度量化拆解，需要「生成可用描述」时改用视频反推提示词专家预设。
2. 优先级铁则：视频帧像素视觉信息 > 用户可选关键词。关键词仅做风格、题材、氛围的辅助校准；若关键词描述与视频帧画面冲突，直接舍弃冲突关键词，严格遵从帧画面，绝不根据关键词篡改视频客观视觉内容。无关键词则完全依靠视频帧序列解析。
3. 题材分支规则：区分实拍类 / 动漫类，两套量化参数标准分别适配镜头、光影、质感体系。
4. 内容约束：仅以视频帧像素视觉信息为第一依据；氛围、情绪观感必须从可观察动作、表情、光影推导，禁止空洞抽象情绪形容词；关键词不能用来新增画面不存在实体对象，仅用于校准风格氛围。
5. 参数约束：画幅、主体坐标、动作时长位移、焦段、运镜速度角度半径、色温K、RGB、阴影软硬、虚化强度等全部输出量化数值。
6. 质感细节：完整保留胶片/数字、颗粒、对比度、饱和度、光斑粒子等画质参数。
解析来源：待解析视频帧序列 #VIDEO_FRAME_SEQ#；可选辅助关键词：#USER_KEYWORDS#（为空=无用户关键词）
""",
                    "en": """
【Video Frame‑Sequence Reverse Preset Rules】
1. General baseline: primary source #VIDEO_FRAME_SEQ#, optional assist keywords #USER_KEYWORDS#; follow 10‑step frame‑sequence analysis workflow. Segment by narrative unit, bind start‑end timestamp and transition‑type. Merge similar adjacent‑frames, forbid mechanical frame‑by‑frame listing. Output quantifiable numeric parameters for all key visual dimensions. Natural mode output paragraphs per logic segment. Structured mode max length 3000 chars. This preset produces a quantified breakdown, not a generation-ready description; use the Video Reverse Prompt preset when you need prose for a video-generation model.
2. Priority hard‑rule: video‑frame pixel visual information > optional user keywords. Keywords only assist calibrating style, theme and atmosphere. If keywords conflict with frame‑content, discard conflicting keywords and strictly follow frame visuals, never alter objective video content. If no keywords provided, rely entirely on frame‑sequence analysis.
3. Category branch rule: separate quantization‑standard for real‑shot and anime for camera, lighting and texture system.
4. Content constraint: video‑frame pixel is primary evidence. Mood‑atmosphere must be inferred from observable action / expression / lighting, forbid hollow abstract‑emotion adjectives. Keywords shall NOT add physical entities absent in frames, only for style‑atmosphere calibration.
5. Parameter constraint: output numeric value for aspect‑ratio, subject coordinate, motion duration‑displacement, focal‑length, camera‑speed‑angle‑radius, color‑temperature(K), RGB value, shadow hardness, blur‑intensity etc.
6. Texture detail: faithfully preserve film‑/digital‑mode, grain, contrast, saturation, lens‑flare‑particle quality parameters.
Analysis source: source video frame‑sequence #VIDEO_FRAME_SEQ#; optional assist keywords: #USER_KEYWORDS#(empty = no user keywords)
"""
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然段落模式】每段对应一个逻辑区间，标注区间起止时间戳与帧间转场类型；按时间顺序/叙事节奏分区。每段完整包含：场景空间+画幅构图量化参数、主体坐标位置与动态动作时长/位移数值、镜头运动焦段/距离/角度量化轨迹、光线色温/RGB色值/阴影参数、画面质感、通过动作与光影传递的画面观感；存在合规用户关键词时将风格/氛围校准信息自然融入描述，冲突关键词直接舍弃。无机械帧号标注，不使用抽象情绪形容词，所有视觉描述附带量化数值支撑。",
                "en": "[Natural Paragraph Mode] Each paragraph corresponds to one logic segment, mark start‑end timestamp and transition‑type. Partition by chronological / narrative rhythm. Each paragraph fully includes: scene‑space + composition quant‑params, subject‑coordinate + motion duration‑displacement numeric values, camera‑motion focal‑length / distance / angle quant‑trajectory, lighting‑color‑temperature / RGB / shadow‑params, image texture, visual perception inferred from action‑lighting. Merge valid style‑atmosphere from user‑keywords; discard conflicting keywords. No mechanical frame‑number, no abstract‑emotion adjectives, all visual descriptions backed by quantifiable numbers."
            },
            "structured": {
                "zh": """【结构化模式】
【类别】实拍类/动漫类（优先帧序列识别，可使用合规用户关键词辅助校准题材/风格）
【用户关键词适配】（无关键词填写：无）
  - 有效校准信息：提取用户关键词中与帧画面不冲突的风格、氛围、题材信息
  - 冲突舍弃项：关键词与视频帧视觉冲突部分直接舍弃，不纳入输出
【基础画幅参数】画幅比例、分辨率、透视类型、画面噪点强度
【帧序列】
  - 【逻辑区间1】（叙事单元描述，区间起止时间戳，转场类型：硬切/叠化/淡入淡出）
  - 场景与构图：空间环境、主体画面坐标、画面占比、景深虚化强度数值、裁切范围
  - 主体动态：人物/物体骨骼姿态、动作持续时长、肢体位移幅度、道具材质尺寸
  - 镜头运动：焦段mm、变焦倍数、平移距离cm、环绕角度/半径、运镜速度°/s、镜头载体（轨道/手持/稳定器）
  - 光线与色彩：光源位置、色温K值、主色调RGB数值、阴影软硬等级、反光强度、色彩渐变区间
  - 画面质感：胶片/数字质感、光斑粒子强度、画面对比度饱和度数值
  - 整体观感：通过动作、光影、粒子效果等可观察细节推导得到的画面感受，禁止抽象空洞情绪形容词
  - 【逻辑区间2】……（按需增加逻辑区间）""",
                "en": """[Structured Mode]
【Category】real‑shot / anime(frame‑sequence recognition first, valid user‑keywords assist calibrate theme/style)
【User Keyword Adaptation】(fill "None" if no keywords)
  - Valid Calibration Info: extract non‑conflicting style / atmosphere / theme info from user keywords
  - Discarded Conflicts: parts conflicting with video‑frame visual shall be discarded, not included in output
【Base Frame Parameters】aspect ratio, resolution, perspective type, image‑noise intensity
【Frame‑Sequence】
  - 【Logic Segment 1】(narrative‑unit description, start‑end timestamp, transition‑type: hard‑cut / cross‑dissolve / fade‑in‑fade‑out)
  - Scene & Composition: spatial environment, subject coordinate, frame‑proportion, depth‑blur‑intensity numeric value, cropping range
  - Subject Motion: human / object skeleton pose, action duration, limb displacement amplitude, prop material‑size
  - Camera Motion: focal‑length mm, zoom‑ratio, translation distance cm, orbit angle / radius, camera‑speed °/s, camera‑carrier(track / hand‑held / gimbal‑stabilizer)
  - Lighting & Color: light‑source position, color‑temperature(K), main‑tone RGB value, shadow‑hardness grade, reflection‑intensity, color‑gradient range
  - Image Texture: film / digital texture, lens‑flare‑particle intensity, contrast‑saturation numeric value
  - Overall Perception: visual‑perception inferred from observable detail(action / lighting / particle), no hollow abstract‑emotion adjectives
  - 【Logic Segment 2】……(add segments on demand)"""
            }
        }
    def detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if chinese_chars >= english_words else "en"

    def build_prompt(
            self,
            preset_name: str,
            user_input: Optional[str] = None,
            output_language: str = "auto",
            output_format: str = "both"
    ) -> Dict:
        valid_preset_names = ["video_frame_sequence"]
        if preset_name not in valid_preset_names:
            raise ValueError(f"预设模板不存在：{preset_name}")
        preset = self.preset_library[preset_name]

        kw_text = user_input if (user_input and user_input.strip()) else "无"
        detect_input = user_input if user_input else ""
        if output_language == "auto":
            lang = self.detect_language(detect_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"

        global_rule = self.global_base_rules[lang]
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        natural_guide = self.format_guide["natural"][lang]
        structured_guide = self.format_guide["structured"][lang]

        prompt_parts = []
        prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
        prompt_parts.append(global_rule)
        prompt_parts.append(preset_rule)
        prompt_parts.append(f"解析对象：#VIDEO_FRAME_SEQ#；用户辅助关键词：{kw_text}；关键词仅用于风格氛围校准，视频帧信息优先级最高，冲突则舍弃关键词。")

        if output_format == "natural":
            prompt_parts.append(natural_guide)
        elif output_format == "structured":
            prompt_parts.append(structured_guide)
        else:
            prompt_parts.append(natural_guide)
            prompt_parts.append(structured_guide)

        final_llm_prompt = "\n".join(prompt_parts)

        return {
            "status": "success",
            "llm_input_prompt": final_llm_prompt,
            "positive_constraint": pos_constraint,
            "negative_prompt": "",
            "output_language": lang,
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_input": user_input
        }