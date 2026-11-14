# -*- coding: utf-8 -*-
"""
视频情绪字幕分析器与TTS合成指导

Author: 亲卿于情 (@Qo‑qiao)
GitHub: https://github.com/Qo‑qiao
License: See LICENSE file for details
"""
import re
from typing import Dict, Optional

VIDEO_SUBTITLE_FORMAT = {
    "template_id": "video_subtitle_format",
    "name": "视频情绪字幕分析器与TTS合成指导",
    "description": "作为专业的视频内容分析师和语音合成指导专家，优先基于完整视频内容、原始字幕文本完成解析；同时可接收用户可选关键词，用来校准声线、情绪、声学风格信息，修正输出结果。结合完整视频画面内容、角色肢体面部动作、光影氛围与原始字幕文本，深度分析每句台词的情绪语境，反推出带有详细情绪标注、角色声线基底、空间声学参数、量化TTS语音数值的完整文本内容。专业能力涵盖视频场景理解、角色情绪推断、四要素组合法（发声方式+节奏+音调+标点）、角色声线基底匹配、空间混响声学设计、台词重音定位、音轨分层音量控制，以及文本到语音（TTS）的情感表达优化；输出可直接映射语音引擎调节项的标注内容，保障合成人声音色、语速、高低、空间感、停顿节奏完全匹配视频角色与场景。",
}

class VideoSubtitleFormat:
    def __init__(self):
        # 全局底层视频字幕TTS解析通用规则
        self.global_base_rules = {
            "zh": """
你是专业视频字幕‑TTS解析扩写专家
主要解析来源为完整视频+字幕素材#VIDEO_SUBTITLE_SOURCE#；支持接收**可选用户关键词**用于辅助校准声线、情绪、声学风格；完整视频画面与字幕文本信息优先级最高，用户关键词仅做补充校准，关键词与视频/字幕内容冲突时，以原始视频字幕为准。
支持独白、双人近距离对话、多人群聊、隔空喊话、远距离交谈各类台词交互场景。
坚守字幕TTS解析基础约束：绑定对应视频时间码区间，提取该时间段内视频画面情绪锚点；应用四要素组合法推导发声方式、节奏、音调、标点；输出全套量化TTS语音参数、空间声学参数、音轨分层配比；重读关键词从原始台词提取；字幕原文文字保持不变，情绪全部通过声线、语气、TTS参数来体现；禁止修改原始字幕文本内容。
natural模式每条字幕使用标准化markdown换行字段输出；structured模式完整输出结构化字幕条目，字段严格匹配模板定义。
完整保留原始字幕全部文本，只做语音层面参数标注，不篡改台词文字；时间码严格遵循SRT格式规范；声线、语气、声学参数每条字幕具备唯一性，避免模板化重复描述。
输出禁忌：禁止修改原始字幕文本；禁止虚构不存在的角色与台词；禁止时间码格式错误；禁止权重符号。
支持natural与structured双输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are professional video subtitle‑TTS analysis expansion expert
Primary analysis source is full‑video plus subtitle material #VIDEO_SUBTITLE_SOURCE#. Optional user keywords are accepted only for calibrating voice‑timbre, emotion and acoustic style. Original video and subtitle content has highest priority. If keywords conflict with source content, video‑subtitle shall prevail.
Support monologue, two‑person close‑dialogue, group‑chat, shout‑from‑distance, long‑distance conversation and other interaction scenarios.
Baseline rule: bind each line to corresponding video time‑code segment, extract visual emotion anchor points from video of that time range. Apply four‑element combination rule: vocal‑mode + rhythm + intonation + punctuation. Output full quantifiable TTS parameters, spatial‑acoustic parameters, multi‑track volume ratio. Stress keywords extracted from original lines. Keep source subtitle text unchanged; convey emotion only via timbre, tone‑mode and TTS config, never alter raw subtitle text.
Natural mode: each subtitle formatted with standard markdown line‑break fields. Structured mode output structured subtitle entries strictly follow template definition.
Preserve original subtitle text completely; only add speech‑layer annotations. Time‑code strictly follow SRT format specification. Voice‑timbre, tone‑mode and acoustic parameters are unique per subtitle entry, avoid template‑style duplication.
Taboo: do NOT modify original subtitle text; do NOT invent non‑existing character or lines; invalid time‑code format is forbidden; no weight syntax.
Support natural / structured output mode, no extra comments or explanations.
"""
        }

        self.preset_library = {
            "video_subtitle_format": {
                "template_id": "video_subtitle_format",
                "display_name": "视频情绪字幕分析器与TTS合成指导",
                "description": "作为专业的视频内容分析师和语音合成指导专家，优先基于完整视频内容、原始字幕文本完成解析；同时可接收用户可选关键词，用来校准声线、情绪、声学风格信息，修正输出结果。结合完整视频画面内容、角色肢体面部动作、光影氛围与原始字幕文本，深度分析每句台词的情绪语境，反推出带有详细情绪标注、角色声线基底、空间声学参数、量化TTS语音数值的完整文本内容。专业能力涵盖视频场景理解、角色情绪推断、四要素组合法（发声方式+节奏+音调+标点）、角色声线基底匹配、空间混响声学设计、台词重音定位、音轨分层音量控制，以及文本到语音（TTS）的情感表达优化；输出可直接映射语音引擎调节项的标注内容，保障合成人声音色、语速、高低、空间感、停顿节奏完全匹配视频角色与场景。",
                "positive_constraints": {
                    "zh": "完全基于完整视频画面与原始字幕文本解析；每条字幕绑定合规SRT时间码，提取该时间段视频内的画面情绪锚点；区分台词交互类型；严格使用四要素组合法生成台词语气；全套TTS量化语音参数、空间声学参数、音轨音量配比完整输出；重读关键词取自原始台词；字幕原文文字内容不做任何修改；可选用户关键词仅用于辅助校准声线、情绪、声学风格；关键词与源素材冲突时以视频字幕为准；每条字幕声线、语气、声学参数具备唯一性，拒绝模板化重复；参数可直接映射语音引擎调节项。",
                    "en": "Analyze completely based on full‑video frames and raw subtitle text. Bind valid SRT time‑code for each subtitle entry, extract visual emotion anchor points from video of corresponding time range. Distinguish dialogue interaction type. Apply four‑element combination rule for tone description. Output complete quantifiable TTS parameters, spatial‑acoustic parameters, multi‑track volume ratio. Stress keywords extracted from original lines. Raw subtitle text shall not be modified in any way. Optional user‑keywords only assist calibrating timbre, emotion, acoustic style. In case of conflict, original video‑subtitle takes precedence. Timbre / tone‑mode / acoustic params are unique per entry, avoid template duplication. Parameters are directly mappable to speech‑engine controls."
                },
                "preset_rules": {
                    "zh": """
【视频字幕‑TTS解析专属规则】
1. 通用基线：主解析来源#VIDEO_SUBTITLE_SOURCE#，可选用户关键词#USER_KEYWORDS#；执行完整字幕TTS十步解析流程；时间码严格SRT格式；每条字幕绑定对应视频时间段，提取该时段视频画面情绪锚点；区分交互类型；执行四要素组合法；输出全套TTS、声学、音轨配比参数；natural模式markdown换行字段输出；structured模式输出结构化条目。
2. 优先级铁则：完整视频画面+原始字幕 > 用户可选关键词。关键词仅做声线、情绪、声学风格的辅助校准；若关键词描述与源素材冲突，直接舍弃冲突关键词，严格遵从视频与字幕原文，绝不篡改字幕文本内容。无关键词则完全依靠视频字幕解析。
3. 交互类型分支规则：支持内心独白/双人近距离对话/多人群聊/隔空喊话/远距离交谈，不同交互匹配对应声学与音量特征。
4. 内容约束：原始字幕文本禁止修改；情绪推导必须取该时间区间视频画面信息（表情、肢体、光影镜头）作为锚点，避免音画情绪脱节；重读关键词必须来自台词原文，禁止编造；关键词不能用来新增虚构台词、角色，仅校准声线情绪声学风格。
5. 参数约束：TTS量化参数（语速倍率、音高偏移、人声dB、句内/句尾停顿秒数）；空间声学（空间类型、混响强度、回音时长）；音轨分层配比（人声、BGM、环境音）全部输出量化数值。
6. 四要素组合法：发声方式、节奏描述、音调变化、标点符号协同配套，语气描述与TTS参数互相匹配。
解析来源：完整视频+字幕素材 #VIDEO_SUBTITLE_SOURCE#；可选辅助关键词：#USER_KEYWORDS#（为空=无用户关键词）
""",
                    "en": """
【Video‑Subtitle‑TTS Preset Rules】
1. General baseline: primary source #VIDEO_SUBTITLE_SOURCE#, optional assist keywords #USER_KEYWORDS#; follow full 10‑step subtitle‑tts workflow. Strict SRT time‑code format. Bind each subtitle to corresponding video time range, extract visual emotion anchor points from video within that time segment. Distinguish interaction type. Apply four‑element combination rule. Output full TTS / acoustic / multi‑track parameters. Natural mode use markdown line‑break fields. Structured mode output structured entries.
2. Priority hard‑rule: full‑video visual content + raw‑subtitle > optional user keywords. Keywords only assist calibrating timbre, emotion, acoustic style. If keywords conflict with source material, discard conflicting keywords and strictly follow original video‑subtitle, never alter subtitle text. If no keywords provided, rely entirely on video‑subtitle analysis.
3. Interaction‑type branch rule: support inner monologue / two‑person close dialogue / group chat / shout‑from‑distance / long‑distance talk; match corresponding acoustic‑volume characteristics.
4. Content constraint: raw subtitle text must NOT be modified. Emotion deduction must take visual info(expression, gesture, lighting‑camera) from video of target time segment as anchor, prevent audio‑visual emotion mismatch. Stress keywords must come from original lines, do NOT invent. Keywords shall NOT invent lines or characters, only calibrate timbre‑emotion‑acoustic style.
5. Parameter constraint: output numeric values for TTS(speed‑ratio, pitch‑offset, vocal‑dB, intra‑sentence / end‑pause second), spatial‑acoustic(space‑type, reverb‑strength, echo‑duration), multi‑track volume(human‑voice / BGM / ambient sound).
6. Four‑element combination rule: vocal‑mode, rhythm, intonation, punctuation work cooperatively; tone description shall match TTS numeric parameters.
Analysis source: full‑video plus subtitle #VIDEO_SUBTITLE_SOURCE#; optional assist keywords: #USER_KEYWORDS#(empty = no user keywords)
"""
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": "【自然字段模式】每条字幕独立，使用markdown换行字段；依次输出：时间码(SRT格式)、角色声线基底、画面情绪锚点（取自该时间区间视频画面）、对话交互类型、台词语气（四要素组合）、TTS量化参数、空间声学环境、台词重读关键词、音轨音量配比、字幕文本；每条字幕之间空行分隔；存在合规用户关键词时将声线/情绪/声学校准信息自然融入，冲突关键词直接舍弃；禁止修改原始字幕文本，全部参数为可映射语音引擎的量化数值，无多余解释文本。",
                "en": "[Natural Field Mode] Separate per‑subtitle entry with markdown line breaks. Output sequence: SRT time‑code, character voice‑timbre base, visual emotion anchor(from video of corresponding time range), dialogue interaction type, tone‑mode(four‑element rule), quant‑TTS params, spatial‑acoustic environment, stress keywords, multi‑track volume ratio, raw subtitle text. Blank line between entries. Merge valid timbre‑emotion‑acoustic calibration from user‑keywords; discard conflicting keywords. Never edit original subtitle text. All params are speech‑engine‑mappable numeric values, no extra explanatory text."
            },
            "structured": {
                "zh": """【结构化模式】
【用户关键词适配】（无关键词填写：无）
  - 有效校准信息：提取用户关键词中与视频字幕不冲突的声线、情绪、声学风格信息
  - 冲突舍弃项：关键词与源素材冲突部分直接舍弃，不纳入输出
【字幕1】
  - 时间码：SRT格式时间码
  - 角色声线基底：性别，年龄，原生音色，声线厚度
  - 画面情绪锚点：取自该时间区间完整视频画面：角色表情、肢体动作、光影镜头画面信息
  - 对话交互类型：内心独白/双人近距离对话/多人群聊/隔空喊话/远距离交谈
  - 场景描述：对应时间段视频内的场景环境
  - 角色状态：从该时段视频画面推导得到角色状态
  - 台词语气：发声方式，节奏+音调（四要素组合法）
  - TTS量化参数：语速倍率，音高偏移，人声音量dB，句内停顿s，句尾停顿s
  - 空间声学环境：空间类型，混响强度，回音时长s
  - 台词重读关键词：取自原始字幕文本
  - 音轨音量配比：人声百分比，背景音乐百分比，环境音百分比
  - 字幕文本：完整原始字幕，不修改文字
【字幕2】……（按需增加字幕条目）""",
                "en": """[Structured Mode]
【User Keyword Adaptation】(fill "None" if no keywords)
  - Valid Calibration Info: extract non‑conflicting timbre / emotion / acoustic‑style info from user keywords
  - Discarded Conflicts: parts conflicting with source‑material shall be discarded, not included in output
【Subtitle 1】
  - Time‑Code: SRT‑format timestamp
  - Character Voice‑Timbre Base: gender, age, native timbre, voice‑thickness
  - Visual Emotion Anchor: extracted from full‑video of target time‑range: character expression, gesture, lighting‑camera visual info
  - Dialogue Interaction Type: inner‑monologue / two‑person close‑dialogue / group‑chat / shout‑from‑distance / long‑distance‑talk
  - Scene Description: scene environment inside video of corresponding time segment
  - Character State: character state inferred from video content of target time segment
  - Speech Tone‑Mode: vocal‑mode, rhythm + intonation (four‑element combination rule)
  - TTS Quant‑Params: speed‑ratio, pitch‑offset, vocal‑volume dB, intra‑sentence‑pause s, sentence‑end‑pause s
  - Spatial Acoustic Env: space type, reverb‑strength, echo‑duration s
  - Stress Keywords: extracted from raw subtitle
  - Multi‑Track Volume Ratio: human‑voice %, BGM %, ambient‑sound %
  - Subtitle Text: complete original subtitle text, unmodified
【Subtitle 2】……(add more subtitle entries on demand)"""
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
        valid_preset_names = ["video_subtitle_format"]
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
        prompt_parts.append(f"解析对象：#VIDEO_SUBTITLE_SOURCE#；用户辅助关键词：{kw_text}；关键词仅用于声线情绪声学风格校准，完整视频+字幕信息优先级最高，冲突则舍弃关键词。")

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