# -*- coding: utf-8 -*-
"""
TTS多人对话生成器与情感合成指导（IndexTTS‑2.5 适配）

Author: 亲卿于情 (@Qo‑qiao)
GitHub: https://github.com/Qo‑qiao
License: See LICENSE file for details

依据 IndexTTS_情感化语音提示词书写指南优化：
内置下游TTS模型内容组织公式库（调用方式与 Ecommerce.model_formula_library 保持一致），
输出全部映射 IndexTTS‑2.5 情感化参数（8维情感向量emo_vector / emo_alpha / duration_factor /
发音标注），确保生成内容可直接用于 IndexTTS‑2.5 引擎。
"""
import re
from typing import Dict


MULTI_SPEAKER_DIALOGUE = {
    "template_id": "multi_speaker_dialogue",
    "name": "TTS多人对话生成器与情感合成指导（IndexTTS‑2.5 适配）",
    "description": "作为专业的TTS音频合成指导专家和剧本内容创作者，优先基于输入原始剧本文本完成解析；同时可接收用户可选关键词，用来校准音色、情绪、声场风格信息，修正输出结果。智能分析对话结构、说话人数量、对话交互关系，为每句台词分配唯一说话人ID、细分音色基底、交互距离、空间声场参数，同步标注台词情感、8维情感向量emo_vector（[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]，总和≤0.8）、情感强度emo_alpha、语速duration_factor、发音标注（中文拼音/英文CMU音素/日语假名）、四要素语气、可直接调用的量化TTS声学参数、重读关键词、情绪过渡逻辑、交叠对话标记、音轨分层配比，支持绑定视频时间码实现音画同步。专业知识涵盖角色音色差异化细分、多角色情绪平滑过渡控制、交叠插话人声处理、空间混响声学设计、台词重音逻辑、四要素组合法（发声方式+节奏+音调+标点），以及多人连续对话的情感连贯性优化，确保合成语音区分每个角色独特音色、情绪起伏自然、多人声场统一流畅，全部参数可直接映射IndexTTS‑2.5引擎。",
}

class MultiSpeakerDialogue:
    def __init__(self):
        # 下游TTS模型内容组织公式库（调用方式与 Ecommerce.model_formula_library 保持一致）
        # Default = 默认输出模型接口：未指定/Auto 时使用的默认优化输出，聚焦四要素组合台词语气等通用文本处理优化
        self.model_formula_library = {
            "Default": {
                "formula_zh": "内容组织顺序：说话人角色ID与音色大类 → 细分声线基底 → 逐句情感判定（8种情感之一或多情感混合）与情绪过渡逻辑 → 四要素组合台词语气（发声方式+节奏+音调+标点四项严格协同，参考【情感↔四要素组合映射表】）→ 量化TTS声学参数（语速倍率、音高偏移值、人声基准音量dB、句内停顿秒数、句尾停顿秒数）→ 重读关键词 → 交叠对话标记 → 空间声学参数（空间类型、混响强度、回音时长）→ 音轨分层配比 → SRT时间码。",
                "formula_en": "Content order: speaker ID & timbre category → fine-grained voice base → sentence-wise emotion (one of 8 emotions or mixed) & emotion-transition logic → four-element combined speech tone (vocal-mode + rhythm + intonation + punctuation strictly cooperate, refer to 【Emotion ↔ Four-Element Mapping Table】) → quantized TTS acoustic params (speed-ratio, pitch-offset, vocal-baseline-dB, intra-sentence-pause-s, sentence-end-pause-s) → stress keywords → overlapping-dialogue mark → spatial-acoustic params (space type, reverb strength, echo duration) → multi-track volume ratio → SRT time-code."
            },
            "IndexTTS-2.5": {
                "formula_zh": "内容组织顺序：说话人角色ID与音色大类 → 细分声线基底 → 逐句情感判定（8维情感向量emo_vector[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]或显式情感描述emo_text，向量总和≤0.8）→ 情感强度emo_alpha（0.0‑1.0，推荐0.6）→ 语速duration_factor（0.5‑2.0）→ 发音控制（中文拼音<行|XING2>/英文CMU音素<minute|M IH1 . N AH0 T>/日语假名<上手|じょうず>）→ 空间声学参数 → 音轨分层配比 → SRT时间码。",
                "formula_en": "Content order: speaker ID & timbre category → fine-grained voice base → sentence-wise emotion (8-dim emo_vector[happy,angry,sad,afraid,disgusted,melancholic,surprised,calm] or explicit emo_text, vector sum ≤0.8) → emotion strength emo_alpha (0.0‑1.0, 0.6 recommended) → speech speed duration_factor (0.5‑2.0) → pronunciation control (Chinese pinyin <行|XING2>/English CMU phonemes <minute|M IH1 . N AH0 T>/Japanese kana <上手|じょうず>) → spatial acoustic params → multi-track volume ratio → SRT time-code."
            },
            "VoxCPM2": {
                "formula_zh": "内容组织顺序：说话人角色ID与音色大类 → 细分声线基底 → 逐句情感判定与情绪过渡逻辑 → 文本规范化控制（normalize=True/False）→ 发音控制（中文拼音<行|XING2>/英文CMU音素<minute|M IH1 . N AH0 T>）→ 标点韵律控制（句号/问号停顿、逗号缩短、省略号迟疑）→ 方言适配（粤语/闽南语/四川话等）→ 空间声学参数 → 音轨分层配比 → SRT时间码。",
                "formula_en": "Content order: speaker ID & timbre category → fine-grained voice base → sentence-wise emotion & emotion-transition logic → text normalization control (normalize=True/False) → pronunciation control (Chinese pinyin <行|XING2>/English CMU phonemes <minute|M IH1 . N AH0 T>) → punctuation prosody control (period/question mark pause, comma shortening, ellipsis hesitation) → dialect adaptation (Cantonese/Minnan/Sichuan etc.) → spatial acoustic params → multi-track volume ratio → SRT time-code."
            }
        }

        # 全局底层TTS多人对话通用规则
        self.global_base_rules = {
            "zh": """
你是专业TTS多人对话解析扩写专家，本模板为【TTS多人对话生成器与情感合成指导（IndexTTS‑2.5 适配）】。
主要解析来源为原始剧本对话文本#DIALOGUE_SOURCE#；支持接收**可选用户关键词**用于辅助校准音色、情绪、声场风格；原始剧本文本信息优先级最高，用户关键词仅做补充校准，关键词与剧本内容冲突时，以原始剧本为准。
支持单人独白、双人对话、多人群聊、插话交叠同步对话各类交互场景；内置完整音色类型参考表。
坚守多人对话TTS解析基础约束：每个说话人分配全局唯一数字ID，不可重复遗漏；同一角色多轮对话音色基底、声场参数保持一致；不同角色音色、语气、声场具备明显区分度；标注对话交互距离；空间声学参数全部量化输出；逐句判定情感与情绪过渡逻辑；交叠对话标记起止区间；**台词原文保持不变，仅通过音色、声学、情绪参数传递情感，禁止修改原始剧本文字**。
适配IndexTTS‑2.5情感化参数库：
支持8种情感：高兴happy/愤怒angry/悲伤sad/害怕afraid/厌恶disgusted/忧郁melancholic/惊讶surprised/平静calm；
情感控制方式：8维情感向量emo_vector（顺序固定[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]，取值0.0‑1.2，向量总和≤0.8）、显式情感描述emo_text（配合use_emo_text=True与emo_alpha=0.6）；
情感强度emo_alpha（0.0‑1.0，推荐0.6，未指定场景时取0.6；有声书0.3‑0.5/广播剧0.6‑0.8/动画配音0.7‑0.9/游戏配音0.8‑1.0/语音助手0.2‑0.4）；
语速duration_factor（0.5‑2.0，默认1.0；快速播报0.7‑0.9/正常对话1.0/慢速朗读1.1‑1.3/诗歌朗诵1.2‑1.5/儿童故事0.9‑1.1）；
发音控制：中文拼音标注<文字|拼音+声调数字>（如<行|XING2>）、英文CMU音素<单词|音素序列>（如<minute|M IH1 . N AH0 T>）、日语假名<汉字|假名>（如<上手|じょうず>）；合法拼音参考checkpoints/pinyin.vocab；
语言代码lang：ZH/EN/JA/ES/AR；
文本情感识别限制：QwenEmotion无法区分"悲伤"与"低落"，检测到"忧郁"关键词（低落/melancholy/depressed/gloomy）时系统自动交换"悲伤"与"低落"向量；明确忧郁需含忧郁关键词，明确悲伤需避免忧郁词汇。
natural模式每条对话条目标准化markdown字段输出，字段严格匹配模板定义。
完整保留剧本全部对话文本，只增加语音合成维度参数标注；时间码严格遵循SRT格式；音色、情绪、声学参数每条对话具备唯一性，避免模板化重复。
输出禁忌：禁止修改原始剧本台词文本；禁止说话人ID重复/缺失；禁止权重符号；禁止情感向量维度错误或总和越界；禁止emo_alpha/duration_factor超出取值范围；禁止情感与文本内容冲突。
支持natural输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are professional multi‑speaker‑dialogue TTS analysis expert. This preset is 【Multi‑Speaker Dialogue Generator & Emotional Synthesis Guide (IndexTTS‑2.5 Adapted)】.
Primary analysis source is raw script dialogue #DIALOGUE_SOURCE#. Optional user keywords are accepted only for calibrating timbre, emotion and acoustic‑field style. Original script content has highest priority. If keywords conflict with script content, original script shall prevail.
Support monologue, two‑person dialogue, group chat, interrupt / overlapping simultaneous‑dialogue scenarios. Built‑in full timbre reference table.
Baseline rule: assign global unique numeric speaker‑ID for each character, no duplicate or missing. Keep consistent timbre‑base & acoustic‑field params for same‑character multi‑turn utterances. Obvious distinction on timbre, tone and acoustic‑field between different speakers. Mark dialogue interaction distance. All spatial‑acoustic params must be quantized. Judge sentence‑wise emotion & emotion‑transition logic. Mark time‑range for overlapping dialogue. **Raw dialogue text shall NOT be modified; convey emotion only via timbre / acoustic / emotion parameters.**
Adapt IndexTTS‑2.5 emotional parameter library:
8 emotions supported: happy / angry / sad / afraid / disgusted / melancholic / surprised / calm;
Emotion control: 8‑dim emo_vector (fixed order [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm], range 0.0‑1.2, vector sum ≤0.8), explicit emo_text (with use_emo_text=True and emo_alpha=0.6);
Emotion strength emo_alpha (0.0‑1.0, 0.6 recommended, 0.6 when no scene specified; audiobook 0.3‑0.5 / drama 0.6‑0.8 / animation 0.7‑0.9 / game 0.8‑1.0 / voice assistant 0.2‑0.4);
Speech speed duration_factor (0.5‑2.0, default 1.0; fast broadcast 0.7‑0.9 / normal dialogue 1.0 / slow reading 1.1‑1.3 / poetry 1.2‑1.5 / children story 0.9‑1.1);
Pronunciation control: Chinese pinyin annotation <char|PINYIN+tone number> (e.g. <行|XING2>), English CMU phonemes <word|phoneme sequence> (e.g. <minute|M IH1 . N AH0 T>), Japanese kana <kanji|kana> (e.g. <上手|じょうず>); valid pinyin reference checkpoints/pinyin.vocab;
Language code lang: ZH/EN/JA/ES/AR;
Text‑emotion recognition limit: QwenEmotion cannot distinguish "sad" from "melancholic"; when melancholic keywords (低落/melancholy/depressed/gloomy) are detected the system auto‑swaps the sad/melancholic vectors; for explicit melancholic include melancholic keywords, for explicit sad avoid melancholic words.
Natural mode output standard markdown fields per‑dialogue entry, strictly follow template definition.
Preserve full original script text, only append TTS‑layer annotations. Time‑code follow SRT specification. Timbre, emotion, acoustic params are unique per utterance, avoid template duplication.
Taboo: do NOT alter raw script lines; duplicate / missing speaker‑ID forbidden; no weight syntax; wrong emo_vector dimension or out‑of‑range sum forbidden; emo_alpha/duration_factor out of range forbidden; emotion conflicting with text content forbidden.
Support natural output mode, no extra comments or explanations.
"""
        }

        # 默认优化输出（Default）全局底层TTS多人对话通用规则
        # 不含IndexTTS‑2.5专属情感参数（emo_vector/emo_alpha/duration_factor/发音标注/lang等），聚焦四要素组合台词语气等通用文本处理优化
        self.default_base_rules = {
            "zh": """
你是专业TTS多人对话解析扩写专家，本模板为【TTS多人对话生成器与情感合成指导（默认优化输出）】。
主要解析来源为原始剧本对话文本#DIALOGUE_SOURCE#；支持接收**可选用户关键词**用于辅助校准音色、情绪、声场风格；原始剧本文本信息优先级最高，用户关键词仅做补充校准，关键词与剧本内容冲突时，以原始剧本为准。
支持单人独白、双人对话、多人群聊、插话交叠同步对话各类交互场景；内置完整音色类型参考与【情感↔四要素组合映射表】。
坚守多人对话TTS解析基础约束：每个说话人分配全局唯一数字ID，不可重复遗漏；同一角色多轮对话音色基底、声场参数保持一致；不同角色音色、语气、声场具备明显区分度；标注对话交互距离；空间声学参数全部量化输出；逐句判定情感（8种情感之一或多情感混合）与情绪过渡逻辑；交叠对话标记起止区间；**台词原文保持不变，仅通过音色、声学、情绪参数传递情感，禁止修改原始剧本文字**；严格使用四要素组合法生成台词语气。
natural模式每条对话条目标准化markdown字段输出，字段严格匹配模板定义。
完整保留剧本全部对话文本，只增加语音合成维度参数标注；时间码严格遵循SRT格式；音色、情绪、声学参数每条对话具备唯一性，避免模板化重复。
输出禁忌：禁止修改原始剧本台词文本；禁止说话人ID重复/缺失；禁止权重符号；禁止台词语气与文本内容冲突。
支持natural输出格式，不添加额外注释、说明、解释。
""",
            "en": """
You are professional multi‑speaker‑dialogue TTS analysis expert. This preset is 【Multi‑Speaker Dialogue Generator & Emotional Synthesis Guide (Default Optimized Output)】.
Primary analysis source is raw script dialogue #DIALOGUE_SOURCE#. Optional user keywords are accepted only for calibrating timbre, emotion and acoustic‑field style. Original script content has highest priority. If keywords conflict with script content, original script shall prevail.
Support monologue, two‑person dialogue, group chat, interrupt / overlapping simultaneous‑dialogue scenarios. Built‑in full timbre reference table and 【Emotion ↔ Four‑Element Mapping Table】.
Baseline rule: assign global unique numeric speaker‑ID for each character, no duplicate or missing. Keep consistent timbre‑base & acoustic‑field params for same‑character multi‑turn utterances. Obvious distinction on timbre, tone and acoustic‑field between different speakers. Mark dialogue interaction distance. All spatial‑acoustic params must be quantized. Judge sentence‑wise emotion (one of 8 emotions or mixed) & emotion‑transition logic. Mark time‑range for overlapping dialogue. **Raw dialogue text shall NOT be modified; convey emotion only via timbre / acoustic / emotion parameters.** Strictly apply four‑element combination rule for speech tone description.
Natural mode output standard markdown fields per‑dialogue entry, strictly follow template definition.
Preserve full original script text, only append TTS‑layer annotations. Time‑code follow SRT specification. Timbre, emotion, acoustic params are unique per utterance, avoid template duplication.
Taboo: do NOT alter raw script lines; duplicate / missing speaker‑ID forbidden; no weight syntax; speech tone conflicting with text content forbidden.
Support natural output mode, no extra comments or explanations.
"""
        }

        # 唯一主预设模板，绑定multi_speaker_dialogue template_id
        self.preset_library = {
            "multi_speaker_dialogue": {
                "template_id": "multi_speaker_dialogue",
                "display_name": MULTI_SPEAKER_DIALOGUE["name"],
                "description": MULTI_SPEAKER_DIALOGUE["description"],
                "positive_constraints": {
                    "zh": "完全基于原始剧本对话文本解析；每个说话人分配全局唯一数字ID；同一角色多轮对话音色基底、声场参数保持一致；不同角色音色语气声场具备明显区分度；标注对话交互距离；空间声学参数量化输出；逐句判定情感并输出8维情感向量emo_vector（顺序[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]，取值0.0‑1.2，总和≤0.8）；标注情感强度emo_alpha（0.0‑1.0，推荐0.6）；标注语速duration_factor（0.5‑2.0）；多音字/生僻字/外来语输出发音标注（中文拼音/英文CMU音素/日语假名）；全套量化TTS声学参数、音轨分层配比完整输出；重读关键词取自对应台词；台词原文文字不做任何修改；可选用户关键词仅用于辅助校准音色、情绪、声场风格；关键词与剧本冲突时以原始剧本为准；每条对话音色情绪声学参数具备唯一性。",
                    "en": "Analyze completely based on raw script dialogue. Assign globally unique numeric speaker‑ID per character. Keep timbre‑base and acoustic‑field parameters consistent for multi‑turn lines of same speaker. Obvious distinction of timbre / tone / acoustic‑field between different roles. Mark dialogue interaction distance. Quantize all spatial‑acoustic parameters. Judge per‑utterance emotion and output 8‑dim emo_vector (order [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm], range 0.0‑1.2, sum ≤0.8). Mark emotion strength emo_alpha (0.0‑1.0, 0.6 recommended). Mark speech speed duration_factor (0.5‑2.0). Output pronunciation annotations for polyphonic / rare / loan words (Chinese pinyin / English CMU phonemes / Japanese kana). Output full quantized TTS acoustic params and multi‑track volume ratio. Stress keywords extracted from corresponding lines. Raw dialogue text must not be modified. Optional user‑keywords only assist calibrating timbre, emotion, acoustic‑field style. When conflict occurs, original script takes precedence. Timbre‑emotion‑acoustic parameters are unique for each dialogue entry."
                },
                "preset_rules": {
                    "zh": """
【TTS多人对话专属规则（IndexTTS‑2.5 适配）】
1. 通用基线：主解析来源#DIALOGUE_SOURCE#，可选用户关键词#USER_KEYWORDS#，目标模型#DOWNSTREAM_MODEL#；执行完整12步对话解析流程；分配全局唯一说话人ID；区分单人/双人/多人/交叠插话；配置音色大类+细分声线基底；标注交互距离；输出量化空间声学参数；逐句情感（8种情感之一或多情感混合）+情绪过渡；全套TTS量化参数与IndexTTS‑2.5情感参数（emo_vector/emo_alpha/duration_factor/发音标注）；标记交叠对话区间；音轨分层配比；可选SRT时间码；natural模式markdown字段输出，字段严格匹配模板定义。
2. 优先级铁则：原始剧本对话文本 > 用户可选关键词。关键词仅做音色、情绪、声场风格的辅助校准；若关键词描述与剧本冲突，直接舍弃冲突关键词，严格遵从原始剧本，绝不篡改台词原文。无关键词则完全依靠剧本文本解析。
3. 角色分支规则：内置音色类型参考表，区分女声/男声/萝莉音/正太音/御姐音/大叔音/老年音，每个角色配置一级音色大类+细分声线基底，实现同大类音色差异化区分。
4. 内容约束：台词原文禁止修改；同一角色多轮对话音色声场保持一致；不同角色具备区分度；交叠对话必须标记起止区间；重读关键词必须取自对应台词文本，禁止编造。
5. 参数约束：TTS量化参数（语速倍率、音高偏移、人声dB、句内停顿秒数、句尾停顿秒数）；空间声学（空间类型、混响强度、回音时长）；音轨分层配比（人声、BGM、环境噪音）全部输出量化数值；IndexTTS‑2.5情感参数：8维情感向量emo_vector（顺序[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]，取值0.0‑1.2，总和≤0.8，维度必须为8维）、情感强度emo_alpha（0.0‑1.0，有声书0.3‑0.5/广播剧0.6‑0.8/动画配音0.7‑0.9/游戏配音0.8‑1.0/语音助手0.2‑0.4，无场景指定默认0.6）、语速duration_factor（0.5‑2.0，默认1.0）、use_emo_text/emo_text显式情感描述、语言代码lang（ZH/EN/JA/ES/AR）；适配视频场景输出SRT标准时间码。
6. 发音控制规则：中文多音字/生僻字用拼音标注<文字|拼音+声调>（如<行|XING2>）；英文一词多音用CMU音素标注<单词|音素序列>（如<minute|M IH1 . N AH0 T>）；日文汉字用假名标注<汉字|假名>（如<上手|じょうず>）；合法拼音参考checkpoints/pinyin.vocab。
7. 情感识别限制：QwenEmotion无法区分"悲伤"与"低落"；明确忧郁需在台词或情感描述中含忧郁关键词（低落/melancholy/depressed/gloomy），明确悲伤需避免忧郁词汇；情感模板参考：happy[0.8,0,0,0,0,0,0,0]、angry[0,0.8,0,0,0,0,0,0]、sad[0,0,0.8,0,0,0,0,0]、melancholic[0,0,0,0,0,0.8,0,0]、calm[0,0,0,0,0,0,0,0.8]等，混合情感可叠加（如sad_melancholic[0,0,0.5,0,0,0.3,0,0]）；情感向量与台词情感必须一致，禁止情感与文本内容冲突。
解析来源：原始剧本对话文本 #DIALOGUE_SOURCE#；可选辅助关键词：#USER_KEYWORDS#（为空=无用户关键词）；目标模型：#DOWNSTREAM_MODEL#
""",
                    "en": """
【Multi‑Speaker‑Dialogue TTS Preset Rules (IndexTTS‑2.5 Adapted)】
1. General baseline: primary source #DIALOGUE_SOURCE#, optional assist keywords #USER_KEYWORDS#, target model #DOWNSTREAM_MODEL#; follow full 12‑step dialogue analysis workflow. Assign globally unique speaker‑ID. Distinguish monologue / two‑party / group / overlapping‑interruption. Configure timbre category + fine‑grained voice base. Mark interaction distance. Output quantized spatial‑acoustic params. Per‑utterance emotion (one of 8 emotions or mixed) & emotion‑transition. Full‑set quantized TTS params and IndexTTS‑2.5 emotional params (emo_vector / emo_alpha / duration_factor / pronunciation annotations). Mark overlapping‑dialogue time‑range. Multi‑track volume ratio. Optional SRT time‑code. Natural mode use markdown fields, strictly follow template definition.
2. Priority hard‑rule: raw‑script‑dialogue > optional user keywords. Keywords only assist calibrating timbre, emotion, acoustic‑field style. If keywords conflict with script content, discard conflicting keywords and strictly follow original script, never alter dialogue text. If no keywords provided, rely entirely on script analysis.
3. Character branch rule: built‑in timbre reference table; distinguish female / male / loli / shota / regal‑lady / middle‑aged‑male / elder voice. Each role gets top‑level timbre category plus fine‑grained voice base to differentiate voices within same category.
4. Content constraint: raw dialogue must NOT be modified. Keep timbre‑acoustic consistency for multi‑turn same‑speaker lines. Ensure distinguishability between different roles. Mark time‑range for overlapping dialogue. Stress keywords must come from corresponding dialogue, do NOT fabricate.
5. Parameter constraint: output numeric values for TTS(speed‑ratio, pitch‑offset, vocal‑dB, intra‑sentence‑pause‑s, sentence‑end‑pause‑s), spatial‑acoustic(space‑type, reverb‑strength, echo‑duration‑s), multi‑track volume(human‑voice / BGM / ambient‑noise). IndexTTS‑2.5 emotional params: 8‑dim emo_vector (order [happy,angry,sad,afraid,disgusted,melancholic,surprised,calm], range 0.0‑1.2, sum ≤0.8, must be 8 dims), emo_alpha (0.0‑1.0; audiobook 0.3‑0.5 / drama 0.6‑0.8 / animation 0.7‑0.9 / game 0.8‑1.0 / voice assistant 0.2‑0.4; default 0.6 when no scene specified), duration_factor (0.5‑2.0, default 1.0), use_emo_text/emo_text explicit emotion description, language code lang (ZH/EN/JA/ES/AR). Output SRT‑format time‑code for video‑adapted scenario.
6. Pronunciation control rule: Chinese polyphonic / rare characters annotated with pinyin <char|PINYIN+tone> (e.g. <行|XING2>); English heteronyms annotated with CMU phonemes <word|phoneme sequence> (e.g. <minute|M IH1 . N AH0 T>); Japanese kanji annotated with kana <kanji|kana> (e.g. <上手|じょうず>); valid pinyin reference checkpoints/pinyin.vocab.
7. Emotion recognition limit: QwenEmotion cannot distinguish "sad" from "melancholic"; for explicit melancholic include melancholic keywords (低落/melancholy/depressed/gloomy) in lines or emotion description, for explicit sad avoid melancholic words; emotion template reference: happy[0.8,0,0,0,0,0,0,0], angry[0,0.8,0,0,0,0,0,0], sad[0,0,0.8,0,0,0,0,0], melancholic[0,0,0,0,0,0.8,0,0], calm[0,0,0,0,0,0,0,0.8] etc.; mixed emotions can stack (e.g. sad_melancholic[0,0,0.5,0,0,0.3,0,0]); emo_vector must match dialogue emotion, emotion conflicting with text content forbidden.
Analysis source: raw script dialogue #DIALOGUE_SOURCE#; optional assist keywords: #USER_KEYWORDS#(empty = no user keywords); target model: #DOWNSTREAM_MODEL#
"""
                },
                "negative_base": {
                    "zh": "修改原始剧本台词文本，说话人ID重复/缺失，同一角色多轮对话音色声场不一致，情感向量维度错误或总和>0.8，emo_alpha/duration_factor超出取值范围，情感向量与文本内容冲突，发音标注错误，权重符号，多余解释文本，时间码非SRT格式，交叠对话未标记起止区间",
                    "en": "Altered raw script lines, duplicate/missing speaker-ID, inconsistent timbre-acoustic across same-speaker multi-turn lines, wrong emo_vector dimension or sum>0.8, emo_alpha/duration_factor out of range, emotion vector conflicting with text content, wrong pronunciation annotations, weight syntax, redundant explanatory text, non-SRT time-code, overlapping dialogue without time-range mark"
                },
                # 默认优化输出（Default）专属：正向约束/专属规则/负向基线（不含IndexTTS‑2.5专属情感参数）
                "default_positive_constraints": {
                    "zh": "完全基于原始剧本对话文本解析；每个说话人分配全局唯一数字ID；同一角色多轮对话音色基底、声场参数保持一致；不同角色音色语气声场具备明显区分度；标注对话交互距离；空间声学参数量化输出；逐句判定情感（8种情感之一或多情感混合）；严格执行四要素组合法生成台词语气（发声方式+节奏+音调+标点四项全输出）；全套量化TTS声学参数、音轨分层配比完整输出；重读关键词取自对应台词；台词原文文字不做任何修改；可选用户关键词仅用于辅助校准音色、情绪、声场风格；关键词与剧本冲突时以原始剧本为准；每条对话音色情绪声学参数具备唯一性。",
                    "en": "Analyze completely based on raw script dialogue. Assign globally unique numeric speaker‑ID per character. Keep timbre‑base and acoustic‑field parameters consistent for multi‑turn lines of same speaker. Obvious distinction of timbre / tone / acoustic‑field between different roles. Mark dialogue interaction distance. Quantize all spatial‑acoustic parameters. Judge per‑utterance emotion (one of 8 emotions or mixed). Strictly apply four‑element combination rule for speech tone (vocal‑mode + rhythm + intonation + punctuation, all four fields). Output full quantized TTS acoustic params and multi‑track volume ratio. Stress keywords extracted from corresponding lines. Raw dialogue text must not be modified. Optional user‑keywords only assist calibrating timbre, emotion, acoustic‑field style. When conflict occurs, original script takes precedence. Timbre‑emotion‑acoustic parameters are unique for each dialogue entry."
                },
                "default_preset_rules": {
                    "zh": """
【TTS多人对话专属规则（默认优化输出）】
1. 通用基线：主解析来源#DIALOGUE_SOURCE#，可选用户关键词#USER_KEYWORDS#，目标模型#DOWNSTREAM_MODEL#；执行完整12步对话解析流程；分配全局唯一说话人ID；区分单人/双人/多人/交叠插话；配置音色大类+细分声线基底；标注交互距离；输出量化空间声学参数；逐句情感（8种情感之一或多情感混合）+情绪过渡；四要素组合法生成语气；全套TTS量化参数；标记交叠对话区间；音轨分层配比；可选SRT时间码；natural模式markdown字段输出，字段严格匹配模板定义。
2. 优先级铁则：原始剧本对话文本 > 用户可选关键词。关键词仅做音色、情绪、声场风格的辅助校准；若关键词描述与剧本冲突，直接舍弃冲突关键词，严格遵从原始剧本，绝不篡改台词原文。无关键词则完全依靠剧本文本解析。
3. 角色分支规则：内置音色类型参考表，区分女声/男声/萝莉音/正太音/御姐音/大叔音/老年音，每个角色配置一级音色大类+细分声线基底，实现同大类音色差异化区分。
4. 内容约束：台词原文禁止修改；同一角色多轮对话音色声场保持一致；不同角色具备区分度；交叠对话必须标记起止区间；参考【情感↔四要素组合映射表】生成台词语气；重读关键词必须取自对应台词文本，禁止编造。
5. 参数约束：TTS量化参数（语速倍率、音高偏移、人声dB、句内停顿秒数、句尾停顿秒数）；空间声学（空间类型、混响强度、回音时长）；音轨分层配比（人声、BGM、环境噪音）全部输出量化数值；台词语气严格按四要素组合法（发声方式+节奏+音调+标点四项全输出）；适配视频场景输出SRT标准时间码。
6. 映射规则：严格使用【情感↔四要素组合映射表】，发声方式、节奏、音调、标点协同配套，台词语气描述与TTS量化参数、情感描述互相匹配；台词语气与台词情感必须一致，禁止语气与文本内容冲突。
7. 【情感↔四要素组合映射表】（生成台词语气时严格查表，发声方式、节奏、音调、标点必须协同配套，禁止任意组合）：
| 情感 | 发声方式 | 节奏 | 音调 | 例句 |
|------|----------|------|------|------|
| 愤怒/命令 | 压低声，喉音下沉 | 节奏顿挫 | 语调下沉 | "你给我出去！" |
| 悲伤/失落 | 气声起，略带哽咽 | 节奏渐慢 | 语调下沉 | "为什么…要离开我…" |
| 喜悦/兴奋 | 明亮高音，提气 | 节奏加快 | 语调上扬 | "真的吗？太好了！" |
| 温柔/安慰 | 气声起，声音轻柔 | 节奏平稳 | 语调平缓 | "别担心，一切都会好的。" |
| 恐惧/紧张 | 气声起，声音发颤 | 节奏卡顿 | 语调失控上扬 | "我…找…不…到…了…" |
| 惊讶/震惊 | 提气，声音拔高 | 节奏突然加快 | 语调剧烈上扬 | "什么？怎么会这样！" |
| 慵懒/不屑 | 气声起，声音拖沓 | 节奏拖慢 | 语调平缓或蜿蜒 | "随便吧…我无所谓~" |
| 严肃/郑重 | 压低声，喉音下沉 | 节奏平稳偏慢 | 语调平稳 | "这件事，必须认真对待。" |
| 亲密/撒娇 | 轻声，声音软糯 | 节奏拖慢 | 语调蜿蜒 | "好不好嘛~人家想要这个~" |
| 紧张/焦虑 | 气声起，声音急促 | 节奏加快 | 语调波动不定 | "怎么办…怎么会这样…" |
8. 四要素组合法详解（台词语气=发声方式+节奏+音调+标点四要素组合，输出格式："发声方式：X；节奏：X；音调：X；标点：X"，四项必须全部给出并与映射表一致）：
   (1) 发声方式：气声起（虚弱/害羞/温柔/亲密）、压低声（压抑/成熟/深沉）、喉音下沉（稳重/权威/严肃）、明亮高音（活力/开朗）、沙哑音（疲惫/沧桑）、颤抖音（恐惧/激动）、提气（紧张/强调）、轻声（亲密/秘密）；
   (2) 节奏：节奏卡顿（紧张/害怕/犹豫）、节奏渐慢（悲伤/沉重/疲惫）、节奏加快（兴奋/紧张/愤怒）、节奏平稳（叙述/平静）、节奏急促（慌张/焦急）、节奏拖慢（慵懒/不屑）、节奏断断续续（哽咽/说不出口）；
   (3) 音调：语调平缓（叙述/平淡）、语调下沉（悲伤/失望）、语调上扬（惊讶/疑问/兴奋）、语调蜿蜒（撒娇/诱惑）、语调顿挫（愤怒/强调）、语调平稳（冷静/客观）、语调波动不定（焦虑/不确定）；
   (4) 标点符号：省略号…（犹豫/哽咽/未尽之意）、波浪号~（尾音拖长/撒娇/轻松）、破折号—（语气延展/停顿/沉思）、感叹号！（强烈情绪/命令/震惊）、问号？（疑问/不确定）。
解析来源：原始剧本对话文本 #DIALOGUE_SOURCE#；可选辅助关键词：#USER_KEYWORDS#（为空=无用户关键词）；目标模型：#DOWNSTREAM_MODEL#
""",
                    "en": """
【Multi‑Speaker‑Dialogue TTS Preset Rules (Default Optimized Output)】
1. General baseline: primary source #DIALOGUE_SOURCE#, optional assist keywords #USER_KEYWORDS#, target model #DOWNSTREAM_MODEL#; follow full 12‑step dialogue analysis workflow. Assign globally unique speaker‑ID. Distinguish monologue / two‑party / group / overlapping‑interruption. Configure timbre category + fine‑grained voice base. Mark interaction distance. Output quantized spatial‑acoustic params. Per‑utterance emotion (one of 8 emotions or mixed) & emotion‑transition. Generate tone description via four‑element rule. Full‑set quantized TTS params. Mark overlapping‑dialogue time‑range. Multi‑track volume ratio. Optional SRT time‑code. Natural mode use markdown fields, strictly follow template definition.
2. Priority hard‑rule: raw‑script‑dialogue > optional user keywords. Keywords only assist calibrating timbre, emotion, acoustic‑field style. If keywords conflict with script content, discard conflicting keywords and strictly follow original script, never alter dialogue text. If no keywords provided, rely entirely on script analysis.
3. Character branch rule: built‑in timbre reference table; distinguish female / male / loli / shota / regal‑lady / middle‑aged‑male / elder voice. Each role gets top‑level timbre category plus fine‑grained voice base to differentiate voices within same category.
4. Content constraint: raw dialogue must NOT be modified. Keep timbre‑acoustic consistency for multi‑turn same‑speaker lines. Ensure distinguishability between different roles. Mark time‑range for overlapping dialogue. Generate speech tone referring to 【Emotion ↔ Four‑Element Mapping Table】. Stress keywords must come from corresponding dialogue, do NOT fabricate.
5. Parameter constraint: output numeric values for TTS(speed‑ratio, pitch‑offset, vocal‑dB, intra‑sentence‑pause‑s, sentence‑end‑pause‑s), spatial‑acoustic(space‑type, reverb‑strength, echo‑duration‑s), multi‑track volume(human‑voice / BGM / ambient‑noise). Speech tone strictly follows four‑element combination rule (vocal‑mode + rhythm + intonation + punctuation, all four fields). Output SRT‑format time‑code for video‑adapted scenario.
6. Mapping rule: strictly follow 【Emotion ↔ Four‑Element Mapping Table】. Vocal‑mode, rhythm, intonation and punctuation cooperate mutually; tone description shall match quantized TTS parameters and emotion description; tone conflicting with dialogue emotion forbidden.
7. 【Emotion ↔ Four-Element Mapping Table】(strictly consult when generating tone description; vocal-mode, rhythm, intonation and punctuation must cooperate, arbitrary combinations forbidden):
| Emotion | Vocal-Mode | Rhythm | Intonation | Example |
|---------|-----------|--------|------------|---------|
| angry/commanding | lowered voice, throat down | staccato | falling | "Get out!" |
| sad/lost | breathy onset, slight sob | slowing | falling | "Why… are you leaving me…" |
| joyful/excited | bright high pitch, raised breath | quickening | rising | "Really? That's great!" |
| gentle/comforting | breathy onset, soft voice | steady | flat | "Don't worry, everything will be fine." |
| fearful/tense | breathy onset, trembling voice | halting | uncontrolled rising | "I… can't… find… it…" |
| surprised/shocked | raised breath, higher pitch | sudden quickening | sharply rising | "What? How could this happen!" |
| lazy/contemptuous | breathy onset, dragging voice | slowed | flat or winding | "Whatever… I don't care~" |
| solemn/serious | lowered voice, throat down | steady-slow | steady | "This matter must be taken seriously." |
| intimate/coquettish | soft voice, tender | slowed | winding | "Pretty please~ I want this~" |
| anxious/nervous | breathy onset, hurried voice | quickening | fluctuating | "What do I do… how…" |
8. Four-element combination rule details (speech tone = vocal-mode + rhythm + intonation + punctuation; output format: "Vocal-mode: X; Rhythm: X; Intonation: X; Punctuation: X", all four must be given and consistent with the mapping table):
   (1) Vocal-mode: breathy onset (weak/shy/gentle/intimate), lowered voice (suppressed/mature/deep), throat down (steady/authoritative/serious), bright high pitch (energetic/cheerful), husky (weary/weather-beaten), trembling (fearful/agitated), raised breath (tense/emphasis), soft voice (intimate/secret);
   (2) Rhythm: halting (nervous/fearful/hesitant), slowing (sad/heavy/tired), quickening (excited/tense/angry), steady (narrative/calm), hurried (flustered/urgent), dragging (lazy/contemptuous), broken-sobbing (sobbing/can't speak);
   (3) Intonation: flat (narrative/bland), falling (sad/disappointed), rising (surprised/questioning/excited), winding (coquettish/tempting), staccato (angry/emphasis), steady (calm/objective), fluctuating (anxious/uncertain);
   (4) Punctuation: ellipsis… (hesitation/sobbing/unfinished), tilde~ (lengthened tail/coquettish/relaxed), dash— (tone extension/pause/pondering), exclamation! (strong emotion/command/shock), question? (questioning/uncertain).
Analysis source: raw script dialogue #DIALOGUE_SOURCE#; optional assist keywords: #USER_KEYWORDS#(empty = no user keywords); target model: #DOWNSTREAM_MODEL#
"""
                },
                "default_negative_base": {
                    "zh": "修改原始剧本台词文本，说话人ID重复/缺失，同一角色多轮对话音色声场不一致，台词语气未按四要素组合法输出或与情感冲突，TTS量化参数缺失或非数值，空间声学参数未量化，音轨配比缺失，重读关键词非台词原文，权重符号，多余解释文本，时间码非SRT格式，交叠对话未标记起止区间",
                    "en": "Altered raw script lines, duplicate/missing speaker-ID, inconsistent timbre-acoustic across same-speaker multi-turn lines, speech tone not following four-element combination rule or conflicting with emotion, missing/non-numeric TTS quantized params, unquantized spatial-acoustic params, missing multi-track ratio, stress keywords not from raw lines, weight syntax, redundant explanatory text, non-SRT time-code, overlapping dialogue without time-range mark"
                },
                # VoxCPM2专属：正向约束/专属规则/负向基线
                "voxcpm2_positive_constraints": {
                    "zh": "完全基于原始剧本对话文本解析；每个说话人分配全局唯一数字ID；同一角色多轮对话音色基底、声场参数保持一致；不同角色音色语气声场具备明显区分度；标注对话交互距离；空间声学参数量化输出；逐句判定情感与情绪过渡逻辑；文本规范化控制（normalize=True处理数字日期/normalize=False保留音素）；发音控制（中文拼音/英文CMU音素）；标点韵律控制（句号问号停顿/逗号缩短/省略号迟疑）；方言适配（粤语用方言词汇书写）；台词原文文字不做任何修改；可选用户关键词仅用于辅助校准音色、情绪、声场风格；关键词与剧本冲突时以原始剧本为准；每条对话音色情绪声学参数具备唯一性。",
                    "en": "Analyze completely based on raw script dialogue. Assign globally unique numeric speaker‑ID per character. Keep timbre‑base and acoustic‑field parameters consistent for multi‑turn lines of same speaker. Obvious distinction of timbre / tone / acoustic‑field between different roles. Mark dialogue interaction distance. Quantize all spatial‑acoustic parameters. Judge per‑utterance emotion & emotion‑transition logic. Text normalization control (normalize=True for numbers/dates, normalize=False to preserve phonemes). Pronunciation control (Chinese pinyin / English CMU phonemes). Punctuation prosody control (period/question mark pause, comma shortening, ellipsis hesitation). Dialect adaptation (write dialect text in dialect vocabulary). Raw dialogue text must not be modified. Optional user‑keywords only assist calibrating timbre, emotion, acoustic‑field style. When conflict occurs, original script takes precedence. Timbre‑emotion‑acoustic parameters are unique for each dialogue entry."
                },
                "voxcpm2_preset_rules": {
                    "zh": """
【VoxCPM2多人对话专属规则】
1. 通用基线：主解析来源#DIALOGUE_SOURCE#，可选用户关键词#USER_KEYWORDS#，目标模型#DOWNSTREAM_MODEL#；执行完整对话解析流程；分配全局唯一说话人ID；区分单人/双人/多人/交叠插话；配置音色大类+细分声线基底；标注交互距离；输出量化空间声学参数；逐句情感+情绪过渡；文本规范化控制；发音控制；标点韵律控制；方言适配；全套TTS量化参数；标记交叠对话区间；音轨分层配比；可选SRT时间码；natural模式markdown字段输出。
2. 优先级铁则：原始剧本对话文本 > 用户可选关键词。关键词仅做音色、情绪、声场风格的辅助校准；若关键词描述与剧本冲突，直接舍弃冲突关键词，严格遵从原始剧本，绝不篡改台词原文。无关键词则完全依靠剧本文本解析。
3. 角色分支规则：内置音色类型参考表，区分女声/男声/萝莉音/正太音/御姐音/大叔音/老年音，每个角色配置一级音色大类+细分声线基底，实现同大类音色差异化区分。
4. 内容约束：台词原文禁止修改；同一角色多轮对话音色声场保持一致；不同角色具备区分度；交叠对话必须标记起止区间；重读关键词必须取自对应台词文本，禁止编造。
5. 文本规范化控制：normalize=True（默认）用于数字、日期、金额等格式自动扩展朗读；normalize=False用于需要精细发音控制的场景（音素输入）。
6. 发音控制规则：中文多音字/生僻字用拼音标注<文字|拼音+声调>（如<行|XING2>）；英文一词多音用CMU音素标注<单词|音素序列>（如<minute|M IH1 . N AH0 T>）；中文使用带音调数字的拼音（如{ni3}{hao3}），英文使用CMUDict风格音素（如{HH AH0 L OW1}）。
7. 标点韵律控制：句号和问号→句尾停顿更清晰；逗号→缩短停顿时间；省略号→迟疑或拖延效应；需更强停顿时拆分短句而非依赖标点。
8. 方言适配规则：生成特定方言语音时，用该方言自己的词汇和表达书写目标文本（如粤语用"伙計，唔該一個A餐"而非"伙计，麻烦来一个A餐"）；不确定时可用大语言模型从普通话翻译。
9. 短文本处理：非常短的输入（如"Hello""好的"）因模型最小音频长度约1秒会听起来很弱，应确保输入自然产生至少几秒钟语音。
10. 参数约束：TTS量化参数（语速倍率、音高偏移、人声dB、句内停顿秒数、句尾停顿秒数）；空间声学（空间类型、混响强度、回音时长）；音轨分层配比（人声、BGM、环境噪音）全部输出量化数值；适配视频场景输出SRT标准时间码。
解析来源：原始剧本对话文本 #DIALOGUE_SOURCE#；可选辅助关键词：#USER_KEYWORDS#；目标模型：#DOWNSTREAM_MODEL#
""",
                    "en": """
【VoxCPM2 Multi-Speaker Dialogue Preset Rules】
1. General baseline: primary source #DIALOGUE_SOURCE#, optional assist keywords #USER_KEYWORDS#, target model #DOWNSTREAM_MODEL#; follow full dialogue analysis workflow. Assign globally unique speaker‑ID. Distinguish monologue / two‑party / group / overlapping‑interruption. Configure timbre category + fine‑grained voice base. Mark interaction distance. Output quantized spatial‑acoustic params. Per‑utterance emotion & emotion‑transition. Text normalization control. Pronunciation control. Punctuation prosody control. Dialect adaptation. Full‑set quantized TTS params. Mark overlapping‑dialogue time‑range. Multi‑track volume ratio. Optional SRT time‑code. Natural mode use markdown fields.
2. Priority hard‑rule: raw‑script‑dialogue > optional user keywords. Keywords only assist calibrating timbre, emotion, acoustic‑field style. If keywords conflict with script content, discard conflicting keywords and strictly follow original script, never alter dialogue text. If no keywords provided, rely entirely on script analysis.
3. Character branch rule: built‑in timbre reference table; distinguish female / male / loli / shota / regal‑lady / middle‑aged‑male / elder voice. Each role gets top‑level timbre category plus fine‑grained voice base to differentiate voices within same category.
4. Content constraint: raw dialogue must NOT be modified. Keep timbre‑acoustic consistency for multi‑turn same‑speaker lines. Ensure distinguishability between different roles. Mark time‑range for overlapping dialogue. Stress keywords must come from corresponding dialogue, do NOT fabricate.
5. Text normalization control: normalize=True (default) for automatic expansion of numbers, dates, amounts; normalize=False for fine pronunciation control (phoneme input).
6. Pronunciation control rule: Chinese polyphonic / rare characters annotated with pinyin <char|PINYIN+tone> (e.g. <行|XING2>); English heteronyms annotated with CMU phonemes <word|phoneme sequence> (e.g. <minute|M IH1 . N AH0 T>); Chinese uses pinyin with tone numbers (e.g. {ni3}{hao3}), English uses CMUDict-style phonemes (e.g. {HH AH0 L OW1}).
7. Punctuation prosody control: period and question mark → clearer sentence-end pause; comma → shorten pause; ellipsis → hesitation or dragging effect; for stronger pause split into shorter sentences rather than relying on punctuation.
8. Dialect adaptation rule: to generate specific dialect speech, write target text in that dialect's own vocabulary and expressions (e.g. Cantonese: "伙計，唔該一個A餐" not "伙计，麻烦来一个A餐"); if unsure use LLM to translate from Mandarin.
9. Short text handling: very short inputs (e.g. "Hello""好的") sound weak due to ~1s minimum audio length; ensure input naturally produces at least several seconds of speech.
10. Parameter constraint: output numeric values for TTS(speed‑ratio, pitch‑offset, vocal‑dB, intra‑sentence‑pause‑s, sentence‑end‑pause‑s), spatial‑acoustic(space‑type, reverb‑strength, echo‑duration‑s), multi‑track volume(human‑voice / BGM / ambient‑noise). Output SRT‑format time‑code for video‑adapted scenario.
Analysis source: raw script dialogue #DIALOGUE_SOURCE#; optional assist keywords: #USER_KEYWORDS#; target model: #DOWNSTREAM_MODEL#
"""
                },
                "voxcpm2_negative_base": {
                    "zh": "修改原始剧本台词文本，说话人ID重复/缺失，同一角色多轮对话音色声场不一致，文本规范化控制错误，发音标注错误，标点韵律控制缺失，方言词汇使用标准普通话替代，短文本未扩写，TTS量化参数缺失或非数值，空间声学参数未量化，音轨配比缺失，重读关键词非台词原文，权重符号，多余解释文本，时间码非SRT格式，交叠对话未标记起止区间",
                    "en": "Altered raw script lines, duplicate/missing speaker-ID, inconsistent timbre-acoustic across same-speaker multi-turn lines, wrong text normalization control, wrong pronunciation annotations, missing punctuation prosody control, dialect text written in standard Mandarin instead of dialect vocabulary, short text not expanded, missing/non-numeric TTS quantized params, unquantized spatial-acoustic params, missing multi-track ratio, stress keywords not from raw lines, weight syntax, redundant explanatory text, non-SRT time-code, overlapping dialogue without time-range mark"
                }
            }
        }

        # 单模型输出格式指引（按下游TTS模型区分，仅支持natural输出）
        # Default = 默认优化输出指引，内置四要素组合台词语气文本处理优化（发声方式/节奏/音调/标点四项强制全输出）
        self.format_guide = {
            "Default": {
                "natural": {
                    "zh": "【默认优化输出·自然字段模式】每条对话独立，使用markdown换行字段；依次输出：时间码(SRT格式)、音色大类、细分声线基底、说话人ID、对话交互距离、空间声学环境、情感（8种之一或多情感）、情绪过渡说明、台词语气（四要素组合，必须严格按【发声方式：X；节奏：X；音调：X；标点：X】格式完整输出四项，严格参考【情感↔四要素组合映射表】与四要素组合法详解）、量化TTS声学参数（语速倍率、音高偏移值、人声基准音量dB、句内停顿秒数、句尾停顿秒数）、台词重读关键词、交叠对话标记、音轨音量配比、对话原文；每条对话之间空行分隔；存在合规用户关键词时将音色/情绪/声学校准信息自然融入，冲突关键词直接舍弃；禁止修改原始剧本台词，无多余解释文本。",
                    "en": "[Default Optimized Output · Natural Field Mode] Separate per-dialogue entry with markdown line breaks. Output sequence: SRT time-code, timbre category, fine-grained voice base, speaker-ID, dialogue interaction distance, spatial-acoustic environment, emotion (one of 8 or mixed), emotion-transition note, speech tone (four-element combination; must strictly output all four fields in format 【Vocal-mode: X; Rhythm: X; Intonation: X; Punctuation: X】, strictly reference the 【Emotion ↔ Four-Element Mapping Table】 and four-element combination rule details), quantized TTS acoustic params (speed-ratio, pitch-offset, vocal-baseline-dB, intra-sentence-pause-s, sentence-end-pause-s), stress keywords, overlapping-dialogue mark, multi-track volume ratio, raw dialogue text. Blank line between entries. Merge valid timbre-emotion-acoustic calibration from user-keywords; discard conflicting keywords. Never edit raw script lines. No extra explanatory text."
                }
            },
            "IndexTTS-2.5": {
                "natural": {
                    "zh": "【IndexTTS‑2.5 自然字段模式】每条对话独立，使用markdown换行字段；依次输出：时间码(SRT格式)、音色大类、细分声线基底、说话人ID、对话交互距离、空间声学环境、情感（8种之一或多情感）、情绪过渡说明、8维情感向量emo_vector（[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]，总和≤0.8）、情感强度emo_alpha（0.0‑1.0）、语速duration_factor（0.5‑2.0）、发音标注（多音字/生僻字/外来语，中文拼音/英文CMU音素/日语假名）、TTS量化参数、台词重读关键词、交叠对话标记、音轨音量配比、对话原文；每条对话之间空行分隔；存在合规用户关键词时将音色/情绪/声学校准信息自然融入，冲突关键词直接舍弃；禁止修改原始剧本台词，全部参数为可映射IndexTTS‑2.5引擎的量化数值，无多余解释文本。",
                    "en": "[IndexTTS‑2.5 Natural Field Mode] Separate per‑dialogue entry with markdown line breaks. Output sequence: SRT time‑code, timbre category, fine‑grained voice base, speaker‑ID, dialogue interaction distance, spatial‑acoustic environment, emotion (one of 8 or mixed), emotion‑transition note, 8‑dim emo_vector ([happy,angry,sad,afraid,disgusted,melancholic,surprised,calm], sum ≤0.8), emo_alpha (0.0‑1.0), duration_factor (0.5‑2.0), pronunciation annotation (polyphonic/rare/loan words, Chinese pinyin/English CMU phonemes/Japanese kana), quant‑TTS params, stress keywords, overlapping‑dialogue mark, multi‑track volume ratio, raw dialogue text. Blank line between entries. Merge valid timbre‑emotion‑acoustic calibration from user‑keywords; discard conflicting keywords. Never edit raw script lines. All params are IndexTTS‑2.5‑engine‑mappable numeric values, no extra explanatory text."
                }
            },
            "VoxCPM2": {
                "natural": {
                    "zh": "【VoxCPM2 自然字段模式】每条对话独立，使用markdown换行字段；依次输出：时间码(SRT格式)、音色大类、细分声线基底、说话人ID、对话交互距离、空间声学环境、情感（8种之一或多情感）、情绪过渡说明、文本规范化标记（normalize=True/False）、发音标注（中文拼音<行|XING2>/英文CMU音素<minute|M IH1 . N AH0 T>/中文拼音数字标记{ni3}{hao3}/英文CMUDict音素{HH AH0 L OW1}）、标点韵律说明（句号问号停顿/逗号缩短/省略号迟疑）、方言标记（如适用）、TTS量化参数（语速倍率、音高偏移值、人声基准音量dB、句内停顿秒数、句尾停顿秒数）、台词重读关键词、交叠对话标记、音轨音量配比、对话原文；每条对话之间空行分隔；存在合规用户关键词时将音色/情绪/声学校准信息自然融入，冲突关键词直接舍弃；禁止修改原始剧本台词，无多余解释文本。",
                    "en": "[VoxCPM2 Natural Field Mode] Separate per-dialogue entry with markdown line breaks. Output sequence: SRT time-code, timbre category, fine-grained voice base, speaker-ID, dialogue interaction distance, spatial-acoustic environment, emotion (one of 8 or mixed), emotion-transition note, text normalization flag (normalize=True/False), pronunciation annotation (Chinese pinyin <行|XING2>/English CMU phonemes <minute|M IH1 . N AH0 T>/Chinese pinyin tone marks {ni3}{hao3}/English CMUDict phonemes {HH AH0 L OW1}), punctuation prosody note (period/question mark pause, comma shortening, ellipsis hesitation), dialect flag (if applicable), quantized TTS acoustic params (speed-ratio, pitch-offset, vocal-baseline-dB, intra-sentence-pause-s, sentence-end-pause-s), stress keywords, overlapping-dialogue mark, multi-track volume ratio, raw dialogue text. Blank line between entries. Merge valid timbre-emotion-acoustic calibration from user-keywords; discard conflicting keywords. Never edit raw script lines. No extra explanatory text."
                }
            }
        }

    def detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return "zh" if chinese_chars >= english_words else "en"

    def build_prompt(
            self,
            user_input,
            preset_name,
            downstream_model=None,
            output_language: str = "auto",
            enable_global_preconstraint: bool = True,
            enable_negative_prompt: bool = True,
            output_format: str = "both"
    ) -> Dict:
        valid_preset_names = ["multi_speaker_dialogue"]
        if preset_name not in valid_preset_names:
            raise ValueError(f"预设模板不存在：{preset_name}")
        # 默认输出模型接口：未指定/Auto/默认 时自动使用默认优化输出（Default）
        if not downstream_model or str(downstream_model).strip().lower() in ("auto", "default", "默认"):
            downstream_model = "Default"
        if downstream_model not in self.model_formula_library:
            raise ValueError(f"不支持的下游TTS模型：{downstream_model}")
        preset = self.preset_library[preset_name]
        model_config = self.model_formula_library[downstream_model]

        detect_input = user_input if user_input else ""
        if output_language == "auto":
            lang = self.detect_language(detect_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"

        # 按输出模式选择规则文本：Default=默认优化输出（不含IndexTTS‑2.5专属情感参数）；IndexTTS-2.5=完整IndexTTS‑2.5适配；VoxCPM2=文本规范化+发音控制+标点韵律+方言适配
        if downstream_model == "Default":
            global_rule = self.default_base_rules[lang] if enable_global_preconstraint else ""
            preset_rule = preset["default_preset_rules"][lang]
            pos_constraint = preset["default_positive_constraints"][lang]
            negative_base = preset["default_negative_base"][lang]
        elif downstream_model == "VoxCPM2":
            global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
            preset_rule = preset["voxcpm2_preset_rules"][lang]
            pos_constraint = preset["voxcpm2_positive_constraints"][lang]
            negative_base = preset["voxcpm2_negative_base"][lang]
        else:
            global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
            preset_rule = preset["preset_rules"][lang]
            pos_constraint = preset["positive_constraints"][lang]
            negative_base = preset["negative_base"][lang]
        formula_hint = model_config[f"formula_{lang}"]
        natural_guide = self.format_guide[downstream_model]["natural"][lang]
        model_display = "默认优化输出（Default）" if downstream_model == "Default" else downstream_model

        prompt_parts = []
        if enable_global_preconstraint:
            prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
            prompt_parts.append(global_rule)
        prompt_parts.append(f"下游TTS模型内容组织公式（{model_display}）：{formula_hint}")
        prompt_parts.append(preset_rule)

        # 默认输出模式覆盖声明：聚焦四要素组合台词语气等通用文本处理优化，不输出任何模型专属情感参数
        if downstream_model == "Default":
            if lang == "zh":
                mode_override = "\n【默认优化输出模式声明】本模式聚焦通用多人对话TTS文本输出：四要素组合台词语气（发声方式/节奏/音调/标点四项全输出）+ 量化TTS声学参数（语速倍率、音高偏移值、人声基准音量dB、句内停顿秒数、句尾停顿秒数）+ 空间声学参数 + 音轨分层配比 + 重读关键词 + 交叠对话标记 + SRT时间码；不输出任何TTS引擎专属情感参数（8维情感向量/情感强度/语速系数/发音标注/语言代码等），也不输出任何音频参考音频、随机采样类参数。"
            else:
                mode_override = "\n[Default Optimized Output Mode Declaration] This mode focuses on generic multi-speaker-dialogue TTS text output: four-element combined speech tone (vocal-mode / rhythm / intonation / punctuation, all four fields) + quantized TTS acoustic params (speed-ratio, pitch-offset, vocal-baseline-dB, intra-sentence-pause-s, sentence-end-pause-s) + spatial-acoustic params + multi-track volume ratio + stress keywords + overlapping-dialogue mark + SRT time-code; do NOT output any TTS-engine-exclusive emotion params (8-dim emotion vector / emotion strength / speed coefficient / pronunciation annotation / language code), nor any audio-reference-audio or random-sampling params."
            prompt_parts.append(mode_override)
        elif downstream_model == "VoxCPM2":
            if lang == "zh":
                mode_override = (
                    "\n【VoxCPM2模式声明】本模式适配VoxCPM2语音合成引擎：文本规范化控制（normalize=True/False）"
                    "+ 发音控制（中文拼音<行|XING2>/英文CMU音素<minute|M IH1 . N AH0 T>/中文拼音数字标记{ni3}{hao3}/英文CMUDict音素{HH AH0 L OW1}）"
                    "+ 标点韵律控制（句号问号停顿/逗号缩短/省略号迟疑）+ 方言适配（粤语用方言词汇书写）"
                    "+ 短文本扩写建议 + 量化TTS声学参数 + 空间声学参数 + 音轨分层配比 + 重读关键词 + 交叠对话标记 + SRT时间码。"
                    "标点符号作为韵律提示：句号和问号让句尾停顿更清晰，逗号缩短停顿，省略号导致迟疑或拖延效应；"
                    "需更强停顿时拆分短句而非依赖标点。方言生成时用该方言自己的词汇和表达书写（如粤语用「伙計，唔該」而非「伙计，麻烦」）。"
                )
            else:
                mode_override = (
                    "\n[VoxCPM2 Mode Declaration] This mode adapts to VoxCPM2 TTS engine: text normalization control (normalize=True/False)"
                    " + pronunciation control (Chinese pinyin <行|XING2>/English CMU phonemes <minute|M IH1 . N AH0 T>/Chinese pinyin tone marks {ni3}{hao3}/English CMUDict phonemes {HH AH0 L OW1})"
                    " + punctuation prosody control (period/question mark pause, comma shortening, ellipsis hesitation)"
                    " + dialect adaptation (write dialect text in dialect vocabulary) + short text expansion suggestion"
                    " + quantized TTS acoustic params + spatial-acoustic params + multi-track volume ratio + stress keywords"
                    " + overlapping-dialogue mark + SRT time-code. Punctuation serves as prosody hints: period and question mark"
                    " make sentence-end pause clearer, comma shortens pause, ellipsis causes hesitation or dragging effect;"
                    " for stronger pause split into shorter sentences rather than relying on punctuation."
                    ' For dialect generation write in that dialect\'s own vocabulary and expressions (e.g. Cantonese: "伙計，唔該" not "伙计，麻烦").'
                )
            prompt_parts.append(mode_override)

        # 仅支持natural输出格式，任何 output_format 一律输出目标模型的natural指引
        prompt_parts.append(natural_guide)

        # 用户输入即原始剧本文本本身，注入主解析来源槽位；可选关键词仅作次级校准
        source_text = detect_input.strip() if detect_input and detect_input.strip() else ("（未提供）" if lang == "zh" else "(not provided)")
        if lang == "zh":
            source_ref = "（见下方【解析对象】）"
            keyword_ref = "（见下方【解析对象】）"
            target_line = (
                f"【解析对象】原始剧本对白（逐句保留原文，禁止改写）：{source_text}；"
                f"可选辅助关键词（仅用于校准音色、情绪、声场风格，与剧本冲突时舍弃）：无；"
                f"目标模型：{model_display}。"
            )
        else:
            source_ref = "(see the 【Analysis Target】 line below)"
            keyword_ref = "(see the 【Analysis Target】 line below)"
            target_line = (
                f"【Analysis Target】Raw script dialogue (preserve verbatim, do not rewrite): {source_text}; "
                f"optional assist keywords (calibrate timbre / emotion / acoustic-field style only, discard on conflict with script): none; "
                f"target model: {model_display}."
            )
        final_llm_prompt = (
            "\n".join(prompt_parts)
            .replace("#DIALOGUE_SOURCE#", source_ref)
            .replace("#USER_KEYWORDS#", keyword_ref)
            .replace("#DOWNSTREAM_MODEL#", model_display)
        )
        final_llm_prompt = f"{final_llm_prompt}\n{target_line}"
        negative_prompt = negative_base if enable_negative_prompt else ""

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
