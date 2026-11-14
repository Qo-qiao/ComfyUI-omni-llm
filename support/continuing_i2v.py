# -*- coding: utf-8 -*-
"""
图生视频预设模块

Author: 亲卿于情 (@Qo-qiao)
GitHub: https://github.com/Qo-qiao
License: See LICENSE file for details
"""
import re
from typing import Dict

CONTINUING_I2V = {
    "template_id": "continuing_i2v",    
    "name": "图生视频连续性导演",
    "description": "专业的图生视频连续性导演，基于首帧图像创造自然流畅、富有情感延续性的动态视频。支持结合用户自定义叙事引导词定向优化动态走向、情绪氛围、运镜节奏；未提供引导词时自动基于图像本身原生构图、光影、主体状态生成贴合物理逻辑的自然画面延续。采用正负向彻底分离架构，正向文本纯加法塑造延续动态，负向统一汇总规避生成通病。内置标准化8步扩写逻辑、固定语义权重优先级、三维度可控镜头视角、70%/25%/5%色彩配比规则与画面精简约束，强化微动作叙事、镜头语言、物理级光影与真实质感，彻底规避风格漂移、画面闪烁、肢体穿模、逻辑混乱、元素堆砌等图生视频常见问题。覆盖实拍类、动漫类两大核心赛道，自动识别首帧类别与风格，根据选择的视频模型输出匹配该模型语义习惯的精准描述，长文本分段控制避免后置约束失效，适配全平台图生视频工作流。",
}

class ContinuingI2V:
    def __init__(self):
        # 视频下游模型配置库：Wan2.2 / LTX2.5 / MiniMax‑H3
        self.video_model_formula_library = {
            "Wan2.2": {
                "formula_zh": "Wan2.2模型约束：原生单段最大时长5秒，推荐4‑5秒（超过5秒易崩）；仅支持单镜头，一个片段控制1‑2个核心微动作，禁止大幅度位移；不生成音频，禁止写台词音效（以上为 Wan2.2 限制；Wan2.6/2.7 支持多镜头与音画分层，选择新版本时按多镜头结构展开）；支持负向提示词。**禁忌特殊符号**：禁止@#$%^&*()等特殊符号、禁止中英文标点混用与全角标点，避免提示词解析混乱失效；动作写法拆分为“速度+方向+身体部位”（如“她以放松的步速走向镜头，左手从耳后撩过头发”）；人物一致性靠主体锚点（≥2个区分特征：疤痕、眼镜、耳饰、发色等），首帧与后续衔接时主体描述一字不改。**景别跨度压缩**：单镜头景别变化不超过两个层级（如远景→中景），最稳范围为远景→中景偏远处，人物禁止跑到近景特写；大跨度+高速动态会引发五官漂移、发型结块、服饰突变，大跨度场景须拆分为多镜头。**动作强度控制**：高风险动态细节一律降强度，“剧烈甩动”“发丝风中飞扬”改为“自然甩动”“长发自然飘动”，避免头发结块、辫子与身体穿模、发型前后不一致；禁止“猛地急停转身”“急速冲刺骤停”等易崩坏动作，爆发力体现在奔跑等过程动作而非骤停。**光影稳定性**：禁止微观高频光影动态（如“叶片光斑持续跳跃”），易致画面忽明忽暗、光影闪烁、光斑静止无细节，改为稳定表达“阳光穿过树叶形成斑驳光影，叶片上有柔和细碎高光”；禁止薄雾+强光柱等高开销光影叠加高速运动。**效果化表达**：禁止评价式模糊指令（“背景虚化处理得当”），改为明确效果（“主体清晰、背景柔和虚化、人物是视觉中心”）；“电影级质感”“电影级光影”等空词须绑定具体画面特征（柔和自然光、绿色背景衬托白色连衣裙、发丝边缘轮廓光、光影过渡自然），不单独空用。**运镜简化**：固定机位最稳，禁止复合叠加运镜（如“推进+横移”“推拉结合”），优先固定机位或单一运镜。**负向过滤**：“无多余装饰元素”类正向剔除句保留作简洁性提示，真正强过滤靠负向提示词。内容组织顺序：首帧锚点延续约束 → 场景基调光影 → 主体微动态 → 镜头运动 → 色彩质感 → 风格标签。natural模式可以使用逗号分隔关键词或连贯短段落；禁止描述音频台词，禁止单片段多镜头切换。",
                "formula_en": "Wan2.2 constraint: native max clip 5s, recommend 4‑5s (over 5s tends to break). Single shot only, keep 1‑2 core micro‑actions per clip, forbid large‑scale displacement. No audio output, do not write dialogue or sound‑effect. Support negative prompt. **Symbol taboo**: forbid special symbols @#$%^&*(), mixed Chinese‑English punctuation and full‑width punctuation to avoid parsing failure; decompose action as \"speed + direction + body part\" (e.g. \"she walks toward camera at relaxed pace, left hand brushing hair behind her ear\"); keep character consistency via subject anchors (≥2 distinguishing features: scar, glasses, earring, hair color, etc.), and keep subject description verbatim between reference frame and following clips. **Shot span compression**: single‑shot framing change limited to two levels (e.g. long → medium), most stable range is long → medium‑far, subject must NOT run into close‑up; large span + fast motion drifts facial features, clumps hair and mutates costume — split large‑span scenes into multiple shots. **Action intensity control**: downgrade high‑risk dynamics, \"violent whipping\" \"hair strands flying in wind\" become \"natural swinging\" \"long hair flowing naturally\", avoid hair clumping, ponytail/body clipping and inconsistent hairstyle; forbid crash‑prone moves like \"abrupt stop‑and‑turn\" \"sudden sprint halt\", put the burst into process actions like running rather than a sudden halt. **Lighting stability**: forbid micro high‑frequency light dynamics (e.g. \"dappled light continuously jumping on leaves\") which cause brightness flicker, shimmering and static glints, use stable phrasing \"sunlight through leaves casts dappled shadows, soft fine highlights on leaves\"; forbid high‑cost combos like thin fog + strong light beam stacked on fast motion. **Effect‑oriented wording**: forbid evaluative vague instructions (\"well‑handled background blur\"), rewrite as explicit effects (\"sharp subject, softly blurred background, subject as visual center\"); bind vague words like \"cinematic quality\" to concrete visual features (soft natural light, green backdrop against white dress, rim light on hair edges, natural light‑shadow transition) instead of using them alone. **Camera simplification**: fixed camera is most stable, forbid compound stacked moves (\"push + pan\" \"dolly zoom\"); prefer fixed camera or a single movement. **Negative filtering**: positive exclusion phrases like \"no extra ornament\" remain as simplicity hints, the real strong filtering relies on the negative prompt. Content order: reference‑frame anchor constraint → scene‑lighting → character micro‑motion → camera movement → color‑texture → style tags. Allow comma‑separated tags or short paragraphs. Forbid audio description and multi‑cut inside one clip."
            },
            "LTX2.5": {
                "formula_zh": "LTX‑2.5模型约束：原生单段最长20秒，稳定推荐3‑12秒；提示词字数必须匹配时长，长视频不能过短提示；原生支持音画同步，可以写台词、环境音、背景音乐；支持负向提示词。**首帧锚点延续**：严格继承首帧构图、主体状态、服装、发型、光影与风格，主体描述一字不改；图生视频从第一帧开始的默认保持连续镜头，除非你有意描述从开头画面切开的剪辑。**两种叙事结构**：①单段连续长镜头——整段写成流畅段落，目标4‑8个描述性句子，动作与运镜使用现在时动词，细节密度与景别匹配（特写比广角需要更多细节），描述相机相对主体的运动；②剧本式场景——含场景标题、角色提示与引号包裹的台词，保持相同原则（现在时、肢体情感线索、引号对话），每一句话都包含具体的视觉或音频细节；长度与复杂度匹配，简单单镜头4‑8句，更长场景只要每句都有具体细节可加长。**多镜头支持（LTX‑2.5新特性）**：可在同一条提示词内用明确剪辑连接2‑4个镜头，整个场景按时间顺序写成连贯段落（禁止镜头列表、编号节拍或剧本字幕，除非用散文描述剪辑）；剪辑名称用自然语言写出——\"硬切至……\"\"视角切至……的特写\"\"匹配剪接连接……\"\"图像溶解成……\"；每次切割后重新建立新镜头（镜头比例、摄像机角度、画面中出现的人或物、灯光是否改变），反复出现的人/物沿用相同视觉标识保持身份一致（如\"穿红大衣的女人，先前在桌边，现在……\"），并说明音乐/对话/环境音在剪辑间是延续还是变化（如\"钢琴配乐贯穿整个剪辑\"\"对话中断，只剩下风\"）；避免地理冲突或剪辑间莫名其妙的服装更换，除非剪辑是为跳跃时间或地点且你已言明。**镜头运动简化**：单镜头只保留一种主导运镜，禁止复合叠加运镜（如\"推进+横移\"\"推拉结合\"），复合运镜易致运镜失控、画面抖动、背景扭曲、人物出画；多镜头时每个镜头保持固定机位或一种运镜。**动作复杂度压缩**：单镜头控制1‑2个核心微动作，动作链按\"起始→过程→结束\"平缓推进，禁止多段连续大动作链；禁止\"猛地急停转身\"\"急速冲刺骤停\"等高崩坏风险动作（急停转身极易肢体崩坏、腿部穿模、姿态僵硬），爆发力应体现在奔跑等过程动作而非骤停；禁止\"定格\"\"冻结帧\"\"画面定格在XX\"指令（模型不支持，写了无效且干扰运动收尾）。**景别跨度压缩**：单镜头景别变化不超过两个层级（如中远景→中景），禁止远景→中景→近景三级跳；大跨度+高速动态叠加会引发发型、服饰、五官漂移，大跨度场景须拆分为多镜头。**场景聚焦与光线一致**：保持场景聚焦，几个清晰的人物和动作比拥挤画面更易读；每张镜头使用一个连贯的光线逻辑（光源、色温、软硬），禁止混合光源与微观高频光影动态（如\"叶片光斑持续跳跃\"），薄雾+强光柱等高开销光影不要叠加人物高速运动（易画面发灰、过曝、光柱闪烁不稳定）。**语法要点**：使用与预期类型相符的摄影术语，加入镜头比例或类别特性优化视觉风格；台词用英文双引号包裹\"台词内容\"，可指定语言/口音；I2V参考素材用@符号引用（如@Image1）；部分工具链支持分段标签[VISUAL]: 画面描述 / [SPEECH]: 台词 / [SOUNDS]: 音效音乐；表情情感不要直接写情绪词（如\"sad\"\"confused\"），要用物理动作体现（如\"他低下头，手指攥紧衣角\"）；角色定义包含年龄、发型、服装与显著特征，以身体信号表达情感；整体写得像cinematographer的镜头描述，越长越具体越好。**屏幕文字与复杂物理**：屏幕文字保持简短且显眼，拼写准确与帧间一致不保证，关键标题、标签、标志在后期添加；高度混沌的运动易引入伪影，更简单合理的运动更可靠。**修辞精简**：删减重复同义描述，同一特征只写一次，不反复堆砌；长文本后段注意力衰减会导致靠后细节被弱化。**瑕疵过滤移入负向提示词**：正向文本只做纯加法塑造画面，不写\"剔除/不要/无多余杂物\"类否定约束，缺陷规避统一写入negative_prompt。内容组织顺序：首帧锚点延续约束 → 镜头设定与场景光影 → 主体时序微动态 → 镜头运动节奏 → 色彩质感 → 音频描述 → 风格标签。natural模式输出流畅叙事段落，不要大量逗号碎片化标签堆砌。",
                "formula_en": "LTX‑2.5 constraint: native max 20s, stable recommend 3‑12s. Prompt length must match clip duration. Native audio‑visual sync, allow dialogue, ambient sound, background music. Support negative prompt. **Reference‑frame anchor continuation**: strictly inherit reference‑frame composition, subject state, costume, hairstyle, light‑shadow and style; keep subject description verbatim; for image‑to‑video starting from the first frame, prefer a continuous shot unless you intentionally describe cutting away from the opening frame. **Two narrative structures**: ①single continuous long‑take — write as one flowing paragraph, aim for 4‑8 descriptive sentences, use present‑tense verbs for action & camera, match detail density to shot size (close‑up needs more detail than wide shot), describe camera movement relative to subject; ②script‑style scene — scene heading, character cues and quoted dialogue, keep the same principles (present tense, physical emotion cues, quoted lines), every sentence containing specific visual or audio detail; length matches complexity — 4‑8 sentences for a simple single shot, longer scenes allowed as long as each sentence carries concrete detail. **Multi‑shot support (new in LTX‑2.5)**: connect 2‑4 shots with explicit cuts inside one prompt, write the whole scene as one chronological paragraph (no shot lists, numbered beats or screenplay captions unless describing the edit in prose); name the transition in natural language — \"hard cut to...\" \"a cut to a close‑up of...\" \"a match cut connects...\" \"the image dissolves into...\"; after every cut re‑establish the new shot (shot size, camera angle, who/what is in frame, whether lighting changed), reuse the same visual identifiers for recurring subjects to keep identity consistent (e.g. \"the woman in the red coat, earlier at the table, now...\"), and state whether music / dialogue / ambience continues or changes across the cut (e.g. \"the piano score continues throughout the cuts\" \"the dialogue cuts off, only wind remains\"); avoid geography conflicts or inexplicable costume changes between cuts unless the cut is a stated time or location jump. **Camera simplification**: keep only ONE dominant camera movement per single shot, forbid compound stacked moves (e.g. \"push + pan\" \"dolly zoom\") — they cause camera instability, jitter, background distortion, subject exiting frame; in multi‑shot keep fixed camera or one movement per shot. **Action complexity compression**: control 1‑2 core micro‑actions per shot, action chain advances smoothly as \"start → process → end\", forbid long multi‑stage action chains; forbid high‑crash actions like \"abrupt stop‑and‑turn\" \"sudden sprint halt\" (limb collapse, leg clipping, stiff pose) — put the burst into running & other process actions, not a sudden halt; forbid \"freeze frame\" \"hold frame\" \"freeze on XX\" (unsupported, ineffective and disrupts motion ending). **Shot span compression**: single‑shot framing change limited to two levels (e.g. medium‑long → medium), forbid long → medium → close‑up triple jump; large span combined with fast motion drifts hair, outfit, facial features — split large‑span scenes into multiple shots. **Scene focus & lighting consistency**: keep the scene focused — a few clear characters and actions read better than a crowded frame; use ONE coherent lighting logic per shot (light source, color temperature, hardness), forbid mixing light sources and micro high‑frequency light dynamics (e.g. \"dappled light continuously jumping on leaves\"); do not stack high‑cost fog + strong light‑beam combos on fast subject motion (gray washout, overexposure, flickering beams). **Syntax points**: use photography terms matching your intended look, add shot size or category traits to refine visual style; wrap dialogue in English double quotes, may specify language/accent; reference I2V material with @ symbol (@Image1); some toolchains support section labels [VISUAL]: visual description / [SPEECH]: lines / [SOUNDS]: sfx & music; do NOT write emotion words (like \"sad\" \"confused\"), convey emotion via physical cues (e.g. \"he lowers his head, fingers clenching the hem of his clothes\"); define characters with age, hairstyle, costume and distinguishing features, express emotion through body signals; write like a cinematographer's shot description, the longer and more specific the better. **Screen text & complex physics**: keep screen text short and prominent — spelling accuracy and frame consistency are not guaranteed, add key titles / labels / logos in post; highly chaotic motion tends to introduce artifacts — simpler, more plausible motion is more reliable. **Rhetoric trimming**: cut repetitive synonymous descriptions, describe each feature once, no repeated stacking; trailing attention decay weakens late details in long text. **Defect filtering to negative prompt**: positive text only adds to the scene, no \"no / without / remove\" negation constraints, defect avoidance all goes into negative_prompt. Content order: reference‑frame anchor constraint → camera‑scene‑lighting → character chronological micro‑motion → camera rhythm → color‑texture → audio description → style tags. Use coherent narrative paragraphs, avoid mass comma‑split tags."
            },
            "MiniMax‑H3": {
                "formula_zh": "MiniMax‑H3模型约束：时长取 4‑15 秒整数值，时间戳与对齐指令统一写作两位小数（如 5.00）；**模型文本只输出三个核心字段，不写 negative_prompt**；负向约束由节点级负向提示词单独输出；原生音画同步，支持台词、环境音、BGM；支持首帧参考图。\n\n**【强制三字段输出结构】MiniMax H3输出时必须包含以下三个独立字段，混写会导致解析混乱和音频分层失效：\n(1) integrated_multimodal_description: 画面描述、[Shot N]镜头编号、镜头运动、角色动作序列（含At时间戳）、台词、画内音乐（收音机/电视/手机播放等diegetic音乐）\n(2) overall_soundscape: **全片环境音+物理动作音效+非语言人声**（风声、雨声、环境底噪、脚步声、衣物摩擦、呼吸声、笑声等），1‑4句一个连续段落；对话、歌声、画内音乐已在integrated字段描述，不在此重复\n(3) non_diegetic_music: 画外配乐BGM，1‑3句，描述乐器、速度、节奏与动态变化，禁止抽象情绪词与情感功能解释；无BGM则写N/A\n**【标签语法】镜头用[Shot N]编号（首镜头[Shot 1]无需时间戳，默认起始00:00.000），后续镜头搭配At时间戳（格式：[Shot N] At MM:SS.ms, 内容），时间戳必须递增且在视频时长内；[Shot 1]开头必须声明整体风格与初始构图（风格词：Cinematic / live-action / 2D-animated / 3D CG / claymation / watercolor / vintage film，示例：[Shot 1] Live-action, cinematic, a medium-wide shot frames...）；镜头切换用the camera cuts to / the shot transitions to / the shot changes to；台词用<d>[语言] 台词</d>标签搭配说话人ID (S1)/(S2)（如 The woman (S1) says: <d>[English] Hello.</d>），多人同时说话写(S1,S2)，说话人描述+ID+动作后接台词；画外音用says in an off-screen voiceover并声明嘴唇闭合（while his lips remain completely closed）；台词跨镜头衔接加<scenetrans>，结尾台词被视频截断加<cutoff>；画面文字用英文双引号包裹并保留原文。\n\n**【首行指令行（必填，且必须放在第一行）】I2VA模式：\"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\"；指令行后空一行再写三字段。\n\n**【时间锚点规范】动作序列必须包含起始、中间、结束3‑4个时间节点，**最后一个时间戳必须与视频总时长严格对应**：\n  - 8秒视频示例：At 00:00.000 起始状态 → At 00:03.000 中间状态 → At 00:06.000 后段状态 → At 00:08.000 最终结束状态\n  - 无结束时间戳会导致后半段速度节奏完全随机失控\n\n**【I2VA推荐结构】按\"首帧锚定 → 动作开始 → 持续发展 → 结果或反应\"组织：首帧完全保持<Picture 1>的外貌、服装、位置与场景布局（preserving her appearance, clothing, seat position, and the carriage layout），再让动作自然展开。\n\n**【音效分层规则】人物动作音效（脚步、衣物摩擦、呼吸、笑声）与环境音（风声、树叶、底噪）统一汇总到overall_soundscape一个连续段落；对话、歌声、画内音乐只在integrated_multimodal_description描述；严禁两字段重复定义同一音效。\n\n**【运镜描述三要素】镜头运动写明运动类型+幅度+速度（如 The camera pushes in with small amplitude at slow speed toward...）；幅度用with small amplitude / with large amplitude，速度用at slow speed / at fast speed；中等幅度与正常速度通常省略。\n\n**【镜头运动简化】**H3对复合镜头运动控制力弱，必须简化：\n  - 禁止复合运动（如\"平移并推进\"\"推拉结合\"），容易导致人物出画、背景扭曲\n  - 优先固定机位：纯靠人物自身运动产生景别变化（最稳定）\n  - 或单一运动：仅保留一种（如仅\"缓慢向前推进\"）\n  - 禁止\"定格\"\"冻结帧\"\"画面定格在XX\"等指令（H3不支持，写了无效且干扰运动收尾）\n\n**【单镜头景别跨度压缩】单镜头景别变化不超过两个层级（如远景→中远景→中景），**禁止远景→中景→近景三级跳**。大跨度场景必须拆分多镜头。\n\n**【人物锚点具体化】至少3个具体区分特征，禁止模糊描述：\n  - 正确：\"黑色齐腰直发\"\"白色圆领\"\"棉麻面料连衣裙\"；错误：\"黑色长发\"\"白色连衣裙\"（太粗，大动态下易漂移）\n  - 禁止\"或\"字二选一（如\"树林或建筑\"→必须明确单一背景\"模糊的树林轮廓\"）\n\n**【风格前置】核心画面风格（光线类型与时段、整体色调、画面清晰度）必须放在[Shot 1]开头首句（顺序：风格类型→场景氛围→镜头运动→主体外貌），后续每个镜头从镜头运动开始；禁止放在段落末尾，长文本末尾注意力衰减会导致风格描述失效。\n\n**【H3禁止项】**禁止百分比坐标；禁止色彩占比数字；禁止画面比例参数；禁止微观纹理要求；禁止无时间锚点的裸动作序列；禁止否定性约束只写正向不写negative_prompt。\n\n内容组织顺序（integrated_multimodal_description内）：首帧对齐指令（第一行）→ [Shot 1] 风格与初始构图 → 首帧锚定延续 → 场景光影与氛围 → 主体外貌（具体锚点） → 时序微动态（At时间戳） → 镜头运动（单一） → 台词。输出时严格按三字段输出，不额外添加解释或注释。",
                "formula_en": "MiniMax‑H3 constraint: duration uses whole seconds 4‑15; timestamps and alignment instructions are written with two decimals (e.g. 5.00). **Model text contains only the three core fields and never negative_prompt**; negative constraints are emitted by the node-level negative prompt. Native audio‑visual sync, support dialogue, ambient sound, BGM. Support first‑frame reference image.\n\n**Mandatory Three‑Field Output Structure — H3 output must include these three separate fields; mixing causes parsing failure and audio layer collapse:\n(1) integrated_multimodal_description: visual description, [Shot N] labels, camera movement, character action sequence (with At timestamps), dialogue, in‑frame (diegetic) music (radio / TV / phone playback, etc.)\n(2) overall_soundscape: **ALL ambient sound + physical action sound + non‑verbal human sound** (wind, rain, environmental noise floor, footsteps, fabric rustle, breathing, laughter, etc.), 1‑4 English sentences as one continuous paragraph; dialogue, singing and in‑frame music are already described in the integrated field, do not repeat\n(3) non_diegetic_music: off‑screen BGM, 1‑3 sentences describing instruments, tempo, rhythm and dynamic changes, no abstract emotion words and no explanation of emotional function; write N/A if no BGM\n**Tag Syntax: label shots with [Shot N] (first shot [Shot 1] has no timestamp, defaults to 00:00.000), subsequent shots paired with At timestamp (format: [Shot N] At MM:SS.ms, content), timestamps must be increasing and within video duration; [Shot 1] must open by declaring overall style and initial composition (style words: Cinematic / live‑action / 2D‑animated / 3D CG / claymation / watercolor / vintage film, e.g. [Shot 1] Live‑action, cinematic, a medium‑wide shot frames...); shot transitions use the camera cuts to / the shot transitions to / the shot changes to; dialogue wrapped in <d>[language] lines</d> with speaker ID (S1)/(S2) (e.g. The woman (S1) says: <d>[English] Hello.</d>), simultaneous speech uses (S1,S2), speaker description + ID + action before the line; off‑screen line uses \"says in an off‑screen voiceover\" while stating lips remain completely closed; cross‑shot dialogue continuation uses <scenetrans>, dialogue truncated by the video end uses <cutoff>; on‑screen text wrapped in English double quotes, keep the original text untranslated.\n\n**Mandatory First‑Line Instruction (I2VA): \"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\" — leave one blank line after it, then write the three fields.\n\n**Time Anchor Specification: action sequences must have start, middle, and end (3‑4 time nodes), with the **final timestamp strictly matching total video duration**:\n  - 8‑second example: At 00:00.000 start state → At 00:03.000 mid state → At 00:06.000 later state → At 00:08.000 final end state\n  - Missing end timestamp causes back‑half pacing to become completely random and uncontrollable\n\n**I2VA Recommended Structure: organize as \"frame anchor → action begins → continuous development → result or reaction\": first fully preserve <Picture 1>'s appearance, clothing, position and scene layout (preserving her appearance, clothing, seat position, and the carriage layout), then let the action unfold naturally.\n\n**Sound Layering Rule: character action sounds (footsteps, fabric rustle, breathing, laughter) and ambient sounds (wind, leaves, noise floor) are all summarized into overall_soundscape as one continuous paragraph; dialogue, singing and in‑frame music are described ONLY in integrated_multimodal_description; NEVER define the same sound in both fields.\n\n**Camera Movement Three Elements: describe camera movement with movement type + amplitude + speed (e.g. The camera pushes in with small amplitude at slow speed toward...); use with small amplitude / with large amplitude, at slow speed / at fast speed; medium amplitude and normal speed usually omitted.\n\n**Camera Movement Simplification: H3 has weak control over compound camera movements, must simplify:\n  - Forbid compound movements (e.g. \"pan and push\" \"dolly zoom\"), tends to cause subject exiting frame, background distortion\n  - Prefer fixed camera: let subject's own motion create perspective change (most stable)\n  - Or single movement: keep only one (e.g. \"slowly push forward\" only)\n  - Forbid \"freeze frame\" \"hold frame\" \"freeze on XX\" instructions (H3 doesn't support, ineffective and disrupts motion ending)\n\n**Single‑Shot Perspective Span Compression: perspective change within single shot limited to two levels (e.g. long → medium‑long → medium), FORBID long → medium → close‑up triple jump. Large‑span scenes must be split into multiple shots.\n\n**Character Anchor Specificity: at least 3 specific distinguishing features, forbid vague descriptions:\n  - Correct: \"waist‑length straight black hair\" \"white round‑neck\" \"linen‑cotton dress\"; Wrong: \"black long hair\" \"white dress\" (too coarse, easily drifts under heavy motion)\n  - Forbid \"or\" ambiguity (e.g. \"woods or buildings\" → use definitive: \"blurred tree silhouettes\")\n\n**Style Front‑loading: core visual style (light type & time of day, overall color palette, image clarity) must be placed at the [Shot 1] opening sentence (order: style → scene atmosphere → camera movement → subject appearance), each following shot starts from camera movement; FORBID placing at paragraph end; attention decay at long text end causes style descriptions to be ignored.\n\n**H3 Forbidden Items: forbid percentage coordinates; forbid color ratio numbers; forbid aspect ratio params; forbid micro‑texture demands; forbid bare action sequences without time anchors; forbid writing negative constraints into the positive prompt alone (the node-level negative prompt handles them).\n\nContent order (within integrated_multimodal_description): frame alignment line (first line) → [Shot 1] style & initial composition → reference‑frame anchor continuation → scene lighting & atmosphere → subject appearance (specific anchors) → chronological micro‑motion (At timestamps) → camera movement (single) → dialogue. Output strictly in three-field format, no extra commentary."
            }
        }

        # 全局底层通用规则（三个视频模型共用基础规则）
        self.global_base_rules = {
            "zh": """
你是专业图生视频首帧延续扩写专家，覆盖实拍类、动漫类。
核心铁则：**首帧锚点优先**，不可篡改原图主体位置、服装、发型、核心道具、场景、光影基调；用户引导词仅修饰动态、情绪、运镜，冲突诉求直接舍弃。优先生成微动作，拒绝大幅度位移。
坚守图生视频生成基础约束：动作具备物理运动逻辑；画面精简，不自动新增无关摆件、路人、杂物；文本权重从前向后逐级递减，首帧约束、主体动态、镜头运动前置，细节参数后置；超长文本分段，防止末尾约束失效。
natural模式300‑600字，2‑3个叙事段落，**严禁帧率、码率、分辨率等数字技术参数**。
区分实拍/动漫质感：实拍保留皮肤肌理；动漫保持画风统一，杜绝画风漂移跳变。
完整保留用户输入（首帧图像信息、可选用户引导词），只做动态细节补充，不篡改原图锚点；光影写明光源、色温、软硬以及随时间的变化；严格执行70%主色‑25%辅助‑5%点缀色彩配比。
重要模型差异化约束会在模型组织公式给出，严格遵守对应模型的时长上限、音频支持、负向提示词能力。
输出禁忌：禁止权重符号；禁止猎奇镜头角度；禁止穿模、闪烁、跳帧、风格漂移；禁止元素堆砌；禁止改动首帧核心锚点。
仅输出自然段落格式，不添加额外注释、说明、解释。
""",
            "en": """
You are professional image‑to‑video continue‑frame prompt expert. Cover real‑shot / anime categories.
Core rule: **reference‑frame anchor highest priority**. Must NOT alter original subject position, costume, hairstyle, key props, scene, light‑shadow baseline. User prompt only modify motion, emotion, camera, conflicting requirements shall be discarded. Prefer micro‑movement, forbid large‑scale displacement.
I2V baseline: physically plausible motion logic; frame‑simplify rule, no auto‑add irrelevant ornaments or passers‑by. Text weight decays from front to back: frame‑anchor constraint & character dynamics & camera motion first, details behind. Split long paragraphs to avoid trailing constraint failure.
Natural mode: 150‑300 words, 2‑3 narrative paragraphs, strictly forbid numeric technical parameters like fps, bitrate, resolution.
Texture distinction: real‑shot preserve skin texture; anime keep consistent art‑style, forbid style drifting.
Fully preserve user input(reference frame info & optional user guide prompt), enrich motion details only, never break frame anchor. Describe light source, color‑temperature, hardness‑softness & temporal light change. Enforce 70%‑25%‑5% color proportion rule.
Strictly follow per‑model constraints on max duration, audio capability, negative‑prompt capability given in model formula.
Taboo: no weight syntax; no grotesque camera angles; no penetration‑clipping, flicker, frame‑skip, style drift; no element over‑stacking; no modifying reference‑frame core anchor.
Output natural paragraph format only, no extra comments or explanations.
"""
        }

        # 唯一主预设模板，绑定CONTINUING_I2V template_id
        self.preset_library = {
            "continuing_i2v": {
                "template_id": "continuing_i2v",
                "display_name": CONTINUING_I2V["name"],
                "description": CONTINUING_I2V["description"],
                "positive_constraints": {
                    "zh": "严格延续首帧构图、主体状态、光影基调与风格质感；用户引导词合规需求自然融入动态叙事，不破坏首帧原生锚点；物理级动态与镜头运动逻辑，自然流畅的微动作与环境变化，精准三维镜头视角与合理透视，真实光影过渡与色彩层次统一。实拍类保留原生皮肤纹理、生活化细节与自然光影过渡；动漫类保持统一画风、流畅线条与稳定色块。画面干净主体突出，仅保留核心叙事元素，节奏舒缓可控，动态前后连贯，全程风格统一，光影随时间自然变化，情绪通过微动作与镜头具象传递。",
                    "en": "Strictly inherit reference‑frame composition, subject state, light‑shadow baseline and art‑style. Reasonable user‑guide demand embedded into motion narrative without breaking frame anchor. Physically‑plausible dynamics & camera logic, smooth micro‑actions & environment motion, accurate 3‑d camera perspective, reasonable perspective, natural light‑shadow transition & color hierarchy. Real‑shot: preserve native skin texture, life‑like details. Anime: consistent art‑style, smooth line‑art, stable color block. Clean frame with prominent subject, only core narrative elements. Coherent motion, unified style, light evolves over time, emotion expressed via micro‑action & camera."
                },
                "preset_rules": {
                    "zh": """
【图生视频首帧延续专属规则】
1. 通用基线：执行8步I2V扩写流程；强制70%‑25%‑5%色彩配比；严格锁定首帧全部锚点（主体位置、服装、场景、光影、外形）；三维镜头视角优先继承首帧视角；镜头运动优先固定或极慢运动；画面精简约束，不自动生成多余摆件装饰。natural模式300‑600字无相机数字参数。
2. 实拍类：保留皮肤自然肌理，光影过渡柔和；以微动作为主，拒绝大幅度位移；动作符合人体生理运动逻辑；镜头运动只为叙事，拒绝无意义炫技运镜。
3. 动漫类：画风全程统一，严防画风漂移跳变；线条色块稳定；肢体以微动态为主，规避大幅度位移。
4. 动态设计：优先微动作、微表情；每个动作写明起始‑过程‑结束，尽量标注时间；环境动态与人物动态节奏匹配，禁止凭空新增首帧不存在的核心物体。
5. 用户引导词处理：解析提取诉求，校验是否与首帧锚点冲突；冲突直接舍弃；仅把合规诉求叠加到动态、情绪、运镜，不得修改原图既定状态；无引导词则仅基于首帧生成原生自然延续。
6. 音频处理：模型不支持音频时，完全删除台词、音效、BGM描述；模型支持音频，可合理加入环境音、台词、背景音乐。
7. 负向提示词处理：MiniMax‑H3 的模型文本只含三个核心字段，负向约束由节点级负向提示词输出；Wan2.2、LTX2.5正常输出负向提示。
8. MiniMax‑H3专用规则：
   a. 强制三字段输出：必须输出integrated_multimodal_description / overall_soundscape / non_diegetic_music 三个字段，禁止混写；negative_prompt 不写入模型文本。
   b. 音效分层：人物动作音效（脚步、衣物、呼吸、笑声）与环境音（风声、树叶、底噪）统一汇总到overall_soundscape（1‑4句一个连续段落）；对话、歌声、画内音乐只在integrated_multimodal_description描述；两字段严禁重复定义同一音效。
   c. 时间锚点：每个[Shot N]搭配At时间戳，动作序列必须包含起始+中间+结束3‑4个节点，最后一个时间戳严格对应视频总时长。
   d. 镜头运动简化：禁止复合镜头运动（"平移并推进"等），优先固定机位或单一运动；运镜写明类型+幅度+速度三要素；禁止"定格""冻结帧"指令。
   e. 景别跨度压缩：单镜头景别变化不超过两个层级，禁止三级跳（远景→中景→近景）；大跨度必须拆分多镜头。
   f. 人物锚点：至少3个具体特征，禁止模糊二选一（"树林或建筑"）。
   g. 风格前置：[Shot 1]开头首句声明整体风格与初始构图（风格词：Cinematic/live-action/2D-animated/3D CG等），核心光线、色调、清晰度紧随其后，禁止放在段落末尾。
   h. 禁止量化描述：不得使用百分比坐标、色彩占比数字、画面比例参数，全部改为自然语言相对描述。
   i. 否定转正向+负向提示：正向文本中所有"不出现XX"转换为正向表达；需规避的缺陷交给节点级负向提示词输出。
   j. 信息精简：每字段控制在200‑400字，不堆砌微观纹理、布料动力学等超出模型能力的描述。
所有题材：首帧锚点拥有最高优先级，用户引导词不能篡改原图设定，仅补充动态细节。
""",
                    "en": """
【I2V Reference‑Frame Continue Preset Rules】
1. General baseline: follow 8‑step I2V expansion workflow. Mandatory color ratio 70%‑25%‑5%. Lock all reference‑frame anchor(subject position, costume, scene, light‑shadow, appearance). Inherit camera view from reference‑frame whenever possible. Prefer static or extremely‑slow camera movement. Frame‑simplify rule: no auto‑generate extra ornaments. Natural mode 150‑300 words without numeric camera params.
2. Real‑shot category: preserve natural skin texture, soft light‑shadow transition. Prioritize micro‑actions, forbid large‑scale displacement; human‑physiology‑compliant motion; camera movement serve narrative, avoid meaningless fancy camera work.
3. Anime category: consistent art‑style, strictly prevent style drift; stable lines & color blocks; mostly micro‑motion, forbid large‑scale displacement.
4. Motion design: prioritize micro‑action & micro‑expression. Describe start‑process‑end for each action, mark time span when possible. Sync environment dynamics with character rhythm. Must NOT add core objects which do not exist on reference‑frame.
5. User guide prompt handling: parse requirement, check conflict against frame anchor. Discard conflicting demand. Only inject valid requirement into motion / emotion / camera, never alter original frame setting. If no guide prompt, generate natural continuation purely based on reference frame.
6. Audio handling: remove dialogue / sfx / BGM for audio‑incapable model; add reasonable ambient sound, dialogue, BGM for audio‑capable model.
7. Negative prompt handling: MiniMax‑H3 model text contains only the three core fields; negative constraints come from the node-level negative prompt. Wan2.2 & LTX2.5 output negative prompt normally.
8. MiniMax‑H3 specific rules:
   a. Mandatory three-field output: must output integrated_multimodal_description / overall_soundscape / non_diegetic_music, no mixing; negative_prompt is not written into the model text.
   b. Sound layering: character action sounds (footsteps, clothing, breathing, laughter) and ambient sounds (wind, leaves, noise floor) are all summarized into overall_soundscape (1‑4 sentences as one continuous paragraph); dialogue, singing and in‑frame music are described ONLY in integrated_multimodal_description; never define the same sound in both fields.
   c. Time anchors: every [Shot N] with At timestamp; action sequences must have start+mid+end (3‑4 nodes), final timestamp strictly matching total video duration.
   d. Camera movement simplification: forbid compound movements ("pan and push" etc.), prefer fixed camera or single movement; describe movement with type+amplitude+speed; forbid "freeze frame" "hold frame" instructions.
   e. Perspective span compression: single‑shot change limited to two levels, forbid triple jump (long→medium→close‑up); large span must split into multiple shots.
   f. Character anchors: at least 3 specific features, forbid vague "or" choices ("woods or buildings").
   g. Style front‑loading: [Shot 1] opens with overall style and initial composition (style words: Cinematic/live‑action/2D‑animated/3D CG etc.), followed by core lighting, color palette, clarity; forbid placing at paragraph end.
   h. Forbid quantitative: no percentage coordinates, color ratio numbers, aspect ratio params; rewrite all as natural‑language descriptions.
   i. Negation + negative prompt: convert all "no XX" to positive phrasing in positive text; defects are handled by the node-level negative prompt.
   j. Information density: each field 200‑400 words max; avoid piling micro‑texture, cloth dynamics beyond model capability.
For all categories: reference‑frame anchor has highest priority. User guide shall not alter original frame setting, only enrich motion details.
"""
                },
                "negative_base": {
                    "zh": "首帧风格漂移，元素丢失突变，画面闪烁跳帧，肢体穿模扭曲，人物比例失调，透视逻辑错误，镜头剧烈抖动，运动轨迹混乱，静态呆滞无动态，人脸崩坏变形，五官飘忽不定，色彩溢出脏污，画面颗粒噪点，低分辨率模糊，塑料虚假质感，平涂无体积感，多余路人乱入，杂物堆砌冗余，无关装饰摆件，字幕水印logo，元素凭空消失，物体漂浮无重力，光影生硬突变，节奏混乱突兀，画风前后不一，边缘锯齿抠图感，过度锐化生硬，卡通低幼失真，场景突兀切换，主体位置偏移，光影基调跳变，强行按照用户提示词改动首帧主体、服装、场景、光影等原生设定，生硬植入用户诉求造成画面逻辑断裂",
                    "en": "reference‑frame style drift, element loss & mutation, flicker & frame skip, body clipping & distortion, bad proportion, wrong perspective, violent camera shake, chaotic motion track, static lifeless pose, broken face, drifting facial feature, muddy overflowing color, grain noise, low‑resolution blur, fake plastic texture, flat shading without volume, random extra people, redundant clutter, irrelevant ornament, subtitle watermark logo, object pop‑in pop‑out, zero‑gravity floating object, harsh sudden light‑shadow change, chaotic rhythm, inconsistent art‑style, jagged cut‑out edge, over‑sharpening, overly childish cartoon distortion, abrupt scene switch, subject position offset, light‑shadow baseline jump, forcibly alter reference‑frame subject / costume / scene / light‑shadow by user prompt, break logic by forcing invalid user demand."
                }
            }
        }

        # 双输出格式指引
        self.format_guide = {
            "natural": {
                "zh": """【自然段落模式】自然段落（2‑3段），首段确立首帧状态、整体基调与光线氛围，次段描述动态延续序列与镜头运动节奏，末段补充光影质感、色彩层次与细节氛围。存在用户引导词时将合规诉求自然融入叙事，无引导词则纯基于图像原生动态创作；语言富有画面感与节奏感，全程无帧率、码率、分辨率类数字技术参数，强化动态连贯性、光影流动感与画面精简约束，保证画面干净聚焦、叙事清晰、风格与首帧高度统一。字数300‑600。

**【MiniMax‑H3专用输出格式】若视频模型为MiniMax‑H3，忽略上方自然段落格式，强制按三字段输出：\nintegrated_multimodal_description:\n（画面描述、[Shot N]镜头编号与At时间戳、镜头运动、角色动作序列、台词<d>[语言]…</d>、画内音乐）\n\noverall_soundscape:\n（全片环境音+人物动作音效+非语言人声，1‑4句一个连续段落；完全静音才写N/A）\n\nnon_diegetic_music:\n（画外配乐，1‑3句，描述乐器/速度/节奏/动态变化；无BGM写N/A）\n\n关键约束：I2VA指令行必须放在第一行（"For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."），指令行后空一行；首镜头[Shot 1]无需时间戳（默认起始00:00.000）且开头声明风格与初始构图，后续每个[Shot N]搭配At时间戳，最后一个时间戳严格对应视频总时长；人物动作音效与环境音统一放overall_soundscape，对话/歌声/画内音乐放integrated字段，严禁交叉重复；禁止复合镜头运动优先固定机位，运镜写明类型+幅度+速度；人物锚点至少3个具体特征禁止"或\
"字二选一；禁止百分比坐标/色彩占比数字/画面比例参数；所有"不出现XX"类否定约束交给节点级负向提示词输出，不写入模型文本。""",
                "en": """[Natural Paragraph Mode] 2‑3 narrative paragraphs: first for reference‑frame state‑space‑lighting‑mood; second for character continuation dynamics & camera rhythm; last for light‑shadow‑color‑detail. Embed valid user‑guide demand if exists, generate pure native motion if no guide prompt. Visual cinematic language. Forbid numeric parameters such as fps, bitrate, resolution. Emphasize motion coherence, flowing light‑shadow, frame‑simplify constraint, style consistent with reference frame. 150‑300 words.

**[MiniMax‑H3 Specific Output Format] If the video model is MiniMax‑H3, ignore the natural paragraph format above and output strictly in three‑field structure:\nintegrated_multimodal_description:\n(visual description, [Shot N] labels with At timestamps, camera movement, character action sequence, dialogue <d>[language]…</d>, in‑frame music)\n\noverall_soundscape:\n(all ambient + character action sounds + non‑verbal human sounds, 1‑4 sentences as one continuous paragraph; write N/A only if completely silent)\n\nnon_diegetic_music:\n(off‑screen BGM, 1‑3 sentences describing instruments/tempo/rhythm/dynamics; write N/A if no BGM)\n\nKey constraints: I2VA instruction line must be the first line ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."), followed by a blank line; first shot [Shot 1] has no timestamp (defaults to 00:00.000) and opens by declaring style & initial composition, every subsequent [Shot N] with At timestamp, final timestamp must strictly match total video duration; character action sounds and ambient sounds are all in overall_soundscape, dialogue/singing/in‑frame music in integrated field, cross‑contamination FORBIDDEN; no compound camera movements, prefer fixed camera, describe movement with type+amplitude+speed; at least 3 specific character anchors, forbid \"or\" ambiguity; forbid percentage/ratio/aspect‑ratio parameters; all \"no XX\" negative constraints are handled by the node-level negative prompt, not written into the model text."""
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
        video_model: str,
        output_language: str = "auto",
        enable_global_preconstraint: bool = True,
        enable_negative_prompt: bool = True,
        output_format: str = "natural"
    ) -> Dict:
        valid_video_models = ["Wan2.2", "LTX2.5", "MiniMax‑H3"]
        if preset_name not in self.preset_library:
            raise ValueError(f"预设模板不存在：{preset_name}")
        if video_model not in valid_video_models:
            raise ValueError(f"不支持的视频模型，可选：{valid_video_models}")

        preset = self.preset_library[preset_name]
        model_config = self.video_model_formula_library[video_model]

        if output_language == "auto":
            lang = self.detect_language(user_input)
        else:
            lang = output_language if output_language in ["zh", "en"] else "zh"

        global_rule = self.global_base_rules[lang] if enable_global_preconstraint else ""
        preset_rule = preset["preset_rules"][lang]
        pos_constraint = preset["positive_constraints"][lang]
        formula_hint = model_config[f"formula_zh" if lang == "zh" else "formula_en"]
        natural_guide = self.format_guide["natural"][lang]

        prompt_parts = []
        if enable_global_preconstraint:
            prompt_parts.append(f"【Hard Precondition Baseline】\n{pos_constraint}")
            prompt_parts.append(global_rule)
        prompt_parts.append(f"视频模型【{video_model}】内容组织公式：{formula_hint}")
        prompt_parts.append(preset_rule)
        prompt_parts.append(f"用户原始需求：{user_input}")
        prompt_parts.append(natural_guide)

        final_llm_prompt = "\n".join(prompt_parts)
        # 所有模型均支持负向提示词输出
        if enable_negative_prompt:
            negative_prompt = preset["negative_base"][lang]
        else:
            negative_prompt = ""

        return {
            "status": "success",
            "llm_input_prompt": final_llm_prompt,
            "positive_constraint": pos_constraint,
            "negative_prompt": negative_prompt,
            "output_language": lang,
            "video_model": video_model,
            "preset_name": preset_name,
            "preset_display_name": preset["display_name"],
            "user_raw_input": user_input,
            "enable_preconstraint": enable_global_preconstraint,
            "enable_negative": enable_negative_prompt
        }