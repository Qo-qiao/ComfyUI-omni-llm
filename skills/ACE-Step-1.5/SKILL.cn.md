---
name: "ACE-Step-1.5"
description: "ACE-Step 1.5 提示词编写指南。把一首歌的需求转成 ComfyUI 中 ACE-Step 节点所需的输入：task_type、caption、lyrics、音乐元数据与音频参考说明。按「音乐简报 → 约束解析 → 风格路由 → 渲染 → 验证」流程产出：先提取简报并分类显式/推断值，再按流派路由到风格词表小节，最后渲染 caption 与 lyrics 并通过自检清单。覆盖 Caption 写作原则、Lyrics 结构标记、风格词表路由、一致性自检与各任务的提示词重点；也适用于 ACE-Step 生成、翻唱、局部重绘、加轨场景下的提示词与歌词创作。"
tag-cn: ACE-Step, 音乐, 提示词, 歌词, Caption, 风格路由
---

# ACE-Step 1.5 提示词编写

把一首歌的需求转成 ComfyUI 里 ACE-Step 节点可直接使用的输入。本 skill **只负责提示词与输入创作**：caption、lyrics、音乐元数据、任务类型与音频参考的选择；不涉及模型下载和推理调参。

全程只用自然语言推理与本 skill 的本地文本文件。不要执行脚本、构建数据库、调用外部 API，也不要逐个读取全部风格词表小节。

## 一、输出契约

每次交付以下内容，字段名与节点输入对应：

1. **task_type**：text2music / cover / repaint / lego / extract / complete
2. **caption**：音乐整体画像（风格、情绪、乐器、音色、人声、制作、结构提示）
3. **lyrics**：时间脚本（纯音乐填 `[Instrumental]`）
4. **音乐元数据**（仅在用户有明确要求或任务需要时给出）：`bpm`、`keyscale`、`timesignature`、`vocal_language`、`duration`
5. **音频参考说明**：是否需要 `reference_audio` / `src_audio`，各自控制什么

**语言**：无特别指定时 caption 用英文（教程示例词表均为英文、风格词最密集）；歌词语言跟随需求（中文需求写中文歌词）；用户明确指定语言时遵从用户。

默认对话模式分段呈现（解析 → 可直接复制的 caption/lyrics 代码块 → 微调建议）；用户要求"JSON/API 格式"时，只输出单行 JSON：`{"task_type": "...", "caption": "...", "lyrics": "...", "bpm": null, "keyscale": null, "timesignature": null, "vocal_language": null, "duration": null}`。

除非用户要求诊断，不展示音乐简报、路由选择或词表小节名。

## 二、工作流

按顺序执行这五个阶段：

1. **构建音乐简报**（第三节）：从输入提取有依据的值并分类。
2. **解析约束**（第四节）：确定优先级，锁定不可反转的显式要求。
3. **风格路由**：读 [references/style-router.md](references/style-router.md) → 选定主家族（仅显式融合时 +1 个副家族，最多两个）→ 按 `load_references` 请求 [references/style-vocabulary.md](references/style-vocabulary.md)，**只使用路由命中的小节**。
4. **渲染**：写 caption 与 lyrics（读 [references/caption-and-lyrics.md](references/caption-and-lyrics.md)）、按需填元数据并判断音频参考（读 [references/inputs-and-metadata.md](references/inputs-and-metadata.md)）。
5. **验证后交付**：过第六节清单；任一项失败先修订一次，只返回修订后的结果。

## 三、构建音乐简报

从输入中只提取有依据或可合理推断的值：

- 宏观流派、子流派与文化/市场风格
- 情绪与情感弧线
- 大致速度感与律动（不发明具体 BPM）
- 人声有无、性别、音色与演绎方式
- 核心乐器与制作质感
- 段落结构与段落变化
- 空间特征与显式排除项

内部把每个值归类为 `explicit`（用户明说）、`inferred`（合理推断）或 `unspecified`（未提供）。

- 当更宽泛的描述已足够时，不要发明精确调性、BPM、人声音域或制作术语。
- 明确的器乐请求保持器乐，不加人声。
- 人声未指定时，采用用户描述与最近风格家族都支持的保守处理。

## 四、约束优先级

按此顺序取舍：

1. 用户显式要求与排除项
2. 歌词方括号标签的段落局部指令（仅在该段落内生效）
3. 用户描述中的强暗示
4. 选中风格词表的特性
5. 保守的音乐默认值

段落标签只改变该段落的局部编排，不替换全局风格；与用户硬性排除项冲突时保留排除项。两个显式指令冲突时，若意图仍清晰，取更具体且靠后的那个，否则做最小的音乐性妥协。**永不静默反转显式的性别、器乐要求、速度限制、必需乐器或禁止元素。**

## 五、任务类型对字段的要求

| 需求 | 任务 | 提示词重点 |
|------|------|-----------|
| 从文字生成一首歌 | `text2music` | caption + lyrics 全自由 |
| 保持结构、换风格/换词（翻唱、Remix、抽卡） | `cover` | 结构由源音频锁定；caption 写**目标**风格，lyrics 可整体替换 |
| 局部改词/改结构/续写（3–90 秒区间） | `repaint` | 只写区间内该发生什么，与上下文风格保持一致 |
| 给已有轨道加乐器 | `lego` | 描述新增轨的乐器与角色 |
| 分离音轨 | `extract` | 无需提示词 |
| 给单轨加伴奏 | `complete` | 描述伴奏的乐器与风格 |

## 六、编写原则

- **Caption 是影响生成的最重要输入**：具体优于模糊，组合风格+情绪+乐器+音色多维度，描述粒度决定自由度（想可控就写详细，想要惊喜就留白）。
- **词表是原料不是清单**：从路由小节挑 3–6 个维度组合，不整段堆砌，不照抄样例句子；简报与词表冲突时以简报为准。
- **避免冲突词汇**：冲突风格（如"古典弦乐"+"硬核金属"）会导致劣化输出。解法：重复强化想要的元素，或把冲突写成**时间上的风格演变**（开头柔和弦乐 → 中段金属 → 结尾 hip-hop）。
- **Caption 与 Lyrics 必须一致**：模型不擅长解决冲突。Caption 写 violin solo，Lyrics 就不要写 `[Guitar Solo]`；人声、情绪、乐器三条线都要对上。
- **结构标记简洁**：`[Chorus - anthemic]` 用 `-` 组合即可，不堆叠多个标记（会被当歌词唱出或让模型困惑）。复杂风格描述放 caption。
- **元数据是引导不是精确指令**：模型把它当锚点在附近采样（设 120 可能得到 118）；数值 BPM/调性/拍号只进元数据参数，caption 只写定性的速度感与律动词。
- **音频能控制的不硬塞文字**：音色、混音、演奏质感交给 `reference_audio`；旋律、和弦、结构交给 Cover 的 `src_audio`——caption 把省下的篇幅花在风格、情绪、编排上。

## 七、自检清单

返回前逐项验证：

- [ ] 每条显式用户约束与排除项都被保留
- [ ] 器乐请求保持器乐；人声性别、必需乐器、速度限制未被静默反转
- [ ] caption 无风格冲突词；冲突已改写成演变或用重复强化
- [ ] caption 的乐器 ↔ lyrics 的器乐标记；caption 的情绪 ↔ 能量标记；caption 的人声 ↔ 人声标记
- [ ] 每行歌词 6–10 音节，同位置行音节数接近（±1–2）
- [ ] 结构标记未堆叠，段落间有空行；可操作的段落标签落在对应段落里
- [ ] BPM/调性/拍号没有出现在 caption；没有发明未被要求的精确元数据
- [ ] 没有整句照抄词表样例；caption 足够具体但不写成论文
- [ ] 与源音频（cover/repaint）的风格、速度不矛盾
- [ ] 歌词无形容词堆砌、押韵混乱、段落串味、单一核心隐喻

任一项失败：修订一次，然后只返回修订后的结果。

## 八、参考文档

- **[style-router.md](references/style-router.md)** — 风格家族路由：契约、家族表、中英别名、融合规则、兜底路由、字段分派
- **[style-vocabulary.md](references/style-vocabulary.md)** — 18 个家族的 caption 词表：术语、情绪、乐器、人声、结构提示与单行样例
- **[caption-and-lyrics.md](references/caption-and-lyrics.md)** — Caption 写作维度与七条原则、结构/人声/能量标记表、歌词技巧、避免 AI 味、完整示例
- **[inputs-and-metadata.md](references/inputs-and-metadata.md)** — 输入字段表、各任务的提示词重点、元数据规则、音频条件对提示词的影响

先读 `style-router.md` 再按需读 `style-vocabulary.md`；回答具体 caption、歌词、元数据问题前，先通过 `load_references` 请求对应 reference 再作答，不要凭记忆猜测文件内容。
