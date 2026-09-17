---
name: "yue2-music"
description: "使用 YuE2 和 SheetSage2/MERT2 生成、翻唱、转录和编辑歌曲。支持 YuE2 full/melody/off 生成、音频转ABC翻唱、风格或歌词修改、乐谱编辑、智能体重和声、旋律保留、可唱歌词适配和可重复的试听对比；也适用于 YuE2 生成、翻唱、改编、改谱、换词和智能体编辑。"
tag-cn: 音乐, 生成, 翻唱, 编辑, YuE2
---

# YuE2 音乐提示词编写

将音乐需求转化为可重复的歌曲和可听的对比。使用已发布的模型接口。在进行更改前保留原始歌曲及其计划。

## 一、选择工作流

| 需求 | 工作流 |
|------|--------|
| 生成可编辑的旋律和和声 | YuE2 `cot="full"` → ABC → 歌曲 |
| 生成带旋律计划的自由伴奏 | YuE2 `cot="melody"` → 无和弦ABC → 歌曲 |
| 无符号规划生成 | YuE2 `cot="off"` → 歌曲；无可编辑ABC |
| 翻唱录音 | SheetSage2 → 检查/修正ABC → 去除和弦 → YuE2 `melody` |
| 翻唱ABC旋律 | 检查/转换原生ABC → 去除和弦 → YuE2 `melody` |
| 修改和声、乐器、速度、结构或歌词 | 复制完整计划 → 编辑ABC/文本 → 重新生成 |
| 智能体编辑 | 导出计划/基线 → 有界编辑智能体 → 检查不变量 → 渲染 → 对比 |
| 分析音乐特征 | 仅在需要连续特征时使用 MERT2 |

```
音频 → SheetSage2 [自动加载 MERT-v2-FullSong] → ABC
风格 + 歌词 → YuE2 full/melody 规划              → ABC
                                              编辑/验证
风格 + 歌词 + ABC → YuE2 语义生成 → 合成 → 潜变量 → VAE → 歌曲
风格 + 歌词      → YuE2 off 生成  → 合成 → 潜变量 → VAE → 歌曲
```

不要将公共 MERT 特征张量作为编解码器令牌馈送给 YuE2。YuE2 不暴露音频参考、音素对齐或局部修复参数。

## 二、设置所需模型

阅读 [references/models-and-setup.md](references/models-and-setup.md)。从官方 GitHub 仓库源安装 YuE2 运行时。由于依赖版本不同，SheetSage2 需要使用单独的环境。下载公共模型快照并记录其修订版本。

使用支持的基线：一次一个请求，支持 BF16 的 NVIDIA GPU，24GB 显存，默认 YuE2 设置。不要静默缩短请求的歌曲或降低推理设置以隐藏 OOM。释放分配或选择合适的硬件；报告更改。

使用 `YuE2-Vae` 进行试听，使用 `YuE2-Vae-legacy` 重现提供的基准协议。保持解码文件分离。

## 三、生成并保留计划

从 [assets/prompt.json](assets/prompt.json) 开始，这是一个原始示例。在 `style` 中放入流派、乐器、人声特征、语言和预期速度；在 `lyrics` 中放入章节标签和实际歌词。将实现笔记排除在歌词之外。

```bash
python scripts/run_yue2.py generate --request assets/prompt.json --output outputs/pop
python scripts/run_yue2.py all-modes --request assets/prompt.json --output outputs/modes
python scripts/run_yue2.py plan --request assets/prompt.json --output outputs/plan
```

检查 `result.json`、截断、`score.abc`、`request.json` 和音频。保留精确的令牌和 `latent.npy`；助手保存原生工件。保留所有请求的模式和失败。

阅读 [references/generation-and-covers.md](references/generation-and-covers.md) 了解 Python、CLI、精确计划继续、CFG、采样和缓存解码。

## 四、翻唱录音

1. 在 SheetSage2 环境中转录。选择人声旋律或完整主旋律，包括器乐段落。
2. 检查警告并修正遗漏的音符、节拍或调号，然后将错误归因于 YuE2。保留源音频和原始转录。
3. 导出无和弦ABC。在删除部分时明确选择保留的声部；仅移除和弦应保留两个旋律声部及其休止符。
4. 使用 `cot="melody"`、目标风格和合适的歌词渲染。这提供符号旋律条件；不保留源歌手的身份或波形。

```bash
# SheetSage2 环境
python scripts/transcribe.py reference.wav --task melody-full --output outputs/transcription
python scripts/abc_tools.py strip-chords outputs/transcription/score.abc outputs/cover.abc

# YuE2 环境；请求提供目标风格和歌词
python scripts/run_yue2.py generate --request assets/prompt.json --cot melody \
  --abc-file outputs/cover.abc --output outputs/cover-song
```

## 五、编辑或委托编辑

阅读 [references/editing-workflows.md](references/editing-workflows.md) 和 [references/abc-editing.md](references/abc-editing.md) 再更改乐谱。

1. 从完整计划渲染基线。冻结其原始目录。
2. 定义不变量：精确音高；音高加节奏；仅轮廓；或有界旋律适配。指定声部/段落、歌词、乐器、速度、节拍和结构。
3. 如果可用委托，向乐谱编辑智能体提供原始ABC、提示词、歌词、请求的更改和 [edit brief](assets/edit-brief.md)。请求新的ABC、修订的风格/歌词（如需），和编辑清单。给独立审查者提供前后工件和约束。
4. 检查音乐事件，而非字符字符串：连线、变音记号和压缩休止符很重要。
5. 更改风格、歌词或ABC后重新生成。旧的声学潜变量可以再次解码，但不能实现音乐或歌词编辑。
6. 对比完整歌曲和编辑附近的短段落。当请求的效果失败时修订；保留每次尝试及其实际提示词。

## 六、交付可听结果

阅读 [references/listening-and-evaluation.md](references/listening-and-evaluation.md)。返回可播放音频、完整提示词/歌词、前后ABC、不变量检查和请求的评估。保持模型/解码器身份和失败可见。

```bash
python scripts/listen.py outputs/pop outputs/jazz --output outputs/comparison
```

这会创建本地HTML播放器，复制音频，并包含精确的请求。不发布或上传。

## 七、参考文档

- **[models-and-setup.md](references/models-and-setup.md)** — 模型、设置和音频到乐谱桥接
- **[generation-and-covers.md](references/generation-and-covers.md)** — 生成、翻唱和可重用阶段
- **[editing-workflows.md](references/editing-workflows.md)** — 音乐编辑和智能体委托
- **[abc-editing.md](references/abc-editing.md)** — 编辑乐谱同时保留音乐含义
- **[listening-and-evaluation.md](references/listening-and-evaluation.md)** — 试听交付和可重复评估

## 八、安全与合规

- 模型权重为 CC BY-NC 4.0。技能许可证不重新许可这些权重或移除非商业条款。
- 代码和依赖保留其适用条款。
- 链接到每个模型的 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`；不要将模型权重、认证材料、缓存数据集或不相关示例捆绑到技能存档中。
