---
name: h3-prompt-writing-zh
description: 为 MiniMax H3 视频生成编写提示词，支持 T2VA、I2VA、FL2VA、L2VA 和 Ref2VA 模式。当需要将多模态请求改写为 H3 提示词结构、编写 integrated_multimodal_description、overall_soundscape 和 non_diegetic_music、对齐关键帧、或为图像、视频和音频定义参考标签时使用。
compatibility: 可移植到任何能够读取本地文件的代理——无需外部 API 调用、MiniMax Hub 工具或专有运行时。agents/openai.yaml 文件仅添加可选的 ChatGPT/Codex UI 元数据；不会将技能限制为 OpenAI 代理。
---

# H3 提示词编写指南

## 工作流程

1. 确定输入模式：T2VA、I2VA、FL2VA、L2VA 或全参考 Ref2VA。
2. 对于基础文本/关键帧模式，阅读 `references/base-zh.txt` 并遵循其最终提示词结构。
3. 对于全参考模式，阅读 `references/ref-zh.txt` 并遵循其六个部分的改写格式。
4. 保留所选指南中的确切字段名称、部分顺序、标签和时间标记。

## 基础模式

- T2VA：从文本构建完整的时间线。
- I2VA：从第一帧开始向前发展。
- FL2VA：描述第一帧和最后一帧之间的连续路径。
- L2VA：推断合理的开头，然后收敛到提供的最后一帧。

按照 `references/base-zh.txt` 中显示的顺序使用 `integrated_multimodal_description`、`overall_soundscape` 和 `non_diegetic_music`。

## 全参考模式

Ref2VA 改写按顺序使用 `subject_definitions`、`summary`、`retention_analysis`、`detailed_description`、`overall_soundscape` 和 `non_diegetic_music`。参考标签在所有部分中保持一致。

阅读 `references/ref-zh.txt` 了解标签规则、保留分析和完整示例。

## 输出规则

- 用英文编写改写部分；保留对话、歌词和可见场景文本的原始语言。
- 通过构图、主体、环境、动作、镜头、声音以及引用内容出现的确切位置来描述每个镜头。
- 避免剧情摘要、未解决的参考标签以及与请求时长不匹配的时间。
## 提升效果的技巧
- 始终使描述的总时长与请求的视频长度（4-15秒）匹配。
- 保持参考标签一致（例如 `<Picture 1>`、`<Video 1>`、`<Audio 1>`）在每个部分中。
- 优先使用具体的视觉和音频细节，而不是"电影感"或"美丽"等抽象词汇。
- 使用关键帧（I2VA / FL2VA / L2VA）时，清楚说明第一帧和/或最后一帧如何连接到时间线。