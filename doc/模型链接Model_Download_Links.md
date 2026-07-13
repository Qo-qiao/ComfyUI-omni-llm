## 中文版
\### 量化级别选择

\- \*\*Q4\_K\_M\*\*：平衡大小和质量（推荐）
\- \*\*Q5\_K\_M\*\*：更高质量，稍大文件
\- \*\*Q3\_K\_M\*\*：更小文件，适合低显存设备
\- \*\*Q2\_K\*\*：最小文件，质量较低


\### 模型准备

1\. \*\*创建模型目录\*\*：

&nbsp;  - 在 `ComfyUI/models/` 目录下创建 `LLM` 文件夹
&nbsp;  - 将下载的模型文件放入此目录，如果你下载多个模型，建议在LLM文件夹内创建子文件夹存放，便于区分


2\. \*\*模型文件格式\*\*：


### 配套文件下载指南

#### 必需配套文件说明

**1. 主模型文件**
- `.gguf` 格式的主模型文件
- 文件名通常包含模型名称和量化级别（如 `Qwen2.5-Omni-7B-Q4_K_M.gguf`）

**2. 视觉配套文件（mmproj）**
- 用于图像和视频处理的视觉投影器
- 文件名通常包含 `mmproj` 关键字
- **必需场景**：图片反推、视频帧处理、OCR识别
- **示例文件名**：`mmproj-model-f16.gguf`

**3. 音频配套文件**
- 推荐使用提供的模型
- 用于音频生成和处理的专用模型，包括语音合成（TTS）、语音识别（ASR）等功能
- 文件名通常包含`tts`、`asr` 关键字
- **必需场景**：语音合成、音频生成、音频理解、语音识别
- **示例文件名**：`Qwen3-TTS-12Hz-1.7B-CustomVoice`

#### .safetensors 模型配套文件

**必需配置文件**（模型无法正常加载）：
- `config.json` - 模型架构配置
- `tokenizer_config.json` - 分词器配置
- `tokenizer.json` - 分词器定义

**可选配置文件**（影响特定功能）：
- `merges.txt` - BPE合并规则（文本生成质量）
- `vocab.json` - 词汇表（多语言支持）
- `special_tokens_map.json` - 特殊标记映射
- `added_tokens.json` - 自定义标记
- `chat_template.json` - 聊天模板（对话质量）
- `generation_config.json` - 生成参数配置
- `preprocessor_config.json` - 预处理配置（图像处理）
- `spk_dict.pt` - 说话人字典（音频生成必需）

#### 配套文件下载方法

**从 Hugging Face 仓库下载**

1. 访问模型的主仓库页面
2. 在文件列表中查找配套文件
3. 下载所有必需的配置文件
4. 将文件放在与主模型相同的目录中


### 主模型文件名保留规则

**模型类型识别依赖**：
- 插件通过主模型文件名中的关键词自动识别模型类型（如 "qwen3.5"、"minicpm-o"、"dreamomni2" 等）
- 文件名中的关键词直接影响模型功能的启用（如音频支持、视觉支持等）
- 重命名主模型文件可能导致模型类型识别错误，从而影响功能使用

**推荐做法**：
- 保持主模型文件的原始名称，不要移除或修改其中的模型标识关键词
- 例如：`Qwen2.5-Omni-7B-Q4_K_M.gguf`、`MiniCPM-o-4_5-Q4_K_M.gguf` 等
- 如果需要重命名，确保保留核心模型标识关键词

### 注意事项
1. **文件名区分大小写**：请保持原始大小写，特别是模型标识关键词
2. **不要重命名配套文件**：否则模型可能无法找到对应的组件
3. **主模型文件名建议**：尽量保持原始名称，确保模型类型正确识别
4. **版本匹配**：不同版本的模型可能需要不同的配套文件，请勿混用
5. **目录结构**：配套文件必须与主模型文件放在同一目录下


## ASR系列（语音识别）

### Qwen系列（只支持提供的模型）

**模型系列**：Qwen-ASR
**具体型号**：Qwen3-ASR-0.6B
**核心特点**：语言覆盖：非常广泛，支持 28+ 种语言 及 20+ 种中国方言（如东北话、四川话、粤语、吴语、闽南语等）。
**音频类型**：不仅能处理纯语音，还支持歌声和带背景音乐的歌曲（抗噪/音乐场景能力强）。
**适配场景**：语音识别、实时语音转文字、会议记录、音频内容提取、字幕生成
**下载链接**：https://huggingface.co/Qwen/Qwen3-ASR-0.6B
**魔搭链接**：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B


**模型系列**：Qwen-ASR
**具体型号**：Qwen3-ASR-1.7B
**核心特点**：语言覆盖广泛，支持 28+ 种语言及 20+ 种中国方言。相比0.6B版本，识别精度更高，对低质量音频和复杂口音的处理能力更强，支持更长音频的连续识别。
**音频类型**：支持纯语音、歌声和带背景音乐的歌曲（抗噪/音乐场景能力强）。
**适配场景**：高精度语音识别、实时语音转文字、会议记录、音频内容提取、字幕生成
**下载链接**：https://huggingface.co/Qwen/Qwen3-ASR-1.7B
**魔搭链接**：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B



**模型系列**：Qwen-ASR
**具体型号**：Qwen3-ForcedAligner-0.6B
**核心特点**：强制对齐。它不是做语音识别，而是将给定的文本与音频进行时间戳对齐，输出每个音素、单词或句子在音频中的起止时间。
**语言覆盖**：仅支持 11 种语言（中、英、粤、法、德、意、日、韩、葡、俄、西），不支持方言。
**音频类型**：仅限纯净语音（不适合歌声或带背景音乐的歌曲）。
**适配场景**：字幕生成
**下载链接**：https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
**魔搭链接**：https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B


## TTS系列（语音合成）

### Qwen系列（只支持提供的模型）


**模型系列**：Qwen-TTS
**具体型号**：Qwen3-TTS-12Hz-1.7B-CustomVoice
**核心特点**：侧重预设与控制。它提供了9种固定的优质音色（覆盖不同性别、年龄、语言和方言），你可以通过指令在这些预设音色的基础上调整说话风格（如语速、情绪等）。
**适配场景**：定制语音合成、个性化语音助手、有声读物、品牌语音形象、角色配音
**下载链接**：https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
**魔搭链接**：https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice


**模型系列**：Qwen-TTS
**具体型号**：Qwen3-TTS-12Hz-1.7B-VoiceDesign
**核心特点**：侧重创造性。它不依赖固定音色，而是根据你输入的文字描述（如“温柔的中年男性”、“带有金属质感的年轻女声”）来实时生成符合描述的音色。
**适配场景**：语音参数调节、情感语音合成、语音设计、交互式语音应用、游戏角色配音
**下载链接**：https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
**魔搭链接**：https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign



## VLM系列（视觉语言模型，支持所有视觉模型文本生成和图像反推）

### ToriiGate系列（XL动漫专用）
**模型系列**：ToriiGate
**具体型号**：ToriiGate-v0.4-7B
**核心特点**：基于 Qwen2-VL 微调，专注动漫和数字艺术领域的高级理解。支持结构化输出、多种 caption 模式（常规摘要、Midjourney 风格、DeviantArt 风格）、Bounding Box 定位、角色名称识别。描述精确、密度高，避免冗长填充词。
**适配场景**：动漫/数字艺术图像描述、角色特征提取、多角色识别、提示词生成、caption 修正与裁剪
**下载链接（nsfw）**：https://huggingface.co/mradermacher/ToriiGate-v0.4-7B-GGUF（含mmproj模型）


### Qwen系列（建议根据clip模型选择提示词生成模型，中文提示词推荐qwen、MiMo系列，英文提示词推荐Llama系列，大显存推荐使用大模型）

**模型系列**：Qwen-VL
**具体型号**：Qwen2.5-7B-VL
**核心特点**：阿里通义千问多模态系列，全尺寸覆盖，多语言OCR与图文推理均衡。
**适配场景**：多语言OCR、图片/视频帧反推、提示词生成、常规图文问答、批量文本提取
**下载链接（nsfw）**：https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF（含mmproj模型）
**魔搭链接**：https://www.modelscope.cn/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF（含mmproj模型）


**具体型号**：Qwen3-VL-8B-Instruct（文本生成与反推，推荐下载）
**核心特点**：Qwen3-VL指令优化款，指令遵循精度最高，支持复杂视觉指令执行（如“提取图像中所有表格并生成文本”）。
**适配场景**：指令型OCR提取、复杂图文反推、专业视觉任务处理、高精度提示词生成
**下载链接（nsfw）**：https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF（含mmproj模型）
**备用链接（nsfw）**：https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive（含mmproj模型）
**魔搭链接**：https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-GGUF（含mmproj模型）


**具体型号**：Qwen3-VL-8B-maid
**核心特点**：Qwen3-VL-8B模型女仆版本
**下载链接**：https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-i1-GGUF（含mmproj模型）
**备用链接**：https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-GGUF（含mmproj模型）
**魔搭链接**：https://www.modelscope.cn/models/Zyi4082/Qwen3-VL-8B-Maid-GGUF（含mmproj模型）

### qwen3.5、qwen3.6等思考类模型使用文本生成模式必须启用加载mmproj模型

**具体型号**：Qwen3.5-4B
**核心特点**：Qwen3.5系列的轻量版本，参数量适中，兼顾推理能力与资源消耗，支持多模态输入和工具调用，响应速度快，适合低显存设备使用。
**适配场景**：低显存设备部署、快速推理、移动端应用、轻量级对话、基础多模态任务、实时交互
**下载链接（nsfw）**：https://huggingface.co/mradermacher/Qwen3.5-4B_Abliterated-GGUF（含mmproj模型）
**备用链接（nsfw）**：https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive（含mmproj模型）


**具体型号**：Qwen3.5-9B
**核心特点**：Qwen3.5系列的平衡版本，参数量适中，兼顾推理能力与资源消耗，支持多模态输入和工具调用，适合中等显存设备。
**适配场景**：中等显存设备部署、高质量多模态推理、复杂图文对话、图像理解分析、视频帧内容识别
**下载链接（nsfw）**：https://huggingface.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF（含mmproj模型）
**备用链接（nsfw）**：https://huggingface.co/llmfan46/Qwen3.5-9B-ultra-uncensored-heretic-v2-GGUF（含mmproj模型）

### Qwen3.5变体系列

**具体型号**：Qwen3.5-9B-DeepSeek-V4-Flash
**核心特点**：Qwen3.5系列的平衡版本，参数量适中，兼顾推理能力与资源消耗，支持多模态输入和工具调用，适合中等显存设备。
**适配场景**：中等显存设备部署、高质量多模态推理、复杂图文对话、图像理解分析、视频帧内容识别
**下载链接**：https://huggingface.co/Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF（含mmproj模型）


### Qwen3.6系列

**具体型号**：Qwen3.6-27B（大显存专用）
**核心特点**：Qwen3.6系列的高性能版本，参数量大、推理能力强，支持复杂多模态任务和长上下文处理，适合大显存设备部署。
**适配场景**：大显存设备部署、复杂图文推理、专业图像分析、长文本理解、视频内容深度分析、高精度生成任务
**下载链接（nsfw）**：https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive（含mmproj模型）


**具体型号**：Qwen3.6-35B-A3B（大显存专用）
**核心特点**：Qwen3.6系列的MoE版本，总参数量35B但仅3B活跃参数参与推理，兼顾高性能与推理效率，A3B架构优化了显存使用效率，支持复杂多模态任务。
**适配场景**：中等显存设备部署、高质量多模态推理、复杂图文分析、长文本理解、视频内容深度分析
**下载链接（nsfw）**：https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive（含mmproj模型）



### OCR专项系列

**模型系列**：Unlimited-OCR
**具体型号**：Unlimited-OCR
**核心特点**：百度新优化模型，基于OCR专项微调，对公式、表格、老旧扫描件识别能力突出，文本提取精度高，无需额外配置即可使用，本地推理效率优异。
**适配场景**：海报文字识别、文档OCR、学术论文公式识别、批量扫描件文本提取、表单信息提取、古籍数字化
**下载链接**：https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF（含mmproj模型）


### MiniCPM系列

**具体型号**：MiniCPM-V-4.6-abliterated-MAX
**核心特点**：MiniCPM-V-4.5的升级版本，视觉识别精度进一步提升，对复杂场景和小目标的识别能力更强，推理速度优化，支持更长上下文的多模态理解。
**适配场景**：高精度提示词生成、图片/视频帧精细反推、多语言OCR、复杂场景图像分析、日常视觉推理
**下载链接（nsfw）**：https://huggingface.co/mradermacher/MiniCPM-V-4.6-abliterated-MAX-GGUF（含mmproj模型）


### GLM系列

**模型系列**：GLM-4
**具体型号**：GLM-4.6V-Flash-abliterated
**核心特点**：智谱大参数量多模态模型，OCR与复杂文档解析能力顶级，支持高分辨率图像识别，本地推理稳定，适配复杂视觉任务。
**适配场景**：专业OCR文本提取、复杂文档/表格识别、学术论文解析、高精度图片反推、多语言图文推理
**下载链接（nsfw）**：https://huggingface.co/seanbailey518/Huihui-GLM-4.6V-Flash-abliterated-GGUF（含mmproj模型）


### JoyCaption系列

**模型系列**：JoyCaption
**具体型号**：llama-joycaption（反推）
**核心特点**：ComfyUI专用提示词生成/图像反推模型，细节捕捉能力强，生成的提示词贴合创作需求。
**适配场景**：ComfyUI提示词生成、图片精准反推、艺术创作类图文关联、批量图像标签生成
**下载链接（nsfw）**：https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF
**mmproj模型**：https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf


### Gemma系列

**模型系列**：Gemma
**具体型号**：google_gemma-3-4b-it
**核心特点**：Google Gemma 3系列多模态模型，具备视觉-语言理解、长上下文处理能力，以及更强的多语言支持能力，单块GPU即可运行，性能超越同参数量模型。
**适配场景**：多语言提示词生成、复杂视觉推理、长文本理解、多模态交互任务
**下载链接**：https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF（含mmproj模型）
**备用链接（nsfw）**：https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF


**具体型号**：Gemma-4-E2B-Uncensored-HauhauCS-Aggressive
**核心特点**：Google Gemma 4系列无审查版本，采用更先进的架构设计，在视觉理解、语言生成和多模态推理方面有显著提升，参数量适中，适合单GPU部署，支持生成更开放的内容。
**适配场景**：高级视觉推理、多语言对话、创意内容生成、复杂多模态任务、需要无审查内容的场景
**下载链接（nsfw）**：https://huggingface.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive（含mmproj模型）


**具体型号**：Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
**核心特点**：Google Gemma 4系列无审查旗舰模型，具备更强的视觉理解能力、更长的上下文处理能力和更准确的多语言支持，性能接近更大规模模型，支持生成更开放的内容。
**适配场景**：复杂视觉推理、长文本理解、多语言翻译、高级多模态交互、需要无审查内容的场景
**下载链接（nsfw）**：https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive（含mmproj模型）


**具体型号**：Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced（大显存专用）
**核心特点**：Google Gemma 4系列的MoE版本，总参数量26B但仅4B活跃参数参与推理，兼顾大模型性能与推理效率，A4B架构优化了显存使用，支持复杂视觉推理和长上下文处理，支持生成更开放的内容。
**适配场景**：高质量视觉推理、长文本理解、多语言翻译、高级多模态交互、需要无审查内容的复杂任务
**下载链接（nsfw）**：https://huggingface.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced（含mmproj模型）


### Phi视觉系列

**模型系列**：Phi-Vision
**具体型号**：Phi-3.5-vision-instruct
**核心特点**：微软迭代版多模态模型，多图/视频帧序列反推优化，指令遵循精度高，适配ComfyUI视频帧处理节点。
**适配场景**：视频帧反推、多图对比反推、指令型提示词生成、常规OCR文本提取
**下载链接**：https://huggingface.co/abetlen/Phi-3.5-vision-instruct-gguf（含mmproj模型）


### LLaMA视觉系列

**模型系列**：LLaMA-Vision
**具体型号**：Llama-3.2-11B-Vision-Instruct-abliterated
**核心特点**：Meta官方多模态模型，全场景均衡，复杂视觉推理能力强，支持高分辨率图像识别，适配ComfyUI各类视觉任务。
**适配场景**：高精度图片反推、复杂文档OCR、视频帧关键信息提取、专业级视觉推理
**下载链接（nsfw）**：https://huggingface.co/case01/Llama-3.2-11B-Vision-Instruct-abliterated-gguf
**mmproj模型**：https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF


### MiMo-VL系列（请使用自定义模板）

**模型系列**：MiMo-VL
**具体型号**：MiMo-VL-7B-RL-2508
**核心特点**：小米开源多模态视觉语言模型，7B参数，基于强化学习训练，四阶段预训练（投影器预热、视觉-语言对齐、通用多模态预训练、长上下文监督微调），小参数量下性能超越百亿级模型。
**适配场景**：中英双语对话、复杂视觉推理、多模态交互、长上下文理解、创意提示词生成
**下载链接**：https://huggingface.co/mradermacher/MiMo-VL-7B-RL-2508-GGUF（含mmproj模型）


### Nex-N2系列（暂不可用）

**模型系列**：Nex-N2
**具体型号**：Huihui-Nex-N2-mini-abliterated-APEX
**核心特点**：基于Nex-N2-mini多模态模型微调的无审查版本，具备优秀的视觉理解能力和图文推理能力，支持图像描述、OCR文本提取和多模态交互，推理速度快，适合日常视觉任务使用。
**适配场景**：图片反推提示词、OCR文本提取、图文问答、日常视觉推理、多模态交互任务
**下载链接（nsfw）**：https://huggingface.co/SC117/Huihui-Nex-N2-mini-abliterated-APEX-GGUF（含mmproj模型）



## english version

### Quantization Level Selection

- **Q4_K_M**: Balanced size and quality (Recommended)
- **Q5_K_M**: Higher quality, slightly larger file
- **Q3_K_M**: Smaller file, suitable for low VRAM devices
- **Q2_K**: Smallest file, lower quality


### Model Preparation

1. **Create Model Directory**:

   - Create `LLM` folder under `ComfyUI/models/` directory
   - Place downloaded model files in this directory. If you download multiple models, it is recommended to create subfolders within the LLM folder for better organization


### Companion File Download Guide

#### Required Companion Files

**1. Main Model File**
- Main model file in `.gguf` or `.safetensors` format
- File name usually contains model name and quantization level (e.g., `Qwen2.5-Omni-7B-Q4_K_M.gguf`)

**2. Visual Companion File (mmproj)**
- Visual projector for image and video processing
- File name usually contains `mmproj` keyword
- **Required Scenarios**: Image reverse prompting, video frame processing, OCR recognition
- **Example File Name**: `mmproj-model-f16.gguf`

**3. Audio Companion Files**
- Specialized models for audio generation and processing, including Text-to-Speech (TTS), Automatic Speech Recognition (ASR), etc.
- File names usually contain `tts`, `asr` keywords
- **Required Scenarios**: Speech synthesis, audio generation, audio understanding, speech recognition
- **Example File Name**: `Qwen3-TTS-12Hz-1.7B-CustomVoice`

#### .safetensors Model Companion Files

**Required Configuration Files** (model cannot load properly without these):
- `config.json` - Model architecture configuration
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Tokenizer definition

**Optional Configuration Files** (affect specific functions):
- `merges.txt` - BPE merge rules (text generation quality)
- `vocab.json` - Vocabulary (multilingual support)
- `special_tokens_map.json` - Special token mapping
- `added_tokens.json` - Custom tokens
- `chat_template.json` - Chat template (conversation quality)
- `generation_config.json` - Generation parameter configuration
- `preprocessor_config.json` - Preprocessing configuration (image processing)
- `spk_dict.pt` - Speaker dictionary (required for audio generation)

#### Companion File Download Method

**Download from Hugging Face Repository**

1. Visit the model's main repository page
2. Find companion files in the file list
3. Download all required configuration files
4. Place files in the same directory as the main model


### Main Model File Name Preservation Rules

**Model Type Recognition Dependency**:
- The plugin automatically recognizes model types through keywords in the main model file name (e.g., "qwen3.5", "minicpm-o", "dreamomni2", etc.)
- Keywords in the file name directly affect the enabling of model functions (e.g., audio support, visual support, etc.)
- Renaming the main model file may cause incorrect model type recognition, affecting functionality

**Recommended Practice**:
- Keep the original name of the main model file, do not remove or modify the model identification keywords
- Examples: `Qwen2.5-Omni-7B-Q4_K_M.gguf`, `MiniCPM-o-4_5-Q4_K_M.gguf`, etc.
- If renaming is necessary, ensure the core model identification keywords are preserved

### Notes
1. **File names are case-sensitive**: Please keep the original case, especially for model identification keywords
2. **Do not rename companion files**: Otherwise the model may not find corresponding components
3. **Main model file name recommendation**: Try to keep the original name to ensure correct model type recognition
4. **Version matching**: Different versions of models may require different companion files, do not mix them
5. **Directory structure**: Companion files must be placed in the same directory as the main model file


## ASR Series (Speech Recognition)

### Qwen Series (Only supports the provided models)

**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ASR-0.6B
**Key Features**: Language coverage: Very extensive, supporting 28+ languages and 20+ Chinese dialects (such as Northeastern Mandarin, Sichuan dialect, Cantonese, Wu dialect, Minnan dialect, etc.).
**Audio Types**: Not only handles pure speech, but also supports singing voices and songs with background music (strong anti-noise/music scene capability).
**Applicable Scenarios**: Speech recognition, real-time speech-to-text, meeting recording, audio content extraction, subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B


**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ASR-1.7B
**Key Features**: Extensive language coverage, supports 28+ languages and 20+ Chinese dialects. Compared to the 0.6B version, it offers higher recognition accuracy, stronger processing capability for low-quality audio and complex accents, and supports continuous recognition of longer audio.
**Audio Types**: Supports pure speech, singing voices, and songs with background music (strong anti-noise/music scene capability).
**Applicable Scenarios**: High-precision speech recognition, real-time speech-to-text, meeting recording, audio content extraction, subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B



**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ForcedAligner-0.6B
**Key Features**: Forced alignment. It does not perform speech recognition, but aligns given text with audio timestamps, outputting the start and end times of each phoneme, word, or sentence in the audio.
**Language Coverage**: Supports only 11 languages (Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish), no dialects.
**Audio Types**: Pure speech only (not suitable for singing or songs with background music).
**Applicable Scenarios**: Subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B


## TTS Series (Text-to-Speech)

### Qwen Series (Only supports the provided models)


**Model Series**: Qwen-TTS
**Specific Model**: Qwen3-TTS-12Hz-1.7B-CustomVoice
**Key Features**: Focus on preset and control. It provides 9 fixed high-quality voices (covering different genders, ages, languages, and dialects), and you can adjust the speaking style (such as speed, emotion, etc.) based on these preset voices through instructions.
**Applicable Scenarios**: Custom speech synthesis, personalized voice assistants, audiobooks, brand voice identity, character voiceover
**Download Link**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice


**Model Series**: Qwen-TTS
**Specific Model**: Qwen3-TTS-12Hz-1.7B-VoiceDesign
**Key Features**: Focus on creativity. It does not rely on fixed voices, but generates voices that match the description in real-time based on your text input (e.g., "gentle middle-aged male", "young female voice with metallic texture").
**Applicable Scenarios**: Voice parameter adjustment, emotional speech synthesis, voice design, interactive voice applications, game character voiceover
**Download Link**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign



## VLM Series (Vision-Language Models, supporting text generation and image reverse prompting for all visual models)

### ToriiGate Series (XL Anime-specialized)
**Model Series**: ToriiGate
**Specific Model**: ToriiGate-v0.4-7B
**Key Features**: Based on Qwen2-VL fine-tuned, focused on advanced understanding of anime and digital art domains. Supports structured output, multiple caption modes (regular summary, Midjourney style, DeviantArt style), Bounding Box localization, and character name recognition. Descriptions are precise, dense, and avoid verbose filler words.
**Applicable Scenarios**: Anime/digital art image description, character feature extraction, multi-character recognition, prompt generation, caption correction and cropping
**Download Link (nsfw)**: https://huggingface.co/mradermacher/ToriiGate-v0.4-7B-GGUF (includes mmproj model)


### Qwen Series (Recommend to use clip model to select prompt reverse prompting model. For Chinese prompts, recommend qwen, MiMo series. For English prompts, recommend Llama series. For large VRAM devices, recommend larger models)

**Model Series**: Qwen-VL
**Specific Model**: Qwen2.5-7B-VL
**Key Features**: Alibaba Tongyi Qianwen multimodal series, full-size coverage, balanced multilingual OCR and image-text reasoning.
**Applicable Scenarios**: Multilingual OCR, image/video frame reverse prompting, prompt generation, regular image-text QA, batch text extraction
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF (includes mmproj model)


**Specific Model**: Qwen3-VL-8B-Instruct (Text Generation & Reverse Prompting, Recommended)
**Key Features**: Qwen3-VL instruction-optimized version, highest instruction following accuracy, supports complex visual instruction execution (e.g., "extract all tables from the image and generate text").
**Applicable Scenarios**: Instruction-based OCR extraction, complex image-text reverse prompting, professional visual task processing, high-precision prompt generation
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-GGUF (includes mmproj model)


**Specific Model**: Qwen3-VL-8B-maid
**Key Features**: Maid version of Qwen3-VL-8B model
**Download Link**: https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-i1-GGUF (includes mmproj model)
**Alternative Link**: https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/Zyi4082/Qwen3-VL-8B-Maid-GGUF (includes mmproj model)

### For Qwen3.5, Qwen3.6 and other thinking models using text generation mode, mmproj model must be enabled. 

**Specific Model**: Qwen3.5-4B (Reverse Prompting)
**Key Features**: Lightweight version of Qwen3.5 series, moderate parameter count, balances reasoning capability with resource consumption, supports multimodal input and tool calling, fast response, suitable for low VRAM devices.
**Applicable Scenarios**: Low VRAM device deployment, fast reasoning, mobile applications, lightweight dialogue, basic multimodal tasks, real-time interaction
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen3.5-4B_Abliterated-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


**Specific Model**: Qwen3.5-9B (Reverse Prompting)
**Key Features**: Balanced version of Qwen3.5 series, moderate parameter count, balances reasoning capability with resource consumption, supports multimodal input and tool calling, suitable for medium VRAM devices.
**Applicable Scenarios**: Medium VRAM device deployment, high-quality multimodal reasoning, complex image-text dialogue, image understanding analysis, video frame content recognition
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF (includes mmproj model)


### Qwen3.5 Variant Series

**Specific Model**: Qwen3.5-9B-DeepSeek-V4-Flash
**Key Features**: Distilled from DeepSeek-V4-Flash, inherits advanced structured reasoning and multi-step problem-solving capabilities. It successfully transfers the high-quality reasoning abilities of DeepSeek-V4 to the efficient Qwen3.5-9B parameter space, providing excellent AI reasoning experience with token efficiency and speed.
**Applicable Scenarios**: Medium VRAM device deployment, high-quality multimodal reasoning, complex image-text dialogue, image understanding analysis, video frame content recognition, structured reasoning tasks, tool-enhanced workflows
**Download Link**: https://huggingface.co/Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF (includes mmproj model)


**Specific Model**: Qwen3.6-27B (Large VRAM Required)
**Key Features**: High-performance version of Qwen3.6 series, large parameter count with strong reasoning capability, supports complex multimodal tasks and long context processing, suitable for large VRAM device deployment.
**Applicable Scenarios**: Large VRAM device deployment, complex image-text reasoning, professional image analysis, long text understanding, video content deep analysis, high-precision generation tasks
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


**Specific Model**: Qwen3.6-35B-A3B (Large VRAM Required)
**Key Features**: MoE version of Qwen3.6 series, 35B total parameters with only 3B active parameters for inference, balances high performance with inference efficiency, A3B architecture optimizes VRAM usage, supports complex multimodal tasks.
**Applicable Scenarios**: Medium VRAM device deployment, high-quality multimodal reasoning, complex image-text analysis, long text understanding, video content deep analysis
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


### OCR Specialized Series

**Model Series**: Unlimited-OCR
**Specific Model**: Unlimited-OCR
**Key Features**: Baidu's newly optimized model, fine-tuned for OCR-specific tasks, outstanding capability in recognizing formulas, tables, and old scanned documents, high text extraction accuracy, no additional configuration required, excellent local inference efficiency.
**Applicable Scenarios**: Poster text recognition, document OCR, academic paper formula recognition, batch scanned document text extraction, form information extraction, ancient text digitization
**Download Link**: https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF (includes mmproj model)


### MiniCPM Series

**Specific Model**: MiniCPM-V-4.6-abliterated-MAX
**Key Features**: Upgraded version of MiniCPM-V-4.5, further improved visual recognition accuracy, stronger capability for recognizing complex scenes and small targets, optimized inference speed, supports longer context multimodal understanding.
**Applicable Scenarios**: High-precision prompt generation, fine-grained image/video frame reverse prompting, multilingual OCR, complex scene image analysis, daily visual reasoning
**Download Link (nsfw)**: https://huggingface.co/mradermacher/MiniCPM-V-4.6-abliterated-MAX-GGUF (includes mmproj model)


### GLM Series

**Model Series**: GLM-4
**Specific Model**: GLM-4.6V-Flash-abliterated
**Key Features**: Zhipu large-parameter multimodal model, top-tier OCR and complex document parsing capabilities, supports high-resolution image recognition, stable local reasoning, adapts to complex visual tasks.
**Applicable Scenarios**: Professional OCR text extraction, complex document/table recognition, academic paper analysis, high-precision image reverse prompting, multilingual image-text reasoning
**Download Link (nsfw)**: https://huggingface.co/seanbailey518/Huihui-GLM-4.6V-Flash-abliterated-GGUF (includes mmproj model)


### JoyCaption Series

**Model Series**: JoyCaption
**Specific Model**: llama-joycaption (Reverse Prompting)
**Key Features**: ComfyUI-specific prompt generation/image reverse prompting model, strong detail capture capability, generated prompts fit creative needs.
**Applicable Scenarios**: ComfyUI prompt generation, precise image reverse prompting, art creation image-text association, batch image tag generation
**Download Link (nsfw)**: https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF
**mmproj Model**: https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf


### Gemma Series

**Model Series**: Gemma
**Specific Model**: google_gemma-3-4b-it
**Key Features**: Google Gemma 3 series multimodal model, features vision-language understanding, long context processing, and enhanced multilingual support, runs on single GPU, outperforms models with similar parameter counts.
**Applicable Scenarios**: Multilingual prompt generation, complex visual reasoning, long text understanding, multimodal interaction tasks
**Download Link**: https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF


**Specific Model**: Gemma-4-E2B-Uncensored-HauhauCS-Aggressive
**Key Features**: Uncensored version of Google Gemma 4 series, uses more advanced architecture design, significant improvements in visual understanding, language generation, and multimodal reasoning, moderate parameter count, suitable for single GPU deployment, supports more open content generation.
**Applicable Scenarios**: Advanced visual reasoning, multilingual dialogue, creative content generation, complex multimodal tasks, scenarios requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


**Specific Model**: Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
**Key Features**: Uncensored flagship model of Google Gemma 4 series, stronger visual understanding, longer context processing, and more accurate multilingual support, performance close to larger-scale models, supports more open content generation.
**Applicable Scenarios**: Complex visual reasoning, long text understanding, multilingual translation, advanced multimodal interaction, scenarios requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


**Specific Model**: Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced (Large VRAM Required)
**Key Features**: MoE version of Google Gemma 4 series, 26B total parameters with only 4B active parameters for inference, balances large model performance with inference efficiency, A4B architecture optimizes VRAM usage, supports complex visual reasoning and long context processing, supports more open content generation.
**Applicable Scenarios**: High-quality visual reasoning, long text understanding, multilingual translation, advanced multimodal interaction, complex tasks requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced (includes mmproj model)


### Phi Vision Series

**Model Series**: Phi-Vision
**Specific Model**: Phi-3.5-vision-instruct
**Key Features**: Microsoft iterative multimodal model, optimized for multi-image/video frame sequence reverse prompting, high instruction following accuracy, adapts to ComfyUI video frame processing nodes.
**Applicable Scenarios**: Video frame reverse prompting, multi-image comparison reverse prompting, instruction-based prompt generation, regular OCR text extraction
**Download Link**: https://huggingface.co/abetlen/Phi-3.5-vision-instruct-gguf (includes mmproj model)


### LLaMA Vision Series

**Model Series**: LLaMA-Vision
**Specific Model**: Llama-3.2-11B-Vision-Instruct-abliterated
**Key Features**: Meta official multimodal model, balanced across all scenarios, strong complex visual reasoning capabilities, supports high-resolution image recognition, adapts to various ComfyUI visual tasks.
**Applicable Scenarios**: High-precision image reverse prompting, complex document OCR, video frame key information extraction, professional visual reasoning
**Download Link (nsfw)**: https://huggingface.co/case01/Llama-3.2-11B-Vision-Instruct-abliterated-gguf
**mmproj Model**: https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF


### MiMo-VL Series (Custom Template Required)

**Model Series**: MiMo-VL
**Specific Model**: MiMo-VL-7B-RL-2508
**Key Features**: Xiaomi open-source multimodal vision-language model, 7B parameters, trained with reinforcement learning, four-stage pre-training (projector warm-up, vision-language alignment, general multimodal pre-training, long-context supervised fine-tuning), outperforms 10-billion-level models with small parameter count.
**Applicable Scenarios**: Chinese-English bilingual dialogue, complex visual reasoning, multimodal interaction, long context understanding, creative prompt generation
**Download Link**: https://huggingface.co/mradermacher/MiMo-VL-7B-RL-2508-GGUF (includes mmproj model)


### Nex-N2 Series (Temporary Unavailable)

**Model Series**: Nex-N2
**Specific Model**: Huihui-Nex-N2-mini-abliterated-APEX
**Key Features**: Uncensored version fine-tuned from Nex-N2-mini multimodal model, excellent visual understanding and image-text reasoning capabilities, supports image description, OCR text extraction and multimodal interaction, fast inference speed, suitable for daily visual tasks.
**Applicable Scenarios**: Image reverse prompting, OCR text extraction, image-text QA, daily visual reasoning, multimodal interaction tasks
**Download Link (nsfw)**: https://huggingface.co/SC117/Huihui-Nex-N2-mini-abliterated-APEX-GGUF (includes mmproj model)
