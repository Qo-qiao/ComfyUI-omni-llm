# **llama-cpp-python 0.3.35 Download Guide & Preset Recommendations**


### Note: Please download according to your needs and hardware environment. Different models are suitable for different scenarios such as ComfyUI prompt generation, images, videos, audio, and reverse prompting. For image and video processing, you need to use the matching mmproj visual model. For audio processing, please refer to official documentation and use matching models (different models have differences). Below are some model download links (models based on qwen, Llava, Llama3 and other underlying architectures generally support visual understanding and text generation functions)

### Credits: @mradermacher, @unsloth, @huihui-ai, @hauhauCS


### Quantization Level Selection

- **Q4_K_M**: Balanced size and quality (Recommended)
- **Q5_K_M**: Higher quality, slightly larger file
- **Q3_K_M**: Smaller file, suitable for low VRAM devices
- **Q2_K**: Smallest file, lower quality


### Model Preparation

1. **Create Model Directory**:

   - Create `LLM` folder under `ComfyUI/models/` directory
   - Place downloaded model files in this directory. If you download multiple models, it is recommended to create subfolders within the LLM folder for better organization

2. **Model File Format**:


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

### Qwen Series (Official original models recommended)

**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ASR-0.6B
**Key Features**: Language coverage: Very extensive, supporting 28+ languages and 20+ Chinese dialects (such as Northeastern Mandarin, Sichuan dialect, Cantonese, Wu dialect, Minnan dialect, etc.).
**Audio Types**: Not only handles pure speech, but also supports singing voices and songs with background music (strong anti-noise/music scene capability).
**Applicable Scenarios**: Speech recognition, real-time speech-to-text, meeting recording, audio content extraction, subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B


**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ASR-1.7B
**Key Features**: Language coverage: Very extensive, supporting 28+ languages and 20+ Chinese dialects (such as Northeastern Mandarin, Sichuan dialect, Cantonese, Wu dialect, Minnan dialect, etc.).
**Audio Types**: Not only handles pure speech, but also supports singing voices and songs with background music (strong anti-noise/music scene capability).
**Applicable Scenarios**: Speech recognition, real-time speech-to-text, meeting recording, audio content extraction, subtitle generation
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

### Qwen Series


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

### Qwen Series (qwen3.5 series models are more divergent, text content generated using preset templates may contain non-prompt information)

**Model Series**: Qwen-VL
**Specific Model**: Qwen2.5-7B-VL
**Key Features**: Alibaba Tongyi Qianwen multimodal series, full-size coverage, balanced multilingual OCR and image-text reasoning.
**Applicable Scenarios**: Multilingual OCR, image/video frame reverse prompting, prompt generation, regular image-text QA, batch text extraction
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF (includes mmproj model)


**Model Series**: Qwen-VL (Text Generation & Reverse Prompting, Recommended)
**Specific Model**: Qwen3-VL-8B-Instruct
**Key Features**: Qwen3-VL instruction-optimized version, highest instruction following accuracy, supports complex visual instruction execution (e.g., "extract all tables from the image and generate text").
**Applicable Scenarios**: Instruction-based OCR extraction, complex image-text reverse prompting, professional visual task processing, high-precision prompt generation
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-GGUF (includes mmproj model)


**Model Series**: Qwen-VL
**Specific Model**: Qwen3-VL-8B-maid
**Key Features**: Maid version of Qwen3-VL-8B model
**Download Link**: https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-i1-GGUF (includes mmproj model)
**Alternative Link**: https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/Zyi4082/Qwen3-VL-8B-Maid-GGUF (includes mmproj model)


**Model Series**: Qwen3.5
**Specific Model**: Qwen3.5-2B
**Key Features**: Ultra-lightweight version of Qwen3.5 series, smallest parameter count, suitable for devices with extremely limited resources, supports multimodal input and basic dialogue capabilities.
**Applicable Scenarios**: Low VRAM device deployment, embedded devices, mobile applications, basic dialogue, simple multimodal tasks
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF (includes mmproj model)
**Multi-part Model**: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-2B-abliterated (nsfw)
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3.5-2B


**Specific Model**: Qwen3.5-4B (Reverse Prompting)
**Key Features**: Lightweight version of Qwen3.5 series, smaller parameter count but retains powerful reasoning capabilities, suitable for low VRAM devices, supports multimodal input and tool calling, fast response.
**Applicable Scenarios**: Low VRAM device deployment, fast reasoning, mobile applications, lightweight dialogue, basic multimodal tasks, real-time interaction
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen3.5-4B-GGUF (includes mmproj model)
**Multi-part Model**: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated (nsfw)


### OCR Specialized Series

**Model Series**: olmOCR
**Specific Model**: olmOCR-2-7B-1025

**Key Features**: Specialized OCR fine-tuning, supports formula, table, and old scanned document recognition, high text extraction accuracy, no additional modifications required, excellent local reasoning efficiency.
**Applicable Scenarios**: Poster text recognition, document OCR, text extraction, academic paper formula recognition, batch scanned document text extraction
**Download Link**: https://huggingface.co/Mungert/olmOCR-2-7B-1025-GGUF (includes mmproj model)


### LLaMA Series

**Model Series**: LLaVA
**Specific Model**: llava-1.6-mistral-7b

**Key Features**: General multimodal benchmark model, stable image-text alignment, balanced visual reasoning capabilities.
**Applicable Scenarios**: Image reverse prompting for prompts, simple OCR text extraction, regular image-text QA, video frame key information recognition
**Download Link**: https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf (includes mmproj model)


**Model Series**: nanoLLaVA
**Specific Model**: nanoLLaVA-1.5
**Key Features**: Lightweight streamlined version of LLaVA, small parameter count, low VRAM usage, fast reasoning speed, extremely low local deployment threshold, strong adaptability for basic visual tasks.
**Applicable Scenarios**: Low-end device image reverse prompting, simple text extraction, batch fast image annotation, edge device local deployment
**Download Link**: https://huggingface.co/qnguyen3/nanoLLaVA-1.5
**mmproj Model**: https://huggingface.co/saiphyohein/nanollava-1.5-gguf


### MiniCPM Series

**Model Series**: MiniCPM-V
**Specific Model**: MiniCPM-V-4.5
**Key Features**: Domestic lightweight multimodal flagship, balances accuracy and speed, supports arbitrary aspect ratio image recognition, excellent OCR capabilities, well-adapted ComfyUI nodes.
**Applicable Scenarios**: Prompt generation, image/video frame reverse prompting, multilingual OCR, batch image text extraction, daily visual reasoning
**Download Link**: https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf (includes mmproj model)


**Model Series**: MiniCPM-V
**Specific Model**: MiniCPM-Llama3-V 2.5
**Key Features**: Optimized based on Llama3 base, lightweight and efficient, improved image-text fusion capabilities, high batch reasoning efficiency, stable local deployment.
**Applicable Scenarios**: Batch image reverse prompting, fast prompt generation, lightweight OCR, low-spec device ComfyUI workflows
**Download Link**: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf (includes mmproj model)


### GLM Series

**Model Series**: GLM-4
**Specific Model**: GLM-4.6V-Flash
**Key Features**: Zhipu large-parameter multimodal model, top-tier OCR and complex document parsing capabilities, supports high-resolution image recognition, stable local reasoning, adapts to complex visual tasks.
**Applicable Scenarios**: Professional OCR text extraction, complex document/table recognition, academic paper analysis, high-precision image reverse prompting, multilingual image-text reasoning
**Download Link**: https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF (includes mmproj model)


### JoyCaption Series

**Model Series**: JoyCaption
**Specific Model**: llama-joycaption (Reverse Prompting)
**Key Features**: ComfyUI-specific prompt generation/image reverse prompting model, strong detail capture capability, generated prompts fit creative needs.
**Applicable Scenarios**: ComfyUI prompt generation, precise image reverse prompting, art creation image-text association, batch image tag generation
**Download Link (nsfw)**: https://huggingface.co/mradermacher/llama-joycaption-beta-one-hf-llava-GGUF
**mmproj Model**: https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf


### Moondream Series

**Model Series**: Moondream
**Specific Model**: Moondream2
**Key Features**: Ultra-lightweight multimodal model (only 1.4B parameters), extremely fast reasoning, very low VRAM usage (runs on 4GB VRAM), sufficient for basic visual and OCR tasks.
**Applicable Scenarios**: Edge device local deployment, batch fast image reverse prompting, simple text extraction, low-spec computer ComfyUI workflows
**Download Link**: https://huggingface.co/Hahasb/moondream2-20250414-GGUF (includes mmproj model)


### Gemma Series

**Model Series**: Gemma
**Specific Model**: google_gemma-3-4b-it
**Key Features**: Google Gemma 3 series multimodal model, features vision-language understanding, long context processing, and enhanced multilingual support, runs on single GPU, outperforms models with similar parameter counts.
**Applicable Scenarios**: Multilingual prompt generation, complex visual reasoning, long text understanding, multimodal interaction tasks
**Download Link**: https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF


**Model Series**: Gemma
**Specific Model**: Gemma-4-E2B-Uncensored-HauhauCS-Aggressive
**Key Features**: Uncensored version of Google Gemma 4 series, uses more advanced architecture design, significant improvements in visual understanding, language generation, and multimodal reasoning, moderate parameter count, suitable for single GPU deployment, supports more open content generation.
**Applicable Scenarios**: Advanced visual reasoning, multilingual dialogue, creative content generation, complex multimodal tasks, scenarios requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


**Model Series**: Gemma
**Specific Model**: Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
**Key Features**: Uncensored flagship model of Google Gemma 4 series, stronger visual understanding, longer context processing, and more accurate multilingual support, performance close to larger-scale models, supports more open content generation.
**Applicable Scenarios**: Complex visual reasoning, long text understanding, multilingual translation, advanced multimodal interaction, scenarios requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive (includes mmproj model)


### Youtu-VL Series

**Model Series**: Youtu-VL
**Specific Model**: Youtu-VL-4B-Instruct
**Key Features**: Tencent Youtu Lab multimodal vision-language model, 4B parameter lightweight design, excellent performance in visual understanding and language generation, supports Chinese-English bilingual processing.
**Applicable Scenarios**: Image understanding and description, visual QA, OCR text extraction, Chinese-English bilingual prompt generation
**Download Link**: https://huggingface.co/tencent/Youtu-VL-4B-Instruct-GGUF (includes mmproj model)


### EraX-VL Series

**Model Series**: EraX-VL
**Specific Model**: EraX-VL-7B-V1.5
**Key Features**: 7B parameter multimodal vision-language model, V1.5 version further improves visual perception and language understanding capabilities, supports high-resolution image processing, strong instruction following capability.
**Applicable Scenarios**: Image reverse prompting, visual QA, complex scene understanding, instruction-based prompt generation
**Download Link**: https://huggingface.co/mradermacher/EraX-VL-7B-V1.5-GGUF (includes mmproj model)


### Phi Vision Series

**Model Series**: Phi-Vision
**Specific Model**: Phi-3.5-vision-instruct
**Key Features**: Microsoft iterative multimodal model, optimized for multi-image/video frame sequence reverse prompting, high instruction following accuracy, adapts to ComfyUI video frame processing nodes.
**Applicable Scenarios**: Video frame reverse prompting, multi-image comparison reverse prompting, instruction-based prompt generation, regular OCR text extraction
**Download Link**: https://huggingface.co/abetlen/Phi-3.5-vision-instruct-gguf (includes mmproj model)


### LLaMA Vision Series

**Model Series**: LLaMA-Vision
**Specific Model**: Llama-3.2-11B-Vision-Instruct
**Key Features**: Meta official multimodal model, balanced across all scenarios, strong complex visual reasoning capabilities, supports high-resolution image recognition, adapts to various ComfyUI visual tasks.
**Applicable Scenarios**: High-precision image reverse prompting, complex document OCR, video frame key information extraction, professional visual reasoning
**Download Link**: https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF (includes mmproj model)


**Model Series**: LLaMA-Vision
**Specific Model**: LLaMA-3.1-Vision-8B
**Key Features**: Predecessor to Llama-3.2-Vision, stable image-text understanding capabilities, suitable as a basic visual model.
**Applicable Scenarios**: Regular image reverse prompting, basic OCR text extraction, simple visual reasoning, backup basic visual model
**Download Link**: https://huggingface.co/FiditeNemini/Llama-3.1-Unhinged-Vision-8B-GGUF (includes mmproj model)


### Yi-VL Series

**Model Series**: Yi-VL
**Specific Model**: Yi-VL-6B
**Key Features**: Domestic long-context multimodal model, balanced Chinese-English OCR capabilities, high image-text understanding accuracy, supports long image-text joint reasoning, adapts to bilingual document scenarios.
**Applicable Scenarios**: Bilingual document OCR, long image-text reverse prompting, Chinese-English bilingual prompt generation, bilingual visual reasoning tasks
**Download Link**: https://huggingface.co/cmp-nct/Yi-VL-6B-GGUF (includes mmproj model)


### LightOnOCR Series

**Model Series**: LightOnOCR
**Specific Model**: LightOnOCR-2-1B
**Key Features**: Ultra-lightweight OCR-specific model, only 1B parameters, extremely fast reasoning, very low VRAM usage (runs on 2GB VRAM), focuses on text extraction tasks, suitable for batch processing.
**Applicable Scenarios**: Batch text extraction, simple OCR tasks, low VRAM devices, fast document scanning, edge device deployment
**Download Link**: https://huggingface.co/noctrex/LightOnOCR-2-1B-GGUF (includes mmproj model)


### MiMo-VL Series

**Model Series**: MiMo-VL
**Specific Model**: MiMo-VL-7B-RL
**Key Features**: Xiaomi open-source multimodal vision-language model, 7B parameters, trained with reinforcement learning, four-stage pre-training (projector warm-up, vision-language alignment, general multimodal pre-training, long-context supervised fine-tuning), outperforms 10-billion-level models with small parameter count.
**Applicable Scenarios**: Chinese-English bilingual dialogue, complex visual reasoning, multimodal interaction, long context understanding, creative prompt generation
**Download Link**: https://huggingface.co/unsloth/MiMo-VL-7B-RL-GGUF (includes mmproj model)


### Jan Series

**Model Series**: Jan-VL
**Specific Model**: Jan-v2-VL-high
**Key Features**: High-performance vision-language model, supports complex visual reasoning and multimodal interaction, excellent image understanding and text generation capabilities.
**Applicable Scenarios**: Image reverse prompting, visual QA, complex scene understanding, instruction-based prompt generation, multimodal interaction tasks
**Download Link**: https://huggingface.co/janhq/Jan-v2-VL-high-gguf (includes mmproj model)