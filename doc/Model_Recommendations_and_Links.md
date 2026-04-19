# **llama-cpp-python 0.3.33 Compatible Models Download and Preset Recommendations**

### Note: please download based on your needs and hardware environment. Different models are suitable for different scenarios such as ComfyUI prompt generation, image/video frame reverse inference, etc. Reverse inference requires matching mmproj visual models.

## Document Structure

This document is organized as follows:
- **Quick Selection Guide**: Models recommended by hardware configuration and task type
- **Quantization Level Selection**: Guidelines for choosing quantization levels
- **Model Preparation**: Instructions for setting up model files
- **ASR Series**: Speech recognition models
- **TTS Series**: Text-to-speech models
- **Omni Series**: All-in-one multimodal models
- **VLM Series**: Visual language models
- **Preset Template Recommended Models**: Models recommended for each preset template

## Quick Selection Guide

### By Hardware Configuration

| Hardware Configuration | Recommended Models | VRAM Requirement | Best For |
|----------------------|-------------------|------------------|----------|
| High-end GPU (24GB+) | Qwen3-VL-8B, GLM-4.6V, LLaVA-1.6 | 24GB+ | Complex visual understanding, professional applications |
| Mid-range GPU (16GB) | LLaVA-1.6, MiniCPM-V-4.5, Qwen2.5-VL | 12-16GB | General visual understanding, balanced tasks |
| Mid-low GPU (12GB) | LLaVA-1.5, MiniCPM-V-2.6, Moondream2 | 8-12GB | Daily visual reasoning, OCR tasks |
| Low-end GPU (8GB) | Moondream2, nanoLLaVA, MobileVLM | 6-8GB | Quick descriptions, low VRAM devices |
| Very Low VRAM (4-6GB) | TinyLLaVA, MiniGPT-v2, LightOnOCR | 4-6GB | Basic tasks, edge device deployment |
| CPU Only | nanoLLaVA, TinyLLaVA | N/A | Maximum compatibility, offline processing |

### By Task Type

| Task Type | Recommended Models | Quantization | Notes |
|-----------|-------------------|--------------|-------|
| General Image Description | Qwen3-VL-8B, LLaVA-1.6, MiniCPM-V-4.5 | Q4_K_M | Balanced quality and speed |
| Fast Inference | Moondream2, nanoLLaVA | Q4_K_M | Prioritize speed |
| High Precision OCR | olmOCR-2-7B-1025, Qwen3-VL-8B | Q5_K_M | Highest accuracy |
| Video Analysis | Qwen2.5-Omni-7B, Phi-3.5-vision-instruct | Q4_K_M | Multi-frame processing |
| Speech Recognition | Qwen3-ASR-0.6B, Qwen3-ASR-1.7B | Original | ASR specialized models |
| Speech Synthesis | Qwen3-TTS-12Hz-1.7B | Original | TTS specialized models |
| Prompt Generation | Qwen3-VL-8B, GLM-4.6V | Q4_K_M | Creative tasks |
| Long Context | Qwen3.5-9B, MiniCPM-V-4.5 | Q4_K_M | 128K context support |



### Quantization Level Selection

- **Q4_K_M**: Balanced size and quality (recommended)
- **Q5_K_M**: Higher quality, slightly larger file
- **Q3_K_M**: Smaller file, suitable for low VRAM devices
- **Q2_K**: Smallest file, lower quality


### Model Preparation

1. **Create model directory**:

   - Create an `LLM` folder under `ComfyUI/models/` directory
   - Place downloaded model files in this directory. If you download multiple models, it's recommended to create subfolders within the LLM folder for easier differentiation


2. **Model file format**:

   - Supports `.gguf` and `.safetensors` formats
   - Visual models require corresponding `mmproj` files


### Required Companion Files

**1. Main Model File**
- `.gguf` or `.safetensors` format main model file
- File name usually contains model name and quantization level (e.g., `Qwen2.5-Omni-7B-Q4_K_M.gguf`)

**2. Visual Companion File (mmproj)**
- Visual projector for image and video processing
- File name usually contains `mmproj` keyword
- **Required scenarios**: Image reverse inference, video frame processing, OCR recognition
- **Example file name**: `mmproj-model-f16.gguf`

**3. Audio Companion Files**
- Specialized models for audio generation and processing, including Text-to-Speech (TTS), Automatic Speech Recognition (ASR), etc.
- File names usually contain `tts`, `asr` keywords
- **Required scenarios**: Speech synthesis, audio generation, audio understanding, speech recognition
- **Example file name**: `Qwen3-TTS-12Hz-1.7B-CustomVoice`




## ASR Series (Speech Recognition)

### Qwen Series (Official original models only, multi-segment models not supported)

**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ASR-0.6B
**Core Features**: Language coverage: Very extensive, supports 28+ languages and 20+ Chinese dialects (such as Northeastern dialect, Sichuan dialect, Cantonese, Wu dialect, Minnan dialect, etc.).
**Audio Type**: Not only handles pure speech but also supports singing voices and songs with background music (strong anti-noise/music scene capability).
**Applicable Scenarios**: Speech recognition, real-time speech-to-text, meeting recording, audio content extraction, subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B


**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ASR-1.7B
**Core Features**: Language coverage: Very extensive, supports 28+ languages and 20+ Chinese dialects (such as Northeastern dialect, Sichuan dialect, Cantonese, Wu dialect, Minnan dialect, etc.).
**Audio Type**: Not only handles pure speech but also supports singing voices and songs with background music (strong anti-noise/music scene capability).
**Applicable Scenarios**: Speech recognition, real-time speech-to-text, meeting recording, audio content extraction, subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B

**Model Series**: Qwen-ASR
**Specific Model**: Qwen3-ForcedAligner-0.6B
**Core Features**: Forced alignment. It is not for speech recognition, but for timestamp alignment between given text and audio, outputting start and end times for each phoneme, word, or sentence in the audio.
**Language Coverage**: Only supports 11 languages (Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish), does not support dialects.
**Audio Type**: Limited to pure speech (not suitable for singing voices or songs with background music).
**Applicable Scenarios**: Subtitle generation
**Download Link**: https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B


## TTS Series (Text-to-Speech)

### Qwen Series (Official original models only, multi-segment models not supported) Note: Download both models simultaneously, do not modify model names, please use arrows on both sides of the node to switch if problems occur

**Model Series**: Qwen-TTS
**Specific Model**: Qwen3-TTS-12Hz-1.7B-CustomVoice
**Core Features**: Focuses on preset and control. It provides 9 fixed high-quality timbres (covering different genders, ages, languages, and dialects), and you can adjust speaking styles (such as speech rate, emotion, etc.) based on these preset timbres through instructions.
**Applicable Scenarios**: Custom voice synthesis, personalized voice assistant, audiobooks, brand voice image, character dubbing
**Download Link**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice


**Model Series**: Qwen-TTS
**Specific Model**: Qwen3-TTS-12Hz-1.7B-VoiceDesign
**Core Features**: Focuses on creativity. It does not rely on fixed timbres, but generates timbres that match the description in real-time based on your input text descriptions (such as "gentle middle-aged male", "young female voice with metallic texture").
**Applicable Scenarios**: Voice parameter adjustment, emotional voice synthesis, voice design, interactive voice applications, game character dubbing
**Download Link**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign


## Omni Series (All-in-One Multimodal, audio processing requires downloading official original multi-segment models, quantized models only support text generation and image reverse inference)

### Qwen Series

**Model Series**: Qwen-Omni
**Specific Model**: Qwen2.5-Omni-3B
**Core Features**: Lightweight version of Alibaba Tongyi Qianwen all-in-one multimodal model, supports text-image-audio multimodal input, suitable for low VRAM devices, balanced multilingual understanding and generation capabilities.
**Applicable Scenarios**: Basic multimodal tasks, low VRAM device deployment, fast inference, general text-image Q&A, simple OCR tasks
**Download Link**: https://huggingface.co/unsloth/Qwen2.5-Omni-3B-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen2.5-Omni-3B-GGUF (includes mmproj model)
**Multi-segment Model**: https://huggingface.co/Qwen/Qwen2.5-Omni-3B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen2.5-Omni-3B


**Specific Model**: Qwen2.5-Omni-7B
**Core Features**: Alibaba Tongyi Qianwen all-in-one multimodal model, supports text-image-audio multimodal input, full-size coverage, balanced multilingual understanding and generation capabilities, suitable for complex multimodal scenarios.
**Applicable Scenarios**: Multilingual OCR, image/video frame reverse inference, audio content understanding, prompt generation, general text-image Q&A, batch text extraction
**Multi-segment Model**: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen2.5-Omni-7B


## VLM Series (Visual Language Models, supports text generation and image reverse inference for all visual models)

### Qwen Series

**Model Series**: Qwen-VL
**Specific Model**: Qwen2.5-VL-7B
**Core Features**: Alibaba Tongyi Qianwen multimodal series, full-size coverage, balanced multilingual OCR and image-text reasoning.
**Applicable Scenarios**: Multilingual OCR, image/video frame reverse inference, prompt generation, general text-image Q&A, batch text extraction
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF (includes mmproj model)



**Model Series**: Qwen-VL (balanced inference speed and text quality, recommended download)
**Specific Model**: Qwen3-VL-8B-Instruct
**Core Features**: Qwen3-VL instruction-optimized version, highest instruction following accuracy, supports complex visual instruction execution (e.g., "extract all tables from images and generate text").
**Applicable Scenarios**: Instruction-based OCR extraction, complex image-text reverse inference, professional visual task processing, high-precision prompt generation
**Download Link (nsfw)**: https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-GGUF (includes mmproj model)

**Specific Model**: Qwen3-VL-8B-maid-i1
**Core Features**: Qwen3-VL-8B model maid version
**Download Link**: https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-i1-GGUF (includes mmproj model)
**Alternative Link**: https://huggingface.co/mradermacher/Qwen3-VL-8B-maid-GGUF (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/Zyi4082/Qwen3-VL-8B-Maid-GGUF (includes mmproj model)


### Qwen3.5 Series

**Model Series**: Qwen3.5
**Specific Model**: Qwen3.5-2B
**Core Features**: Ultra-lightweight version of Qwen3.5 series, smallest parameter count, suitable for devices with extremely limited resources, supports multimodal input and basic dialogue capabilities.
**Applicable Scenarios**: Extremely low VRAM device deployment, embedded devices, mobile applications, basic dialogue, simple multimodal tasks
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF (includes mmproj model)
**Multi-segment Model**: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-2B-abliterated (nsfw)
**ModelScope Link**: https://www.modelscope.cn/models/Qwen/Qwen3.5-2B


**Specific Model**: Qwen3.5-4B
**Core Features**: Lightweight version of Qwen3.5 series, smaller parameter count but retains powerful reasoning capabilities, suitable for low VRAM devices, supports multimodal input and tool calling, fast response speed.
**Applicable Scenarios**: Low VRAM device deployment, fast inference, mobile applications, lightweight dialogue, basic multimodal tasks, real-time interaction
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen3.5-4B-GGUF (includes mmproj model)
**Multi-segment Model**: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated (nsfw)


**Specific Model**: Qwen3.5-9B
**Core Features**: Alibaba Tongyi Qianwen new generation dialogue model, comprehensively upgraded based on Qwen3, supports longer context (up to 128K), stronger reasoning capabilities, significantly improved code generation and mathematical calculation capabilities, supports tool calling and complex instruction execution.
**Applicable Scenarios**: Long text processing, code generation, mathematical reasoning, complex dialogue, prompt optimization, batch text processing, intelligent assistant
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive (includes mmproj model)
**ModelScope Link**: https://www.modelscope.cn/models/unsloth/Qwen3.5-9B-GGUF (includes mmproj model)



### OCR Specialized Series

**Model Series**: olmOCR (professional OCR model, recommended download)
**Specific Model**: olmOCR-2-7B-1025

**Core Features**: OCR-specialized fine-tuned, supports formula, table, and old scanned document recognition, high text extraction accuracy, no additional modifications needed, excellent local inference efficiency.
**Applicable Scenarios**: Poster text recognition, document OCR, text extraction, academic paper formula recognition, batch scanned document text extraction
**Download Link**: https://huggingface.co/Mungert/olmOCR-2-7B-1025-GGUF (includes mmproj model)

**Model Series**: LightOnOCR
**Specific Model**: LightOnOCR-2-1B
**Core Features**: Ultra-lightweight OCR-specific model, only 1B parameters, extremely fast inference speed, very low VRAM usage (runs on 2GB VRAM), focused on text extraction tasks, suitable for batch processing.
**Applicable Scenarios**: Batch text extraction, simple OCR tasks, low VRAM devices, fast document scanning, edge device deployment
**Download Link**: https://huggingface.co/noctrex/LightOnOCR-2-1B-GGUF (includes mmproj model)



### LLaMA Series

**Model Series**: LLaVA
**Specific Model**: llava-1.6-mistral-7b
**Core Features**: General multimodal benchmark model, stable image-text alignment, balanced visual reasoning capabilities.
**Applicable Scenarios**: Image reverse prompt generation, simple OCR text extraction, general image-text Q&A, video frame key information recognition
**Download Link**: https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf (includes mmproj model)

**Model Series**: nanoLLaVA
**Specific Model**: nanoLLaVA-1.5
**Core Features**: LLaVA lightweight compact version, small parameter count, low VRAM usage, fast inference speed, extremely low local deployment threshold, strong adaptability for basic visual tasks.
**Applicable Scenarios**: Low-end device image reverse inference, simple text extraction, batch fast image annotation, edge device local deployment
**Download Link**: https://huggingface.co/qnguyen3/nanoLLaVA-1.5
**mmproj model**: https://huggingface.co/saiphyohein/nanollava-1.5-gguf

**Model Series**: LLaMA-Vision
**Specific Model**: Llama-3.2-11B-Vision-Instruct
**Core Features**: Meta official multimodal model, balanced across all scenarios, strong complex visual reasoning capabilities, supports high-resolution image recognition, adapts to various ComfyUI visual tasks.
**Applicable Scenarios**: High-precision image reverse inference, complex document OCR, video frame key information extraction, professional-level visual reasoning
**Download Link**: https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF (includes mmproj model)

**Specific Model**: LLaMA-3.1-Vision
**Core Features**: Predecessor of Llama-3.2-Vision, stable image-text understanding capabilities, suitable for use as a basic visual model.
**Applicable Scenarios**: General image reverse inference, basic OCR text extraction, simple visual reasoning, backup basic visual model
**Download Link**: https://huggingface.co/FiditeNemini/Llama-3.1-Unhinged-Vision-8B-GGUF (includes mmproj model)



### MiniCPM-V Series

**Model Series**: MiniCPM-V
**Specific Model**: MiniCPM-V-4.5
**Core Features**: Domestic lightweight multimodal flagship, balances accuracy and speed, supports arbitrary aspect ratio image recognition, excellent OCR capabilities, complete ComfyUI node adaptation.
**Applicable Scenarios**: Prompt generation, image/video frame reverse inference, multilingual OCR, batch image text extraction, daily visual reasoning
**Download Link**: https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf (includes mmproj model)


**Model Series**: MiniCPM-V
**Specific Model**: MiniCPM-Llama3-V 2.5
**Core Features**: Optimized based on Llama3 base, lightweight and efficient, improved image-text fusion capabilities, high batch inference efficiency, strong local deployment stability.
**Applicable Scenarios**: Batch image reverse inference, fast prompt generation, lightweight OCR, low-config device ComfyUI workflows
**Download Link**: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf (includes mmproj model)


















### Gemma Series

**Model Series**: Gemma
**Specific Model**: google_gemma-3-4b-it
**Core Features**: Google Gemma 3 series multimodal model, featuring visual-language understanding, long context processing capabilities, and enhanced multilingual support, runs on a single GPU, outperforming models with the same parameter count.
**Applicable Scenarios**: Multilingual prompt generation, complex visual reasoning, long text understanding, multimodal interaction tasks
**Download Link**: https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF (includes mmproj model)
**Alternative Link (nsfw)**: https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF

**Model Series**: Gemma
**Specific Model**: Gemma-4-E2B-Uncensored-HauhauCS-Aggressive
**Core Features**: Google Gemma 4 series uncensored version, featuring more advanced architecture design, significant improvements in visual understanding, language generation, and multimodal reasoning, moderate parameter count, suitable for single GPU deployment, supports generating more open content.
**Applicable Scenarios**: Advanced visual reasoning, multilingual dialogue, creative content generation, complex multimodal tasks, scenarios requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive (includes mmproj model)

**Model Series**: Gemma
**Specific Model**: Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
**Core Features**: Google Gemma 4 series uncensored flagship model, 9B parameter scale, featuring stronger visual understanding capabilities, longer context processing capabilities, and more accurate multilingual support, performance close to larger scale models, supports generating more open content.
**Applicable Scenarios**: Complex visual reasoning, long text understanding, multilingual translation, advanced multimodal interaction, scenarios requiring uncensored content
**Download Link (nsfw)**: https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive (includes mmproj model)



### Youtu-VL Series

**Model Series**: Youtu-VL
**Specific Model**: Youtu-VL-4B-Instruct
**Core Features**: Tencent YouTu Lab multimodal visual-language model, 4B parameter lightweight design, excellent performance in visual understanding and language generation, supports bilingual (Chinese-English) processing.
**Applicable Scenarios**: Image understanding and description, visual question answering, OCR text extraction, bilingual prompt generation
**Download Link**: https://huggingface.co/tencent/Youtu-VL-4B-Instruct-GGUF (includes mmproj model)



### EraX-VL Series

**Model Series**: EraX-VL
**Specific Model**: EraX-VL-7B-V1.5
**Core Features**: 7B parameter multimodal visual-language model, V1.5 version further enhances visual perception and language understanding capabilities, supports high-resolution image processing, strong instruction following ability.
**Applicable Scenarios**: Image reverse inference, visual question answering, complex scene understanding, instruction-based prompt generation
**Download Link**: https://huggingface.co/mradermacher/EraX-VL-7B-V1.5-GGUF (includes mmproj model)



### MiMo-VL Series

**Model Series**: MiMo-VL
**Specific Model**: MiMo-VL-7B-RL
**Core Features**: Xiaomi open-source multimodal visual-language model, 7B parameters, trained based on reinforcement learning, four-stage pre-training (projector warm-up, visual-language alignment, general multimodal pre-training, long context supervised fine-tuning), outperforms 100B+ models with small parameter count.
**Applicable Scenarios**: Chinese-English bilingual dialogue, complex visual reasoning, multimodal interaction, long context understanding, creative prompt generation
**Download Link**: https://huggingface.co/unsloth/MiMo-VL-7B-RL-GGUF (includes mmproj model)



### DreamOmni Series

**Model Series**: DreamOmni
**Specific Model**: DreamOmni2
**Core Features**: Cross-modal multimodal model, supports image-text combined reasoning, prompt generation fits creative needs, adapts to ComfyUI creative workflows.
**Applicable Scenarios**: Creative prompt generation, image reverse inference, audio-image-text fusion creation (audio functionality requires additional components), complex visual scene reasoning
**Download Link**: https://huggingface.co/rafacost/DreamOmni2-7.6B-GGUF (includes mmproj model)




### Phi Vision Series

**Model Series**: Phi-Vision
**Specific Model**: Phi-3.5-vision-instruct
**Core Features**: Microsoft iterative multimodal model, optimized for multi-image/video frame sequence reverse inference, high instruction following accuracy, adapts to ComfyUI video frame processing nodes.
**Applicable Scenarios**: Video frame reverse inference, multi-image comparison reverse inference, instruction-based prompt generation, general OCR text extraction
**Download Link**: https://huggingface.co/abetlen/Phi-3.5-vision-instruct-gguf (includes mmproj model)





### LLaMA Vision Series

**Model Series**: LLaMA-Vision
**Specific Model**: Llama-3.2-11B-Vision-Instruct
**Core Features**: Meta official multimodal model, balanced across all scenarios, strong complex visual reasoning capabilities, supports high-resolution image recognition, adapts to various ComfyUI visual tasks.
**Applicable Scenarios**: High-precision image reverse inference, complex document OCR, video frame key information extraction, professional-level visual reasoning
**Download Link**: https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF (includes mmproj model)



**Model Series**: LLaMA-Vision
**Specific Model**: LLaMA-3.1-Vision
**Core Features**: Predecessor of Llama-3.2-Vision, stable image-text understanding capabilities, suitable for use as a basic visual model.
**Applicable Scenarios**: General image reverse inference, basic OCR text extraction, simple visual reasoning, backup basic visual model
**Download Link**: https://huggingface.co/FiditeNemini/Llama-3.1-Unhinged-Vision-8B-GGUF (includes mmproj model)



### Yi-VL Series

**Model Series**: Yi-VL
**Specific Model**: Yi-VL-6B
**Core Features**: Domestic long-context multimodal model, balanced Chinese-English OCR capabilities, high image-text understanding accuracy, supports long image-text joint reasoning, adapts to bilingual document scenarios.
**Applicable Scenarios**: Bilingual document OCR, long image-text reverse inference, Chinese-English bilingual prompt generation, bilingual visual reasoning tasks
**Download Link**: https://huggingface.co/cmp-nct/Yi-VL-6B-GGUF (includes mmproj model)



### LightOnOCR Series

**Model Series**: LightOnOCR
**Specific Model**: LightOnOCR-2-1B
**Core Features**: Ultra-lightweight OCR-specific model, only 1B parameters, extremely fast inference speed, very low VRAM usage (runs on 2GB VRAM), focused on text extraction tasks, suitable for batch processing.
**Applicable Scenarios**: Batch text extraction, simple OCR tasks, low VRAM devices, fast document scanning, edge device deployment
**Download Link**: https://huggingface.co/noctrex/LightOnOCR-2-1B-GGUF (includes mmproj model)



## Preset Template Recommended Models

### Basic Templates

**Empty Template**: Empty template, fully customizable

**Applicable Models**: None (select all models as needed)



**Simple Description**: Simply describe image content

**Recommended Models**: Qwen3-VL-8B-Instruct, LLaVA-1.6, nanoLLaVA-1.5, Moondream2, LLaMA-3.1-Vision-8B, llama-joycaption, LightOnOCR-2-1B, Qwen2.5-Omni-7B, Qwen3.5-4B, Qwen3-VL-8B-maid-i1



### Prompt Style Templates

**Tag Style**: Generate image tag lists, suitable for models like SDXL, outputs up to 60 unique tags

**Recommended Models**: Qwen3-VL-8B-Instruct, MiniCPM-V-4.5, llama-joycaption, Qwen3-VL-8B-maid-i1, Qwen2.5-Omni-7B, Qwen3.5-4B



**Concise Description**: Concise image description (within 300 words), enhances clarity and expressiveness

**Recommended Models**: Qwen3-VL-8B-Instruct, nanoLLaVA-1.5, Moondream2, LLaMA-3.1-Vision-8B, Qwen2.5-VL-7B, llama-joycaption, GLM-4.6V-Flash, Youtu-VL-4B-Instruct, Qwen2.5-Omni-7B, Qwen3.5-4B, Qwen3-VL-8B-maid-i1



**Detailed Description**: Detailed image description (within 500 words), adds specific details for each element

**Recommended Models**: Qwen3-VL-8B-Instruct, LLaVA-1.6, MiniCPM-V-4.5, GLM-4.6V-Flash, EraX-VL-7B-V1.5, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



**Detailed Expansion**: Detailed prompt expansion (within 800 words), enhances clarity and expressiveness

**Recommended Models**: Qwen3-VL-8B-Instruct, Llama-3.2-11B-Vision-Instruct, GLM-4.6V-Flash, gemma-3-4b-it, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



**Optimized Expansion**: Optimize and expand prompts to make them more expressive and visually rich

**Recommended Models**: Qwen3-VL-8B-Instruct, GLM-4.6V-Flash, MiMo-VL-7B-RL, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



### Visual Templates

**Bounding Box Detection**: Generate object detection bounding boxes, output JSON format coordinate lists

**Recommended Models**: LLaVA-1.6, Llama-3.2-11B-Vision-Instruct, Qwen3-VL-8B-Instruct, Qwen3.5-4B




### OCR Templates

**OCR Enhanced**: Professional poster OCR text recognition, accurately extracts text content and style attributes, adapts to prompt reverse inference needs

**Recommended Models**: olmOCR-2-7B-1025, GLM-4.6V-Flash, Qwen3-VL-8B-Instruct, MiniCPM-V-4.5, LightOnOCR-2-1B, Qwen3.5-4B



### Multilingual Templates

**Chinese-English Bilingual Generation**: Chinese-English bilingual prompt generation, adapts to cross-border creation / bilingual document scenarios

**Recommended Models**: Qwen2.5-VL-7B, GLM-4.6V-Flash, Qwen2.5-Omni-7B, Qwen3.5-4B



### High Resolution Templates

**Ultra HD Reverse Inference**: Ultra HD image prompt reverse inference, supports 4K/8K resolution image detail extraction

**Recommended Models**: Llama-3.2-11B-Vision-Instruct, GLM-4.6V-Flash, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1, Qwen3.5-4B




### Video Templates

**Video Reverse Inference**: Video reverse inference prompts, generate detailed video descriptions based on video content

**Recommended Models**: Qwen3-VL-8B-Instruct, Phi-3.5-vision-instruct, GLM-4.6V-Flash, MiMo-VL-7B-RL, Qwen2.5-Omni-7B, Qwen3.5-4B



**Video Scene Deconstruction**: Detailed video scene deconstruction, provides complete details for each scene in chronological order

**Recommended Models**: Qwen3-VL-8B-Instruct, Phi-3.5-vision-instruct, Llama-3.2-11B-Vision-Instruct, GLM-4.6V-Flash, MiMo-VL-7B-RL, Qwen2.5-Omni-7B, Qwen3.5-4B



**Video Subtitle Format**: Generate standard format video subtitles, including timecodes and synchronized text

**Recommended Models**: Phi-3.5-vision-instruct, Qwen2.5-VL-7B, Youtu-VL-4B-Instruct, Qwen2.5-Omni-7B, Qwen3.5-4B



### Professional Model Templates

**ZIMAGE-Turbo**: Designed specifically for Z-Image-Turbo model, creates efficient and high-quality image generation prompts, uses 8-step Turbo inference to quickly generate 1080P HD images

**Recommended Models**: Qwen3-VL-8B-Instruct, GLM-4.6V-Flash, Youtu-VL-4B-Instruct, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



**FLUX2-Klein**: Designed specifically for FLUX.2 Klein model, creates concise and expressive prompts (within 200 words)

**Recommended Models**: Qwen3-VL-8B-Instruct, nanoLLaVA-1.5, Moondream2, MiniCPM-Llama3-V 2.5, Qwen2.5-VL-7B, GLM-4.6V-Flash, Youtu-VL-4B-Instruct, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



**LTX-2**: Designed specifically for LTX-2 model, creates detailed and dynamic video generation prompts, supports high-quality, audio-visual synchronized 4K video

**Recommended Models**: Phi-3.5-vision-instruct, Llama-3.2-11B-Vision-Instruct, GLM-4.6V-Flash, EraX-VL-7B-V1.5, Qwen2.5-Omni-7B, Qwen3.5-4B



**Qwen Layered Image**: Designed specifically for Qwen-Image-Layered model, creates detailed layered prompts, handles complex compositions and multiple elements

**Recommended Models**: Qwen2.5-VL-7B, Qwen3-VL-8B-Instruct, GLM-4.6V-Flash, EraX-VL-7B-V1.5, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



**Qwen Image Edit**: Comprehensive edit prompt enhancer, used for image editing tasks, supports adding, deleting, replacing operations

**Recommended Models**: Qwen3-VL-8B-Instruct, GLM-4.6V-Flash, MiMo-VL-7B-RL, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1



**Qwen2512**: Designed specifically for Qwen Image2512 model, creates high-quality image generation prompts

**Recommended Models**: Qwen2.5-VL-7B, Qwen3-VL-8B-Instruct, LLaVA-1.6, GLM-4.6V-Flash, Youtu-VL-4B-Instruct, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



### Film Style Templates

**WAN Text to Video**: Film director style, adds film elements to original prompt (time, light source, light intensity, light angle, tone, shooting angle, shot size, composition, etc.)

**Recommended Models**: Llama-3.2-11B-Vision-Instruct, Qwen3-VL-8B-Instruct, GLM-4.6V-Flash, MiMo-VL-7B-RL, Qwen2.5-Omni-7B, Qwen3.5-4B



**WAN Image to Video**: Video description prompt rewriting expert, rewrites video descriptions based on images and input prompts, emphasizes dynamic content

**Recommended Models**: Phi-3.5-vision-instruct, Qwen3-VL-8B-Instruct, GLM-4.6V-Flash, EraX-VL-7B-V1.5, Qwen2.5-Omni-7B, Qwen3.5-4B



**WAN Image to Video Empty Template**: Video description prompt writing expert, generates video descriptions based on images using imagination

**Recommended Models**: llama-joycaption, GLM-4.6V-Flash, Qwen2.5-Omni-7B, Qwen3-VL-8B-maid-i1, Qwen3.5-4B



**WAN First/Last Frame to Video**: Prompt optimizer, optimizes and rewrites prompts based on video first/last frame images, emphasizes motion information and camera movement

**Recommended Models**: Phi-3.5-vision-instruct, Llama-3.2-11B-Vision-Instruct, GLM-4.6V-Flash, EraX-VL-7B-V1.5, Qwen2.5-Omni-7B, Qwen3.5-4B


### Audio Templates (omni models require official original multi-segment models for audio processing, quantized models only support text generation and image reverse inference, need to use asr and tts models)

**Audio-Text Conversion**: Convert between audio and subtitles, generate synchronized audio parameters or subtitle text

**Recommended Models**: Qwen2.5-Omni-3B, Qwen2.5-Omni-7B


**Video Audio-Text Generation**: Generate audio descriptions and synchronized subtitles based on video content

**Recommended Models**: Qwen2.5-Omni-3B, Qwen2.5-Omni-7B


**Audio Analysis**: Analyze emotional, style, rhythm and other characteristics of audio

**Recommended Models**: Qwen2.5-Omni-3B, Qwen2.5-Omni-7B
