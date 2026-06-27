# ComfyUI-omni-llm

Run LLM/VLM models natively in ComfyUI based on llama.cpp, supporting multimodal inference, visual language understanding, and full-stack audio processing capabilities.

**[📃中文](./README_zh.md)**

## Project Overview

ComfyUI-omni-llm is a comprehensive ComfyUI plugin, deeply refactored and enhanced based on ComfyUI-llama-cpp-vlm, focusing on providing localized and efficient multimodal AI inference capabilities. The plugin supports ASR speech recognition and TTS speech synthesis, and supports over 40 VLM models and 200+ LLM models.

### Core Advantages

- **Full-Stack Multimodal**: Seamlessly integrates text, image, video, and audio processing capabilities
- **Smart Hardware Adaptation**: Automatic parameter tuning, supports NVIDIA, AMD, and other mainstream GPUs
- **Rich Model Ecosystem**: Supports multiple mainstream VLM/LLM models, automatically adapts to new models
- **Efficient Inference Performance**: Introduces parallel processing and caching mechanisms, significantly improving runtime efficiency
- **Professional Prompt System**: Built-in rich scenario-based preset templates
- **Powerful Audio Capabilities**: Integrated ASR speech recognition and TTS speech synthesis functions

[Tip] Models with thinking mode cannot provide accurate thought content filtering. It is recommended to use non-thinking mode models. All generated prompt content is for reference only and cannot guarantee perfect image generation. Please verify and optimize content based on your actual needs.


## Chinese Translation

Place the zh-CN files into the corresponding folder of translation plugins (ComfyUI-Chinese-Translation/AIGODLIKE-ComfyUI-Translation/ComfyUI-DD-Translation) to overwrite. It is recommended to install the [ComfyUI-Chinese-Translation](https://github.com/a63976659/ComfyUI-Chinese-Translation) plugin for more comprehensive Chinese localization and faster translation updates.

## Recommended Companion Plugins

Due to the different functional positioning of preset templates, some preset templates and prompts have been updated to [ComfyUI-prompt-storage](https://github.com/Qo-qiao/ComfyUI-prompt-storage), which can be used with this plugin to increase extensibility.



## Installation Instructions

If you are not familiar with installing dependencies, it is recommended to use the [ComfyNexus Launcher](https://github.com/Allen-xxa/ComfyNexus/releases/). The launcher provides a more intuitive installation method, making it easier to view installed dependency files and resolve dependency conflicts.

### 1. Basic Installation

1. **Extract dependency files from the plugin's site-packages folder**: [Download dependency package](https://github.com/Qo-qiao/ComfyUI-omni-llm/releases)
   - Extract the dependency files from the plugin's site-packages folder to `ComfyUI/custom_nodes/ComfyUI-omni-llm/site-packages` directory
   - Directory reference format:
     ```
     ComfyUI/custom_nodes/ComfyUI-omni-llm/site-packages/
     ├── qwen-tts
     ├── qwen_asr
     ├── transformers
     ├── ... other dependencies
     ```
2. **Install remaining dependencies**:
   - Install:
     ```bash
     # Run in ComfyUI root directory
     pip install -r custom_nodes/ComfyUI-omni-llm/requirements.txt
     ```

3. **Install llama-cpp-python** (Required):
   - Download the latest version 0.3.35 manually from [llama_cpp_python_wheels](https://github.com/JamePeng/llama-cpp-python/releases) or directly download the zip package

   **Wheel Selection Guide:**
   - **Python Version Match**: `cp312` in the filename indicates Python 3.12, select the file matching your Python version
   - **CUDA Version Match**: `cu128` in the filename indicates CUDA 12.8, select the file compatible with your CUDA version
   - **OS Match**: `win_amd64` indicates Windows 64-bit system, ensure you select the file suitable for your operating system
   - **Function Version**: `basic` indicates basic version, `full` includes more features (such as OpenCL)

   **Installation Command:**
   - Install using: `pip install downloaded_filename.whl`
   - Example: `pip install llama_cpp_python-0.3.32+cu128.basic-cp312-cp312-win_amd64.whl`

   **Version Selection Recommendations:**
   - **CUDA 12.8**: Select `cu128` version for best performance
   - **CUDA 12.x**: Select `cu124` version for backward compatibility


### 2. Model Preparation

1. **Create Model Directory**:
   - Create an `LLM` folder under `ComfyUI/models/` directory
   - Place downloaded model files in this directory

2. **Model File Format**:
   - LLM/VLM models support `.gguf` format
   - Visual models require corresponding `mmproj` files
   - ASR/TTS models require downloading full model files

## Workflow Examples (Reference examples, please modify parameters according to actual situation)

### Workflow Files
- [Audio Mode Workflow](./workflows/omni-llm(audio).json)
- [Text or Image Mode Workflow](./workflows/omni-llm(text_or_image).json)
- [Video Mode Workflow](./workflows/omni-llm(video).json)

### Workflow Example Images

#### Text Generation
![Text Generation Workflow Example](./workflows/文本生成（Text_Generation）.png)

#### Image Processing
![Image Reverse Engineering Workflow Example](./workflows/图像反推（Image_reverse_engineering）.png)

#### Video Processing
![Video Reverse Engineering Workflow Example](./workflows/视频反推（Video_reverse_engineering）.png)


#### Audio Processing
![Text to Audio Workflow Example](./workflows/文本转音频（Text_to_Audio）.png)

![Audio to Text Workflow Example](./workflows/音频转文本（Audio_to_text）.png)

![Multi-person Dialogue Audio Generation Workflow Example](./workflows/多人对话音频生成（Multi-person_dialogue_audio_generation）.png)


## Documentation (For reference only)

Please check [Model Download Links](./doc/Model_Download_Links.md)
Please check [Node Parameter Guide](./doc/Node_Parameter_Guide.md)


## Changelog

#### v3.1.6 (2026-06-27)

The following updates are based on llama-cpp-python version 0.3.40, please install this version of llama-cpp-python first

- **Preset Template Optimization**: Optimized preset templates to improve out-image quality and user experience. Added detailed preset templates examples for different categories (portrait, design, video, etc.).
- **Rewrite Prompt Style preset information**

<details>
<summary>📋 Older Versions</summary>

#### v3.1.5 (2026-06-18)

The following updates are based on llama-cpp-python version 0.3.40, please install this version of llama-cpp-python first

- **Optimized Node Parameters**: Improved Qwen3 series inference logic. For Qwen3.5, Qwen3.6 and other thinking models using text generation mode, mmproj model must be enabled. Some preset templates may not work, please test and adjust or use non-thinking models
- **Fixed Inference Issues**: Fixed inference failures or content mismatch issues caused by preset template function modifications. Added example guidance to reduce the chance of generated content deviating from expected results
- **Preset Template Optimization**: Improved preset template examples (different preset templates have different optimization directions, e.g., use zimage preset template for portraits, use other models for generation, please adjust according to actual needs)

#### v3.1.4 (2026-06-13)

The following updates are based on llama-cpp-python version 0.3.40, please install this version of llama-cpp-python first

**I. Node Optimization**
- **Optimized Node Parameters**: Removed redundant optional parameters from ASR and TTS nodes, streamlined code to retain only qwen-tts model parameters to avoid parameter errors during use
- **Reasoning Budget Parameter**: Added `reasoning_budget` parameter to support thinking mode control for Qwen3.5-Thinking and similar models (-1=unlimited, 0=disable thinking, N=limit to N tokens), thinking models can now use text generation mode normally

**II. Dependency Management Optimization**
- **Multi-Platform Support**: Optimized `requirements.txt` to automatically select precompiled wheels for Windows/Linux/macOS (manual installation recommended to avoid automatic installation failures)

**III. Code Quality Improvement**
- **Exception Handling Optimization**: Replaced `exit(1)` forced exit with custom exception classes `LlamaCppDependencyError` and `ComfyUINotFoundError` to prevent program crashes
- **Complete Parameter Passing**: Improved parameter passing for `create_chat_completion()`, supporting `presence_penalty`, `typical_p`, `mirostat` and other parameters

**IV. Preset Template Optimization**
- **Template Optimization**: Based on Seedance2.0 model prompts, added emotion and expression-related prompt content. Note: wan and other open-source models may not be suitable, generated content may not achieve expected results, please test and adjust yourself. (Same applies to previously adjusted image and audio preset templates)

#### v3.1.3 (2026-06-06)

**I. Model Support Optimization**
- **Qwen3.6 Model Support**: Added Qwen3.6 series model support, including Qwen3.6-27B and Qwen3.6-35B-A3B large VRAM models, optimized context length configuration (default 8192) and GPU layer allocation (not tested due to limited personal hardware, please understand if runtime errors occur)
- **Gemma4 Model Support**: Improved Gemma-4 series model support, including Gemma-4-E2B, Gemma-4-E4B, and Gemma4-26B-A4B model automatic recognition and loading

**II. Preset Template Adjustment**
- **Template Optimization**: 1.Optimized repetitive instruction content, resolved text generation content confusion caused by logic disorder
2.Restructured output callback to JSON output format for precise image generation control
3.Preset templates now feature intelligent category processing. Please add relevant keyword information such as poster, illustration, photography to prompts for model-based generation and reduced errors
4.Illustrious preset template added ToriiGate model exclusive optimization, ToriiGate model recommended
- **Template Addition**: Added Ideogram-4 model exclusive preset template (usage according to model requirements, generated results may not be ideal, please test yourself, recommended to use qwen3 series models)

#### v3.1.2 (2026-05-31)

**I. Node Optimization**
- **Enhanced MoE Model Support**: Optimized tensor split logic for Mixture of Experts models, improved expert layer allocation for better performance
- **Added Example Output Interface**: New example output interface to help users write better prompts based on preset template examples, improving generated text quality

**II. Dependency Optimization**
- **Dependency Detection Prompt**: New dependency detection feature to notify users if required dependencies are installed, avoiding missing dependency issues during use

**III. Preset Template Adjustment**
- **Template Optimization**: Image templates optimized for shot descriptions and AI-compatible logic descriptions; video templates optimized for content coherence and character expression descriptions; audio templates optimized for emotional expression of audio content (not yet tested, feedback welcome, templates continuously optimized)

#### v3.1.1 (2026-05-10)

**I. Template Optimization**
- **Template Content Adjustment**: Add frame sequence analysis template, optimize video reverse prompt template, enhance facial expression and detail descriptions for all templates involving characters, optimize ERNIE_IMAGE and QWEN_IMAGE_2512 output text classification confusion problem, and adjust some templates to be more suitable for European and American audiences (currently still updating, not completed yet, please understand if the generated results are not ideal)
- **Inference Time Increase**: To achieve more comprehensive prompt generation, the templates output more text content, which will increase the inference time (the amount of text content determines the inference time), please adjust parameters or use custom templates according to actual situation.

**II. Code Maintenance**
- **Bug Fixes**: Fix known issues found in multi-image transmission and video processing
- **Template Optimization**: Optimize template output format, fix output content mismatch problem (some templates only have one output format, switching after output results are consistent)

#### v3.1 (2026-04-30)

**I. Streamlining and Optimization**
- **Node Streamlining**: Remove all nodes that did not meet expectations, remove impractical templates, remove acceleration modules with unsatisfactory results
- **Completion Optimization**: Add parameter optimization for Intel graphics cards

**II. Code Optimization**
- **Code Simplification**: Optimize code structure, reduce redundancy and complexity, improve runtime efficiency, fix TTS voice mismatch and qwen3.5 output issues

**III. Template Enhancement**
- **New Optimized Templates**: Add Illustrious, Anima, ERNIE-Image dedicated preset templates, strengthen video-related template content, adjust qwen-image2512 template positioning

**IV. Documentation Simplification**
- **Documentation Simplification**: Simplify supporting documentation content, improve practicality, remove redundant and unnecessary explanatory content

#### v3.0 (2026-04-19)

**I. Prompt Format Optimization**
- **JSON Format Output**: Multi-image input node prompt output changed to JSON format, including complete task type, theme, narrative style, content focus, target audience, story length, video model type, and specific content fields

**II. Preset Template Adjustment and Optimization**
- **Streamlining Optimization**: Remove impractical templates, retain frequently used and stable templates
- **Classification Reorganization**: Optimize template positioning, reintegrate into a clearer classification system
- **Portrait Template Enhancement**: Add portrait-specific templates, optimize portrait feature capture, expression depiction, pose description and other functions
- **Poster Template Enhancement**: Strengthen poster template's layout, font style, color matching and other design element support

**III. New Model Support**
- **Gemma4 Model Support**: Add Google Gemma 4 series multimodal model support (quantized versions only). Non-quantized Gemma 4 versions cannot be compatible due to dependency limitations and have unsatisfactory performance, so support is not added
- **LFM2.5-VL Model Support**: Add LFM 2.5 Vision series model support (quantized versions only), expand visual understanding capabilities

**IV. Conflict Dependency Handling**
- **Conflict Dependency Handling**: Conflict dependencies embedded in the plugin to avoid conflicts with other user plugins, ensuring stable plugin operation


#### v2.0 (2026-04-03)

**I. Modular Inference Architecture**
- **Smart Hardware Adaptation**: Improve NVIDIA/AMD/CPU three-end adaptation, automatically adjust parameters based on VRAM size
- **Error Recovery Mechanism**: Enhance error handling and recovery capabilities to ensure inference continuity and stability

**II. Image Inference Node Optimization and Adjustment**
- **Unified Inference Node**: Integrate text, image, and audio processing capabilities, support 5 inference modes (text generation, image understanding, audio to text, text to audio, video understanding), achieve cross-modal understanding and generation
- **API Model Node System**: Including API configuration management node, API configuration selector, API configuration manager, and API model switcher, supports multiple API providers including online OpenAI, offline Ollama, llms-py, vllm-omni (requires Linux), enabling remote deployment and distributed inference
- **Parameter Configuration Node**: Add more efficient inference parameter configurations such as Flash Attention and KV Cache to improve inference speed

**III. ASR+TTS Audio Processing System**
- **ASR Model Loader**: Supports Qwen3-ASR speech recognition model, supports multilingual recognition, audio reverse engineering, and timestamp generation
- **TTS Model Loader**: Supports Qwen3-TTS speech synthesis model, supports multiple voice selection and emotion control
- **Forced Aligner Model Loader**: Supports Qwen3-ForcedAligner model, achieves high-precision audio-text alignment and timestamp generation
- **Forced Aligner Inference Node**: Processes audio-text alignment, generates precise timestamp information
- **Multi-Model TTS Node**: Supports multi-role dialogue synthesis, can assign different voices to different roles
- **TTS Alignment Node**: Optimizes audio alignment effect of TTS output

**IV. Video Processing System**
- **Video Loader Node**: Supports video file input, extracts video frame sequence for analysis
- **Video Frame Sampling**: Supports two modes: automatic uniform sampling and manual frame index specification
- **Video Understanding Mode**: Supports video content analysis, storyboard decomposition, video reverse engineering, and subtitle generation
- **Video to Audio and Subtitles**: Extract audio from video and generate synchronized subtitles

**V. New Audio Preset Templates**
- **Audio Subtitle Conversion**: Convert audio content to standard format subtitles, including time codes and synchronized text
- **Video Audio Subtitle Generation**: Generate complete subtitles with audio descriptions based on video content
- **Text to Audio**: Convert text to natural speech, supports multi-person dialogue synthesis and voice selection
- **Audio Analysis**: Analyze audio content, extract key information and emotional features

**VI. Multi-Image Input Node Interface Expansion**
- Double the multi-image input node interfaces, support more image inputs to meet the creation of richer and more coherent story content

#### v1.4.0 (2026-02-23)
- Add Multi-Image Input node with the following features:
  - Dual-mode operation: Image mode analyzes multiple images and creates stories, Text mode generates prompts through option settings
  - Multi-image input support: Supports 1-6 images input, automatic preprocessing and encoding
  - Rich content creation types: Supports 10 types including coherent story, storyboard description, scene analysis, character development, emotional progression, creative writing, script writing, advertising copy, product introduction, educational content
  - Flexible length control: Supports four length options - Short (200 words), Medium (400 words), Detailed (600 words), Complete (1000 words)
  - Multilingual support: Supports Chinese and English output
  - Rich theme selection: Supports 12 themes including Adventure, Romance, Mystery, Sci-Fi, Fantasy, Daily Life, History, Future Technology, Business Marketing, Educational Popular Science, Entertainment Comedy
  - Diverse narrative styles: Supports 4 styles including First Person, Third Person, Omniscient Perspective, Multiple Perspectives Switching
  - Content focus control: Supports 6 focuses including Balanced Development, Emphasize Plot, Emphasize Characters, Emphasize Emotion, Emphasize Visual, Emphasize Dialogue
  - Target audience customization: Supports 5 audiences including General Public, Teenagers, Children, Professionals, Specific Groups
  - Video model optimization: Optimize prompt format for different video generation models such as WAN2.2, LTX2, General Video, Custom
  - Custom prompts: Supports adding custom prompts to guide content creation
  - Image description control: Option to include description of each image before the story
- Add Multi-Image Input node usage documentation link

#### v1.3.0 (2026-02-08)
- Add multiple preset prompt templates: Chinese-English bilingual prompt generation, ultra-high definition image prompt reverse engineering
- Optimize model loading and inference process, improve runtime efficiency
- Improve Chinese translation, enhance localization support
- Add video interface, support video input, add templates for video reverse engineering function
- Add OCR enhancement function, support poster text recognition and style restoration, adapt to prompt reverse engineering needs
- Implement intelligent model detection system, automatically discover and support new VL models added by llama_cpp_python
- Optimize model name inference logic, automatically generate model names according to ChatHandler naming rules
- Expand model support list to ensure backward compatibility with all previously supported models
- Implement model list deduplication function, keep interface clean and organized
- Add support for multiple models
- Add dynamic support function, llama_cpp_python updates and releases support for new model versions, models can be downloaded and used immediately
- Fix known bugs

#### v1.2.0 (2026-01-29)
- Refactor file directory, delete old version files during installation, do not overwrite
- Add comprehensive preset prompt templates for professional AI models:
  - **ZIMAGE - Turbo**: Optimized for Z-Image-Turbo model, supports 8-step Turbo inference to quickly generate 1080P HD images
  - **FLUX2 - Klein**: Designed for FLUX.2 Klein model, creates concise and expressive prompts
  - **LTX-2**: Customized for LTX-2 video generation model, supports dynamic video prompts, can generate high-quality, audio-visual synchronized 4K videos
  - **Qwen - Image Layered**: Created for Qwen-Image-Layered model, supports detailed layered prompts, handles complex compositions and multiple elements
  - **Qwen - Image Edit Combined**: Edit prompt enhancer for image editing tasks
  - **Qwen - Image Dual**: Designed for Qwen Image 2512 model, supports high-resolution generation capabilities
  - **Video - Reverse Prompt**: Video reverse prompt generator, creates detailed video descriptions based on video content (600-1000 words)
  - **WAN - T2V**: Film director style template, adds film elements (time, light source, light intensity, light angle, color tone, shooting angle, lens size, composition)
  - **WAN - I2V**: Video description prompt rewriting expert, emphasizes dynamic content
  - **WAN - FLF2V**: Prompt optimizer, optimizes prompts based on first and last frame images of video, emphasizes motion information and camera movement
- Optimize Chinese-English switching function, improve language adaptability
- Provide bilingual preset templates (English and Chinese), better compatibility for different language models (exclusive presets have word limits to meet model generation requirements while ensuring efficient results. If not meeting requirements, please input in preset box or use external custom)
- Add Chinese-English switching function for generation results

#### v1.1.0 (2026-01-24)
- Refactor node file directory
- Add parameter recommendation settings documentation to help users understand the impact of each parameter on generation results
- Add support types for MiniCPM-V-4.5, LFM2.5-VL-1.6B, GLM-4.6V models
- Add Chinese-English switching function for easy switching between different types of reverse engineering models
- Model loading only supports .gguf format model files

#### v1.0.0 (2026-01-17)
- Add support type for llama-joycaption reverse engineering model
- Add mmproj model switch to support text-only generation
- Add session cleanup node (releases resources occupied by current conversation, reduces occurrence of no results)
- Add model unload node (reduces VRAM usage)
- Add hardware optimization module, adapts to different performance hardware, improves inference speed, ensures smooth use on different hardware

</details>

## Acknowledgments
* [ComfyUI-llama-cpp_vlm](https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm) @lihaoyun6
* [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous