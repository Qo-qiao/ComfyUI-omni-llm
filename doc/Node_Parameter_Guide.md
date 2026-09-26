# ComfyUI-omni-llm Node Parameter Guide & Recommended Settings

This document provides detailed descriptions of the parameters and recommended settings for each node in the ComfyUI-omni-llm plugin, helping users better utilize and adjust the models.

## Dynamic Model Support

This plugin implements an intelligent model detection system that can automatically discover and support new VL models added by llama_cpp_python. When llama_cpp_python updates and supports new model versions, users can directly download and use them without modifying the plugin code.

## Document Structure

This document is organized as follows:
- **Node Parameter Guide**: Detailed descriptions of each node's parameters and their functions
- **Device Performance Adaptation Recommendations**: Recommended settings based on different hardware configurations
- **Supported Models Description**: Features and applicable scenarios of various models
- **FAQ & Solutions**: Common issues and solutions during usage
- **Usage Tips & Best Practices**: Techniques to improve model performance and result quality
- **Prompt Weighting Techniques**: Methods and tips for adding weights to prompts

## 1. Model Loader Nodes

### 1.1 Llama-cpp Model Loader (llama_cpp_model_loader)

### Function
Load and initialize LLM (Large Language Model)/VLM (Vision Language Model) models, serving as the foundation for all other nodes. Supports .gguf format and automatically detects multi-part models.

### Parameter Description

#### Basic Parameters

**Parameter Name: Model File**
- Function: Select the LLM model file to load
- Recommended Setting: Choose an appropriate GGUF model as needed
- Supported Format: .gguf

**Parameter Name: Enable Multimodal (enable_mmproj)**
- Function: When enabled, automatically selects the chat format processor and enables multimodal functionality
- Recommended Setting: Set to True when processing images or needing chat format, set to False for text-only generation
- Note: Requires selecting the corresponding visual encoding model when enabled

**Parameter Name: Visual Encoding Model (mmproj)**
- Function: Select the corresponding visual encoding model file
- Recommended Setting: Select matching model when multimodal is enabled
- Supported Format: .gguf
- Notes: Different models require corresponding mmproj files, ensure version compatibility

**Parameter: ASR Speech Recognition**
- Function: ASR does not need to be manually enabled in the model loader
- Usage: Connect the "Omni LLM ASR Model Loader" node in the workflow and provide an audio input; transcription runs automatically. In Audio-to-Text mode the transcript is the direct output, while in other modes it is automatically merged into the prompt

#### Runtime Mode Parameters

**Parameter Name: Context Length (n_ctx)**
- Range: 1024-327680
- Function: Context length, affecting the length of text that can be processed and preset template completeness
- Recommended Setting:
  - 24GB+ VRAM (5090/4090): 16384
  - 16GB VRAM (4080): 8192
  - 12GB VRAM (4070 Ti/3080): 6144
  - 8GB VRAM (3070/3060): 4096
  - 4-6GB VRAM: 2048
- **Important Note**: Qwen3 series recommended ≤4096, Qwen3.5 recommended ≤2048
- **Key Impact**: Insufficient context length will truncate preset templates, affecting prompt generation quality

**Parameter Name: GPU Model Layers (n_gpu_layers)**
- Range: -1-1000
- Function: Number of model layers loaded to GPU, -1=all layers (GPU mode only)
- Recommended Setting:
  - 24GB+ VRAM: -1 (all layers)
  - 16GB VRAM: -1 (all layers)
  - 12GB VRAM: -1 (all layers)
  - 8GB VRAM: 30 (partial loading)

**Parameter Name: VRAM Limit (vram_limit)**
- Range: -1-24
- Function: VRAM limit (GB), -1=no limit (GPU mode only)
- Recommended Setting: Usually set to -1 or actual VRAM minus 1

#### Image Processing Parameters

**Parameter Name: Maximum Image Encoding Tokens (image_max_tokens)**
- Range: 0-4096
- Function: Maximum number of tokens for image encoding
- Recommended Setting: Keep default 0 (auto)
- **Auto Optimization**: MiMo-VL models automatically set to 256

#### Advanced Parameters

**Parameter Name: Attention Type (attention_type)**
- Range: Auto/Standard/Flash/XFormers
- Function: Select attention mechanism type for the model, affecting inference speed and memory usage
- Recommended Setting: Flash (default, Flash Attention automatically enabled for NVIDIA GPU)

### Smart Recommendations
- The system automatically recommends default values for device_mode, n_ctx, and n_gpu_layers based on hardware performance
- Low-performance devices automatically reduce default parameters to ensure smooth operation
- CPU mode automatically ignores GPU-related parameters, no manual adjustment needed
- Batch parameters (n_batch, n_ubatch, n_threads, n_threads_batch) are now automatically adjusted based on hardware performance
- **Auto Optimization Items**: image_min_tokens, cache_prompt are now automatically determined in the background, no manual setting needed

## 2. Llama-cpp Unified Inference Node (llama_cpp_unified_inference)

### Function
Performs LLM/VLM/Omni model inference, supporting text-only, single image, multiple images, audio, and video inputs. Serves as a unified inference entry point.

**Note**: This node supports all model types (LLM, VLM, Omni), no need to distinguish between model types.

#### Model Configuration

**Parameter Name: Llama Model**
- Function: Select loaded LLM/VLM/Omni model for inference processing
- Recommended Setting: Connect to model loader node
- Support: All model types (LLM, VLM, Omni)

#### Inference Mode

**Parameter Name: Inference Mode (inference_mode)**
- **Impact Level: High**
- Function: Select inference mode to determine input type and task type
- Options:
  - `[Basic] Text Generation`: Text-only mode, processes text input only, generates prompts
  - `[Basic] Image Understanding`: Image processing mode, processes single or multiple images, reverse-engineers prompts
  - `[Basic] Batch Image Understanding`: Processes multiple images at once, reduces inference calls
  - `[Basic] Audio to Text`: Uses ASR model to convert audio to text
  - `[Advanced] Video Understanding`: Processes video files, extracts frames for analysis

#### Prompt Configuration (Core Parameters)

**Parameter Name: Preset Prompt Template (preset_prompt)**
- **Impact Level: Very High**
- Function: Select preset prompt template, determines the structure and style of prompt generation

**Parameter Name: System Prompt (system_prompt)**
- **Impact Level: High**
- Function: Define AI assistant role and behavior, may include preset template placeholders
- Default: "You are an excellent AI prompt processing expert."
- **Key Impact**: System prompt guides the model's behavioral patterns, affecting output style

**Parameter Name: User Input Text (text_input)**
- **Impact Level: High**
- Function: User input text, serves as the user message content in the conversation
- **Optimization Tips**:
  - Image reverse engineering: Provide detailed visual element descriptions (character, environment, lighting, composition)
  - Prompt expansion: Provide core creative ideas and scene descriptions, including emotional tone
  - Multi-domain design: Clearly specify design type (keywords like poster, UI, portrait)

#### Language Settings

**Parameter Name: Preset Template Language (prompt_language)**
- Function: Call preset template in corresponding language
- Options: Chinese / English

**Parameter Name: Response Language (response_language)**
- Function: Select output text language
- Options: Chinese / English

**Parameter Name: ASR Language (asr_language)**
- Function: Target language for ASR speech recognition
- Options: Auto Detect / Chinese / English / Japanese / Korean / French / German / Spanish

#### Output Format Settings

**Parameter Name: Output Format (output_format)**
- **Impact Level: Medium**
- Function: Control output text format
- Options:
  - `natural`: Output pure text content in natural paragraph format
  - `structured`: Output JSON structured text content

#### Video Processing Parameters

**Parameter Name: Maximum Frames (video_max_frames)**
- Range: 2-1024
- Function: Maximum frames for video processing
- Recommended Setting: Between 16-32
- Performance Impact: More frames provide more comprehensive analysis but take longer to process

**Parameter Name: Frame Sampling Mode (video_sampling)**
- Function: Video frame sampling method
- Options:
  - `Auto Uniform Sampling`: Uniformly extract specified number of frames from video
  - `Manual Frame Indices`: Custom frame indices to extract
- Recommended Setting: Auto Uniform Sampling (default)

**Parameter Name: Manual Frame Indices (video_manual_indices)**
- Function: Frame indices in manual mode, only effective in manual sampling
- Recommended Setting: Input frame indices as needed, e.g., "0,5,10,15" or "0-10"
- Format: Comma-separated numbers

#### Image Processing Parameters

**Parameter Name: Maximum Size (image_max_size)**
- Range: 128-16384
- Function: Maximum edge length for image processing (pixels)
- Recommended Setting:
  - Low-performance devices: 256
  - High-performance devices: 512
- Balance: Larger size captures better details but requires more VRAM

#### Generation Parameters

**Parameter Name: Random Seed (seed)**
- Range: 0-0xffffffffffffffff
- Function: Random seed for reproducible results
- Recommended Setting: 0 (random) or fixed value

**Parameter Name: Force Offload (force_offload)**
- Function: Force unload model to release VRAM after inference
- Recommended Setting: Usually set to False
- Use Case: Only use when immediate VRAM release is needed

**Parameter Name: Save States (save_states)**
- Function: Save model state for later recovery
- Recommended Setting: Set to True for continuous conversations
- Advantage: Maintains conversation context, improves multi-turn interaction coherence

#### Optional Inputs

**Parameter Name: Parameters**
- Function: Additional generation parameter configuration
- Recommended Setting: Optional, uses default parameters when not connected
- Source: Connect to parameter setting node

**Parameter Name: Images**
- Function: Image input (for image understanding mode)
- Recommended Setting: Provide when processing images

**Parameter Name: Video**
- Function: Video input (for video understanding mode)
- Recommended Setting: Provide when processing video

**Parameter Name: Audio**
- Function: Audio input (for ASR recognition)
- Recommended Setting: Provide when processing audio

**Parameter Name: TTS Model (tts_model)**
- Function: TTS model input (for speech synthesis)
- Recommended Setting: Connect to TTS loader when speech output is needed

**Parameter Name: ASR Model (asr_model)**
- Function: ASR model input (for speech recognition)
- Recommended Setting: Connect to ASR loader when speech recognition is needed

### Output Description

- **output**: Generated text output
- **output_list**: Generated text list (supports batch output)
- **state_uid**: Conversation state ID
- **audio**: Generated audio data (only valid in text-to-audio mode)

## Output Format Options: Natural Paragraph vs Structured Output

This plugin supports two output format types for prompt templates, providing flexibility for different use cases.

### 2.1 Natural Paragraph Output (自然段落)

Natural paragraph output organizes all prompt elements into a coherent, flowing English paragraph without explicit field markers.

**Characteristics:**
- All visual elements (subject, lighting, composition, color, atmosphere, etc.) are integrated into seamless paragraphs
- No field labels or markers like 【】 or **
- Ideal for direct use in image/video generation prompts
- Easier to read and understand at a glance
- Best for users who prefer narrative-style descriptions

**Example:**
```
A beautiful anime girl with long flowing black hair, gentle smile, standing in a cherry blossom garden. Masterpiece quality with cel shading and clean line art. Soft lighting creates dreamy atmosphere. She wears a white sailor uniform with red accents, dynamic pose with wind blowing through her hair. Background features falling cherry blossom petals against a gradient pink sky.
```

### 2.2 Structured Output (结构化输出)

Structured output uses field markers with 【】 brackets to organize information into distinct categories, making it easier to locate specific details.

**Characteristics:**
- Each field is clearly labeled with 【】 brackets
- Fields include: 【Subject Description】【Art Style】【Lighting】【Composition】【Color Palette】【Details】【Technical Parameters】【Clothing】, etc.
- Ideal for systematic analysis and review
- Easier to modify specific elements without rewriting entire prompts
- Best for users who need precise control over prompt components

**Example:**
```
【Subject Description】Beautiful anime girl, long flowing black hair, gentle smile
【Art Style】Masterpiece, best quality, ultra-detailed, anime style, cel shading, clean line art
【Lighting】Soft lighting, gentle sunlight, rim lighting, dreamy atmosphere
【Composition】Dynamic pose, wind blowing through hair, full body, looking at viewer
【Color Palette】White, red, pink gradient, soft pastel tones
【Clothing】White sailor uniform with red accents, pleated skirt, frills
【Background】Cherry blossom garden, falling petals, gradient pink sky
```

## 3. Llama-cpp Parameters Node (llama_cpp_parameters)

### Function
Set detailed parameters for LLM inference to control the quality and style of generated text.

### Parameter Description

#### Core Parameters (Highly Recommended to Understand)

**Parameter Name: Maximum Tokens (max_tokens)**
- **Impact Level: Very High**
- Range: 0-4096
- Function: Maximum number of generated tokens, directly affecting output text length
- Recommended Setting:
  - Short answers: 256-512
  - Detailed descriptions: 768-1024
  - Long text generation: 1024-2048
- **Key Impact**: Too small will truncate prompts, too large increases inference time

**Parameter Name: Temperature**
- **Impact Level: Very High**
- Range: 0.0-2.0
- Function: Generation temperature, higher values are more random, lower values are more deterministic
- Recommended Setting: 0.6-0.8
- **Detailed Guide**:
  - Low temperature (0.1-0.4): Suitable for scenarios requiring accurate answers, e.g., QA, code generation
  - Medium temperature (0.5-0.8): Suitable for most scenarios, balances accuracy and creativity
  - High temperature (0.9-1.5): Suitable for creative tasks, e.g., story generation, poetry

**Parameter Name: Top-P Sampling (top_p)**
- **Impact Level: High**
- Range: 0.0-1.0
- Function: Nucleus sampling threshold, controls generation diversity
- Recommended Setting: 0.85-0.9
- **Effect Comparison**:
  - Low top_p (<0.8): More conservative generation, higher accuracy
  - High top_p (>0.9): More open generation, stronger diversity

**Parameter Name: Top-K Sampling (top_k)**
- **Impact Level: High**
- Range: 0-1000
- Function: Number of sampling candidates, smaller values produce more focused generation
- Recommended Setting: High-performance devices: 30, Low-performance devices: 20
- **Impact Analysis**:
  - Low top_k (<20): More deterministic generation, suitable for factual tasks
  - High top_k (>50): More diverse generation, suitable for creative tasks

#### Secondary Parameters (Adjust as Needed)

**Parameter Name: Repeat Penalty (repeat_penalty)**
- Range: 0.0-10.0
- Function: Repeat penalty, prevents repetitive content generation
- Recommended Setting: 1.0
- Use Case: When repetitive content occurs, gradually increase to 1.1-1.3

**Parameter Name: Presence Penalty (presence_penalty)**
- Range: 0.0-2.0
- Function: Presence penalty, encourages new content generation
- Recommended Setting: 1.0
- Use Case: Avoid topic drift in long text generation

**Parameter Name: Minimum Probability (min_p)**
- Range: 0.0-1.0
- Function: Minimum sampling probability, prevents complete neglect of low-probability tokens
- Recommended Setting: 0.05

**Parameter Name: Frequency Penalty (frequency_penalty)**
- Range: 0.0-1.0
- Function: Frequency penalty, reduces high-frequency token occurrence
- Recommended Setting: 0.0

**Parameter Name: Typical Sampling (typical_p)**
- Range: 0.0-1.0
- Function: Typical sampling threshold, controls generation typicality
- Recommended Setting: 1.0

#### Advanced Parameters (For Advanced Users)

**Parameter Name: Mirostat Mode (mirostat_mode)**
- Range: 0-2
- Function: Mirostat sampling mode: 0=off, 1=basic, 2=version 2
- Recommended Setting: 0 (off)
- Use Case: Try mode 1 or 2 for more consistent generation quality

**Parameter Name: Mirostat Eta (mirostat_eta)**
- Range: 0.0-1.0
- Function: Mirostat learning rate
- Recommended Setting: 0.1

**Parameter Name: Mirostat Tau (mirostat_tau)**
- Range: 0.0-10.0
- Function: Mirostat target perplexity
- Recommended Setting: 5.0

#### Session Management Parameters

**Parameter Name: Conversation State ID (state_uid)**
- Range: -1-999999
- Function: Conversation state ID, -1=use node unique ID
- Recommended Setting: -1
- Session Management: Use different state_uid to maintain multiple independent sessions

**Parameter Name: Reasoning Budget (reasoning_budget)**
- Range: -1-1024
- Function: Reasoning budget (for Qwen3.5-Thinking and other thinking-mode supported models)
- Recommended Setting: -1 (unlimited)
- Use Case: 0=disable thinking mode, N=limit to N thinking tokens

### Parameter Adjustment Tips

1. **Control Output Length**: Adjust `max_tokens`. Larger values produce longer output but consume more resources
2. **Control Randomness**: `temperature` is the most commonly used parameter. Lower = more deterministic, higher = more creative
3. **Balance Diversity and Accuracy**: `top_p` and `top_k` are usually used together
4. **Avoid Repetitive Content**: Increase `repeat_penalty` to reduce repetition
5. **Parameter Priority Recommendations**:
   - Primary adjustment: `temperature`, `max_tokens`, `top_p`/`top_k`
   - Secondary adjustment: `repeat_penalty`, `presence_penalty`
   - Advanced users: `min_p`, `mirostat` related parameters
6. **Common Scenario Settings**:
   - **Factual QA**: temperature=0.1-0.3, top_k=10, top_p=0.7
   - **Creative Writing**: temperature=0.8-1.2, top_k=50, top_p=0.95
   - **Code Generation**: temperature=0.2-0.4, top_k=20, top_p=0.8
   - **Dialogue**: temperature=0.6-0.8, top_k=30, top_p=0.9

## 3.5 Llama-cpp Model Cleanup Node (llama_cpp_clean_states)

### Function
Clean up model states and VRAM resources, combining the original cleanup states and unload model functionality.

### Parameter Description

**Parameter Name: State UID (state_uid)**
- Range: -1-999999
- Function: Clean up specific session state, -1=clean all states
- Recommended Setting: -1

**Parameter Name: Clean LLM (clean_llm)**
- Function: Clean up LLM model
- Recommended Setting: True

**Parameter Name: Clean ASR (clean_asr)**
- Function: Clean up ASR model
- Recommended Setting: True

**Parameter Name: Clean TTS (clean_tts)**
- Function: Clean up TTS model
- Recommended Setting: True

**Parameter Name: Clean Aligner (clean_aligner)**
- Function: Clean up forced aligner model
- Recommended Setting: True

**Parameter Name: Unload All ComfyUI Models (unload_all_comfyui_models)**
- Function: Unload all ComfyUI models (not just omni-llm models)
- Recommended Setting: False
- **Note**: When enabled, calls ComfyUI's mm.unload_all_models(), releasing more VRAM


## 4. ASR Model Parameters

### 4.1 ASR Model Loader Parameters (llama_cpp_asr_loader)

#### Node Display Options (Manually Adjustable)

- **ASR Model**: Select ASR model file, choose appropriate ASR model as needed, only qwen3-asr is currently supported
- **GPU Model Layers**: Number of model layers loaded to GPU: 24GB+ VRAM: -1, 16GB VRAM: -1, 12GB VRAM: -1, 8GB VRAM: 20, Range: -1-1000
- **Language**: Recognition language, select based on audio content, options: auto/zh/en/ja/ko/fr/de/es
- **Task**: Task type: transcribe/translate (translate to English), options: transcribe/translate

#### Audio Input Requirements

- **Supported Formats**: WAV, MP3, FLAC and other common audio formats
- **Recommended Sample Rate**: 16000Hz or 22050Hz
- **Channels**: Mono or stereo (auto-handled)

## 5. Multi-Image Input Node Guide

### 5.1 Node Overview

**Multi-Image Input (Story Creation)** node supports two working modes:

1. **Image Mode**: Analyze multiple images and create story content, supports multiple video generation models (WAN2.2, LTX2, etc.)
2. **Text Mode**: Generate prompts through option settings, no image input required

### 5.2 Node Functions

#### Main Functions
1. **Dual Mode Support**: Flexible switching between image mode and text mode
2. **Multi-Image Input**: Supports input of multiple images (image mode)
3. **Auto Preprocessing**: Automatic scaling and encoding of images
4. **Story Creation**: Generate story content suitable for video generation
5. **Flexible Configuration**: Supports multiple story types, lengths, and styles
6. **Multiple Applications**: Supports story creation, script writing, advertising copy, and other content types
7. **Image Output**: Pass image data to inference node

### 5.3 Input Parameters

#### Working Mode
- **mode** (dropdown): Working mode
  - Image Mode: Analyze image content for story creation
  - Text Mode: Generate prompts through option settings

#### Image Input (Image Mode Only)
- **image1** ~ **image12** (IMAGE): Input 1-12 images
  - Can connect one or more images
  - At least one image required (image mode)
  - Auto-detect image count
  - Auto preprocessing and encoding

#### Configuration Parameters
- **story_type** (dropdown): Content creation type
  - Coherent Story/Storyboard Description/Scene Analysis/Character Development/Emotional Progression/Creative Writing/Script Writing/Advertising Copy/Product Introduction/Educational Content

- **story_length** (dropdown): Content length
  - Short (within 200 words)/Medium (within 400 words)/Detailed (within 600 words)/Complete (within 1000 words)

- **language** (dropdown): Output language (Chinese / English)

- **max_size** (integer): Maximum image size (pixels), range 128-512, default 256

- **custom_prompt** (text): Custom prompt

- **include_image_descriptions** (boolean): Include image descriptions (image mode only)

- **story_theme** (dropdown): Content theme
  - No Specific Theme/Adventure/Romance/Mystery/Sci-Fi/Fantasy/Daily Life/Historical/Future Tech/Business Marketing/Educational/Comedy

- **narrative_style** (dropdown): Narrative style
  - First Person/Third Person/Omniscient/Multiple Perspectives

- **content_focus** (dropdown): Content focus
  - Balanced/Emphasize Plot/Emphasize Characters/Emphasize Emotion/Emphasize Visual/Emphasize Dialogue

- **target_audience** (dropdown): Target audience
  - General Public/Teenagers/Children/Professionals/Specific Group

- **video_model** (dropdown): Video generation model type
  - WAN2.2: Emphasizes scene description and visual elements
  - LTX2: Focuses on detailed description and emotional expression
  - General Video: Balances scene description and narrative fluency
  - Custom: Custom video generation model

### 5.4 Output Parameters

- **prompt** (STRING): Generated content creation prompt
- **images** (IMAGE): Image data (returns preprocessed images in image mode, None in text mode)

### 5.5 Usage Methods

#### Image Mode Examples

**Example 1: Coherent Story Creation**
- Mode: Image Mode
- Story Type: Coherent Story
- Story Length: Medium (within 400 words)
- Language: Chinese
- Story Theme: Adventure
- Narrative Style: First Person

**Example 2: Storyboard Description**
- Mode: Image Mode
- Story Type: Storyboard Description
- Story Length: Detailed (within 600 words)
- Language: Chinese
- Story Theme: No Specific Theme
- Narrative Style: Third Person

WAN2.2 output format: 3-4 shots, 3-5 seconds each
LTX2 output format: 5-6 shots, 5-10 seconds each

#### Text Mode Examples

**Example 3: Creative Writing**
- Mode: Text Mode
- Story Type: Creative Writing
- Story Length: Medium (within 400 words)
- Language: Chinese
- Story Theme: Sci-Fi
- Narrative Style: First Person
- Content Focus: Emphasize Plot
- Target Audience: Teenagers

**Example 4: Script Writing**
- Mode: Text Mode
- Story Type: Script Writing
- Story Length: Short (within 200 words)
- Language: Chinese
- Story Theme: Business Marketing
- Narrative Style: Third Person
- Content Focus: Emphasize Dialogue
- Target Audience: General Public

### 5.6 Best Practices

#### Mode Selection Recommendations
- **Image Mode**: Have specific image materials to analyze
- **Text Mode**: No image materials, need to create from scratch

#### Image Selection Recommendations
1. **Coherence**: Select images with coherent content
2. **Moderate Quantity**: Recommend 1-12 images
3. **Quality Priority**: Use high-quality images
4. **Diverse Scenes**: Include different scenes and angles

#### Parameter Setting Recommendations
1. **Content Type**:
   - WAN2.2: Recommend "Coherent Story" or "Storyboard Description"
   - LTX2: Recommend "Character Development" or "Emotional Progression"
2. **Content Length**:
   - Short video (10-30 seconds): Within 200 words
   - Medium video (30-60 seconds): Within 400 words
   - Long video (60-120 seconds): Within 600 words

### 5.7 FAQ

**Q1: What's the difference between Image Mode and Text Mode?**
- **Image Mode**: Requires images, model analyzes image content and creates stories
- **Text Mode**: No images needed, generates prompts through option settings

**Q2: Why is the story not coherent enough?**
Possible reasons: Image content not coherent/Inappropriate story type selection/Unclear custom prompt
Solutions: Select more coherent images/Try different story types/Add more specific custom prompts

**Q3: How to correctly connect image data to inference node?**
- **Image Mode**: prompt→custom_prompt, images→images
- **Text Mode**: prompt→custom_prompt, no need to connect images

## 6. Prompt Weighting Techniques & Methods

### 6.1 Semantic Priority Ordering

The preset templates in this plugin use a **semantic priority ordering** mechanism, which automatically sorts and emphasizes different elements by importance when generating prompts.

#### Core Priority Rules

Taking the `THICKPAINT_ROLE_ZH` (Next-gen CG Thick Paint 3D Character Portrait) template as an example, its semantic weight priority is:

| Priority | Element Category | Description |
|----------|-----------------|-------------|
| 1 (Highest) | Character Structure & Material Quality | Body proportions, skeletal structure, PBR materials, subsurface scattering, etc. |
| 2 | Pose, Expression & Styling | Dynamic poses, emotional expression, clothing matching, accessory design |
| 3 | Lighting, Color & Volume | Main light direction, rim light, ambient light, color matching |
| 4 | Scene Environment | Indoor/outdoor, spatial structure, background elements |
| 5 (Lowest) | Rendering/Brush Parameters | Technical parameters, style tags, etc. |

#### Practical Effects

When you input "Ancient Chinese fantasy girl, blue hanfu", the template will:
1. **Prioritize**: Character facial features refinement, hanfu material quality, skin translucency
2. **Secondarily represent**: Girl's pose (e.g., hands clasped in prayer), expression (gentle devotion), hairstyle design
3. **Thirdly process**: Lighting atmosphere (soft morning light, golden rim light), color matching (light blue + gold + white)
4. **Finally supplement**: Scene environment (ancient courtyard, falling cherry blossoms), style tags (beautiful, mysterious, artistic hand-painted)

### 6.2 Emphasizing Elements via text_input

You can use the following techniques in the `User Input Text (text_input)` field to emphasize or de-emphasize specific elements:

> **Note**: The following techniques are used to **influence the LLM's prompt generation behavior**, making the model pay more attention to certain elements during generation. They are not direct weight syntax for image generation models (SD/Flux, etc.), but indirectly achieve emphasis by guiding the LLM to output more detailed descriptions. Effectiveness varies by model; experimentation is recommended.

#### Method 1: Repeat Keywords

Increase importance by repeating keywords:

```
Ancient Chinese fantasy girl, Ancient Chinese fantasy girl, blue hanfu, blue hanfu, exquisite facial features
```

#### Method 2: Use Emphasis Markers

Add emphasis markers before keywords:

```
[Emphasis] Ancient Chinese fantasy girl, [Must] blue hanfu, [Strongly Required] exquisite facial features, [De-emphasize] background environment
```

#### Method 3: Adjective Stacking

Enhance weight by describing the same element with multiple adjectives:

```
An extremely beautiful, exquisitely featured, well-proportioned, elegant ancient Chinese fantasy girl
```

#### Method 4: Explicit Instructions

Directly tell the model which elements are more important:

```
Ancient Chinese fantasy girl, focus on facial expression and clothing details, simplify background processing
```

### 6.3 SD/Flux Style Weight Syntax

Although prompts generated by this plugin do not directly include weight syntax, you can manually add weights after generation or when passing prompts to downstream image generation nodes.

#### Common Weight Syntax

| Syntax Format | Example | Effect Description |
|---------------|---------|-------------------|
| `(word:1.5)` | `(masterpiece:1.5)` | Increase "masterpiece" weight to 1.5x |
| `(word:0.5)` | `(background:0.5)` | Decrease "background" weight to 0.5x |
| `[word:1.5]` | `[beautiful eyes:1.5]` | SDXL format weight syntax |
| `word++` | `beautiful++` | Slightly increase weight (supported by some models) |
| `word--` | `background--` | Slightly decrease weight (supported by some models) |

#### Weight Value Reference

| Weight Value | Effect | Application Scenario |
|--------------|--------|---------------------|
| 0.3-0.5 | Significantly reduced | Elements to completely avoid, backgrounds to de-emphasize |
| 0.6-0.8 | Moderately reduced | Secondary elements, parts needing reduced presence |
| 1.0 | Default weight | Normal elements |
| 1.2-1.5 | Moderately increased | Important elements, features needing emphasis |
| 1.6-2.0 | Significantly increased | Core elements, must-emphasize features |
| 2.0+ | Extremely increased | Very critical elements, use with caution |

#### Usage Example

Combine prompts generated by this plugin with weight syntax:

```
Original prompt:
A beautiful ancient Chinese fantasy girl, blue hanfu, exquisite facial features, gentle smile, ancient courtyard, falling cherry blossoms

With weights added:
(masterpiece:1.5), (best quality:1.5), (ultra-detailed:1.2), 
An extremely beautiful ancient Chinese fantasy girl, (blue hanfu:1.3), (exquisite facial features:1.4), (gentle smile:1.3),
ancient courtyard, (falling cherry blossoms:1.2), (background:0.6)
```

### 6.4 Negative Prompt Weighting

Negative prompts can also use weight syntax to control their strength:

```
negative prompt:
(ugly:1.5), (duplicate:1.3), (morbid:1.2), (mutilated:1.2),
(tranny:1.3), mutated hands, (poorly drawn hands:1.4),
(blurry:1.2), (bad anatomy:1.3), (bad proportions:1.2)
```

### 6.5 Practical Application Recommendations

#### Scenario 1: Emphasize Character Features

When you want the model to focus more on facial features and clothing:

```
text_input: Ancient Chinese fantasy girl, focus on exquisite facial features and clothing details
After generation manual adjustment: (exquisite facial features:1.5), (clothing details:1.4), (background:0.6)
```

#### Scenario 2: Control Background Strength

When you want to simplify the background and highlight the subject:

```
text_input: Sci-fi warrior, simplified background, emphasize subject
After generation manual adjustment: (sci-fi warrior:1.3), (background:0.5), (environmental details:0.6)
```

#### Scenario 3: Emphasize Style Characteristics

When you want to strengthen a specific artistic style:

```
text_input: Cyberpunk style, strong neon effects
After generation manual adjustment: (cyberpunk:1.5), (neon effects:1.4), (neon lights:1.3)
```

### 6.6 Notes

1. **Avoid excessively high weights**: Extremely high weight values (e.g., >3.0) may cause unstable generation or produce strange results
2. **Balance positive and negative weights**: Maintain a reasonable ratio between positive and negative prompt weights
3. **Test different combinations**: Recommend multiple tests to find the optimal weight combination for your needs
4. **Model differences**: Different models support weight syntax to varying degrees; adjust based on actual results
5. **Avoid conflicts**: Do not set high weights for contradictory elements simultaneously (e.g., emphasizing both "realistic" and "cartoon")