# ComfyUI-omni-llm Node Parameter Description and Recommended Settings

This document details the parameter functions and recommended settings for each node in the ComfyUI-omni-llm plugin, helping users better understand, configure, and optimize the models.

## Dynamic Model Support

This plugin implements an intelligent model detection system that can automatically discover and support new VL models added in llama_cpp_python. When llama_cpp_python updates and supports new model versions, users can directly download and use them without modifying plugin code.

## Document Structure

This document is organized as follows:
- **Node Parameter Descriptions**: Detailed introduction to each node's parameters and their functions
- **Device Performance Adaptation Recommendations**: Recommended settings based on different hardware configurations
- **Supported Models Description**: Characteristics and applicable scenarios of various models
- **Common Questions and Solutions**: Problems that may be encountered during use and their solutions
- **Usage Suggestions and Best Practices**: Tips to improve model performance and result quality

## 1. Model Loading Nodes

### 1.1 Llama-cpp Model Loading Node (llama_cpp_model_loader)

### Function
Loads and initializes LLM (Large Language Model)/VLM (Vision Language Model) models, serving as the foundation for all other nodes. Supports .gguf and .safetensors formats, with automatic multi-segment model detection.

### Parameter Description

Parameter name: model
  Function description: Select the LLM model file to load
  Recommended setting: Choose an appropriate GGUF model as needed
  Examples:
  - Lightweight tasks: Moondream2, MobileVLM
  - Medium tasks: LLaVA-1.6, MiniCPM-V-4.5
  - Complex tasks: Qwen3-VL, GLM-4.6V

Parameter name: auto_config
  Function description: Automatically selects dialogue format handler and enables multimodal functionality when activated. Recommended to turn off for prompt generation.
  Recommended setting: Set to True for image processing or dialogue format needs, False for text-only generation

Parameter name: mmproj
  Function description: Corresponding visual encoding model file
  Recommended setting: Select a matching model when multimodal functionality is enabled
  Notes: Different models require corresponding mmproj files, ensure version matching

Parameter name: device_mode
  Function description: Runtime mode selection, CPU or GPU
  Recommended setting:
  - 8GB+ VRAM: GPU (recommended)
  - <8GB VRAM: CPU
  - No GPU: CPU

Parameter name: n_ctx
  Adjustment range: 1024-327680
  Function description: Context length, affecting the length of text that can be processed
  Recommended setting:
  - 24GB+ VRAM (5090/4090): 16384
  - 16GB VRAM (4080): 8192
  - 12GB VRAM (4070 Ti/3080): 6144
  - 8GB VRAM (3070/3060): 4096
  - 4-6GB VRAM: 2048
  Rule of thumb: Context length is directly proportional to task complexity and inversely proportional to hardware performance

Parameter name: n_gpu_layers
  Adjustment range: -1-1000
  Function description: Number of model layers loaded to GPU (-1 = load all layers, only effective in GPU mode)
  Recommended setting:
  - GPU mode:
    - 24GB+ VRAM (5090/4090): -1 (load all layers)
    - 16GB VRAM (4080): -1 (load all layers)
    - 12GB VRAM (4070 Ti/3080): -1 (load all layers)
    - 8GB VRAM (3070/3060): 30 (partial load)
  - CPU mode: Automatically ignored, no need to set

Parameter name: vram_limit
  Adjustment range: -1-24
  Function description: VRAM limit (GB, -1 = no limit, only effective in GPU mode)
  Recommended setting:
  - GPU mode: Usually set to -1 or actual VRAM size minus 1 (e.g., 15 for 16GB VRAM)
  - CPU mode: Automatically ignored, no need to set

Parameter name: n_batch
  Adjustment range: 512-8192
  Function description: Maximum batch size for prompt processing, affecting inference speed
  Recommended setting:
  - NVIDIA GPU: 2048 (default)
  - AMD GPU: 1024
  - Low-performance devices: 1024
  Performance impact: Larger value means faster inference speed, but higher VRAM usage

Parameter name: n_ubatch
  Adjustment range: 128-2048
  Function description: Physical batch size, affecting memory usage
  Recommended setting:
  - NVIDIA GPU: 512 (default)
  - AMD GPU: 512
  - Low-performance devices: 256
  Performance impact: Smaller value means lower VRAM usage, but may affect inference speed

Parameter name: n_threads
  Adjustment range: 0-64
  Function description: Number of generation threads, 0 = auto (usually CPU cores/2)
  Recommended setting: 0 (auto)
  Performance impact: Too many threads may cause contention, auto mode is recommended

Parameter name: n_threads_batch
  Adjustment range: 0-64
  Function description: Number of batch processing threads, 0 = auto (usually CPU cores)
  Recommended setting: 0 (auto)
  Performance impact: Affects batch processing performance, auto mode is recommended

Parameter name: flash_attention
  Adjustment range: Auto/Enabled/Disabled
  Function description: Flash Attention acceleration (effective for NVIDIA GPU)
  Recommended setting: Auto (default)
  Performance impact:
  - Enabled: Significantly improves inference speed (NVIDIA GPU)
  - Disabled: Disables Flash Attention
  - Auto: Automatically selects based on GPU

Parameter name: offload_kqv
  Adjustment range: True/False
  Function description: Offload K, Q, V to GPU to improve performance
  Recommended setting: True (default)
  Performance impact: Significantly improves inference speed when enabled

Parameter name: use_mmap
  Adjustment range: True/False
  Function description: Use memory mapping to load model (reduces memory usage)
  Recommended setting: True (default)
  Performance impact: Reduces memory usage when enabled, but may slightly increase loading time

Parameter name: use_mlock
  Adjustment range: True/False
  Function description: Force model to be locked in memory (prevents swapping to disk)
  Recommended setting: False (default)
  Usage scenario: Enable to improve performance when memory is sufficient

Parameter name: f16_kv
  Adjustment range: True/False
  Function description: Use half-precision KV cache (reduces VRAM usage)
  Recommended setting: True (default)
  Performance impact: Reduces VRAM usage by about 50% when enabled

Parameter name: low_vram
  Adjustment range: True/False
  Function description: Low VRAM mode (sacrifices some performance to reduce VRAM usage)
  Recommended setting: False (default)
  Usage scenario: Enable when VRAM is insufficient

Parameter name: image_min_tokens
  Adjustment range: 0-4096
  Function description: Minimum number of image encoding tokens
  Recommended setting: Keep default value 0

Parameter name: image_max_tokens
  Adjustment range: 0-4096
  Function description: Maximum number of image encoding tokens
  Recommended setting: Keep default value 0

### Intelligent Recommendations
- The system automatically recommends default values for device_mode, n_ctx and n_gpu_layers based on hardware performance
- Low-performance devices will automatically reduce default parameters to ensure smooth operation
- CPU mode automatically ignores GPU-related parameters, no need to manually adjust
- Batch processing parameters (n_batch, n_ubatch, n_threads, n_threads_batch) are now automatically adjusted based on hardware performance

### 1.2 CPU/GPU Runtime Mode Details

#### device_mode Parameter Description

The `device_mode` parameter allows users to choose the runtime mode of the model, providing flexible hardware adaptation solutions.

#### GPU Mode (Recommended)

**Applicable Scenarios**:
- Available GPU VRAM
- Need fast inference
- Processing large models or complex tasks

**Features**:
- Fast inference speed, suitable for real-time applications
- Supports larger models and longer contexts
- Automatic VRAM estimation and optimization
- Intelligent parameter adjustment based on VRAM size

**Parameter Settings**:
- `n_gpu_layers`: Controls the number of model layers loaded to GPU, -1 means load all
- `vram_limit`: VRAM limit (GB), -1 means no limit
- `n_batch`: Batch size, affecting inference speed (NVIDIA: 2048, AMD: 1024)
- `n_ubatch`: Physical batch size, affecting memory usage (NVIDIA: 512, AMD: 512)
- `n_threads`: Number of generation threads, 0 = auto
- `n_threads_batch`: Number of batch processing threads, 0 = auto
- `flash_attention`: Flash Attention acceleration (Auto/Enabled/Disabled)
- `offload_kqv`: Offload K, Q, V to GPU
- `use_mmap`: Use memory mapping to load model
- `use_mlock`: Force model to be locked in memory
- `f16_kv`: Use half-precision KV cache
- `low_vram`: Low VRAM mode

**Recommended Configuration**:
- 24GB+ VRAM: n_gpu_layers = -1, vram_limit = 24
- 16GB VRAM: n_gpu_layers = -1, vram_limit = 16
- 12GB VRAM: n_gpu_layers = -1, vram_limit = 12
- 8GB VRAM: n_gpu_layers = 30, vram_limit = 8

**Performance**:
- Inference speed: Fast (depends on GPU performance)
- VRAM usage: High (depends on model size)
- Suitable tasks: All types, especially real-time applications

#### CPU Mode

**Applicable Scenarios**:
- No GPU or insufficient GPU VRAM
- Need to use CPU for inference
- Low-performance hardware (<8GB VRAM)
- Need maximum compatibility

**Features**:
- Does not depend on GPU VRAM
- Automatically ignores GPU-related parameters
- Slower inference speed, but good compatibility
- Suitable for all hardware configurations

**Parameter Settings**:
- In CPU mode, n_gpu_layers and vram_limit parameters are automatically ignored
- No need to manually adjust these parameters
- System automatically sets n_gpu_layers = 0, vram_limit = -1

**Recommended Configuration**:
- Suitable for all hardware configurations
- Suitable for small models and simple tasks
- Alternative solution when VRAM is insufficient

**Performance**:
- Inference speed: Slow (depends on CPU performance)
- VRAM usage: Low (only uses system memory)
- Suitable tasks: Simple tasks, offline processing

#### Smart Defaults

The plugin automatically selects appropriate runtime mode based on hardware performance:

- **High-performance hardware** (8GB+ VRAM): Defaults to GPU mode
- **Low-performance hardware** (<8GB VRAM): Defaults to CPU mode
- **No GPU detected**: Defaults to CPU mode

#### Usage Recommendations

1. **Prioritize GPU Mode**:
   - If GPU VRAM is sufficient, prioritize GPU mode for better performance
   - GPU mode inference speed is usually 5-10 times faster than CPU mode

2. **Switch to CPU When VRAM is Insufficient**:
   - If you encounter OOM errors, try switching to CPU mode
   - CPU mode is slower but not limited by VRAM

3. **Flexible Switching**:
   - You can switch runtime modes anytime based on task requirements
   - Different tasks can use different runtime modes

4. **Monitor Performance**:
   - When using GPU mode, monitor VRAM usage
   - Use task manager or GPU monitoring tools to check VRAM usage

5. **Parameter Optimization**:
   - GPU mode: You can adjust n_gpu_layers and vram_limit to optimize performance
   - CPU mode: No need to adjust GPU-related parameters, focus on other inference parameters

#### Common Questions

**Q: When should I use CPU mode?**
A: CPU mode is recommended when:
- No available GPU
- GPU VRAM is insufficient to load the model
- Need to process multiple large models, VRAM is insufficient
- Need maximum compatibility

**Q: How much performance difference is there between GPU and CPU modes?**
A: Typically, GPU mode inference speed is 5-10 times faster than CPU mode, depending on hardware configuration and model size.

**Q: Does the n_gpu_layers parameter still have effect in CPU mode?**
A: No. CPU mode automatically ignores n_gpu_layers and vram_limit parameters, forcing pure CPU execution.

**Q: How do I know if I'm currently using GPU or CPU mode?**
A: Check the console log output, which will display the current device mode:
- GPU mode: `Using GPU mode (n_gpu_layers=X, vram_limit=XGB)`
- CPU mode: `Using CPU mode (n_gpu_layers and vram_limit parameters ignored)`

**Q: Can I dynamically switch between GPU and CPU modes at runtime?**
A: Yes. Each time you reload the model, you can switch the device_mode parameter. However, note that switching modes will reload the model, which may take some time.

### 1.3 Performance Optimization Parameters Details

This plugin provides various performance optimization parameters that can be finely adjusted according to hardware configuration and usage scenarios to achieve optimal performance and VRAM usage efficiency.

#### Auto-adjusted Batch Processing Parameters

The following parameters are now automatically adjusted based on hardware performance, no manual adjustment needed:

- **n_batch (Batch Size)**: Automatically set based on hardware performance level (1024-4096)
- **n_ubatch (Physical Batch Size)**: Automatically set to 1/4 of n_batch
- **n_threads (Number of Generation Threads)**: Automatically set based on CPU core count
- **n_threads_batch (Number of Batch Processing Threads)**: Automatically set based on CPU core count

#### Hardware Configuration Parameter Recommendation Table

| Hardware Configuration | n_batch | n_gpu_layers | vram_limit | n_ctx |
|----------------------|--------|--------------|------------|-------|
| High Performance (24GB+ VRAM) | 4096 | -1 | 24 | 16384 |
| Mid-High (16GB VRAM) | 3072 | -1 | 16 | 8192 |
| Mid-Range (12GB VRAM) | 2048 | -1 | 12 | 6144 |
| Mid-Low (8GB VRAM) | 1536 | 30 | 8 | 4096 |
| Low ( <8GB VRAM) | 1024 | 0 | -1 | 2048 |

#### Performance Optimization Configuration Examples

##### High Performance Configuration (24GB+ VRAM)
```
n_gpu_layers: -1
vram_limit: 24
```

##### Medium Performance Configuration (12-16GB VRAM)
```
n_gpu_layers: -1
vram_limit: 16
```

##### Low Performance Configuration (8GB VRAM)
```
n_gpu_layers: 30
vram_limit: 8
```

##### CPU Mode Configuration
```
device_mode: CPU
```
Note: GPU-related parameters are automatically ignored in CPU mode

#### GPU Vendor Adaptation
- **NVIDIA GPU**: Flash Attention is automatically enabled
- **AMD GPU**: Batch processing parameters are automatically optimized
- Different GPUs require different optimization strategies

#### Performance Optimization Tips

1. **Prioritize VRAM-related parameters**:
   - Adjust n_gpu_layers (GPU model layers)
   - Adjust vram_limit (VRAM limit)

2. **Gradual optimization**:
   - Start from default configuration
   - Adjust only one parameter at a time
   - Observe performance changes to find optimal configuration

3. **Monitor resource usage**:
   - Use task manager or GPU monitoring tools
   - Observe VRAM usage and inference speed
   - Adjust parameters based on actual situation

4. **Balance performance and quality**:
   - When VRAM is insufficient, prioritize availability
   - When VRAM is sufficient, prioritize performance
   - Flexibly adjust according to task requirements

## 2. Llama-cpp Image Inference Node

### Function
Performs LLM/VLM inference, supporting text-only, single-image, multi-image, and video inputs.

### Parameter Description

Parameter name: llama_model
  Function description: Loaded LLM model
  Recommended setting: Connect to the model loading node

Parameter name: preset_prompt
  Function description: Preset prompt template
  Recommended setting: Select as needed (e.g., "Normal - Describe" for image description)
  Common template selection guide:
  - Image description: "Normal - Describe" or "Prompt Style - Detailed"
  - Generate tags: "Prompt Style - Tags"
  - Video analysis: "Video - Reverse Prompt"

Parameter name: custom_prompt
  Function description: Custom prompt
  Recommended setting: Use in conjunction with preset prompts
  Prompt optimization tips:
  - Be specific and clear: Provide detailed instructions and context
  - Clear formatting: Use punctuation and line breaks to improve readability
  - Example guidance: Provide examples of expected output

Parameter name: system_prompt
  Function description: System prompt that defines model behavior
  Recommended setting: Modify according to requirements
  Examples:
  - Image description task: "You are a professional image description assistant. Describe the content of the image in detail, including characters, scenes, objects, and other details."
   - Video analysis task: "You are a professional video analysis assistant. Analyze the content of the video in detail."

Parameter name: inference_mode
  Function description: Select inference mode: text=text-only, one by one=process individually, images=image processing, video=video processing
  Recommended setting: "images" for single image, "one by one" for multiple images, "video" for video
  Mode explanation:
  - images: Process single image or multiple images as a whole
  - one by one: Process multiple images one by one, suitable for scenarios requiring individual descriptions
  - video: Process video files, extract frames for analysis

Parameter name: preset_prompts_language
  Function description: Select language of preset prompts
  Recommended setting: Select Chinese or English as needed

Parameter name: output_language
  Function description: Select output text language
  Recommended setting: Select Chinese or English as needed

Parameter name: max_frames
  Adjustment range: 2-1024
  Function description: Maximum number of frames for video processing
  Recommended setting: Between 16-32
  Performance impact: More frames provide more comprehensive analysis, but longer processing time

Parameter name: max_size
  Adjustment range: 128-16384
  Function description: Maximum size of images/videos
  Recommended setting: Low-performance devices: 256, high-performance devices: 512
  Balance suggestion: Larger size captures more details but uses more VRAM

Parameter name: seed
  Adjustment range: 0-0xffffffffffffffff
  Function description: Random seed
  Recommended setting: 0 (random) or a fixed value
  Notes: Setting a fixed value yields reproducible results

Parameter name: force_offload
  Function description: Force model offloading after inference
  Recommended setting: Usually set to False
  Usage scenario: Only use when you need to immediately release VRAM

Parameter name: save_states
  Function description: Save conversation state
  Recommended setting: Set to True for continuous dialogue
  Benefits: Maintains conversation context, improves coherence of multi-turn interactions

Parameter name: parameters
  Function description: Connect to the parameter setting node
  Recommended setting: Optional; use default parameters if not connected

Parameter name: images
  Function description: Input images
  Recommended setting: Provide when processing images

### Preset Prompt Template Description

Preset prompt templates provide various common inference scenarios. Here are detailed descriptions:

#### Basic Templates

- **Empty - Nothing**: Empty template, using only custom prompts, suitable for scenarios requiring fully customized prompts

- **Normal - Describe**: Briefly describe image content, suitable for basic image description tasks

#### Prompt Style Templates

- **Prompt Style - Tags**: Generate image tag lists, suitable for SDXL model prompts, outputting comma-separated visual element tags (maximum 60 unique tags)

- **Prompt Style - Simple**: Generate concise image descriptions (within 300 words), suitable for quickly getting an image overview

- **Prompt Style - Detailed**: Generate detailed image descriptions (within 500 words), suitable for scenarios requiring in-depth analysis of image content

#### Vision Task Templates

- **Vision - Bounding Box**: Generate bounding boxes for object detection, output JSON-formatted bounding box coordinates

#### Model-Specific Templates

- **ZIMAGE - Turbo**: Designed for Z-Image-Turbo model, creates efficient and high-quality image generation prompts, utilizing 8-step Turbo inference for fast 1080P HD image generation
- **FLUX2 - Klein**: Designed for FLUX series (including Flux.1 and FLUX.2 Klein) models, creates concise and expressive prompts
- **LTX-2**: Designed for LTX-2 model, creates detailed and dynamic video generation prompts
- **Qwen - Image Layered**: Designed for Qwen-Image-Layered model, creates detailed layered prompts, utilizing the model's ability to handle complex compositions and multiple elements
- **Qwen - Image Edit Combined**: Comprehensive editing prompt enhancer, generates precise, concise, direct, and specific editing prompts based on user-provided instructions and image input conditions
- **Qwen - Image Dual**: Designed for Qwen Image series (including Qwen Image and Qwen Image 2512) models, creates high-quality image generation prompts

#### Video Processing Templates

- **Video - Reverse Prompt**: Video reverse prompt expert, analyzes and generates detailed video description prompts based on video content provided by users

- **Video - Detailed Scene Breakdown**: Detailed video scene breakdown, describes each scene's content, time points, scenes, subjects, actions, lighting, colors, and camera movements in chronological order

- **Video - Subtitle Format**: Video subtitle format generation, adds standard format time codes and corresponding text content for video content

- **WAN - Text to Video**: Cinematic director style video generation prompt, adds cinematic elements to user's original prompt

- **WAN - Image to Video**: Video description prompt rewriting expert, rewrites provided video description prompts based on user input images, emphasizing potential dynamic content

- **WAN - Image to Video Empty**: Video description prompt writing expert, generates video descriptions based on user input images (no prompt needed), using reasonable imagination to make images come alive

- **WAN - FLF to Video**: First and Last Frame Prompt optimizer, references user input image detail content, rewrites user input prompts into high-quality prompts

#### OCR Enhancement Templates

- **OCR - Enhanced**: Specialized prompt template designed for poster OCR text recognition, accurately extracting text content and style attributes
  - Supports recognition of text font, size, color, layout style and other detailed attributes
  - Compatible with direct prompt reverse calling, improving image generation accuracy
  - Suitable for analyzing images containing text such as posters, billboards, and packaging

#### Multilingual Templates

- **[Multilingual] Bilingual Prompt Generate**: Professional bilingual prompt generation expert, specializing in creating high-quality Chinese-English bilingual prompts for cross-border creation and bilingual document scenarios
  - Maintains semantic consistency between Chinese and English prompts, ensuring both languages convey the same visual information and creative intent
  - For Chinese prompts, uses vocabulary and sentence structures that conform to Chinese expression habits, focusing on artistic conception and aesthetics
  - For English prompts, uses vocabulary and sentence structures that conform to English expression habits, focusing on accuracy and fluency
  - Output format: Outputs Chinese prompt first, then English prompt, separated by blank line

#### High Resolution Templates

- **[HighRes] Ultra HD Image Reverse**: Professional ultra HD image prompt reverse expert, specializing in extracting detailed visual information from 4K/8K ultra HD images
  - Carefully analyzes all details in ultra HD images, including subjects, scenes, materials, textures, lighting, colors, composition, etc.
  - Identifies minute details and complex structures in images, such as fabric textures, skin textures, environmental details, etc.
  - Analyzes lighting conditions, color distribution, and spatial relationships in images, ensuring prompts accurately reflect these elements
  - Considers the resolution characteristics of ultra HD images, generating prompts with sufficient detail descriptions to adapt to high-resolution image generation

## 3. Llama-cpp Parameter Setting Node

### Function
Sets detailed parameters for LLM inference to control the quality and style of generated text.

### Parameter Description

Parameter name: max_tokens
  Adjustment range: 0-4096
  Function description: Maximum number of generated tokens, controlling the maximum length of generated text and directly affecting the completeness of output content
  Recommended setting: Low-performance devices: 512-1024, high-performance devices: 1024-2048
  Practical suggestions:
  - Short answers: 256-512
  - Detailed descriptions: 768-1024
  - Long text generation: 1024-2048

Parameter name: top_k
  Adjustment range: 0-1000
  Function description: Number of candidate tokens for sampling, controlling the number of vocabulary items considered per generation (smaller = more focused, larger = more diverse)
  Recommended setting: Low-performance devices: 20, high-performance devices: 30
  Impact analysis:
  - Low top_k (<20): More deterministic generation, suitable for factual tasks
  - High top_k (>50): More diverse generation, suitable for creative tasks

Parameter name: top_p
  Adjustment range: 0.0-1.0
  Function description: Nucleus sampling threshold, balancing diversity and accuracy of generated content (larger = more diverse)
  Recommended setting: 0.85-0.9
  Effect comparison:
  - Low top_p (<0.8): More conservative generation, higher accuracy
  - High top_p (>0.9): More open generation, stronger diversity

Parameter name: min_p
  Adjustment range: 0.0-1.0
  Function description: Minimum sampling probability, preventing low-probability tokens from being completely ignored to increase generation richness (default value is usually sufficient)
  Recommended setting: 0.05
  Applicable scenarios: Use when you need to increase generation diversity

Parameter name: typical_p
  Adjustment range: 0.0-1.0
  Function description: Typicality sampling parameter, controlling the typicality of generated content (1.0 = disable this feature; most users do not need adjustment)
  Recommended setting: 1.0

Parameter name: temperature
  Adjustment range: 0.0-2.0
  Function description: Generation temperature, controlling randomness (one of the most commonly used parameters; higher = more random, lower = more deterministic)
  Recommended setting: 0.6-0.8
  Temperature guide:
  - Low temperature (0.1-0.4): Suitable for scenarios requiring accurate answers, such as Q&A, code generation
  - Medium temperature (0.5-0.8): Suitable for most scenarios, balancing accuracy and creativity
  - High temperature (0.9-1.5): Suitable for creative tasks, such as story generation, poetry creation

Parameter name: repeat_penalty
  Adjustment range: 0.0-10.0
  Function description: Repetition penalty to avoid duplicate content and improve text quality (one of the commonly used parameters)
  Recommended setting: 1.0
  Adjustment suggestions:
  - Gradually increase to 1.1-1.3 when duplicate content appears
  - Should not be too high, otherwise it may cause incoherent generated content

Parameter name: frequency_penalty
  Adjustment range: 0.0-1.0
  Function description: Frequency penalty to reduce the probability of high-frequency tokens (0.0 = disable; usually no active adjustment needed)
  Recommended setting: 0.0

Parameter name: presence_penalty
  Adjustment range: 0.0-2.0
  Function description: Presence penalty to encourage new content generation and increase text richness (one of the commonly used parameters)
  Recommended setting: 1.0
  Usage scenarios:
  - Avoid topic drift during long text generation
  - Encourage new viewpoints in conversations

Parameter name: mirostat_mode
  Adjustment range: 0-2
  Function description: Mirostat sampling mode for controlling perplexity of generated text (0 = disabled, 1 = basic mode, 2 = version 2; most users do not need to use)
  Recommended setting: 0
  Advanced user suggestions: Try mode 1 or 2 for more consistent generation quality

Parameter name: mirostat_eta
  Adjustment range: 0.0-1.0
  Function description: Mirostat learning rate (only effective when mirostat_mode is enabled), controlling adjustment speed
  Recommended setting: 0.1

Parameter name: mirostat_tau
  Adjustment range: 0.0-10.0
  Function description: Mirostat target perplexity (only effective when mirostat_mode is enabled), controlling generation diversity
  Recommended setting: 5.0

Parameter name: state_uid
  Adjustment range: -1-999999
  Function description: Dialogue state ID for saving and restoring conversation states (only useful for maintaining continuous dialogue context)
  Recommended setting: -1
  Session management:
  - Use different state_uid to maintain multiple independent sessions simultaneously
  - Keep the same state_uid for continuous dialogue to maintain context

### Parameter Adjustment Tips

1. **Control Output Length**: Adjust `max_tokens` (larger values = longer output but higher resource usage). Set according to actual needs to avoid performance issues from overly large values.

2. **Control Randomness**: `temperature` is one of the most commonly used parameters (lower = more deterministic for factual Q&A, higher = more creative for story/poetry generation).

3. **Balance Diversity and Accuracy**:
   - `top_p` and `top_k` are usually used together
   - Lower `top_k` and higher `top_p` improve accuracy while maintaining diversity
   - Higher `top_k` and lower `top_p` increase generation diversity

4. **Avoid Duplicate Content**:
   - Increase `repeat_penalty` to reduce duplicates
   - Combine with `presence_penalty` to encourage new content generation

5. **Parameter Priority Recommendations**:
   - Priority adjustment: `temperature`, `max_tokens`, `top_p`/`top_k`
   - Secondary adjustment: `repeat_penalty`, `presence_penalty`
   - Advanced users: `min_p`, Mirostat-related parameters

6. **Common Scenario Parameter Settings**:
   - **Factual Q&A**: temperature=0.1-0.3, top_k=10, top_p=0.7
   - **Creative Writing**: temperature=0.8-1.2, top_k=50, top_p=0.95
   - **Code Generation**: temperature=0.2-0.4, top_k=20, top_p=0.8
   - **Dialogue Interaction**: temperature=0.6-0.8, top_k=30, top_p=0.9

## 4. Llama-cpp Clear States Node

### Function
Clears specified dialogue session states and releases resources.

### Parameter Description

Parameter name: any_input
  Function description: Any type of input to trigger node execution
  Recommended setting: Any connection

Parameter name: state_uid
  Adjustment range: -1-999999
  Function description: ID of dialogue state to clear, -1 = clear all states
  Recommended setting: -1 (clear current session) or a specified ID

## 5. Llama-cpp Unload Model Node

### Function
Unloads currently loaded model, releasing memory and VRAM resources.

### Parameter Description

Parameter name: any_input
  Function description: Any type of input to trigger node execution
  Recommended setting: Any connection

## 6. Multi-Image Input Node

### Function
Processes multiple image inputs, supporting image mode and text mode.

### Parameter Description

Parameter name: image1
  Function description: First input image
  Recommended setting: Provide first image

Parameter name: image2
  Function description: Second input image
  Recommended setting: Optional, provide second image

Parameter name: image3
  Function description: Third input image
  Recommended setting: Optional, provide third image

Parameter name: image4
  Function description: Fourth input image
  Recommended setting: Optional, provide fourth image

Parameter name: image5
  Function description: Fifth input image
  Recommended setting: Optional, provide fifth image

Parameter name: image6
  Function description: Sixth input image
  Recommended setting: Optional, provide sixth image

Parameter name: mode
  Function description: Select working mode: image mode analyzes image content, text mode generates prompts based on options
  Recommended setting: Select image mode or text mode as needed

Parameter name: story_type
  Function description: Select content creation type
  Recommended setting: Select as needed

Parameter name: story_length
  Function description: Select content length
  Recommended setting: Select as needed

Parameter name: language
  Function description: Select output language
  Recommended setting: Select Chinese or English as needed

Parameter name: max_size
  Function description: Maximum image size (pixels)
  Recommended setting: 512 (default)

Parameter name: custom_prompt
  Function description: Custom prompt for guiding content creation
  Recommended setting: Optional, for guiding content creation

Parameter name: include_image_descriptions
  Function description: Whether to include descriptions of each image before story
  Recommended setting: Select as needed

Parameter name: story_theme
  Function description: Select content theme
  Recommended setting: Select as needed

Parameter name: narrative_style
  Function description: Select narrative style
  Recommended setting: Select as needed

Parameter name: content_focus
  Function description: Content focus
  Recommended setting: Select as needed

Parameter name: target_audience
  Function description: Select target audience
  Recommended setting: Select as needed

Parameter name: video_model
  Function description: Select video generation model type, different models require different prompt formats
  Recommended setting: Select as needed

## 7. JSON to Bounding Box Node

### Function
Converts JSON-formatted bounding box data generated by the model into visual bounding boxes.

### Parameter Description

Parameter name: json
  Function description: JSON string containing bounding box information
  Recommended setting: Connect to the output of the inference node

Parameter name: mode
  Function description: Bounding box parsing mode
  Recommended setting: Select according to the model (Qwen3-VL or Qwen2.5-VL)

Parameter name: label
  Function description: Filter bounding boxes for specific labels
  Recommended setting: Leave blank for no filtering

Parameter name: image
  Function description: Original image
  Recommended setting: Optional, used for drawing bounding boxes


## 8. API Model Nodes

### 8.1 API Configuration Manager (llama_cpp_api_model_config)

#### Function
Creates and manages API model configurations, supporting multiple external API providers.

#### Parameter Description

**Required Parameters:**
- `model_name`: API model name for configuration identification
- `api_provider`: API provider, options: OpenAI/Ollama/llms-py/vllm-omni/Custom, etc.
- `api_base`: API base URL
- `api_key`: API key (if required)
- `max_tokens`: Maximum generated tokens (default 1024)
- `temperature`: Temperature parameter (default 0.7)

### 8.2 API Inference Node (llama_cpp_api_inference)

#### Function
Performs multimedia inference using APIs, supporting image, video, audio inputs, and text, audio outputs.

#### Parameter Description

**Required Parameters:**
- `api_config`: API configuration (connect to API Configuration Manager node)
- `inference_mode`: Inference mode, options: Text Generation/Image Understanding/Audio to Text/Text to Audio/Multimodal Integration/Video Understanding
- `preset_prompt`: Select preset prompt template
- `system_prompt`: System prompt
- `text_input`: User input text
- `prompt_language`: Language of preset prompts, options: Chinese/English
- `response_language`: AI response language, options: Auto Detect/Chinese/English
- `video_max_frames`: Video mode: Maximum frames to extract (default 16)
- `video_sampling`: Video frame sampling method, options: Auto Uniform Sampling/Manual Frame Indices
- `video_manual_indices`: Frame indices in manual mode
- `image_max_size`: Maximum size for image processing (default 256)
- `tts_voice`: TTS voice selection
- `tts_emotion`: TTS emotion style
- `tts_speed`: TTS speed (default 1.0)
- `seed`: Random seed (default 101)

**Optional Parameters:**
- `images`: Image input (for image understanding mode)
- `video`: Video input (for video understanding mode)
- `audio`: Audio input (for audio understanding mode)

**Output:**
- `text`: Generated text
- `audio`: Generated audio (when Text to Audio or Multimodal Integration mode is selected)



## II. Device Performance Adaptation Recommendations

### 1. Performance Classification Based on VRAM Size

#### 24GB+ VRAM Devices (e.g., 5090, 4090, 3090)

- **Recommended Models**: Qwen3-VL, GLM-4.6V, LLaVA-1.6, MiniCPM-V-4_5, Gemma3
- **Model Loading Node**:
  - n_ctx: 16384
  - n_gpu_layers: -1 (load all layers to GPU)
  - n_batch: 4096
  - n_ubatch: 1024
  - flash_attention: Enabled
  - offload_kqv: True
  - use_mlock: True
  - f16_kv: False
  - low_vram: False
  - max_size: 512-1024

- **Parameter Setting Node**:
  - max_tokens: 1024-2048
  - temperature: 0.8
  - top_k: 30
  - top_p: 0.9

#### 16GB VRAM Devices (e.g., 4080)

- **Recommended Models**: LLaVA-1.6, MiniCPM-V-4_5, Qwen2.5-VL, LFM2.5-VL
- **Model Loading Node**:
  - n_ctx: 8192
  - n_gpu_layers: -1 (load all layers to GPU)
  - n_batch: 2048
  - n_ubatch: 512
  - flash_attention: Auto
  - offload_kqv: True
  - use_mmap: True
  - f16_kv: True
  - low_vram: False
  - max_size: 512

- **Parameter Setting Node**:
  - max_tokens: 1024-1536
  - temperature: 0.7
  - top_k: 25
  - top_p: 0.85

#### 12GB VRAM Devices (e.g., 4070 Ti, 3080)

- **Recommended Models**: LLaVA-1.5, MiniCPM-v2.6, Moondream2, LFM2.5-VL
- **Model Loading Node**:
  - n_ctx: 6144
  - n_gpu_layers: -1 (load all layers to GPU)
  - n_batch: 2048
  - n_ubatch: 512
  - flash_attention: Auto
  - offload_kqv: True
  - use_mmap: True
  - f16_kv: True
  - low_vram: False
  - max_size: 384

- **Parameter Setting Node**:
  - max_tokens: 768-1024
  - temperature: 0.65
  - top_k: 25
  - top_p: 0.85

#### 8GB VRAM Devices (e.g., 3070, 3060)

- **Recommended Models**: Moondream2, nanoLLaVA, LFM2.5-VL-1.6B, MobileVLM, TinyLLaVA
- **Model Loading Node**:
  - n_ctx: 4096
  - n_gpu_layers: 30 (partial load)
  - n_batch: 1024
  - n_ubatch: 256
  - flash_attention: Auto
  - offload_kqv: True
  - use_mmap: True
  - f16_kv: True
  - low_vram: True
  - max_size: 256

- **Parameter Setting Node**:
  - max_tokens: 512-768
  - temperature: 0.6
  - top_k: 20
  - top_p: 0.85

#### 4-6GB VRAM Devices (e.g., 3050, 2060)

- **Recommended Models**: nanoLLaVA, TinyLLaVA, MiniGPT-v2
- **Model Loading Node**:
  - n_ctx: 2048
  - n_gpu_layers: 0 (CPU only)
  - n_batch: 512
  - n_ubatch: 128
  - flash_attention: Disabled
  - offload_kqv: False
  - use_mmap: True
  - f16_kv: True
  - low_vram: True
  - max_size: 192

- **Parameter Setting Node**:
  - max_tokens: 256-512
  - temperature: 0.5
  - top_k: 15
  - top_p: 0.8

#### CPU Mode (No GPU or GPU VRAM Insufficient)

- **Recommended Models**: nanoLLaVA, TinyLLaVA, MiniGPT-v2
- **Model Loading Node**:
  - device_mode: CPU
  - n_ctx: 2048
  - n_gpu_layers: 0 (CPU only)
  - n_batch: 512
  - n_ubatch: 128
  - flash_attention: Disabled
  - offload_kqv: False
  - use_mmap: True
  - f16_kv: True
  - low_vram: True
  - max_size: 192

- **Parameter Setting Node**:
  - max_tokens: 256-512
  - temperature: 0.5
  - top_k: 15
  - top_p: 0.8

## III. Supported Models Description

### 1. Core Model Support

#### Large Models (Suitable for 24GB+ VRAM)
- **Qwen3-VL**: Powerful multimodal model with complex visual understanding capabilities
- **GLM-4.6V**: High-performance visual language model supporting multiple tasks
- **LLaVA-1.6**: Advanced visual assistant model with strong comprehension abilities

#### Medium Models (Suitable for 12GB+ VRAM)
- **MiniCPM-V-4_5**: Efficient visual language model with balanced performance
- **Qwen2.5-VL**: Lightweight version of Qwen3-VL, suitable for mid-range devices
- **LLaVA-1.5**: Classic visual assistant model, stable and reliable

#### Small Models (Suitable for 8GB+ VRAM)
- **Moondream2**: Lightweight visual model with fast inference speed
- **nanoLLaVA**: Ultra-small visual model, suitable for low VRAM devices
- **LFM2.5-VL-1.6B**: Very small model, suitable for edge devices

### 2. New Lightweight Models (Updated 2026-01-24)

- **MobileVLM**: Visual model optimized for mobile devices, extremely VRAM-efficient
- **TinyLLaVA**: Micro visual language model, suitable for devices with 4GB+ VRAM
- **MiniGPT-v2**: Lightweight multimodal model prioritizing speed

### Model Selection Recommendations

#### By Task Type

| Task Type | Recommended Models | Applicable Scenarios |
|-----------|------------------|---------------------|
| Simple Image Description | Moondream2, nanoLLaVA | Quick image analysis, simple descriptions |
| Complex Visual Tasks | Qwen3-VL, GLM-4V | Detailed image understanding, multi-image analysis |
| ASR Speech Recognition | Whisper Large-v3, Qwen-Asr | Speech to text, audio content extraction |
| TTS Speech Synthesis | Qwen3-TTS | Text to speech, speech synthesis |
| Text-only Tasks | Llama 3.2 1B, Mistral 7B | Chat, writing, Q&A |
| Multimodal Tasks | Qwen2.5-Omni, Qwen3.5 | Simultaneous text, image, audio processing |

#### By Hardware Configuration

| Hardware Configuration | Recommended Models | Runtime Mode |
|----------------------|-------------------|--------------|
| High-end GPU (24GB+) | Any model, recommended Qwen3.5 series | GPU Mode |
| Mid-range GPU (12-16GB) | Qwen2.5-Omni-7B, LLaVA-1.6 | GPU Mode |
| Mid-low GPU (8GB) | Qwen2.5-Omni-3B, Moondream2 | GPU Mode |
| Low-end GPU (<8GB) | Lightweight models or use CPU mode | CPU Mode |
| No GPU | Lightweight models | CPU Mode |

### Model Characteristics Comparison

| Model | VRAM Requirement | Strengths | Best For |
|-------|----------------|-----------|----------|
| Qwen3-VL | 24GB+ | Complex visual understanding, multimodal reasoning | Complex tasks, professional applications |
| GLM-4.6V | 24GB+ | High performance, multiple task support | High-quality generation, diverse applications |
| LLaVA-1.6 | 16GB+ | Strong comprehension, stable performance | General visual understanding tasks |
| MiniCPM-V-4_5 | 12GB+ | Balanced performance, efficient | Mid-range devices, balanced tasks |
| Qwen2.5-VL | 12GB+ | Lightweight, good performance | Mid-range devices, general tasks |
| LLaVA-1.5 | 12GB+ | Stable, reliable | Classic visual assistant tasks |
| Moondream2 | 8GB+ | Fast inference | Quick descriptions, real-time applications |
| nanoLLaVA | 8GB+ | Ultra-small, efficient | Low VRAM devices, edge computing |
| MobileVLM | 8GB+ | Mobile optimized, VRAM efficient | Mobile devices, resource-constrained environments |
| TinyLLaVA | 4GB+ | Micro size | Minimal VRAM devices, basic tasks |
| MiniGPT-v2 | 4GB+ | Speed prioritized | Fast inference, simple tasks |

### 3. Model Selection Best Practices

1. **Select Models Based on VRAM**:
   - 24GB+: Any model
   - 16GB: Medium models
   - 12GB: Small to medium models
   - 8GB: Small models
   - 4-6GB: Ultra-small models

2. **Select Models Based on Tasks**:
   - Complex visual understanding: Qwen3-VL, GLM-4.6V
   - Fast inference: Moondream2, nanoLLaVA
   - Mobile devices: MobileVLM, TinyLLaVA

3. **Quantization Level Selection**:
   - Priority: Q4_K_M (balanced size and quality)
   - High quality: Q5_K_M, Q6_K
   - Small size: Q3_K_M, Q2_K

## IV. Common Issues and Solutions

### 1. Out of Memory (OOM)

**Symptoms**: "out of memory" or "OOM" error during inference

**Solutions**:
- Reduce `n_gpu_layers` to lower GPU load
- Decrease `n_ctx` to reduce context length
- Lower `max_tokens` to reduce generated text length
- Use smaller model files
- Reduce image `max_size`

### 2. Slow Inference Speed

**Solutions**:
- Increase `n_gpu_layers` (if sufficient VRAM is available)
- Enable `flash_attention` (NVIDIA GPU) for 2-3x speed improvement
- Increase `n_batch` to improve processing speed (NVIDIA: 2048-4096, AMD: 1024-2048)
- Increase `n_ubatch` to improve memory efficiency (NVIDIA: 512-1024, AMD: 512)
- Enable `offload_kqv` to offload K, Q, V to GPU
- Reduce `n_ctx` to reduce context length
- Use smaller models
- Close unnecessary applications to free up system resources
- Adjust `n_threads` and `n_threads_batch` for optimal CPU utilization

**Performance Optimization Tips**:
- For NVIDIA GPU: Set `flash_attention` to "Enabled" for best performance
- For AMD GPU: Adjust `n_batch` and `n_ubatch` to optimize performance
- For CPU mode: Set `n_threads` to CPU cores/2, `n_threads_batch` to CPU cores
- Use `low_vram` mode when VRAM is insufficient, but expect slower performance

### 3. Poor Generation Quality

**Symptoms**: Low-quality, repetitive, irrelevant, or unexpected generated text

**Solutions**:

1. **Optimize Prompts**: High-quality prompts are more important than parameter adjustments. Ensure prompts are clear, specific, and provide sufficient context.

2. **Adjust Generation Parameters**:
   - For overly random/irrelevant content: Lower `temperature`, reduce `top_p`/`top_k`
   - For overly repetitive content: Increase `repeat_penalty`, raise `presence_penalty`
   - For insufficiently rich content: Raise `temperature`, increase `top_p`/`top_k`, consider adjusting `min_p`

3. **Increase Output Length**: If content is truncated, increase `max_tokens`

4. **Use Higher-Quality Models**: Try larger or more specialized models

5. **Check Model Compatibility**: Ensure `chat_handler` matches the model type

### 4. Errors When Processing Images

**Solutions**:
- Ensure `mmproj` is enabled and the correct visual encoding model is selected
- Verify `chat_handler` matches the model
- Check if image size is too large; reduce `max_size`

### 5. Model Loading Fails

**Symptoms**: Model fails to load or crashes during loading

**Solutions**:
- Verify model file integrity (check file size and hash)
- Ensure sufficient VRAM or system memory
- Try loading the model in CPU mode first
- Check if the model format is supported (GGUF format required)
- Verify that mmproj file matches the main model version

### 6. Inconsistent Output Quality

**Symptoms**: Output quality varies significantly between runs

**Solutions**:
- Set a fixed `seed` value for reproducible results
- Adjust `temperature` to a stable range (0.6-0.8)
- Use `mirostat_mode` for more consistent generation quality
- Ensure stable system resources (close other applications)

## V. Usage Recommendations and Best Practices

### 1. Basic Usage Suggestions

1. **Gradual Adjustment**: Start with default parameters and adjust incrementally based on actual results
2. **Model Matching**: Ensure `chat_handler` matches the model type
3. **Resource Monitoring**: Use task manager to monitor memory and VRAM usage
4. **Prompt Optimization**: High-quality prompts are more important than parameter adjustments; describe requirements in detail
5. **Session Management**: Regularly clear session states during long-term use to release resources

### 2. Model Selection Best Practices

1. **Select Models Based on VRAM**:
   - 24GB+: Any model
   - 16GB: Medium models
   - 12GB: Small to medium models
   - 8GB: Small models
   - 4-6GB: Ultra-small models

2. **Select Models Based on Tasks**:
   - Complex visual understanding: Qwen3-VL, GLM-4.6V
   - Fast inference: Moondream2, nanoLLaVA
   - Mobile devices: MobileVLM, TinyLLaVA

3. **Quantization Level Selection**:
   - Priority: Q4_K_M (balanced size and quality)
   - High quality: Q5_K_M, Q6_K
   - Small size: Q3_K_M, Q2_K

### 3. Performance Optimization Tips

1. **VRAM Management**:
   - Use the unload model node when switching between models
   - Clean session states after large tasks
   - Avoid loading multiple large models simultaneously
   - Enable `f16_kv` to reduce VRAM usage by about 50%
   - Use `low_vram` mode when VRAM is insufficient
   - Adjust `n_batch` and `n_ubatch` to balance performance and VRAM usage

2. **Speed Optimization**:
   - Enable `flash_attention` for NVIDIA GPU (2-3x speed improvement)
   - Increase `n_batch` to improve processing speed (NVIDIA: 2048-4096, AMD: 1024-2048)
   - Enable `offload_kqv` to offload K, Q, V to GPU
   - Appropriately reduce n_ctx to improve inference speed
   - Use smaller models for simple tasks
   - Adjust image_max_tokens to reduce visual processing overhead
   - Optimize `n_threads` and `n_threads_batch` for CPU utilization

3. **Quality Optimization**:
   - Use appropriate prompt templates for specific tasks
   - Fine-tune parameters based on task type (factual vs. creative)
   - Leverage system prompts to guide model behavior
   - Adjust `temperature`, `top_p`, `top_k` for different scenarios

4. **Memory Optimization**:
   - Enable `use_mmap` to reduce memory usage for large models
   - Enable `use_mlock` when memory is sufficient to improve performance
   - Disable `use_mmap` for faster model loading when memory is sufficient
   - Use `low_vram` mode as the last optimization method

### 4. Prompt Engineering Best Practices

1. **Be Specific and Clear**:
   - Provide detailed instructions and context
   - Specify desired output format
   - Include examples when possible

2. **Use Appropriate Templates**:
   - Image description: "Normal - Describe" or "Prompt Style - Detailed"
   - Tag generation: "Prompt Style - Tags"
   - Video analysis: "Video - Reverse Prompt"

3. **Combine Preset and Custom Prompts**:
   - Use preset templates as a foundation
   - Add custom prompts for specific requirements
   - Leverage system prompts to define model behavior

### 5. Troubleshooting Workflow

1. **Identify the Problem**:
   - Check error messages in console
   - Monitor resource usage (VRAM, memory)
   - Test with simplified parameters

2. **Systematic Debugging**:
   - Start with default settings
   - Change one parameter at a time
   - Document successful configurations

3. **Resource Planning**:
   - Estimate VRAM requirements before loading models
   - Plan model switching sequences
   - Schedule resource-intensive tasks appropriately

### 6. Advanced Usage Tips

1. **Multi-Session Management**:
   - Use different `state_uid` values for independent conversations
   - Clear sessions when switching contexts
   - Save important session states for reuse

2. **Batch Processing**:
   - Process multiple images with consistent parameters
   - Use "one by one" mode for individual image analysis
   - Adjust max_frames for video processing efficiency

3. **Integration with Other Tools**:
   - Combine with image processing nodes in ComfyUI
   - Use output for downstream generation tasks
   - Integrate with video processing workflows

### Practical Use Cases

#### Case 1: Image Description and Prompt Generation

**Applicable Scenario**: Generate detailed prompts for AI painting

**Configuration Steps**:
1. **Model Selection**: Use Qwen2.5-Omni-3B or Moondream2
2. **Inference Mode**: Select "images" mode
3. **Preset Template**: Select "Prompt Style - Detailed"
4. **System Prompt**: "You are a professional prompt engineer for AI image generation. Create detailed, vivid prompts that will produce high-quality images."
5. **Text Input**: "Based on this image, generate detailed prompts suitable for SDXL models, with realistic style and natural lighting."
6. **Parameter Settings**: temperature=0.7, max_tokens=1024

**Expected Effect**: Generate detailed prompts suitable for AI painting

#### Case 2: Video Content Analysis

**Applicable Scenario**: Analyze video content and generate summary

**Configuration Steps**:
1. **Model Selection**: Use Qwen2.5-Omni-7B or higher version
2. **Inference Mode**: Select "video" mode
3. **Preset Template**: Select "Video - Detailed Scene Breakdown"
4. **System Prompt**: "You are a professional video analyst. Analyze video content and provide comprehensive summaries of key events and visual elements."
5. **Text Input**: "Analyze the content of this video, focusing on main events and visual effects."
6. **Video Parameters**: max_frames=32, video_sampling=Auto Uniform Sampling
7. **Parameter Settings**: temperature=0.6, max_tokens=2048

**Expected Effect**: Generate detailed video content analysis and summary

#### Case 3: Multimodal Dialogue

**Applicable Scenario**: Complex dialogue processing text, images and audio simultaneously

**Configuration Steps**:
1. **Model Selection**: Use Qwen3.5 or Qwen2.5-Omni-7B
2. **Inference Mode**: Select "Multimodal Integration" mode
3. **Preset Template**: Select "Empty - Nothing"
4. **System Prompt**: "You are an excellent multimodal assistant. You can understand and respond to text, images, and audio inputs."
5. **Text Input**: Enter based on specific dialogue content
6. **Parameter Settings**: temperature=0.7, max_tokens=1024, save_states=True

**Expected Effect**: Ability to understand and process multiple input modalities simultaneously

### Performance Optimization Best Practices

1. **VRAM Management**:
   - Regularly check VRAM usage
   - Use force_offload to release VRAM after large model inference
   - Avoid running multiple large models simultaneously

2. **Inference Speed Optimization**:
   - Use GPU mode for best performance
   - Adjust n_gpu_layers to fully utilize GPU resources
   - Select appropriate batch processing parameters

3. **Quality Optimization**:
   - Adjust temperature parameter based on task type
   - Use appropriate prompt templates and system prompts
   - Provide detailed context information

4. **Troubleshooting**:
   - When encountering OOM, try reducing n_ctx or switching to CPU mode
   - When inference is slow, check GPU utilization and driver version
   - When generation quality is poor, optimize prompts and parameter settings

We hope this document helps you better use the ComfyUI-omni-llm plugin. For additional questions, refer to the README file or submit an issue.