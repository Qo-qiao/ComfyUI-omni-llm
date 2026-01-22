# Llama-cpp-vlmforQo Node Parameter Description and Recommended Settings

This document details the parameter functions and recommended settings of each node in the ComfyUI-llama-cpp-vlmforQo plugin, helping users better use and adjust the model.

## 1. Llama-cpp Model Loading Node

### Function
Loads and initializes LLM/VLM models, serving as the foundation for all other nodes.

### Parameter Description

Parameter name: model
  Function description: Select the LLM model file to load
  Recommended setting: Choose an appropriate GGUF model as needed

Parameter name: enable_mmproj
  Function description: Enables image input processing when activated
  Recommended setting: Set to True for image processing, False for text-only scenarios

Parameter name: mmproj
  Function description: Corresponding visual encoding model file
  Recommended setting: Select a matching model when mmproj is enabled

Parameter name: chat_handler
  Function description: Select a dialogue format handler suitable for the model
  Recommended setting: Match the model (e.g., select Qwen3-VL for the Qwen3-VL model)

Parameter name: n_ctx
  Adjustment range: 1024-327680
  Function description: Context length, affecting the length of text that can be processed
  Recommended setting: 4096 for low-performance devices, 8192 for high-performance devices

Parameter name: n_gpu_layers
  Adjustment range: -1-1000
  Function description: Number of model layers loaded to GPU (-1 = load all layers)
  Recommended setting: -1 for high-performance devices, 0 (CPU mode) for low-performance devices

Parameter name: vram_limit
  Adjustment range: -1-24
  Function description: VRAM limit (GB, -1 = no limit)
  Recommended setting: Usually set to -1; personally recommended to set as actual VRAM size minus 1 (e.g., 15 for 16GB VRAM)

Parameter name: image_min_tokens
  Adjustment range: 0-4096
  Function description: Minimum number of image encoding tokens
  Recommended setting: Keep default value 0

Parameter name: image_max_tokens
  Adjustment range: 0-4096
  Function description: Maximum number of image encoding tokens
  Recommended setting: Keep default value 0

### Intelligent Recommendations
- The system automatically recommends default values for n_ctx and n_gpu_layers based on hardware performance
- Low-performance devices will automatically reduce default parameters to ensure smooth operation

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

Parameter name: custom_prompt
  Function description: Custom prompt
  Recommended setting: Use in conjunction with preset prompts

Parameter name: system_prompt
  Function description: System prompt that defines model behavior
  Recommended setting: Modify according to requirements

Parameter name: inference_mode
  Function description: Inference mode
  Recommended setting: "images" for single image, "one by one" for multiple images, "video" for video

Parameter name: max_frames
  Adjustment range: 2-1024
  Function description: Maximum number of frames for video processing
  Recommended setting: Between 16-32

Parameter name: max_size
  Adjustment range: 128-16384
  Function description: Maximum size of images/videos
  Recommended setting: 256 for low-performance devices, 512 for high-performance devices

Parameter name: seed
  Adjustment range: 0-0xffffffffffffffff
  Function description: Random seed
  Recommended setting: 0 (random) or a fixed value

Parameter name: force_offload
  Function description: Force model offloading after inference
  Recommended setting: Usually set to False

Parameter name: save_states
  Function description: Save conversation state
  Recommended setting: Set to True for continuous dialogue

Parameter name: parameters
  Function description: Connect to the parameter setting node
  Recommended setting: Optional; use default parameters if not connected

Parameter name: images
  Function description: Input images
  Recommended setting: Provide when processing images

### Preset Prompt Template Description

Preset prompt templates cover various common inference scenarios:

- **Empty - Nothing**: Empty template, using only custom prompts
- **Normal - Describe**: Briefly describe image content
- **Prompt Style - Tags**: Generate image tag lists (suitable for SDXL model prompts)
- **Prompt Style - Simple**: Generate concise image descriptions (within 300 words)
- **Prompt Style - Detailed**: Generate detailed image descriptions (within 600 words)
- **Prompt Style - Comprehensive Expansion**: Detailed prompt expansion to enhance clarity and expressiveness in AI generation tasks, ensuring output language matches input language (within 1000 words)
- **Creative - Detailed Analysis**: Detailed analysis of image content
- **Creative - Summarize Video**: Summarize key events and narrative points of video content
- **Creative - Short Story**: Generate short stories based on images or videos
- **Creative - Refine & Expand Prompt**: Optimize and expand user prompts for greater expressiveness and visual richness
- **Vision - *Bounding Box**: Generate bounding boxes for object detection

## 3. Llama-cpp Parameter Setting Node

### Function
Sets detailed parameters for LLM inference to control the quality and style of generated text.

### Parameter Description

Parameter name: max_tokens
  Adjustment range: 0-4096
  Function description: Maximum number of generated tokens, controlling the maximum length of generated text and directly affecting the completeness of output content
  Recommended setting: 512-1024 for low-performance devices, 1024-2048 for high-performance devices

Parameter name: top_k
  Adjustment range: 0-1000
  Function description: Number of candidate tokens for sampling, controlling the number of vocabulary items considered per generation (smaller = more focused, larger = more diverse)
  Recommended setting: 20 for low-performance devices, 30 for high-performance devices

Parameter name: top_p
  Adjustment range: 0.0-1.0
  Function description: Nucleus sampling threshold, balancing diversity and accuracy of generated content (larger = more diverse)
  Recommended setting: 0.85-0.9

Parameter name: min_p
  Adjustment range: 0.0-1.0
  Function description: Minimum sampling probability, preventing low-probability tokens from being completely ignored to increase generation richness (default value is usually sufficient)
  Recommended setting: 0.05

Parameter name: typical_p
  Adjustment range: 0.0-1.0
  Function description: Typicality sampling parameter, controlling the typicality of generated content (1.0 = disable this feature; most users do not need adjustment)
  Recommended setting: 1.0

Parameter name: temperature
  Adjustment range: 0.0-2.0
  Function description: Generation temperature, controlling randomness (one of the most commonly used parameters; higher = more random, lower = more deterministic)
  Recommended setting: 0.6-0.8

Parameter name: repeat_penalty
  Adjustment range: 0.0-10.0
  Function description: Repetition penalty to avoid duplicate content and improve text quality (one of the commonly used parameters)
  Recommended setting: 1.0

Parameter name: frequency_penalty
  Adjustment range: 0.0-1.0
  Function description: Frequency penalty to reduce the probability of high-frequency tokens (0.0 = disable; usually no active adjustment needed)
  Recommended setting: 0.0

Parameter name: presence_penalty
  Adjustment range: 0.0-2.0
  Function description: Presence penalty to encourage new content generation and increase text richness (one of the commonly used parameters)
  Recommended setting: 1.0

Parameter name: mirostat_mode
  Adjustment range: 0-2
  Function description: Mirostat sampling mode for controlling perplexity of generated text (0 = disabled, 1 = basic mode, 2 = version 2; most users do not need to use)
  Recommended setting: 0

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

## 4. Llama-cpp Clear Session Node

### Function
Clears specified dialogue session states and releases resources.

### Parameter Description

Parameter name: any
  Function description: Placeholder parameter that can connect to any node
  Recommended setting: Any connection

Parameter name: state_uid
  Adjustment range: -1-999999
  Function description: ID of the dialogue state to clear
  Recommended setting: -1 (clear current session) or a specified ID

## 5. Llama-cpp Unload Model Node

### Function
Unloads the currently loaded model and releases memory/VRAM resources.

### Parameter Description

Parameter name: any
  Function description: Placeholder parameter that can connect to any node
  Recommended setting: Any connection

## 6. JSON to Bounding Box Node

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

## II. Device Performance Adaptation Recommendations

### Low-Performance Devices (e.g., RTX 3060 6GB)

- **Model Loading Node**:
  - n_ctx: 4096
  - n_gpu_layers: 0 (use CPU)
  - max_size: 256

- **Parameter Setting Node**:
  - max_tokens: 512-1024
  - temperature: 0.6
  - top_k: 20
  - top_p: 0.85

### High-Performance Devices (e.g., RTX 4090)

- **Model Loading Node**:
  - n_ctx: 8192
  - n_gpu_layers: -1 (load all layers to GPU)
  - max_size: 512-1024

- **Parameter Setting Node**:
  - max_tokens: 1024-2048
  - temperature: 0.8
  - top_k: 30
  - top_p: 0.9

## III. Common Issues and Solutions

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
- Reduce `n_ctx`
- Use smaller models
- Close unnecessary applications to free up system resources

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

## IV. Usage Recommendations

1. **Gradual Adjustment**: Start with default parameters and adjust incrementally based on actual results
2. **Model Matching**: Ensure `chat_handler` matches the model type
3. **Resource Monitoring**: Use task manager to monitor memory and VRAM usage
4. **Prompt Optimization**: High-quality prompts are more important than parameter adjustments; describe requirements in detail
5. **Session Management**: Regularly clear session states during long-term use to release resources

We hope this document helps you better use the Llama-cpp-vlmforQo plugin. For additional questions, refer to the README file or submit an issue.