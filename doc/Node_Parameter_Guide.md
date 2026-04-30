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
- Notes:
  - Different models require corresponding mmproj files, ensure version compatibility

**Parameter Name: Enable ASR (enable_asr)**
- Function: Enable ASR speech recognition functionality
- Recommended Setting: Set to True when speech recognition is needed
- Prerequisite: Must be used with ASR model loader

**Parameter Name: Enable TTS (enable_tts)**
- Function: Enable TTS text-to-speech functionality
- Recommended Setting: Set to True when speech synthesis is needed
- Prerequisite: Must be used with TTS model loader

#### Runtime Mode Parameters

**Parameter Name: Context Length (n_ctx)**
- Range: 1024-327680
- Function: Context length, affecting the length of text that can be processed
- Recommended Setting:
  - 24GB+ VRAM (5090/4090): 16384
  - 16GB VRAM (4080): 8192
  - 12GB VRAM (4070 Ti/3080): 6144
  - 8GB VRAM (3070/3060): 4096
  - 4-6GB VRAM: 2048
- Special Model Requirements:
  - Qwen3 series: GPU mode recommended ≤4096, Qwen3.5 recommended ≤2048
- Rule of Thumb: Context length is proportional to task complexity and inversely proportional to hardware performance

**Parameter Name: GPU Model Layers (n_gpu_layers)**
- Range: -1-1000
- Function: Number of model layers loaded to GPU, -1=all layers (GPU mode only)
- Recommended Setting:
  - GPU Mode:
    - 24GB+ VRAM (5090/4090): -1 (all layers)
    - 16GB VRAM (4080): -1 (all layers)
    - 12GB VRAM (4070 Ti/3080): -1 (all layers)
    - 8GB VRAM (3070/3060): 30 (partial loading)

**Parameter Name: VRAM Limit (vram_limit)**
- Range: -1-24
- Function: VRAM limit (GB), -1=no limit (GPU mode only)
- Recommended Setting:
  - GPU Mode: Usually set to -1 or actual VRAM minus 1 (e.g., set 15 for 16GB)

#### Image Processing Parameters

**Parameter Name: Minimum Image Encoding Tokens (image_min_tokens)**
- Range: 0-4096
- Function: Minimum number of tokens for image encoding
- Recommended Setting:
  - Default: 0 (auto)
  - Qwen3 series: ≥1024 (auto-adjusted)
  - MiniCPM-o-4.5: ≥768 (auto-adjusted)
- Notes: Some models require minimum tokens for proper image processing

**Parameter Name: Maximum Image Encoding Tokens (image_max_tokens)**
- Range: 0-4096
- Function: Maximum number of tokens for image encoding
- Recommended Setting: Keep default 0 (auto)
- Notes: If set, must be ≥ image_min_tokens

#### Advanced Parameters

**Parameter Name: Attention Type (attention_type)**
- Range: Auto/Standard/Flash/XFormers
- Function: Select attention mechanism type for the model, affecting inference speed and memory usage
- Recommended Setting: Auto (default)
- Options:
  - Auto: Automatic selection (recommended, Flash Attention automatically enabled for NVIDIA GPU)
  - Flash: Flash Attention (NVIDIA GPU optimization, fastest inference)
  - XFormers: XFormers optimization (AMD, Intel GPU optimization, more memory efficient)
- Use Case: Manually adjust when encountering attention mechanism compatibility issues or needing performance optimization

### Smart Recommendations
- The system automatically recommends default values for device_mode, n_ctx, and n_gpu_layers based on hardware performance
- Low-performance devices automatically reduce default parameters to ensure smooth operation
- CPU mode automatically ignores GPU-related parameters, no manual adjustment needed
- Batch parameters (n_batch, n_ubatch, n_threads, n_threads_batch) are now automatically adjusted based on hardware performance

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
- Function: Select inference mode to determine input type and task type
- Options:
  - `[Basic] Text Generation`: Text-only mode, processes text input only
  - `[Basic] Image Understanding`: Image processing mode, processes single or multiple images
  - `[Basic] Audio to Text`: Uses ASR model to convert audio to text
  - `[Basic] Text to Audio`: Generates text then converts to speech using TTS model
  - `[Advanced] Video Understanding`: Processes video files, extracts frames for analysis

#### Language Settings

**Parameter Name: Preset Template Language (prompt_language)**
- Function: Call preset template in corresponding language
- Options: Chinese / English
- Recommended Setting: Select as needed

**Parameter Name: Response Language (response_language)**
- Function: Select output text language
- Options: Chinese / English
- Recommended Setting: Select as needed

**Parameter Name: ASR Language (asr_language)**
- Function: Target language for ASR speech recognition
- Options: Auto Detect / Chinese / English / Japanese / Korean / French / German / Spanish
- Recommended Setting: Select based on audio content

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

## 3. Llama-cpp Parameters Node (llama_cpp_parameters)

### Function
Set detailed parameters for LLM inference to control the quality and style of generated text.

### Parameter Description

**Parameter Name: Maximum Tokens (max_tokens)**
- Range: 0-4096
- Function: Maximum number of generated tokens, affecting output text length
- Recommended Setting: Low-performance devices: 512-1024, High-performance devices: 1024-2048
- Practical Tips:
  - Short answers: 256-512
  - Detailed descriptions: 768-1024
  - Long text generation: 1024-2048

**Parameter Name: Top-K Sampling (top_k)**
- Range: 0-1000
- Function: Number of sampling candidates, smaller values produce more focused generation
- Recommended Setting: Low-performance devices: 20, High-performance devices: 30
- Impact Analysis:
  - Low top_k (<20): More deterministic generation, suitable for factual tasks
  - High top_k (>50): More diverse generation, suitable for creative tasks

**Parameter Name: Top-P Sampling (top_p)**
- Range: 0.0-1.0
- Function: Nucleus sampling threshold, controls generation diversity
- Recommended Setting: 0.85-0.9
- Effect Comparison:
  - Low top_p (<0.8): More conservative generation, higher accuracy
  - High top_p (>0.9): More open generation, stronger diversity

**Parameter Name: Minimum Probability (min_p)**
- Range: 0.0-1.0
- Function: Minimum sampling probability, prevents complete neglect of low-probability tokens
- Recommended Setting: 0.05
- Use Case: Use when increased generation diversity is needed

**Parameter Name: Typical Sampling (typical_p)**
- Range: 0.0-1.0
- Function: Typical sampling threshold, controls generation typicality
- Recommended Setting: 1.0

**Parameter Name: Temperature**
- Range: 0.0-2.0
- Function: Generation temperature, higher values are more random
- Recommended Setting: 0.6-0.8
- Temperature Guide:
  - Low temperature (0.1-0.4): Suitable for scenarios requiring accurate answers, e.g., QA, code generation
  - Medium temperature (0.5-0.8): Suitable for most scenarios, balances accuracy and creativity
  - High temperature (0.9-1.5): Suitable for creative tasks, e.g., story generation, poetry

**Parameter Name: Repeat Penalty (repeat_penalty)**
- Range: 0.0-10.0
- Function: Repeat penalty, prevents repetitive content generation
- Recommended Setting: 1.0
- Adjustment Tips:
  - When repetitive content occurs, gradually increase to 1.1-1.3
  -不宜过高，否则可能导致生成内容不连贯
  - Avoid setting too high, may cause incoherent generation

**Parameter Name: Frequency Penalty (frequency_penalty)**
- Range: 0.0-1.0
- Function: Frequency penalty, reduces high-frequency token occurrence
- Recommended Setting: 0.0
- Use Case: Use when needing to reduce high-frequency word repetition

**Parameter Name: Presence Penalty (presence_penalty)**
- Range: 0.0-2.0
- Function: Presence penalty, encourages new content generation
- Recommended Setting: 1.0
- Use Case:
  - Avoid topic drift in long text generation
  - Encourage new viewpoints in conversations

**Parameter Name: Mirostat Mode (mirostat_mode)**
- Range: 0-2
- Function: Mirostat sampling mode: 0=off, 1=basic, 2=version 2
- Recommended Setting: 0
- Advanced User Tip: Try mode 1 or 2 for more consistent generation quality

**Parameter Name: Mirostat Eta (mirostat_eta)**
- Range: 0.0-1.0
- Function: Mirostat learning rate
- Recommended Setting: 0.1

**Parameter Name: Mirostat Tau (mirostat_tau)**
- Range: 0.0-10.0
- Function: Mirostat target perplexity
- Recommended Setting: 5.0

**Parameter Name: Conversation State ID (state_uid)**
- Range: -1-999999
- Function: Conversation state ID, -1=use node unique ID
- Recommended Setting: -1
- Session Management:
  - Use different state_uid to maintain multiple independent sessions simultaneously
  - Keep same state_uid for continuous conversations to maintain context

### Parameter Adjustment Tips

1. **Control Output Length**: Adjust `max_tokens`. Larger values produce longer output but consume more resources. Set according to actual needs, avoid setting too large to prevent performance issues

2. **Control Randomness**: `temperature` is one of the most commonly used parameters. Lower values are more deterministic (suitable for accurate answers), higher values are more creative (suitable for story/poetry generation)

3. **Balance Diversity and Accuracy**:
   - `top_p` and `top_k` are usually used together
   - Decreasing `top_k` while increasing `top_p` improves accuracy while maintaining diversity
   - Increasing `top_k` while decreasing `top_p` increases generation diversity

4. **Avoid Repetitive Content**:
   - Increase `repeat_penalty` to reduce repetition
   - Combine with `presence_penalty` to encourage new content generation

5. **Parameter Priority Recommendations**:
   - Primary adjustment: `temperature`, `max_tokens`, `top_p`/`top_k`
   - Secondary adjustment: `repeat_penalty`, `presence_penalty`
   - Advanced users: `min_p`, `mirostat` related parameters

6. **Common Scenario Settings**:
   - **Factual QA**: temperature=0.1-0.3, top_k=10, top_p=0.7
   - **Creative Writing**: temperature=0.8-1.2, top_k=50, top_p=0.95
   - **Code Generation**: temperature=0.2-0.4, top_k=20, top_p=0.8
   - **Dialogue**: temperature=0.6-0.8, top_k=30, top_p=0.9

## 4. API Model Nodes

### 4.1 API Configuration Manager (llama_cpp_api_model_config)

#### Function
Create and manage API model configurations, supporting multiple external API providers.

#### Node Parameters

**Required Parameters:**
- `model_name`: API model name, used to identify this configuration
- `api_provider`: API provider, options include OpenAI/Ollama/llms-py/vllm-omni/Custom, etc.
- `api_base`: API base URL
- `api_key`: API key (if required)
- `max_tokens`: Maximum generated tokens (default 1024)
- `temperature`: Temperature parameter (default 0.7)

### 4.2 API Inference Node (llama_cpp_api_inference)

#### Function
Perform multimedia inference using API, supporting image, video, audio inputs, and text, audio outputs.

#### Node Parameters

**Required Parameters:**
- `api_config`: API configuration (connect to API Configuration Manager node)
- `inference_mode`: Inference mode, options: Text Generation/Image Understanding/Audio to Text/Text to Audio/Multimodal Integration/Video Understanding
- `preset_prompt`: Select preset prompt template
- `system_prompt`: System prompt
- `text_input`: User input text
- `prompt_language`: Language of preset prompt, options: Chinese/English
- `response_language`: AI response language, options: Chinese/English
- `video_max_frames`: Video mode: Maximum extracted frames (default 16)
- `video_sampling`: Video frame sampling method, options: Auto Uniform Sampling/Manual Frame Indices
- `video_manual_indices`: Frame indices in manual mode
- `image_max_size`: Maximum edge length for image processing (default 256)
- `tts_voice`: TTS voice selection
- `tts_emotion`: TTS emotion style
- `tts_speed`: TTS speed (default 1.0)
- `seed`: Random seed (default 101)

**Optional Parameters:**
- `images`: Image input (for image understanding mode)
- `video`: Video input (for video understanding mode)
- `audio`: Audio input (for audio understanding mode)

#### Output

- `text`: Generated text
- `audio`: Generated audio (when Text to Audio or Multimodal Integration mode is selected)

## 5. ASR/TTS Model Parameters

### 5.1 ASR Model Loader Parameters (llama_cpp_asr_loader)

#### Node Display Options (Manually Adjustable)

- **ASR Model**: Select ASR model file, choose appropriate ASR model as needed, only qwen3-asr is currently supported
- **GPU Model Layers**: Number of model layers loaded to GPU: 24GB+ VRAM: -1, 16GB VRAM: -1, 12GB VRAM: -1, 8GB VRAM: 20, Range: -1-1000
- **Language**: Recognition language, select based on audio content, options: auto/zh/en/ja/ko/fr/de/es
- **Task**: Task type: transcribe/translate (translate to English), options: transcribe/translate

#### Audio Input Requirements

- **Supported Formats**: WAV, MP3, FLAC and other common audio formats
- **Recommended Sample Rate**: 16000Hz or 22050Hz
- **Channels**: Mono or stereo (auto-handled)

### 5.2 TTS Model Loader Parameters (llama_cpp_tts_loader)

#### Node Display Options (Manually Adjustable)

- **TTS Model**: Select TTS model file, only qwen3-tts is currently supported
- **GPU Model Layers**: Number of model layers loaded to GPU: 24GB+ VRAM: -1, 16GB VRAM: -1, 12GB VRAM: -1, 8GB VRAM: 20, Range: -1-1000
- **Sample Rate**: Audio sample rate (Hz), Qwen3-TTS recommends 24000Hz, other models 22050Hz, Range: 8000-48000
- **Voice**: Select voice type, choose based on content and scenario, select based on model-supported voices
- **Emotion**: Select emotion style, choose based on content emotion, options: Default/Happy/Sad/Angry/Surprised/Calm/Excited/Gentle

#### Optional Parameters

- **temperature**: Generation temperature, higher values are more random, default 0.7, Range: 0.1-2.0
- **top_p**: Nucleus sampling parameter, controls generation diversity, default 0.9, Range: 0.1-1.0
- **top_k**: Top-K sampling parameter, default 50, Range: 1-100
- **repetition_penalty**: Repeat penalty, higher values avoid repetition more, default 1.1, Range: 1.0-2.0
- **max_new_tokens**: Maximum generated tokens, default 2048, Range: 100-4096
- **use_cache**: Enable KV cache for inference acceleration, default True, options: True/False
- **ref_audio_path**: Voice cloning reference audio path, default "" (empty), WAV format file path
- **pitch**: Pitch offset, adjusted based on reference audio, default 0.0, Range: -5.0-5.0
- **volume**: Volume, default 1.0, Range: 0.1-2.0

#### Emotion Mapping

- **Default**: emotion value default, neutral emotion
- **Happy**: emotion value happy, upbeat tone, positive and cheerful
- **Sad**: emotion value sad, low tone, sad and heavy
- **Angry**: emotion value angry, strong tone, excited and angry
- **Surprised**: emotion value surprised, fluctuating tone, surprised and unexpected
- **Calm**: emotion value calm, steady tone, objective and calm
- **Excited**: emotion value excited, high tone, excited
- **Gentle**: emotion value gentle, soft tone, kind and warm

### 5.3 Qwen3-TTS Special Instructions

#### Model Variants

- **VoiceDesign**: Detects keywords voicedesign, voice_design, supports voice design, controls voice through natural language instructions
- **CustomVoice**: Detects keywords customvoice, custom_voice, supports voice cloning and standard voices

#### Sample Rate Notes

- **12Hz version**: 12000Hz, auto-detected when path contains "12hz"
- **Standard version**: 24000Hz, default

#### Voice Cloning Reference Audio Requirements

- **Format**: WAV format
- **Duration**: 3-60 seconds optimal
- **Channels**: Mono
- **Sample Rate**: 48000Hz
- **Quality**: Clear, no noise, no background music

### 5.4 Using TTS in Inference Node

#### Method 1: TTS Voice Enhancement Output Mode

1. Select `Audio Output Mode` as `TTS Voice Enhancement Output` in inference node
2. Connect TTS model to `tts_model` input port
3. Select appropriate `tts_voice`, `tts_emotion`, and `tts_speed`
4. The `audio` output of inference node will contain generated audio

#### Method 2: Independent TTS Synthesis

1. Load TTS model using TTS model loader
2. Call `synthesize` method of TTS model:
   ```python
   audio_output = tts_model.synthesize(
       text="Text to synthesize",
       speaker_id=0,  # Voice ID
       speed=1.0,     # Speed
       emotion="default"  # Emotion style
   )
   ```
3. Save audio using audio save node

## 6. Multi-Speaker Conversation Nodes

### 6.1 Multi-Speaker TTS Node (multi_speaker_tts)

Multi-speaker TTS functionality allows you to generate dialogue audio with multiple speakers in ComfyUI, supporting different voices and emotional expressions.

#### Dialogue Text Formats

Three dialogue text formats are supported:

##### Format 1: Bracket Format (Recommended)

```
[Female1] Hello, how's the weather today?
[Male1] The weather is nice today, sunny.
[Female1] Shall we go out for a walk?
[Male1] Sure, let's go to the park.
```

##### Format 2: Colon Format

```
Female1: Hello, how's the weather today?
Male1: The weather is nice today, sunny.
Female1: Shall we go out for a walk?
Male1: Sure, let's go to the park.
```

##### Format 3: Dash Format

```
Female1 - Hello, how's the weather today?
Male1 - The weather is nice today, sunny.
Female1 - Shall we go out for a walk?
Male1 - Sure, let's go to the park.
```

#### Emotion Tags

Emotion tags can be added after speaker names:

```
[Female1] Hello, how's the weather today?
[Male1] The weather is nice today, sunny.
[Female1(Happy)] Great! Shall we go out for a walk?
[Male1(Calm)] Sure, let's go to the park.
```

#### Speaker List

| Name | Description |
|------|-------------|
| Female1 | Clear female voice |
| Female2 | Gentle female voice |
| Male1 | Steady male voice |
| Male2 | Deep male voice |

#### Emotion List

| Name | Description |
|------|-------------|
| Default | Normal tone |
| Happy | Cheerful tone |
| Sad | Low tone |
| Angry | Intense tone |
| Surprised | Upbeat tone |
| Calm | Peaceful tone |
| Excited | Enthusiastic tone |
| Gentle | Soft tone |

#### Quick Parameter Settings

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| speed | 0.6-1.8 | 1.0 | Speech speed |
| pitch | -4.0-4.0 | 0.0 | Pitch offset |
| volume | 0.2-2.0 | 1.0 | Volume |
| pause_duration | 0.0-2.0 | 0.5 | Pause duration (seconds) |

#### Common Scenario Configurations

##### Daily Conversation
- speed: 1.0
- pitch: 0.0
- pause_duration: 0.5

##### Emotional Expression
- speed: 0.8
- pitch: 0.5
- pause_duration: 0.7

##### Fast Dialogue
- speed: 1.4
- pitch: 0.0
- pause_duration: 0.3

##### Narrative Reading
- speed: 0.9
- pitch: -0.5
- pause_duration: 0.8

#### Example Dialogues

##### Simple Dialogue

```
[Female1] Hello, can I help you?
[Male1] I'd like to learn about your products.
[Female1] Sure, our products have many advantages.
[Male1] Can you introduce them in detail?
[Female1] No problem, let me explain in detail.
```

##### Dialogue with Emotions

```
[Female1(Happy)] What a nice day!
[Male1(Calm)] Yes, perfect for a walk.
[Female1(Excited)] Let's go to the park!
[Male1(Gentle)] Okay, I'll go with you.
```

##### Multi-Character Dialogue

```
[Female1] Hello everyone, welcome to today's meeting.
[Male1] Thank you host, I'm glad to attend.
[Female2] I'm also looking forward to today's discussion.
[Male2] Let's start, time is limited.
[Female1] Okay, let's first discuss the first topic.
```

#### Quick Checklist

Check before use:
- [ ] TTS model loaded
- [ ] Dialogue text format correct
- [ ] Speaker names correct
- [ ] Emotion names correct
- [ ] Output filename set

#### FAQ & Solutions

| Issue | Solution |
|-------|----------|
| Some segments not generated | Check speaker and emotion names |
| Audio unnatural | Adjust pause time and speed |
| Format parsing failed | Use bracket format |
| Volume too low | Adjust volume parameter |

## 7. Audio Time Alignment Nodes

### 7.1 Forced Aligner Model Loader (llama_cpp_forced_aligner_loader)

Forced aligner model loader node is used to load Qwen3-ForcedAligner model, supporting automatic detection and downloading of missing files.

#### Node Parameters

**Required Parameters:**
- `forced_aligner_model`: Select forced aligner model file, supports Qwen3-ForcedAligner-0.6B model
- `n_gpu_layers`: Number of model layers loaded to GPU, -1 means all layers to GPU

**Optional Parameters:**
- `precision`: Model precision selection, options: float16/bfloat16/float32, default float16
- `batch_size`: Batch size, default 32, range 1-128
- `flash_attention`: Enable FlashAttention2 optimization, requires flash-attn installation

#### Supported Models
- **Qwen3-ForcedAligner-0.6B**: Alibaba Qwen forced alignment model, supports precise time alignment between Chinese speech and text

#### Model File Requirements
- `model.safetensors`: Model weights file
- `config.json`: Model configuration file
- `tokenizer.json`: Tokenizer configuration file
- `preprocessor_config.json`: Preprocessor configuration file

#### Auto Download Feature
The node automatically checks for required files, and downloads missing files from ModelScope or Hugging Face based on network connectivity.

### 7.2 Forced Aligner Inference Node (llama_cpp_forced_aligner_inference)

Forced aligner inference node inputs audio and text to ForcedAligner, outputs fine-grained timestamps (phoneme/word level).

#### Node Parameters

**Required Parameters:**
- `aligner_model`: Connect output of forced aligner model loader
- `audio`: Input audio data
- `text`: Text content to align

**Optional Parameters:**
- `sample_rate`: Audio sample rate, default 16000Hz, range 8000-48000Hz
- `output_format`: Output format, options: Text/SRT/VTT/JSON, default Text

#### Output Description

The node outputs three values:
1. **aligned_text**: Aligned text
2. **timestamps**: Timestamp data (JSON format), contains paragraph, word, and phoneme level time information
3. **subtitle_text**: Subtitle format text (SRT or VTT format)

#### Timestamp Data Structure
```json
{
"task_id": "Task ID",
"audio_info": {
    "audio_duration": Audio duration (seconds),
    "audio_sample_rate": Sample rate
},
"segments": [
    {
    "start": Start time (seconds),
    "end": End time (seconds),
    "text": "Segment text",
    "words": [
        {
        "word": "Word",
        "start": Start time (seconds),
        "end": End time (seconds),
        "phonemes": [
            {
            "phoneme": "Phoneme",
            "start": Start time (seconds),
            "end": End time (seconds)
            }
        ]
        }
    ]
    }
],
"status": "success/failed",
"error_msg": "Error message"
}
```

### 7.3 TTS Alignment Node (tts_align)

TTS alignment node uses timestamp data from forced aligner inference node to generate voice-over with precise text timeline alignment, supporting two modes: exact timing alignment and natural flow alignment.

#### Node Parameters

**Required Parameters:**
- `tts_model`: TTS model
- `timestamps`: Timestamp data, can be obtained from forced aligner inference node or imported from JSON file
- `text`: Text content to synthesize
- `sample_rate`: Audio sample rate, default 24000Hz, range 8000-48000Hz

**Optional Parameters:**
- `speed`: TTS speed, default 1.0, range 0.5-2.0
- `align_mode`: Alignment mode, options: Exact Timing/Natural Flow, default Natural Flow
- `silence_padding`: Silence padding after segments (seconds), default 0.1, range 0.0-1.0

#### Alignment Mode Description
- **Exact Timing**: Strictly generate audio according to timestamps, ensuring each segment duration matches timestamps exactly
- **Natural Flow**: Maintain natural speech rhythm, do not force audio duration adjustment

#### Output Description
The node outputs two values:
1. **aligned_audio**: Aligned audio
2. **alignment_info**: Alignment information, containing alignment status and duration for each segment

### 7.4 Role Configuration Node (RoleConfig)

Role configuration node is used to configure role parameters, including name, voice, speed, pitch, volume, etc., supporting local TTS models.

#### Node Parameters

**Required Parameters:**
- `role_name`: Role name

**Local Model Parameters:**
- `voice`: Local model voice, options:
  - Vivian - Bright, slightly sharp young female voice (Chinese)
  - Serena - Warm, soft young female voice (Chinese)
  - Uncle_Fu - Deep, rich mature male voice (Chinese)
  - Dylan - Clear, natural Beijing dialect male voice (Chinese)
- `emotion`: Local model emotion, options: default/happy/sad/angry/surprised/calm/excited/gentle, default default

**General Parameters:**
- `speed`: Speech speed, default 1.0, range 0.5-2.0
- `pitch`: Pitch offset, default 0.0, range -5.0-5.0
- `volume`: Volume, default 1.0, range 0.1-2.0

#### Output Description
The node outputs one value:
- **role_config**: Role configuration dictionary, containing all configuration parameters

## 8. FAQ & Solutions

### 8.1 TTS Audio Output Failed

**Reasons:**
- TTS model does not support specified voice or emotion
- Text length exceeds model limit
- Missing required audio processing dependencies

**Solutions:**
- Check supported voice and emotion range for TTS model
- Reduce input text length
- Install required audio processing libraries

### 8.2 TTS Voice Cloning Poor Quality

**Reasons:**
- Poor reference audio quality
- Incorrect reference audio format
- Inappropriate reference audio duration

**Solutions:**
- Use clear reference audio, avoid noise and background music
- Ensure reference audio is WAV format, mono, 48000Hz
- Control reference audio duration between 3-60 seconds

### 8.3 Low ASR Recognition Accuracy

**Reasons:**
- Poor audio quality
- Mismatched language settings
- Inappropriate model selection

**Solutions:**
- Use clear audio files
- Set correct language parameters
- Select appropriate ASR model for language and scenario

### 8.4 Audio Save Failed

**Reasons:**
- Incorrect audio data format
- Output path does not exist
- Unsupported format

**Solutions:**
- Ensure audio data contains waveform and sample_rate fields
- Check if output path exists
- Use supported formats (wav/mp3/flac/ogg)

## 9. Usage Flow Examples

### 9.1 TTS Speech Synthesis

1. **Load Models**:
   - Use `llama_cpp_model_loader` node to load LLM model
   - Use `llama_cpp_tts_loader` node to load TTS model (e.g., Qwen3-TTS)

2. **Configure Unified Inference Node**:
   - Use `llama_cpp_unified_inference` node
   - Select inference mode: `[Basic] Text to Audio`
   - Connect LLM model to `Llama Model` input
   - Connect TTS model to `tts_model` input
   - Set TTS voice (tts_voice), emotion (tts_emotion), and speed (tts_speed)

3. **Provide Text Input**:
   - Enter text to synthesize in `Text Input`
   - Select appropriate preset template

4. **Save Audio**:
   - Use `llama_cpp_audio_saver` audio save node
   - Connect `audio` output of inference node to audio save node
   - Select save format (WAV recommended)

### 9.2 ASR Speech Recognition

1. **Load Models**:
   - Use `llama_cpp_model_loader` node to load LLM model
   - Use `llama_cpp_asr_loader` node to load ASR model (e.g., Qwen3-ASR)

2. **Configure Unified Inference Node**:
   - Use `llama_cpp_unified_inference` node
   - Select inference mode: `[Basic] Audio to Text`
   - Connect LLM model to `Llama Model` input
   - Connect ASR model to `asr_model` input
   - Set ASR language (asr_language)

3. **Provide Audio Input**:
   - Connect audio input to `audio` port

4. **Run Recognition**:
   - Get recognized text output

### 9.3 Multi-Speaker Conversation Synthesis

1. **Load TTS Model**:
   - Use `llama_cpp_tts_loader` node to load TTS model (e.g., Qwen3-TTS)

2. **Configure Multi-Speaker TTS Node**:
   - Use `multi_speaker_tts` node
   - Connect TTS model to `tts_model` input
   - Enter multi-speaker dialogue text in `dialogue_text`
   - Set `output_name` output filename

3. **Set Dialogue Parameters**:
   - Set `default_speaker` default speaker
   - Set `default_emotion` default emotion
   - Adjust `speed` parameter
   - Set `pause_duration` pause time between speakers

4. **Generate Dialogue Audio**:
   - Run workflow
   - Audio file will be saved in audio subdirectory of ComfyUI output directory

## 10. Multi-Image Input Node Guide

### 10.1 Node Overview

**Multi-Image Input (Story Creation)** node supports two working modes:

1. **Image Mode**: Analyze multiple images and create story content, supports multiple video generation models (WAN2.2, LTX2, etc.)
2. **Text Mode**: Generate prompts through option settings, no image input required

### 10.2 Node Functions

#### Main Functions
1. **Dual Mode Support**: Flexible switching between image mode and text mode
2. **Multi-Image Input**: Supports input of multiple images (image mode)
3. **Auto Preprocessing**: Automatic scaling and encoding of images
4. **Story Creation**: Generate story content suitable for video generation
5. **Flexible Configuration**: Supports multiple story types, lengths, and styles
6. **Multiple Applications**: Supports story creation, script writing, advertising copy, and other content types
7. **Image Output**: Pass image data to inference node

### 10.3 Input Parameters

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

### 10.4 Output Parameters

- **prompt** (STRING): Generated content creation prompt
- **images** (IMAGE): Image data (returns preprocessed images in image mode, None in text mode)

### 10.5 Usage Methods

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

### 10.6 Best Practices

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

### 10.7 FAQ

**Q1: What's the difference between Image Mode and Text Mode?**
- **Image Mode**: Requires images, model analyzes image content and creates stories
- **Text Mode**: No images needed, generates prompts through option settings

**Q2: Why is the story not coherent enough?**
Possible reasons: Image content not coherent/Inappropriate story type selection/Unclear custom prompt
Solutions: Select more coherent images/Try different story types/Add more specific custom prompts

**Q3: How to correctly connect image data to inference node?**
- **Image Mode**: prompt→custom_prompt, images→images
- **Text Mode**: prompt→custom_prompt, no need to connect images