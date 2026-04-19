# ComfyUI-omni-llm ASR/TTS Model Parameter Explanation and Recommended Settings

This document provides detailed explanations of parameter settings and recommended configurations for ASR models, TTS models, and multi-speaker conversation nodes in the ComfyUI-omni-llm plugin, distinguishing between node-displayed options and background-adjusted parameters.

## I. ASR Model Parameter Explanation

### 1. ASR Model Loader Node Parameters (llama_cpp_asr_loader)

#### Node Display Options (Manually Adjustable)

- **ASR Model**: Select ASR model file. Choose appropriate ASR models as needed, supports .gguf, .safetensors, .pt, .pth, .bin
- **Run Mode**: Select run mode, GPU mode recommended (when sufficient VRAM), options: GPU/CPU
- **GPU Model Layers**: Number of model layers to load to GPU, 24GB+ VRAM: -1, 16GB VRAM: -1, 12GB VRAM: -1, 8GB VRAM: 20, range: -1-1000
- **Language**: Recognition language, choose based on audio content, options: auto/zh/en/ja/ko/fr/de/es
- **Task**: Task type, transcribe (transcription)/translate (translate to English), options: transcribe/translate

#### Supported ASR Model Types

- **Whisper**: Detects keyword "whisper", excellent multilingual support, high accuracy
- **Wav2Vec2**: Detects keywords "wav2vec", "wav2vec2", suitable for specific language optimization
- **Qwen Audio**: Detects keywords "qwen" + "audio/asr", excellent Chinese support
- **FunASR**: Detects keywords "fun" + "asr", Chinese speech recognition
- **GLM-ASR**: Detects keywords "glm" + "asr", supports MLX format

#### Audio Input Requirements

- **Supported Formats**: WAV, MP3, FLAC, and other common audio formats
- **Recommended Sample Rate**: 16000Hz or 22050Hz
- **Channels**: Mono or stereo (automatically processed)

### 2. Using ASR in Inference Node

In unified inference node:
1. Connect ASR model to `asr_model` input port
2. Select inference mode as `[Basic] Audio to Text`
3. Provide audio input to `audio` port
4. Set appropriate `asr_language`

## II. TTS Model Parameter Explanation

### 1. TTS Model Loader Node Parameters (llama_cpp_tts_loader)

#### Node Display Options (Manually Adjustable)

- **TTS Model**: Select TTS model file, choose appropriate TTS models as needed, supports .gguf, .safetensors, .pt, .pth, .bin
- **GPU Model Layers**: Number of model layers to load to GPU, 24GB+ VRAM: -1, 16GB VRAM: -1, 12GB VRAM: -1, 8GB VRAM: 20, range: -1-1000
- **Sample Rate**: Audio sample rate (Hz), Qwen3-TTS recommends 24000Hz, other models 22050Hz, range: 8000-48000
- **Voice**: Select voice type, choose based on content and scenario, options: Female 1/Female 2/Male 1/Male 2
- **Emotion**: Select emotion style, choose based on content emotion, options: Default/Happy/Sad/Angry/Surprised/Calm/Excited/Gentle

#### Optional Parameters (Displayed when expanded)

- **temperature**: Generation temperature, higher value means more random, default: 0.7, range: 0.1-2.0
- **top_p**: Nucleus sampling parameter, controls generation diversity, default: 0.9, range: 0.1-1.0
- **top_k**: Top-K sampling parameter, default: 50, range: 1-100
- **repetition_penalty**: Repetition penalty, higher value avoids repetition, default: 1.1, range: 1.0-2.0
- **max_new_tokens**: Maximum generated tokens, default: 2048, range: 100-4096
- **use_cache**: Whether to use KV cache for acceleration, default: True, options: True/False
- **ref_audio_path**: Voice cloning reference audio path, default: "" (empty), WAV format file path
- **pitch**: Pitch offset, adjusted based on reference audio, default: 0.0, range: -5.0-5.0
- **volume**: Volume, default: 1.0, range: 0.1-2.0

#### Voice to speaker_id Mapping

- **Female 1**: speaker_id 0, suitable for female characters, gentle content
- **Female 2**: speaker_id 1, suitable for female characters, lively content
- **Male 1**: speaker_id 2, suitable for male characters, formal content
- **Male 2**: speaker_id 3, suitable for male characters, gentle content

#### Emotion to emotion Mapping

- **Default**: emotion value "default", neutral emotion
- **Happy**: emotion value "happy", rising intonation, positive and cheerful
- **Sad**: emotion value "sad", low intonation, sad and heavy
- **Angry**: emotion value "angry", strong intonation, excited and angry
- **Surprised**: emotion value "surprised", fluctuating intonation, surprised and unexpected
- **Calm**: emotion value "calm", steady intonation, objective and calm
- **Excited**: emotion value "excited", high intonation, excited and enthusiastic
- **Gentle**: emotion value "gentle", soft intonation, kind and warm

### 2. Supported TTS Model Types

- **Qwen3-TTS**: Detects keywords "qwen" + "tts", high-quality Chinese TTS, supports VoiceDesign and CustomVoice variants
- **Qwen3-TTS-MLX**: Detects keywords "qwen" + "tts" + "mlx", MLX format Qwen3-TTS, requires MLX environment
- **Bark**: Detects keyword "bark", multilingual support, high naturalness
- **XTTS**: Detects keyword "xtts", supports voice cloning, requires reference audio
- **VITS**: Detects keyword "vits", lightweight, fast speed
- **Coqui**: Detects keywords "coqui", "your_tts", open-source TTS
- **Supertonic**: Detects keyword "supertonic", ONNX format
- **Supertonic-2**: Detects keywords "supertonic-2", "supertonic2", improved Supertonic

### 3. Qwen3-TTS Model Special Instructions

#### Model Variants

- **VoiceDesign**: Detects keywords "voicedesign", "voice_design", supports voice design, controls voice through natural language instructions
- **CustomVoice**: Detects keywords "customvoice", "custom_voice", supports voice cloning and standard voices

#### Sample Rate Instructions

- **12Hz version**: 12000Hz, automatically detected if path contains "12hz"
- **Standard version**: 24000Hz, default usage

#### Voice Cloning Reference Audio Requirements

- **Format**: WAV format
- **Duration**: 3-60 seconds optimal
- **Channels**: Mono
- **Sample Rate**: 48000Hz
- **Quality**: Clear, no noise, no background music

### 4. Using TTS in Inference Node

#### Method 1: TTS Voice Enhancement Output Mode

1. In inference node, select `Audio Output Mode` as `TTS Voice Enhancement Output`
2. Connect TTS model to `tts_model` input port
3. Select appropriate `tts_voice`, `tts_emotion`, and `tts_speed`
4. The `audio` output of inference node will contain generated audio

#### Method 2: Independent TTS Synthesis

1. Load TTS model using TTS model loader
2. Call TTS model's `synthesize` method:
   ```python
   audio_output = tts_model.synthesize(
       text="Text to synthesize",
       speaker_id=0,  # Voice ID
       speed=1.0,     # Speech speed
       emotion="default"  # Emotion style
   )
   ```
3. Use audio save node to save audio

## III. Multi-speaker Conversation Related Nodes

### 1. Multi-speaker TTS Node (multi_speaker_tts)

Multi-speaker TTS functionality allows you to generate dialogue audio with multiple speakers in ComfyUI, supporting different voices and emotion expressions.

#### Feature Characteristics

- Supports 4 speakers: Female 1, Female 2, Male 1, Male 2
- Supports 8 emotions: Default, Happy, Sad, Angry, Surprised, Calm, Excited, Gentle
- Supports multiple dialogue text formats
- Adjustable speed, pitch, volume
- Configurable pause time between speakers
- Automatic splicing of multiple audio segments

#### Node Parameters

**Required Parameters:**
- `tts_model`: Connect output of TTS loader node
- `dialogue_text`: Multi-speaker dialogue text
- `output_name`: Output file name

**Optional Parameters:**
- `default_speaker`: Default speaker (used when not specified in dialogue)
- `default_emotion`: Default emotion (used when not specified in dialogue)
- `speed`: Speech speed (0.6-1.8, default 1.0)
- `pitch`: Pitch offset (-4.0 to 4.0, default 0.0)
- `volume`: Volume (0.2-2.0, default 1.0)
- `pause_duration`: Pause time between speakers (seconds, default 0.5)
- `add_silence`: Whether to add silence (default True)

#### Dialogue Text Format

Supports three dialogue text formats:

##### Format 1: Bracket Format (Recommended)

```
[Female 1] Hello, how's the weather today?
[Male 1] The weather is nice today, sunny.
[Female 1] Shall we go for a walk?
[Male 1] Sure, let's go to the park.
```

##### Format 2: Colon Format

```
Female 1: Hello, how's the weather today?
Male 1: The weather is nice today, sunny.
Female 1: Shall we go for a walk?
Male 1: Sure, let's go to the park.
```

##### Format 3: Dash Format

```
Female 1 - Hello, how's the weather today?
Male 1 - The weather is nice today, sunny.
Female 1 - Shall we go for a walk?
Male 1 - Sure, let's go to the park.
```

#### Emotion Markers

Emotion markers can be added after speakers:

```
[Female 1] Hello, how's the weather today?
[Male 1] The weather is nice today, sunny.
[Female 1(Happy)] Great! Shall we go for a walk?
[Male 1(Calm)] Sure, let's go to the park.
```

#### Speaker List

| Name | Description |
|------|------------|
| Female 1 | Clear female voice |
| Female 2 | Gentle female voice |
| Male 1 | Stable male voice |
| Male 2 | Deep male voice |

#### Emotion List

| Name | Description |
|------|------------|
| Default | Normal intonation |
| Happy | Cheerful intonation |
| Sad | Low intonation |
| Angry | Intense intonation |
| Surprised | Rising intonation |
| Calm | Peaceful intonation |
| Excited | Enthusiastic intonation |
| Gentle | Soft intonation |

#### Quick Parameter Settings

| Parameter | Range | Default | Description |
|----------|-------|---------|------------|
| speed | 0.6-1.8 | 1.0 | Speech speed |
| pitch | -4.0-4.0 | 0.0 | Pitch offset |
| volume | 0.2-2.0 | 1.0 | Volume |
| pause_duration | 0.0-2.0 | 0.5 | Pause duration (seconds) |

#### Common Scenario Configurations

##### Daily Dialogue
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

#### Output Explanation

Node outputs two values:

1. **audio_path**: Generated audio file path
   - Format: WAV
   - Location: audio subdirectory of ComfyUI output directory
   - File name: `{output_name}.wav`

2. **dialogue_info**: Dialogue information dictionary
   - `total_segments`: Total number of dialogue segments
   - `total_duration`: Total duration (seconds)
   - `sample_rate`: Sample rate
   - `segments`: Detailed information for each segment
     - `speaker`: Speaker
     - `emotion`: Emotion
     - `text`: Text content
     - `duration`: Segment duration
   - `pause_duration`: Pause duration
   - `add_silence`: Whether to add silence

#### Example Dialogues

##### Simple Dialogue

```
[Female 1] Hello, how can I help you?
[Male 1] I'd like to learn about your products.
[Female 1] Of course, our products have many advantages.
[Male 1] Can you introduce them in detail?
[Female 1] No problem, let me explain in detail.
```

##### Dialogue with Emotions

```
[Female 1(Happy)] The weather is so nice today!
[Male 1(Calm)] Yes, perfect for a walk.
[Female 1(Excited)] Let's go to the park!
[Male 1(Gentle)] Okay, I'll go with you.
```

##### Multi-character Dialogue

```
[Female 1] Hello everyone, welcome to today's meeting.
[Male 1] Thank you host, I'm happy to participate.
[Female 2] I also look forward to today's discussion.
[Male 2] Let's start, time is limited.
[Female 1] Okay, let's discuss the first topic.
```

#### Quick Checklist

Check before use:
- [ ] TTS model loaded
- [ ] Dialogue text format correct
- [ ] Speaker names correct
- [ ] Emotion names correct
- [ ] Output file name set

#### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Some segments not generated | Check speaker and emotion names |
| Audio not natural | Adjust pause time and speed |
| Format parsing failed | Use bracket format |
| Volume too low | Adjust volume parameter |

## IV. Recommended Configurations for Different Models

### TTS Model Recommended Configurations

#### High-performance TTS Configuration (16GB+ VRAM)
- **Recommended Models**: Qwen3-TTS, Bark
- **Configuration Parameters**:
  ```
  device_mode: GPU
  n_gpu_layers: -1
  sample_rate: 24000
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  ```

#### Balanced TTS Configuration (8-16GB VRAM)
- **Recommended Models**: ChatTTS, XTTS
- **Configuration Parameters**:
  ```
  device_mode: GPU
  n_gpu_layers: 20
  sample_rate: 22050
  temperature: 0.7
  ```

#### Lightweight TTS Configuration (<8GB VRAM)
- **Recommended Models**: VITS, Supertonic
- **Configuration Parameters**:
  ```
  device_mode: CPU
  sample_rate: 22050
  ```

### ASR Model Recommended Configurations

#### High-precision ASR Configuration
- **Recommended Models**: Whisper Large-v3, Qwen-Audio
- **Configuration Parameters**:
  ```
  device_mode: GPU
  n_gpu_layers: -1
  language: auto
  task: transcribe
  ```

#### Fast ASR Configuration
- **Recommended Models**: Whisper Base/Small, FunASR
- **Configuration Parameters**:
  ```
  device_mode: GPU
  n_gpu_layers: 20
  language: zh  # Specifying language improves speed
  task: transcribe
  ```

## V. Performance Optimization Suggestions

### 1. GPU Mode Optimization
- Set appropriate `n_gpu_layers` to ensure model fully loaded to GPU
- Flash Attention automatically enabled based on GPU type (enabled for NVIDIA, disabled for AMD)
- Batch processing parameters (n_batch, n_ubatch, n_threads, n_threads_batch) automatically optimized based on hardware performance

### 2. CPU Mode Optimization
- Thread count automatically set based on CPU cores
- Memory mapping enabled by default to reduce memory usage
- Choose lightweight models to improve inference speed

### 3. VRAM Management
- Monitor VRAM usage to avoid OOM errors
- For large models, consider using CPU mode

### 4. Audio Processing Optimization
- **Sample Rate Selection**:
  - 8000Hz: Phone quality, smallest file size
  - 16000Hz: Standard quality, suitable for general applications
  - 22050Hz: Balanced quality and file size (recommended)
  - 44100Hz: CD quality, suitable for high-quality audio
  - 48000Hz: Professional quality, recommended for TTS reference audio

- **TTS Speed Adjustment**:
  - 0.5-0.7: Very slow, suitable for hearing-impaired users
  - 0.8-0.9: Slow, suitable for learning materials
  - 1.0: Normal speed, suitable for most scenarios
  - 1.1-1.3: Fast, suitable for news broadcasting
  - 1.4-2.0: Very fast, suitable for special effects

### 5. Audio Processing Task Optimization
- **ASR+TTS Combination**: Can be used for speech translation, voice conversion, etc.
- **Multi-speaker Conversation**: Use multi-speaker TTS node for multi-role dialogue synthesis

## VI. Common Issues and Solutions

### 1. TTS Audio Output Failure

**Causes:**
- TTS model does not support specified voice or emotion
- Text length exceeds model limit
- Missing necessary audio processing dependencies

**Solutions:**
- Check supported voice and emotion ranges for TTS model
- Shorten input text length
- Install necessary audio processing libraries

### 2. Poor TTS Voice Cloning Effect

**Causes:**
- Poor reference audio quality
- Incorrect reference audio format
- Inappropriate reference audio duration

**Solutions:**
- Use clear reference audio, avoid noise and background music
- Ensure reference audio is WAV format, mono, 48000Hz
- Control reference audio duration between 3-60 seconds

### 3. Low ASR Recognition Accuracy

**Causes:**
- Poor audio quality
- Language setting mismatch
- Inappropriate model selection

**Solutions:**
- Use clear audio files
- Set correct language parameters
- Choose appropriate ASR model for language and scenario

### 4. Audio Save Failure

**Causes:**
- Incorrect audio data format
- Output path does not exist
- Unsupported format

**Solutions:**
- Ensure audio data contains waveform and sample_rate fields
- Check if output path exists
- Use supported formats (wav/mp3/flac/ogg)

## VII. Audio Time Alignment Related Nodes

### 1. Forced Aligner Model Loader (llama_cpp_forced_aligner_loader)

Forced aligner model loader node is used to load Qwen3-ForcedAligner model, supporting automatic detection and download of missing files.

#### Node Parameters

**Required Parameters:**
- `forced_aligner_model`: Select forced aligner model file, supports Qwen3-ForcedAligner-0.6B model
- `n_gpu_layers`: Number of model layers to load to GPU, -1 means load all to GPU

**Optional Parameters:**
- `precision`: Model precision selection, options: float16/bfloat16/float32, default float16
- `batch_size`: Batch size, default 32, range 1-128
- `flash_attention`: Whether to enable FlashAttention2 optimization, requires flash-attn installation

#### Supported Models
- **Qwen3-ForcedAligner-0.6B**: Alibaba Qwen forced alignment model, supports precise time alignment between Chinese speech and text

#### Model File Requirements
- `model.safetensors`: Model weight file
- `config.json`: Model configuration file
- `tokenizer.json`: Tokenizer configuration file
- `preprocessor_config.json`: Preprocessor configuration file

#### Automatic Download Function
Node automatically checks required files and downloads missing ones from ModelScope or Hugging Face based on network connection.

### 2. Forced Aligner Inference Node (llama_cpp_forced_aligner_inference)

Forced aligner inference node inputs audio and text into ForcedAligner, outputs fine-grained timestamps (phoneme/word level).

#### Node Parameters

**Required Parameters:**
- `aligner_model`: Connect output of forced aligner model loader
- `audio`: Input audio data
- `text`: Text content to align

**Optional Parameters:**
- `sample_rate`: Audio sample rate, default 16000Hz, range 8000-48000Hz
- `output_format`: Output format, options: Text/SRT/VTT/JSON, default Text

#### Output Explanation

Node outputs three values:
1. **aligned_text**: Aligned text
2. **timestamps**: Timestamp data (JSON format), contains paragraph, word, and phoneme level timing information
3. **subtitle_text**: Subtitle format text (SRT or VTT format)

#### Timestamp Data Structure
```json
{
"task_id": "Task ID",
"audio_info": {
    "audio_duration": Audio duration(seconds),
    "audio_sample_rate": Sample rate
},
"segments": [
    {
    "start": Start time(seconds),
    "end": End time(seconds),
    "text": "Paragraph text",
    "words": [
        {
        "word": "Word",
        "start": Start time(seconds),
        "end": End time(seconds),
        "phonemes": [
            {
            "phoneme": "Phoneme",
            "start": Start time(seconds),
            "end": End time(seconds)
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

### 3. TTS Alignment Node (tts_align)

TTS alignment node uses timestamp data output from forced aligner inference node to generate voiceover according to text timeline, supporting two modes: exact timing alignment and natural flow alignment.

#### Node Parameters

**Required Parameters:**
- `tts_model`: TTS model
- `timestamps`: Timestamp data, can be obtained from forced aligner inference node or imported from JSON format file
- `text`: Text content to synthesize
- `sample_rate`: Audio sample rate, default 24000Hz, range 8000-48000Hz

**Optional Parameters:**
- `speed`: TTS speed, default 1.0, range 0.5-2.0
- `align_mode`: Alignment mode, options: Exact Timing/Natural Flow, default Natural Flow
- `silence_padding`: Silence padding after segments (seconds), default 0.1, range 0.0-1.0

#### Alignment Mode Explanation
- **Exact Timing**: Strictly generate audio according to timestamps, ensuring each segment duration matches timestamps exactly
- **Natural Flow**: Maintain natural speech rhythm, no forced adjustment of audio duration

#### Output Explanation
Node outputs two values:
1. **aligned_audio**: Aligned audio
2. **alignment_info**: Alignment information, containing alignment status and duration information for each segment



### 5. Role Configuration Node (RoleConfig)

Role configuration node is used to configure role parameters, including name, voice, speed, pitch, volume, etc., supporting local TTS models.

#### Node Parameters

**Required Parameters:**
- `role_name`: Role name

**Local Model Parameters:**
- `voice`: Local model voice, options:
  - Vivian - Bright young female voice with slight edge (Chinese)
  - Serena - Warm and gentle young female voice (Chinese)
  - Uncle_Fu - Deep and rich mature male voice (Chinese)
  - Dylan - Clear and natural Beijing young male voice (Chinese Beijing dialect)
- `emotion`: Local model emotion, options: default/happy/sad/angry/surprised/calm/excited/gentle, default default

**General Parameters:**
- `speed`: Speech speed, default 1.0, range 0.5-2.0
- `pitch`: Pitch offset, default 0.0, range -5.0-5.0
- `volume`: Volume, default 1.0, range 0.1-2.0

#### Output Explanation
Node outputs one value:
- **role_config**: Role configuration dictionary, contains all configuration parameters

## VIII. Usage Flow Examples

### Example 1: TTS Speech Synthesis

1. **Load Models**:
   - Use `llama_cpp_model_loader` node to load LLM model
   - Use `llama_cpp_tts_loader` node to load TTS model (e.g., Qwen3-TTS)

2. **Configure Unified Inference Node**:
   - Use `llama_cpp_unified_inference` node
   - Select inference mode: `[Basic] Text to Audio`
   - Connect LLM model to `Llama模型` input
   - Connect TTS model to `tts_model` input
   - Set TTS voice (tts_voice), emotion (tts_emotion), and speed (tts_speed)

3. **Provide Text Input**:
   - Input text to synthesize in `文本输入`
   - Select appropriate preset template

4. **Save Audio**:
   - Use `llama_cpp_audio_saver` audio save node
   - Connect `audio` output of inference node to audio save node
   - Select save format (recommended WAV)

### Example 2: ASR Speech Recognition

1. **Load Models**:
   - Use `llama_cpp_model_loader` node to load LLM model
   - Use `llama_cpp_asr_loader` node to load ASR model (e.g., Qwen3-ASR)

2. **Configure Unified Inference Node**:
   - Use `llama_cpp_unified_inference` node
   - Select inference mode: `[Basic] Audio to Text`
   - Connect LLM model to `Llama模型` input
   - Connect ASR model to `asr_model` input
   - Set ASR language (asr_language)

3. **Provide Audio Input**:
   - Connect audio input to `audio` port

4. **Run Recognition**:
   - Get recognition text output

### Example 3: Multi-speaker Conversation Synthesis

1. **Load TTS Model**:
   - Use `llama_cpp_tts_loader` node to load TTS model (e.g., Qwen3-TTS)

2. **Configure Multi-speaker TTS Node**:
   - Use `multi_speaker_tts` node
   - Connect TTS model to `tts_model` input
   - Input multi-speaker dialogue text in `dialogue_text`
   - Set `output_name` output file name

3. **Set Dialogue Parameters**:
   - Set `default_speaker` default speaker
   - Set `default_emotion` default emotion
   - Adjust `speed` parameter
   - Set `pause_duration` pause time between speakers

4. **Generate Dialogue Audio**:
   - Run workflow
   - Audio file will be saved in audio subdirectory of ComfyUI output directory

## IX. Summary

This document provides detailed parameter explanations and recommended settings for ASR models, TTS models, and multi-speaker conversation nodes in the ComfyUI-omni-llm plugin, distinguishing between node-displayed options and background-adjusted parameters.

### Key Points

1. **ASR Models**:
   - Support multiple ASR architectures (Whisper, Qwen-Audio, etc.)
   - Choose appropriate language and task type
   - Audio quality directly affects recognition accuracy

2. **TTS Models**:
   - Support multiple TTS architectures (Qwen3-TTS, Bark, XTTS, etc.)
   - Note sample rate settings and reference audio format
   - Voice cloning requires high-quality reference audio

3. **Multi-speaker Conversation**:
   - Support multi-role dialogue synthesis
   - Support different voices and emotion expressions
   - Support multiple dialogue text formats
   - Adjustable speed, pitch, volume, and pause time

### Usage Recommendations

1. Start from default configuration and adjust gradually based on actual results
2. Prioritize hardware performance, choose suitable models and parameters
3. Monitor resource usage to avoid resource exhaustion
4. Choose appropriate model and parameter combinations based on task type
5. For complex tasks, consider using higher-performance hardware or more suitable models