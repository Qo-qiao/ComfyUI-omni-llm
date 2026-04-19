# Multi-Image Input Node User Guide

## Node Overview

The **Multi-Image Input (Story Creation)** node supports two working modes:

1. **Image Mode**: Analyzes multiple images and creates story content, supporting various video generation models (WAN2.2, LTX2, etc.)
2. **Text Mode**: Generates prompts through option settings without requiring image input

## Node Features

### Main Features
1. **Dual Mode Support**: Flexible switching between Image Mode and Text Mode
2. **Multi-Image Input**: Supports input of multiple images (Image Mode)
3. **Automatic Preprocessing**: Automatically scales and encodes images
4. **Story Creation**: Generates story content suitable for video generation
5. **Flexible Configuration**: Supports multiple story types, lengths, and styles
6. **Multiple Applications**: Supports story creation, script writing, advertising copy, and other content types
7. **Image Output**: Passes image data to inference nodes

### Input Parameters

#### Working Mode
- **mode** (Dropdown): Working mode
  - Image Mode: Analyzes image content for story creation
  - Text Mode: Generates prompts through option settings

#### Image Input (Image Mode Only)
- **image1** (IMAGE): First input image
- **image2** (IMAGE): Second input image
- **image3** (IMAGE): Third input image
- **image4** (IMAGE): Fourth input image
- **image5** (IMAGE): Fifth input image
- **image6** (IMAGE): Sixth input image
- **image7** (IMAGE): Seventh input image
- **image8** (IMAGE): Eighth input image
- **image9** (IMAGE): Ninth input image
- **image10** (IMAGE): Tenth input image
- **image11** (IMAGE): Eleventh input image
- **image12** (IMAGE): Twelfth input image
  - Can connect only one or multiple images
  - At least one image must be provided (Image Mode)
  - Automatically detects image count
  - Automatically preprocesses and encodes

#### Configuration Parameters
- **story_type** (Dropdown): Content creation type
  - Coherent Story: Creates a coherent and complete story
  - Storyboard Description: Describes each scene in chronological order
  - Scene Analysis: Deeply analyzes details of each scene
  - Character Development: Describes character growth and changes
  - Emotional Progression: Shows emotional changes and development
  - Creative Writing: Engages in creative writing, showcasing unique ideas and creativity
  - Script Writing: Creates a complete script with dialogue and scene descriptions
  - Advertising Copy: Creates engaging advertising copy
  - Product Introduction: Writes detailed product introductions
  - Educational Content: Creates educational content that is easy to understand and learn

- **story_length** (Dropdown): Content length
  - Short (200 words or less): Quick, concise content
  - Medium (400 words or less): Balanced content length
  - Detailed (600 words or less): Rich content
  - Complete (1000 words or less): Complete content narrative

- **language** (Dropdown): Output language
  - 中文: Generates Chinese content
  - English: Generates English content

- **max_size** (Integer): Maximum image size (pixels)
  - Range: 128-512
  - Default: 256
  - Used to control image scaling (Image Mode only)

- **custom_prompt** (Text): Custom prompt
  - Used to guide content creation
  - Can add specific requirements or styles
  - Multi-line text input

- **include_image_descriptions** (Boolean): Whether to include image descriptions
  - True: Include descriptions of each image before the story
  - False: Create story directly without image descriptions
  - Valid only for Image Mode

- **story_theme** (Dropdown): Content theme
  - No Specific Theme: No theme restrictions
  - Adventure Story: Adventure theme, full of challenges and exploration
  - Romance Story: Romance theme, showcasing emotions and relationships
  - Mystery Story: Mystery theme, setting suspense and puzzles
  - Sci-Fi Story: Science fiction theme, showcasing future technology
  - Fantasy Story: Fantasy theme, containing magic and supernatural elements
  - Daily Life: Daily life theme, showcasing beauty in the ordinary
  - Historical Story: Historical theme, showcasing past cultures and eras
  - Future Technology: Future technology theme, showcasing advanced technology and human development
  - Business Marketing: Business marketing theme, highlighting product or service advantages
  - Educational Popularization: Educational popularization theme, spreading knowledge and information
  - Entertainment Comedy: Entertainment comedy theme, bringing joy and relaxation

- **narrative_style** (Dropdown): Narrative style
  - First Person: Uses first-person perspective (I)
  - Third Person: Uses third-person perspective (he/she)
  - Omniscient Perspective: Uses omniscient perspective, knowing all characters' thoughts
  - Multi-Perspective Switching: Switches perspectives between different characters

- **content_focus** (Dropdown): Content focus
  - Balanced Development: Balanced development of plot, characters, emotions, and other aspects
  - Emphasize Plot: Highlights plot development and twists
  - Emphasize Characters: Focuses on character personality and growth
  - Emphasize Emotions: Focuses on emotional changes and expressions
  - Emphasize Visuals: Focuses on visual elements and scene details
  - Emphasize Dialogue: Advances story through dialogue

- **target_audience** (Dropdown): Target audience
  - General Public: Suitable for general public reading and understanding
  - Teenagers: Suitable for teenagers, positive and uplifting content
  - Children: Suitable for children, simple and easy to understand language
  - Professionals: Suitable for professionals, professional and in-depth content
  - Specific Groups: Suitable for specific groups, highly targeted

- **video_model** (Dropdown): Video generation model type
  - WAN2.2: WAN2.2 video generation model, emphasizes scene descriptions and visual elements, uses concise and powerful language
  - LTX2: LTX2 video generation model, focuses on detail descriptions and emotional expressions, highlights key scene transitions
  - General Video: General video generation model, balances scene descriptions and narrative flow
  - Custom: Custom video generation model, adjust according to specific model requirements

### Output Parameters

- **prompt** (STRING): Generated content creation prompt
  - Image Mode: Contains image analysis requirements
  - Text Mode: Contains detailed creation guidance
  - Suitable for direct connection to llama_cpp_instruct_adv node's custom_prompt

- **images** (IMAGE): Image data
  - Image Mode: Returns preprocessed image data
  - Text Mode: Returns None
  - Suitable for direct connection to llama_cpp_instruct_adv node's images input

## Usage

### Image Mode Usage

#### Basic Workflow

```
1. Prepare images (1-12)
   ↓
2. Multi-Image Input node (Image Mode)
   ↓
3. Llama-cpp Image Inference node
   ↓
4. Get story content
   ↓
5. For video generation (WAN2.2, LTX2, etc.)
```

#### Detailed Steps

1. **Prepare Images**:
   - Prepare 1-12 images (at least 1)
   - Ensure image content is coherent and has story potential
   - Images can be continuous scenes or fragments of different scenes

2. **Configure Multi-Image Input Node**:
   - Select `mode` as "Image Mode"
   - Connect image inputs to `image1` to `image12` ports (at least one connection required)
   - Select `story_type` (recommend "Coherent Story" or "Storyboard Description")
   - Select `story_length` (usually 400 words or less)
   - Select `language` (Chinese or English)
   - Select `video_model` (WAN2.2/LTX2/General Video/Custom)
   - Adjust other parameters (theme, narrative style, etc.)

3. **Connect to Llama-cpp Image Inference Node**:
   - Connect `prompt` output to `custom_prompt` input
   - Connect `images` output to `images` input
   - In Llama-cpp Image Inference node:
     - `inference_mode` select "images"
     - `preset_prompt` select "[Video Analysis] Video - Detailed Scene Breakdown"
     - `output_language` select the same language as Multi-Image Input

4. **Get Story Content**:
   - Run the workflow
   - Wait for model processing
   - Get the generated story content

5. **For Video Generation**:
   - Copy story content to the prompt input of the corresponding video generation model (WAN2.2, LTX2, etc.) based on the selected `video_model` type
   - Adjust video parameters based on story content
   - Generate video

### Text Mode Usage

#### Detailed Steps

1. **Configure Multi-Image Input Node**:
   - Select `mode` as "Text Mode"
   - No need to connect any images
   - Select `story_type` (according to needs)
   - Select `story_length` (according to content length needs)
   - Select `language` (Chinese or English)
   - Configure other parameters (theme, narrative style, content focus, target audience)
   - Optional: Add `custom_prompt` for custom requirements

2. **Connect to Llama-cpp Image Inference Node**:
   - Connect `prompt` output to `custom_prompt` input
   - No need to connect `images` output (Text Mode returns None)
   - In Llama-cpp Image Inference node:
     - `inference_mode` select "text"
     - `preset_prompt` select "[Video Analysis] Video - Detailed Scene Breakdown" or other appropriate preset
     - `output_language` select the same language as Multi-Image Input

3. **Get Created Content**:
   - Run the workflow
   - Wait for model processing
   - Get the generated content

4. **Use Generated Content**:
   - For video generation (WAN2.2, LTX2, etc.)
   - For other text generation applications
   - For creative writing or content creation

## Usage Examples

### Image Mode Examples

#### Example 1: Coherent Story Creation

**Scenario**: Create a story about travel adventure

**Images**: 4 continuous travel scene images

**Configuration**:
- mode: Image Mode
- story_type: Coherent Story
- story_length: Medium (400 words or less)
- language: 中文
- story_theme: Adventure Story
- narrative_style: First Person

**Expected Output**:
```
Image 1: A young person at the airport preparing to depart
Image 2: Airplane flying in the clouds
Image 3: Arriving at a foreign airport
Image 4: Enjoying night view from hotel room

Generated story:
I stood at the airport gate, holding my passport and ticket tightly. This was the beginning of the adventure I had always dreamed of. The plane slowly ascended, passing through layers of clouds, and my heart raced. A few hours later, the plane landed on foreign soil. Walking out of the airport, the unfamiliar language and architecture made me feel both excited and nervous. In my hotel room, I stood by the window, watching the city's night view, with neon lights flashing and cars flowing through the streets. I knew this was just the first day of my adventure, and there were countless stories waiting for me to discover.
```

#### Example 2: Storyboard Description

**Scenario**: Create storyboard descriptions for video

**Images**: 6 different scene images

**Configuration**:
- mode: Image Mode
- story_type: Storyboard Description
- story_length: Detailed (600 words or less)
- language: 中文
- story_theme: No Specific Theme
- narrative_style: Third Person

**Expected Output (WAN2.2 - 3-4 shots, 3-5 seconds each)**:
```
Image 1: Golden beach at sunrise
Image 2: Waves hitting the shore
Image 3: Seagulls soaring in the sky
Image 4: Footprints on the beach

Generated storyboard description:
Shot 1 (0:00-0:04): Golden beach at sunrise, waves gently lapping the shore. The sun has just risen, the sea surface glows with orange-red light, and footprints extend into the distance on the beach.

Shot 2 (0:04-0:08): Waves hitting the shore, water splashing everywhere. Waves rhythmically surge up and retreat, leaving white foam on the beach.

Shot 3 (0:08-0:12): Seagulls soaring in the sky, making crisp calls. They circle in the air, sometimes diving toward the sea surface, sometimes flying high.

Shot 4 (0:12-0:16): Footprints on the beach, recording someone's tracks. Footprints extend from the seaside into the distance, seemingly telling a story.
```

**Expected Output (LTX2 - 5-6 shots, 5-10 seconds each)**:
```
Image 1: Golden beach at sunrise
Image 2: Waves hitting the shore
Image 3: Seagulls soaring in the sky
Image 4: Footprints on the beach
Image 5: Distant horizon
Image 6: Sunset

Generated storyboard description:
Shot 1 (0:00-0:08): Golden beach at sunrise, waves gently lapping the shore. The sun has just risen, the sea surface glows with orange-red light, and footprints extend into the distance on the beach.

Shot 2 (0:08-0:16): Waves hitting the shore, water splashing everywhere. Waves rhythmically surge up and retreat, leaving white foam on the beach.

Shot 3 (0:16-0:24): Seagulls soaring in the sky, making crisp calls. They circle in the air, sometimes diving toward the sea surface, sometimes flying high.

Shot 4 (0:24-0:32): Footprints on the beach, recording someone's tracks. Footprints extend from the seaside into the distance, seemingly telling a story.

Shot 5 (0:32-0:40): Distant horizon, where sky and sea merge. The horizon is clearly visible, sky and ocean merge into one, showing a broad view.

Shot 6 (0:40-0:48): Sunset, the sky dyed orange-red. The sun slowly sets, the sea surface reflects golden light, the entire scene is full of warmth and tranquility.
```

### Text Mode Examples

#### Example 3: Creative Writing (No Images)

**Scenario**: Create a sci-fi short story

**Configuration**:
- mode: Text Mode
- story_type: Creative Writing
- story_length: Medium (400 words or less)
- language: 中文
- story_theme: Sci-Fi Story
- narrative_style: First Person
- content_focus: Emphasize Plot
- target_audience: Teenagers
- custom_prompt: The story takes place in 2050, the protagonist is a young scientist

**Expected Output**:
```
In the early morning of 2050, I stood in front of the floor-to-ceiling window of the laboratory, watching the traffic suspended in the sky outside. As the youngest scientist at the Quantum Research Institute, I was about to complete an invention that would change human history.

"Amy, is the system ready?" I turned to ask my AI assistant.

"Yes, Doctor. The quantum teleportation device is ready, awaiting your instructions."

I took a deep breath and pressed the start button. Blue ripples appeared in the air, and a light ball slowly formed in the center of the device. This was the first attempt at quantum teleportation in human history.

Suddenly, an alarm sounded. "Warning! Energy fluctuation anomaly!"

I rushed to the console, fingers flying across the keyboard. "Stabilize the energy field!"

The light ball began to become unstable, and the surrounding equipment started to vibrate. I had to make a choice: continue the experiment or stop immediately?

"Doctor, safety protocol recommends immediate termination." Amy's voice was calm and firm.

"No, we've come this far." I gritted my teeth and adjusted the parameters.

The light ball stabilized, and a flash of light passed. When I opened my eyes again, I found myself standing in another laboratory, with a 2050 calendar hanging on the wall.

Success! I made history.
```

#### Example 4: Script Writing (No Images)

**Scenario**: Create a short advertising script

**Configuration**:
- mode: Text Mode
- story_type: Script Writing
- story_length: Short (200 words or less)
- language: 中文
- story_theme: Business Marketing
- narrative_style: Third Person
- content_focus: Emphasize Dialogue
- target_audience: General Public
- custom_prompt: The product is a smart watch

**Expected Output**:
```
[Scene: Modern office, bright]

(Camera: Young office worker Xiao Wang is busy at his desk, phone rings)

Xiao Wang: (Looking at phone) Oh no, I'm going to be late again!

(Camera: Xiao Wang hurriedly packs up, preparing to leave)

Voiceover: (Gentle female voice) Still worried about missing important moments?

(Camera: Smart watch on Xiao Wang's wrist lights up, showing schedule reminder)

Smart Watch: Xiao Wang, you have 15 minutes to reach the meeting room.

Xiao Wang: (Relieved) Good thing I have you.

(Camera: Xiao Wang confidently walks into the meeting room, watch shows steady heart rate)

Voiceover: Smart watch, making life more从容.

(Camera: Product close-up, showing various functions)

Voiceover: Health monitoring, schedule management, smart reminders, your personal assistant.

(Camera: Xiao Wang smiles looking at watch)

Xiao Wang: This is the life I want.

(Camera: Brand logo and slogan)

Voiceover: Smart watch, control every moment.
```

## Best Practices

### Mode Selection Suggestions
1. **Image Mode**:
   - Have specific image materials that need analysis
   - Need to create stories based on image content
   - Image content is coherent and has story potential

2. **Text Mode**:
   - No image materials, need to create from scratch
   - Need to quickly generate content
   - Need flexible control over content type and style

### Image Selection Suggestions (Image Mode Only)
1. **Coherence**: Choose coherent images to help create better stories
2. **Appropriate Quantity**: Recommend 1-12 images, choose appropriate quantity based on needs
3. **Quality First**: Use high-quality images to help model understanding
4. **Diverse Scenes**: Include different scenes and angles to enrich story content

### Parameter Setting Suggestions
1. **Working Mode**:
   - Have image materials: Select Image Mode
   - No image materials: Select Text Mode

2. **Content Type**:
   - WAN2.2 video generation: Recommend "Coherent Story" or "Storyboard Description"
   - LTX2 video generation: Recommend "Character Development" or "Emotional Progression"
   - General video generation: Recommend "Coherent Story" or "Scene Analysis"
   - Creative writing: Recommend "Character Development" or "Emotional Progression"
   - Business applications: Recommend "Advertising Copy" or "Product Introduction"
   - Educational applications: Recommend "Educational Content"

3. **Content Length**:
   - Short video (10-30 seconds): 200 words or less
   - Medium video (30-60 seconds): 400 words or less
   - Long video (60-120 seconds): 600 words or less
   - Short content: 200 words or less
   - Medium content: 400 words or less
   - Long content: 600 words or less

4. **Narrative Style**:
   - First Person: Suitable for personalized stories
   - Third Person: Suitable for objective descriptions
   - Omniscient Perspective: Suitable for complex stories
   - Multi-Perspective Switching: Suitable for multi-character stories

5. **Content Focus**:
   - Balanced Development: Suitable for most scenarios
   - Emphasize Plot: Suitable for fast-paced content
   - Emphasize Characters: Suitable for character-driven stories
   - Emphasize Emotions: Suitable for emotional content
   - Emphasize Visuals: Suitable for video generation
   - Emphasize Dialogue: Suitable for script writing

6. **Target Audience**:
   - General Public: Suitable for widespread dissemination
   - Teenagers: Suitable for young groups
   - Children: Suitable for educational content
   - Professionals: Suitable for professional content
   - Specific Groups: Suitable for targeted content

### Theme Selection Suggestions
1. **Adventure Story**: Suitable for travel, exploration, challenge themes
2. **Romance Story**: Suitable for love, emotion, relationship themes
3. **Mystery Story**: Suitable for puzzles, solving mysteries, tension themes
4. **Sci-Fi Story**: Suitable for future, technology, space themes
5. **Fantasy Story**: Suitable for magic, supernatural, fantasy themes
6. **Business Marketing**: Suitable for product promotion, brand publicity
7. **Educational Popularization**: Suitable for knowledge dissemination, skill teaching
8. **Entertainment Comedy**: Suitable for relaxed, humorous content

## FAQ

### Q1: What's the difference between Image Mode and Text Mode?
**A**:
- **Image Mode**: Requires providing images, the model analyzes image content and creates stories based on images
- **Text Mode**: No images required, generates prompts through option settings, the model creates content based on prompts
- **Selection Recommendation**: Use Image Mode if you have image materials, use Text Mode if you don't

### Q2: Why is the model's output story not coherent enough?
**A**: Possible reasons:
1. Image content is not coherent enough (Image Mode)
2. Story type selection is inappropriate
3. Custom prompt is not clear enough
4. Parameter settings are unreasonable

**Solutions**:
1. Choose more coherent images (Image Mode)
2. Try different story types
3. Add more specific custom prompts
4. Adjust parameter settings (content focus, narrative style, etc.)

### Q3: How to improve story quality?
**A**: Suggestions:
1. **Image Mode**:
   - Use high-quality images
   - Choose appropriate story length
   - Add specific custom prompts
   - Choose suitable story theme and narrative style

2. **Text Mode**:
   - Set parameters in detail (type, length, theme, style, etc.)
   - Add specific custom prompts
   - Choose suitable content focus and target audience
   - Configure narrative style reasonably

### Q4: What scenarios can Text Mode be used for?
**A**: Text mode is suitable for:
1. Creative Writing: Novels, short stories, poetry
2. Script Writing: Movie scripts, advertising scripts, short video scripts
3. Business Copy: Advertising copy, product introductions, marketing copy
4. Educational Content: Teaching materials, popular science articles, knowledge explanations
5. Content Generation: Social media content, blog posts, press releases

### Q5: Video generation quality is poor?
**A**: Possible reasons:
1. Story content is not specific enough
2. Story length is inappropriate
3. Language mismatch
4. Content focus selection is inappropriate
5. Video model selection is inappropriate

**Solutions**:
1. Optimize story content to make it more specific
2. Adjust story length to match video length
3. Ensure language matches video content
4. Select "Emphasize Visuals" content focus
5. Use Image Mode, provide visual reference
6. Select appropriate `video_model` parameter based on target video generation model:
   - WAN2.2: Emphasizes scene descriptions and visual elements, uses concise and powerful language
   - LTX2: Focuses on detail descriptions and emotional expressions, highlights key scene transitions
   - General Video: Balances scene descriptions and narrative flow
   - Custom: Adjust according to specific model requirements

### Q6: How to correctly connect image data to inference node?
**A**:
1. **Image Mode**:
   - Connect `prompt` output to `custom_prompt` input
   - Connect `images` output to `images` input
   - Ensure both connections are established correctly

2. **Text Mode**:
   - Connect `prompt` output to `custom_prompt` input
   - No need to connect `images` output (Text Mode returns None)
   - Only need to connect prompt output

## Workflow Examples

### Complete Workflow Example

```
[Load Image] → [image1]         →                  ↓
[Load Image] → [image2] → [Multi-Image Input] → [Llama-cpp Image Inference] → [Display Text]
[Load Image] → [image3]         →                  ↓                        ↓
           1-12 images                     Generate prompts       Get story          Display results
```

### Integration with Video Generation Model Example

```
[Load Image] → [image1]         →                  ↓                  ↓
[Load Image] → [image2] → [Multi-Image Input] → [Llama-cpp Image Inference] → [Copy Text] → [Video Generation Model]
[Load Image] → [image3]         →                  ↓                  ↓              ↓
           1-12 images                     Generate prompts       Get story      Copy to clipboard    Generate video
```

**Note**:
- [Video Generation Model] can be WAN2.2, LTX2, or other video generation models
- Use the corresponding video generation model based on the `video_model` parameter selected in the Multi-Image Input node

## Technical Notes

### Image Preprocessing
- Automatically scales to specified size
- Encodes to base64 format
- Maintains image quality
- Supports multiple image formats

### Prompt Generation
- Generates structured prompts based on parameters
- Supports multi-language output
- Customizable prompt templates
- Optimized for different models

### Output Format
- Formatted prompt text
- Image data (Image Mode)
- Suitable for direct connection to inference nodes

## Summary

**Key Advantages**:
- ✅ Supports multi-image input
- ✅ Automatic preprocessing and encoding
- ✅ Flexible configuration
- ✅ Maintains compatibility
- ✅ Easy to use
- ✅ Image data transmission

**Applicable Scenarios**:
- WAN2.2 video generation
- LTX2 video generation
- Other video generation models
- Story creation
- Scene analysis
- Multi-image content understanding
- Creative writing
- Script writing
- Business copywriting
- Educational content creation

---

**Version**: 1.0
**Last Updated**: 2025
**For more information, please refer to the project documentation or contact the development team.**
