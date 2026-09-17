---
name: Image Prompt Reverse
description: AI image prompt reverse engineering tool. Analyze uploaded images and generate precise descriptive prompts, supporting output for 19+ mainstream AI drawing models (SD1.5, SDXL, Flux1, Flux2-Klein, Midjourney, Pony Diffusion, GPT-Image2, HunyuanImage_2.1, etc.), with positive quality prompts and negative prompt recommendations. Triggered when users need to "reverse image prompts", "analyze image prompts", "image to prompts", "extract image descriptions", "generate AI drawing prompts".
tag-cn: 反推, 图像, 提示词
---

# Image Prompt Reverse

## Overview

This skill analyzes user-uploaded images and generates precise, comprehensive AI drawing prompts. It uses a systematic analysis method with emphasis on accurate gender identification, supporting 19+ mainstream AI drawing model format outputs, helping users reproduce similar images on different AI drawing platforms.

## Core Capabilities

1. **Deep Image Analysis** - Systematically extract visual elements using layered checklists
2. **Precise Gender Identification** - Multi-verification mechanism to avoid misidentification
3. **Scene-Specific Recognition** - Specialized analysis for portraits, anime, landscapes, cities, still life, fantasy, and more
4. **Multi-Model Format Output** - Adapt prompts for 19+ mainstream AI drawing models
5. **Intelligent Prompt Generation** - Auto-generate optimal prompt structures based on scene type
6. **Character Recognition** - Identify common anime characters and celebrities
7. **Quality Word Recommendation** - Auto-recommend positive quality words and negative prompts

## Supported AI Drawing Models

| Model | Features | Use Cases |
|-------|----------|-----------|
| SD1.5 | Weight syntax, highly controllable | Fine-tuning, professional creation |
| SDXL | Improved understanding, native high-res | High-quality realistic, rich details |
| Anima | Anime style optimization | Anime, illustration creation |
| Flux1 | Natural language + style words | Realistic portraits, high-quality output |
| Krea2 | Real-time generation, style mixing | Fast iteration, style exploration |
| ZImage | Multi-style fusion, detail preservation | High-quality images, style fusion |
| Qwen-Image-2512 | Multi-modal understanding, bilingual | Complex scenes, Chinese/English |
| Mage-Flow | Streaming generation, real-time preview | Fast prototyping, style adjustment |
| HunyuanImage_2.1 | Chinese optimization, Guofeng support | Chinese style, Eastern aesthetics |
| HiDream-O1-Image | High-quality generation, rich details | High-quality creation, detail preservation |
| ERNIE-Image | Chinese understanding, knowledge enhanced | Chinese scenes, knowledge enhanced |
| Boogu-Image | Stylized generation, strong colors | Artistic creation, concept design |
| LongCat-Image | High resolution, long text | High-quality images, complex scenes |
| Flux2-Klein | Enhanced understanding, detail preservation | High-quality creation, complex scenes |
| GLM-Image | Chinese optimization, knowledge enhanced | Chinese scenes, diverse styles |
| GPT-Image2 | Conversational generation, multi-modal | Commercial design, concept iteration |
| Nanobanana | Lightweight fast, diverse styles | Fast prototyping, style exploration |
| Pony Diffusion | Tag format, anime optimization | Anime characters, anime creation |
| Midjourney | Natural language + parameters | Artistic style, concept design |

## Workflow

### Step 1: Receive Image

Confirm the user has uploaded an image. If not, prompt the user to provide one.

### Step 2: Scene Type Identification

First identify the image's scene type to determine analysis focus:

**Scene Types**:
- **Portrait** - Person >50%, face clear
- **Environmental Portrait** - Person 20-50%, environment visible
- **Anime/2D** - Clear lines, large eyes, simplified features
- **Landscape/Nature** - Natural scenery dominant, person <20%
- **Urban/Architecture** - Building structures, geometric lines
- **Still Life/Product** - Single object, professional lighting
- **Fantasy/Sci-Fi** - Surreal elements, CG rendering feel
- **Poster Design** - Graphic design, text+image combination
- **Book Cover** - Book binding, typography+illustration
- **Movie Poster** - Film promotion, dramatic composition
- **Game Art** - Game aesthetics, character/scene concept art
- **Product Photography** - Commercial products, professional lighting
- **Food Photography** - Cuisine shooting, plating art
- **Fashion Photography** - Clothing/jewelry, magazine style
- **UI/UX Design** - Interface design, APP/web screenshots
- **Album Cover** - Music album, visual art
- **Manga/Illustration** - Comic panels, picture book illustrations
- **Ink/Chinese Painting** - Traditional Eastern painting style
- **Oil/Watercolor** - Traditional painting techniques
- **3D Rendering** - CG rendering, modeling works
- **Pixel Art** - Retro pixel style

Refer to [scene-recognition.md](references/scene-recognition.md) for specialized analysis.

### Step 2.5: Character Recognition (For Images with People)

When the image contains people, attempt to identify known anime characters or celebrities:

**Anime Character Recognition**:
- Extract character features: hairstyle, hair color, eyes, clothing, accessories, signature elements
- Match known characters through feature combinations
- Refer to [character-recognition.md](references/character-recognition.md)

**Celebrity Recognition**:
- Extract facial features: face shape, facial proportions, signature facial features
- Combine hairstyle, body type, and temperament for judgment
- Refer to [celebrity-recognition.md](references/celebrity-recognition.md)

**Recognition Confidence**:

| Confidence | Output Method |
|------------|---------------|
| High | Directly state character/celebrity name |
| Medium | "Suspected to be XX" |
| Low | Do not guess, describe visible features in detail |

### Step 3: Deep Image Analysis

**Pre-analysis Preparation**:
1. Assess image quality (clarity, lighting, obstruction)
2. Determine image type (portrait/landscape/still life/anime etc.)

**Five-Layer Analysis Method**:

```
Layer 1: Overall Impression - Style, atmosphere, emotional tone
Layer 2: Subject Content - Person/object features, quantity, position
Layer 3: Environment Background - Scene, location, background elements
Layer 4: Technical Details - Lighting, perspective, composition, depth of field
Layer 5: Quality Features - Resolution, detail level, image texture
```

**Gender Identification Notes**:

Gender identification is a common error point. Follow these principles:

1. **Multi-verification Principle** - Do not rely on single features, cross-verify with at least 2-3 features
2. **Clothing Priority Principle** - Clothing style is usually the most obvious gender indicator
3. **Uncertainty Labeling** - Use neutral descriptions or clearly mark uncertainty when uncertain

**Gender Identification Priority**:
```
High Priority (Decisive Features):
- Clothing style (dress/suit etc.) - Certainty: Very High
- Facial hair (beard) - Certainty: Very High
- Accessories (tie/necklace etc.) - Certainty: High

Medium Priority (Auxiliary Features):
- Hairstyle length and style - Certainty: Medium
- Facial contours (realistic style) - Certainty: Medium

Low Priority (Reference Features):
- Color preferences
- Decorative elements
```

Refer to [analysis-method.md](references/analysis-method.md) and [gender-identification.md](references/gender-identification.md) for detailed analysis methods.

**Use Checklist for Completeness**:

Must refer to [precision-checklist.md](references/precision-checklist.md) for item-by-item checking:

- [ ] Layer 1: Overall impression (atmosphere, tone, style, emotion)
- [ ] Layer 2: Subject content (detailed features of person/object/scene)
- [ ] Layer 3: Environment background (space, time, weather, background elements)
- [ ] Layer 4: Technical details (lighting, color, perspective, depth of field)
- [ ] Layer 5: Quality features (detail level, artistic processing)

### Step 4: Verify Analysis Results

**Cross-verification Rules**:

- **Gender Verification**: Must satisfy Clothing features + (Hairstyle features OR Facial features), or clearly gendered clothing
- **Style Verification**: Must satisfy at least 2 consistent style features
- **Lighting Verification**: Shadow direction + Highlight position + Light source direction must be consistent

**Self-check Questions**:
1. Does my determined gender have at least 2 supporting features?
2. Can all described features be seen in the image?
3. Is the lighting description consistent with shadows?
4. Have I covered all five layers of analysis?

### Step 5: Generate Prompts

**Select Template Based on Scene Type**:

Use the corresponding template from [prompt-templates.md](references/prompt-templates.md):

- **Portrait** - Use portrait photography template
- **Anime Character** - Use anime character template (with character recognition results)
- **Celebrity Portrait** - Use celebrity portrait template (with recognition results)
- **Landscape Photography** - Use landscape photography template
- **Urban Architecture** - Use urban architecture template
- **Still Life Product** - Use still life product template
- **Fantasy/Sci-Fi** - Use fantasy/sci-fi template
- **Poster Design** - Use graphic design template
- **Book Cover** - Use book cover template
- **Movie Poster** - Use movie poster template
- **Game Art** - Use game art template
- **Product Photography** - Use product photography template
- **Food Photography** - Use food photography template
- **Fashion Photography** - Use fashion photography template
- **UI/UX Design** - Use interface design template
- **Album Cover** - Use album cover template
- **Ink/Chinese Painting** - Use Eastern traditional painting template
- **Oil/Watercolor** - Use traditional painting template
- **3D Rendering** - Use 3D rendering template
- **Pixel Art** - Use pixel art template

**Prompt Generation Principle**:

```
[Subject] → [Features] → [Environment] → [Lighting] → [Style] → [Technical] → [Quality]
```

- Specific over abstract: "flowing silver hair" > "nice hair"
- Quantifiable over vague: "35mm lens" > "normal lens"
- Sort by importance, core subject has highest weight

**Weight Distribution Strategy**:

| Element Type | SD Weight | MJ Weight | Description |
|--------------|-----------|-----------|-------------|
| Core Subject | 1.3-1.5 | ::3-5 | Most important identification features |
| Key Features | 1.2-1.3 | ::2-3 | Important appearance/feature descriptions |
| Style Words | 1.1-1.2 | ::1-2 | Artistic style modifiers |
| Environment Words | 0.9-1.1 | ::0.5-1 | Background environment description |
| Quality Words | 1.1-1.3 | Naturally integrated | Quality enhancement words |

### Step 6: Model Adaptation Output

**Ask the user which model formats they need**, or default to outputting these common models:

| Model | Features |
|-------|----------|
| SD1.5 | Weight syntax, needs quality words |
| SDXL | Improved understanding, high resolution |
| Flux1 | Natural language + style words |
| Flux2-Klein | Enhanced understanding, detail preservation |
| Midjourney | Parameter syntax, natural description |
| Pony Diffusion | Tag format, anime optimization |
| GPT-Image2 | Conversational generation |

**Auto-adaptation Rules**:

1. **SD → Midjourney**: Remove weight syntax, keep core description, add parameters
2. **SD → Flux**: Simplify weights, keep style words, natural language focus
3. **SD → Pony Diffusion**: Convert to tag format, add score tags
4. **SD → GPT-Image2**: Expand to full sentences, add connecting words, detailed description

See [models.md](references/models.md) for detailed format specifications.

### Step 7: Output Quality Words

Auto-generate matching quality enhancement words and negative prompts. See [quality-words.md](references/quality-words.md).

## Output Format

### Standard Output Template

```markdown
## Image Analysis Result

### Overall Description
[One sentence summarizing the image content]

### Scene Type
[Identified scene type: Portrait/Anime/Landscape/City/Still Life/Fantasy/Poster/Cover/Product Photography/Food/Fashion/UI Design/3D Rendering etc.]

### Detailed Analysis
- **Subject**: [Subject description, including gender/quantity/features]
- **Character Recognition**: [Identified anime character/celebrity name, or "No known character identified"]
- **Style**: [Artistic style]
- **Lighting**: [Lighting type]
- **Perspective**: [Perspective composition]
- **Atmosphere**: [Emotional atmosphere]

---

## Prompt Output

### SD1.5
**Positive Prompt:**
```
[SD1.5 format prompt]
```

**Negative Prompt:**
```
[Negative prompt]
```

---

### SDXL
**Positive Prompt:**
```
[SDXL format prompt]
```

**Negative Prompt:**
```
[Negative prompt]
```

---

### Midjourney
```
[Midjourney format prompt with parameters]
```

---

### Flux1
```
[Flux1 format prompt]
```

---

### Flux2-Klein
```
[Flux2-Klein format prompt]
```

---

### Pony Diffusion
**Positive Prompt:**
```
[Pony Diffusion format prompt]
```

**Negative Prompt:**
```
[Negative prompt]
```

---

### GPT-Image2
```
[Natural language description]
```

---

## Quality Word Recommendations

### Positive Quality Words
```
[Recommended quality enhancement words]
```

### Negative Prompt Template
```
[Common negative prompts]
```

### Style Modifier Suggestions
[Style words recommended based on image style]
```

## Prompt Writing Principles

### Precise Description

- **Specific over abstract**: Use "flowing silver hair" not "nice hair"
- **Quantifiable over vague**: Use "35mm lens" not "normal lens"
- **Professional terminology**: Use "chiaroscuro lighting" not "dramatic lighting"
- **Avoid subjective**: Don't describe "beautiful", describe "symmetrical facial features"

### Weight Distribution

| Element Type | Recommended Weight |
|--------------|-------------------|
| Core Subject | 1.2-1.4 |
| Style Words | 1.0-1.2 |
| Quality Words | 1.1-1.3 |
| Background Elements | 0.8-1.0 |

### Length Recommendations

| Model | Recommended Length |
|-------|-------------------|
| SD1.5 | 50-150 tokens |
| SDXL | 50-150 tokens |
| Flux1 | 50-150 words |
| Flux2-Klein | 50-150 words |
| Midjourney | 30-80 words |
| Pony Diffusion | Tag list |
| GPT-Image2 | 100-400 words |
| Other Models | 50-200 words |

### Uncertainty Handling

When certain features cannot be determined:

- **Gender uncertain**: Use "a person", "androgynous appearance", or clearly mark "gender features unclear"
- **Age uncertain**: Use "young adult" or "age difficult to determine"
- **Details unclear**: Mark "details blurry, possibly..."
- **Severe obstruction**: Mark "partially obscured, visible..."

## Common Error Prevention

### Gender Misidentification Prevention

**High-risk Scenarios**:
- Anime-style characters (male/female facial features similar)
- Neutral clothing (T-shirts, jeans, etc.)
- Back/side views (cannot see face and front clothing)
- Non-traditional gender expression

**Prevention Measures**:
1. Prioritize clothing (dress/suit etc. as decisive features)
2. Multi-feature cross-verification (clothing + hairstyle + accessories)
3. Use neutral descriptions when uncertain ("a person")
4. Avoid judgment based on single features

### Style Misidentification Prevention

**Easily Confused Styles**:
- 3D rendering vs Photo realistic
- Oil painting vs Digital painting
- Anime vs Cartoon

**Prevention Measures**:
1. Check texture details (brushstrokes, pixels, smoothness)
2. Check edge processing methods
3. Observe light/shadow physical accuracy
4. Note color naturalness

### Detail Omission Prevention

**Easily Omitted Details**:
- Small accessories
- Text/signs in background
- Special material effects
- Emotional micro-expressions

**Prevention Measures**:
1. Check item by item according to checklist
2. Pay attention to edges and corners
3. Check reflections and shadows for information

## Reference Resources

- **[character-recognition.md](references/character-recognition.md)** - Anime character recognition guide
- **[celebrity-recognition.md](references/celebrity-recognition.md)** - Celebrity recognition guide
- **[scene-recognition.md](references/scene-recognition.md)** - Scene recognition guide
- **[analysis-method.md](references/analysis-method.md)** - Image analysis methodology
- **[gender-identification.md](references/gender-identification.md)** - Gender identification guide
- **[precision-checklist.md](references/precision-checklist.md)** - Precision analysis checklist
- **[prompt-templates.md](references/prompt-templates.md)** - Prompt generation strategies and templates
- **[models.md](references/models.md)** - Model prompt format specifications
- **[quality-words.md](references/quality-words.md)** - Quality words and negative prompts
