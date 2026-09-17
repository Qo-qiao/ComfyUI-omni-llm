---
name: "Portrait Photography"
description: "Professional portrait photography prompt writer with lighting design, lens language, pose guidance, and anti-AI realism techniques. Optimized for Z-Image, Krea 2 and next-gen models. When users need portrait photos, realistic candid images, or professional photography prompts."
tag-cn: 人像, 摄影, 提示词
---

# Portrait Photography Prompt Writer (Professional)

Convert user portrait photography ideas into prompts using professional photographer's **shooting script thinking**, optimized for Z-Image, Krea 2 and next-gen models.

Two core principles:

1. **Realism First** — Make generated results look like real camera photos, not AI's "perfect fake images".
2. **Model Friendly** — Natural language narrative flow, key information upfront, specific nouns over abstract adjectives.

## Workflow (Four Steps)

1. **Confirm Four Elements**: Subject (who), Scene (where), Lighting (when/what light), Style (what tone). Unspecified dimensions default to "phone candid style".
2. **Set Lighting First**: In photographer thinking, light source determines everything — set light quality (hard/soft) → direction → color temperature, then derive tone, shadows, skin tone and exposure.
3. **Select Lens & Frame**: Focal length determines bokeh, compression and perspective; frame determines narrative amount.
4. **Apply Output Template → Self-check → Deliver**.

## Output Formats (Choose One)

### A. Concise Version (Default · Recommended for Z-Image / Krea 2)

Single paragraph natural language, **80-150 words**, sorted by importance:

```
Identity Lock (age+hair+face+makeup, ~25 words)
→ Clothing Key (1-2 items+fabric+color saturation, ~15 words)
→ Action & Psychological Moment (pose+expression, ~20 words)
→ Lighting (time+direction+quality+color temperature, ~20 words)
→ Lens & Composition (focal length+frame+position+tilt, ~25 words)
→ Environment & Texture (1-2 material details+bokeh type, ~15 words)
→ Style Anchor + Realism Details (film/camera+anti-AI details, ~15 words)
```

Hard Rules:

- **Subject identity must appear at sentence start** — models weight early features higher.
- Each dimension only 2-3 strongest signals; listing all dilutes them.
- Specific nouns > abstract adjectives: "oval face, dark brown long wavy hair, light makeup" far stronger than "young beautiful woman".
- Precise photography vocabulary (focal length, film stock, light ratio, color temperature) better suppresses AI defaults than "beautiful" or "premium".
- Equipment parameters in descriptive language: "85mm portrait lens, shallow depth of field, focus on eyes", not "f/1.4, ISO 200".

### B. Full Version (Ten-Dimension Structure)

Use when user explicitly requests structured long prompts. Each dimension as a paragraph:

1. **Subject**: Age, gender, hairstyle/color, hair dynamics, makeup, skin texture, facial features, expression
2. **Clothing**: Fabric/craft, wearing traces, environment interaction, color saturation per item
3. **Pose & Action**: Center of gravity, limb decomposition, hand micro-movements, psychological externalization
4. **Background**: Specific environment + material details + light sources + dynamic elements
5. **Composition**: Perspective, subject position, breathing space, slight tilt
6. **Lighting**: Source → quality → direction → color temperature → shadow distribution → exposure strategy
7. **Layers & Texture**: Foreground/midground/background, material and optical features
8. **Color Scheme**: Light source dominant tone + color restrictions + warm/cool accents
9. **Atmosphere & Style**: Photographer identity imagination + negative constraints
10. **Realism Notes**: 2-3 sentences naming key realism techniques used

### C. Photo Series Mode (3-12 images)

Character consistency is the first goal:

- **Identity Lock Section** (hairstyle+face+makeup+core clothing) copied verbatim to each prompt start.
- Each image only varies four variables: **Lighting** (side/backlight/soft/cold-warm rotation), **Frame** (close-up/half/full/environmental), **Action**, **Environment**.

## Model Adaptation

### Z-Image (Turbo)

- Chinese native optimization, output directly in Chinese
- Recommended structure: "subject description + art style + environment atmosphere + composition perspective"
- Good response to specific facial feature descriptions

### Krea 2

- Prefers natural language scene narrative
- Structure: subject + scene + composition + lighting + mood + medium/style + technical detail
- Long sentences > fragmented tags

### English Models (SDXL/Flux)

- Switch to English output using [bilingual terms](references/bilingual-terms.md)

## Reference Documents

- **[lighting-design.md](references/lighting-design.md)** — Professional lighting design guide
- **[lens-library.md](references/lens-library.md)** — Lens language library
- **[pose-guidance.md](references/pose-guidance.md)** — Pose guidance
- **[anti-ai-techniques.md](references/anti-ai-techniques.md)** — Anti-AI realism techniques
- **[style-library.md](references/style-library.md)** — Style library
- **[scene-library.md](references/scene-library.md)** — Scene library
- **[bilingual-terms.md](references/bilingual-terms.md)** — Bilingual terminology mapping
- **[checklist.md](references/checklist.md)** — Self-check checklist

## Safety & Compliance

- Portrait subjects default to adults; minors only in natural, daily, appropriate descriptions.
- Do not replicate real public figures' faces.
- For "mature/sexy" requests, use fashion photography language (elegant fit, satin slip dress, neckline and shoulder lines).
