---
name: "qwen-image2.1"
description: "Qwen-Image-2.1 prompt rewriter. Routes by input: text-to-image follows the official PE-T2I rules (compact or long observational style), image editing follows the official PE-I2I rules. Supports text-heavy images/posters/infographics, transparent RGBA output, common failure-pattern fixes. Use when rewriting prompts in Qwen-Image 2.1 format, writing text-to-image prompts, rewriting image-editing instructions, or rendering readable text inside images."
tag-cn: Qwen-Image, 提示词, 改写, 图像
---

# Qwen-Image 2.1 Prompt Rewriter

Qwen-Image-2.1 (released 2026-09-20, 7B single-stream DiT, Qwen3-VL 8B text encoder, 64-channel RGBA VAE, native 2K resolution) unifies **text-to-image generation** and **image editing** behind **fluent natural language** prompts — not SD-style tag lists, and not `(term:1.5)` weight syntax. This Skill turns vague user requests into prompts that Qwen-Image-2.1 actually understands, together with the aspect-ratio field that the pipeline reads separately.

## Core Rules (Always Apply)

1. **Natural language, no tag lists.** Express emphasis descriptively: "with vibrant, striking red hair".
2. **Subject first.** Early concrete tokens carry the highest weight: subject → environment → style → composition → lighting.
3. **Text that must appear in the image is kept verbatim in double quotes.** Reproduce casing, punctuation, and line breaks exactly; never rewrite, translate, or "polish" user strings.
4. **Edit = change + preservation clause.** "Change X to Y. Keep everything else unchanged."; one logical edit per pass, and split complex changes into a chain of 2–3 small edits.
5. **Aspect ratio and resolution live only in `wh_ratio` / `ratio_follow`, never in the prompt body.**
6. **No quality-inflation words** — "masterpiece", "8K", "highly detailed", "award-winning" are not encouraged for 2.1; describe what is visible.
7. **Text-to-image descriptions are always in English** (in-image text keeps its original script); editing prose follows the language determination in edit-rules.md.

## Workflow

1. **Determine the track**: no input image → **t2i**; one or more input images → **edit / compositing**.
2. **Read the corresponding reference before writing** — do not answer from memory; request it through `load_references` and wait for loading to complete:
   - t2i → `references/t2i-rules.md`
   - edit → `references/edit-rules.md`
   The output schema, field names, mutual-exclusion rules, and self-check list in the reference all apply verbatim.
3. **If readable text inside the image is involved**, layer on the quoting discipline from `references/text-rendering.md`.
4. **If the input can be inspected visually** (environment supports a vision tool): first examine the input image's text, pose, clothing, and layout before rewriting; for plain text input, ask only when a key invariant (such as which image is the canvas) cannot be reasonably inferred.
5. **Self-check against the core rules**, then deliver in the requested output mode.

## Track Overview

### A. Text-to-Image (t2i)

- **Output contract**: `{"rewritten_prompt": "<description>", "wh_ratio": "<e.g. 3:2>"}`.
- **Style selection (decided before writing)**:

| Situation | Style |
|---|---|
| A single well-defined subject (portrait, still life, quiet scene) | **Compact**: 1–3 fluent natural-language sentences, subject → environment → style → composition → lighting, about 30 words |
| Posters, layouts, UI screens, infographics, storyboards, multi-element scenes, or anything needing precise placement or multiple text blocks | **Long observational**: the official eight-step long passage, about 20 sentences / 300–500 words |

When unsure, go long for layout and text-heavy work; go compact for simple subjects.
- Present tense, third person, declarative sentences. No imperative tone, no quality-inflation words. Hedge where uncertain, and be categorical only about elements the user specified.
- Aspect ratio goes only in `wh_ratio`, never in the description body. Default to landscape `3:2` / portrait `2:3` unless the user specifies otherwise.

### B. Image Editing (edit)

- **Output contract**: `{"rewritten_prompt": "<instruction>", "wh_ratio": "<ratio or empty>", "ratio_follow": "<imageN or empty>"}` — three fields, with `wh_ratio` and `ratio_follow` mutually exclusive.
- **Attribute decoupling**: edit only the attributes the user names, and change them all the way; lock everything else to the input as-is with preservation clauses (this prevents leakage and under-editing). The lock protects **content**, not editing strength.
- **Identity is the hardest invariant**: faces, signature accessories, product design, and rendering medium are always preserved when not explicitly targeted; identity coming from a reference image is pointed at with image tags, not described.
- **Multi-image tags**: with N ≥ 2 you must use `<image1>`, `<image2>`…; for a single image, use natural phrasing. State each image's role clearly (canvas / source-material provider).
- **Edits are short imperative sentences + preservation clauses**; split complex tasks into 2–3 small passes, re-verifying identity and tag fidelity after each round.
- **Local edits**: if the user marks a region (red circle / selection / mask), confine the instruction to that region: "Within the red box, … Leave everything else unchanged."

## Output Modes

### 1. Default Mode (Conversation-Friendly)

Standard chat requests are presented in three parts, **with no JSON output**:

1. **Prompt optimization analysis**: 2–4 bullet points — subject concept, aspect ratio and rationale, lighting/composition/material decisions, language determination.
2. **Prompt (ready to copy)**: a single code block containing **only the final prompt string**, clean enough to copy in one click into DashScope / WebUI / ComfyUI / a generation form; do not repeat the ratio label or a subtitle.
3. **Fine-tuning suggestions**: 2–3 actionable adjustments (style variants, different ratios, different rendered text).

Aspect-ratio information appears in part 1 because it never enters the prompt body.

### 2. Strict JSON Mode (On Request Only)

When the user explicitly asks for "API format", "JSON only", or "script format", or when running an automated pipeline, **output a single line of JSON only**:

- t2i: `{"rewritten_prompt": "<description>", "wh_ratio": "<e.g. 3:2>"}`
- edit: `{"rewritten_prompt": "<instruction>", "wh_ratio": "<ratio or empty>", "ratio_follow": "<imageN or empty>"}`

Field rules, mutual exclusion, and language self-checks follow the reference.

## Common Problems → Fixes

| Symptom | Fix |
|---|---|
| Tag/weight syntax is ignored | Rewrite as fluent natural sentences |
| Subject is vague | Put the subject first and add concrete detail |
| Long prompt yields a muddled composition | Compress to 1–3 sentences and lead with the subject — **unless** it is a layout/poster/text-dense image, in which case the long style is correct and the usual problem is unspecified placement |
| In-image text comes out garbled | Put strings in quotes, raise true_cfg to 6–8 and steps to 35–50, and simplify numerals and symbols; for Japanese, use the edit-tracing route |
| Edits bleed into untouched regions | Add or extend preservation clauses, and split into smaller edits |
| Face/identity drift | State "Preserve face/clothing features" explicitly, and point at reference images with tags instead of describing them |
| Broken fingers/hands | Negative "extra fingers, deformed hands" + positive "natural hand posture, five fingers" |
| Early 2.1: heavy noise, stiff poses | A known issue at launch with no mature fix — retry with a different seed and shorten the prompt |

## Reference Documents

- **[t2i-rules.md](references/t2i-rules.md)** — Official PE-T2I eight-step text-to-image rewriting spec (long observational + compact styles)
- **[edit-rules.md](references/edit-rules.md)** — Official PE-I2I editing/compositing rewriting spec (language determination, attribute decoupling, aspect-ratio selection)
- **[text-rendering.md](references/text-rendering.md)** — In-image text rendering: quoting discipline, presentation control, multi-block text, Japanese text tracing, failure fixes
- **[cheat-sheet.md](references/cheat-sheet.md)** — Vocabulary quick reference: mediums/styles, materials, physical textures, spatial terms, lighting, aspect-ratio cheat sheet, transparent-image templates
- **[examples.md](references/examples.md)** — Curated official and community examples (generation, text-dense, multi-reference, transparent, editing)
