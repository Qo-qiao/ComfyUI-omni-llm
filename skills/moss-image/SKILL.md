---
name: "moss-image"
description: "MOSS-Image prompt enhancer. Expand an image-generation request into a detailed MOSS-Image caption or pixel Layout, and compile a local JSON draft into a ready-to-paste positive prompt. Use for prompt enhancement, image composition, poster/panel text placement, bounding-box layout, or MOSS-Image 提示词扩写、画面描述、排版布局、文字定位; this skill produces prompts only and does not generate images."
tag-cn: MOSS-Image, 提示词, 排版, 图像
---

# MOSS-Image Prompt Enhancer

Turn the user's idea into a concrete image description that MOSS-Image can consume as its positive prompt. You make the creative decisions; the bundled Python tool handles coordinate arithmetic and serialization. This skill never submits image generation or publishes anything.

## Choose the deliverable

Honor an explicit request for a caption or Layout. Otherwise use a caption for ordinary scenes, photographs and illustrations. Use Layout when specific text placement, a poster, several panels, or controlled object relationships would benefit from bounding boxes. A detailed image does not automatically need Layout.

Preserve the user's subjects, counts, relationships, medium, style, exclusions and intended visible wording. Resolve ordinary creative gaps yourself; ask only when a missing decision would materially change the requested image. Do not invent factual event logistics or specifications as if the user supplied them.

Default to English for descriptive prose, following a requested output language when given. Preserve visible text verbatim in its original language, including punctuation, capitalization and requested line breaks. A quoted sentence in an instruction is not necessarily text to render: distinguish visible wording from reference material and constraints.

## Develop the image

Describe visible choices: subject and action, relevant appearance, setting, camera/framing, medium, lighting, materials, palette and mood. Add detail where it makes the requested image more specific. Avoid redundant quality tags, unrequested style changes, arbitrary word budgets and invisible process notes. Keep an explicit minimalist brief minimalist; develop a rich brief without turning it into a generic title/hero/footer template.

For designed images, compose the information and imagery together. Establish reading order, relative prominence, coherent groups and readable text at the chosen pixels. Write the intended copy rather than placeholders such as "headline here" or "some explanatory text". Do not import unrelated names, watermarks or copy from a reference. Use supplied references when useful; outside research is optional unless needed for factual accuracy or requested by the user.

## Caption

Write the caption as cohesive prose, with spatial relations in words and any visible wording quoted exactly. It may contain paragraphs; do not add coordinates or layout fields to a caption just to make it look structured. Deliver it in a code block so it can be pasted straight into the MOSS-Image positive prompt.

## Layout

Write a readable JSON draft with `summary`, a flat `items` array, concrete `width` and `height`, `bbox_coord_space="abs"` and `bbox_format="xyxy"`.

**Load the schema first — never draft from memory.** Request `references/layout.md` through `load_references` in your state tag and wait for it to appear in the loaded references before writing any Layout draft.

**Draft boxes use `[x_min,y_min,x_max,y_max]` in actual pixels, from the canvas top-left.** Keep x in `0..width` and y in `0..height`. Do not use normalized coordinates or manually center, reorder or quantize the boxes.

Honor supplied dimensions. With only an aspect ratio, choose compatible pixel dimensions; without either, use 1024×1024 unless the requested medium suggests another shape. MOSS-Image dimensions must be positive multiples of 16. Do not silently change explicit dimensions that are incompatible; surface the conflict. Use the same width and height in the image-generation workflow.

Use boxes for coherent visible units, splitting only when independent placement matters. Grouping should preserve content and relationships, not erase them. Natural occlusion and overlap are allowed. A box is a placement annotation, not an instruction to draw a border, panel or sticker. Describe appearance in captions and global treatment in `summary.caption`; let boxes agree with any spatial descriptions. Keep exact visible strings in text items' `text` fields.

## Compile and deliver

Save the draft JSON to a working file, then run the tool bundled beside this SKILL.md. Resolve the script path from the skill location (`skills/moss-image/scripts/`), not the user's working directory:

```bash
python skills/moss-image/scripts/compile_prompt.py draft.json --output prompt.txt
```

Keep the editable JSON draft and compiled output separate. Read the result and check the subject, exact text, intended geometry and chosen canvas before delivering. Fix validation errors in the draft and rerun; do not bypass a failed conversion by hand-writing centered JSON — the model must never calculate `abs_centralized` / `yxyx` coordinates itself.

If this environment cannot run shell commands, deliver the draft JSON plus the exact compile command above, and state that the compiled prompt comes from running the tool; never emit guessed centered coordinates.

For captions, the output is plain text. For Layout, it is one physical line of JSON with flat `items`, integer `abs_centralized` / `yxyx` boxes and no canvas metadata. The tool encloses fractional pixel edges before subtracting the center. It rejects invalid or out-of-bounds boxes instead of silently clipping them. String newlines are escaped in the JSON while remaining actual newlines in the decoded text.

Return the ready-to-paste prompt in a code block. For Layout also state the required width×height outside the prompt; the model prompt itself does not contain these dimensions. If showing Layout JSON inline, use the compiled output without reformatting it. Only the caption or compiled Layout belongs in MOSS-Image's positive prompt.

## Reference Files

- **[layout.md](references/layout.md)** — Layout draft schema: field definitions, the full example, centering math explanation, and composition checks. Load before writing any Layout draft.
