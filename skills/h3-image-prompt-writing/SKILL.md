---
name: h3-image-prompt-writing
description: Write H3Image prompts for text-to-image generation and reference-image editing. Use to turn an idea or brief into a complete image prompt, refine an existing prompt, or describe edits and reference-guided compositions.
compatibility: Portable to any agent that can read local files — no external API calls, image-generation runtime, or proprietary tools required. This skill only writes prompts; rendering is handled by the host application.
---

# H3Image Prompt Writing

Turn ideas, briefs, existing prompts, and reference images into ready-to-use H3Image prompts.

## Workflow

1. Identify the mode: T2I for a text-only image, or Edit when supplied reference images must be preserved, changed, or borrowed, including for a new composition.
2. Read the matching guide and its examples:
   - [T2I](references/t2i-en.txt): compose an image from text.
   - [Edit](references/edit-en.txt): write the single-paragraph source–relationship–change prompt. Inspect the relevant images in their supplied order.
3. Develop the brief into concrete visible content and complete ordinary unspecified details. For an existing prompt, apply the requested revision while retaining its other constraints.
4. Ask only when missing information prevents a useful prompt. Do not require a finished draft from someone asking you to write one, or invent source facts to fill an evidence gap.
5. Check the finished prompt against the user's content, exact wording, counts, spatial relationships, and exclusions.
6. When the host or downstream tool needs generation settings, give the aspect ratio separately. Preserve explicit ratios or derive it from requested dimensions.

## Image Modes

### T2I

Build a finished scene or designed artifact from text. Choose a clear visual focus, honor specified entities, counts, style, medium, and layout, and describe only what appears in the image. Use the composition, physical-readability, and artifact-completion rules in `references/t2i-en.txt`.

### Edit

Write one continuous plain-text paragraph that defines each referenced source, states the target and each source relationship, and then completes the changed or new content. Keep `<Picture N>` in supplied image order and `<Subject N>` stable across the prompt. Do not print the guide's section headings or issue editing commands.

## Output Rules

- Deliver the prompt itself. This skill writes prompts; it does not generate images.
- Use English visual narration; visible lettering and authored copy retain their required language.
- Organize T2I descriptions naturally. Follow the Edit guide's single-paragraph source–relationship–change structure.
- Preserve literal text, counts, spatial relationships, and exclusions exactly as requested; do not invent real statistics, prices, certifications, or specifications.

## Tips for Better Results

- Establish the main composition before adding scenery; choose framing from the deliverable and its content.
- Keep one frozen moment unless a sequence is requested, and keep contact, gaze, screen direction, and viewpoint consistent.
- Give every requested region inspectable content: actual copy, chart axes and units, UI states, or diagram labels.
- Use concrete visual choices instead of generic quality claims, hidden backstory, or extra principal subjects.
- In Edit mode, preserve unmentioned content by default and keep full-preservation statements to their exact defined scope.
