---
name: wan2.2-video
description: Wan video prompt optimizer. Rewrites a video idea into Tongyi Wan (Wan2.2–2.7) ready prompts using the official formulas — basic, advanced, image-to-video, sound, reference-generation, and multi-shot — plus video-editing instruction techniques, image/video reference notation, and a cinematic aesthetics vocabulary. Use when the user wants to write, rewrite, or optimize Wan text/image/reference-to-video prompts, edit video by instruction (change elements, environment, style, dialogue, replicate, continue), or asks about Wan prompt formulas, camera movement, shot sizes, color tone, emotion, or style wording.
tag-cn: 万相, Wan, 提示词, 视频, 编辑
---

# Wan Video Prompt Optimizer

Rewrites a video idea into the prompt Tongyi Wan (Wan2.2–2.7) expects: pick the formula for the task, use plain semantic instructions for editing tasks, and add aesthetics/style vocabulary only as needed. The more complete, precise, and rich the description, the closer the result; image-to-video and editing tasks should only describe what that task adds, never restate what the image/video already fixes.

Full formulas, editing techniques, and vocabulary live in the three reference files. Load the relevant one before writing.

## Workflow

1. **Classify the task** — text-to-video / image-to-video / reference-generation (@character or subject reference) / video editing / continuation · first-last frame / storyboard (multi-grid) / with sound.
2. **Pick the formula** ([prompt-formulas.md](references/prompt-formulas.md)):

   | Task | Formula |
   |---|---|
   | Text-to-video (basic) | subject + scene + motion |
   | Text-to-video (advanced) | subject + scene + motion + aesthetics + stylization |
   | Image-to-video | motion + camera movement |
   | With sound | subject + scene + motion + sound (voice / SFX / BGM) |
   | Reference-generation | @character + action + dialogue + scene |
   | Multi-shot (2.6) | overview + shot numbers + timestamps + per-shot content |
   | Story clip / storyboard | storyboard-style structure or "this is a film clip centered on …" |

3. **Editing tasks** ([video-editing.md](references/video-editing.md)) — describe the semantic change plainly; append "keep everything else unchanged" to preserve the rest; when changing dialogue, write "keep the original voice tone and pace" and quote the new line.
4. **Add vocabulary** ([style-vocabulary.md](references/style-vocabulary.md)) — pick from light source, lighting, time of day, shot size, composition, focal length, camera angle, camera type, color tone, motion, emotion, camera movement, style, and sound terms; use only the dimensions that fit the scene.
5. **Add version control keywords** when needed — "Generate single shot." / "No dialogue." / "No background music." (Wan 2.7).
6. **Check against the checklist**, revise once, and return only the revised prompt.

## Language & Reference Notation

- Write the prompt in Chinese for a Chinese request, English for an English request; quote dialogue.
- Referring to images/videos: Chinese uses "图1, 视频1" (no space before surrounding text), English uses "Image 1, Video 1" (space between letters and digits, capitalized); order matches the submitted array, images and videos counted separately; at most 5 references total.
- Reference-generation uses @A/@B for characters (up to 2 characters, referenced as often as needed).

## Checklist

- Correct formula with all its elements (omit only elements the task lacks)
- Image-to-video / editing prompts add only new information; never restate fixed subject, scene, or style
- Editing instructions are simple and explicit; "keep everything else unchanged" where parts must be preserved; dialogue changes keep voice tone and pace
- Reference numbering matches submission order and prompt language
- Multi-shot prompts carry shot numbers and timestamps; continuation describes only motion and never combines with driving audio
- Voice = content + emotion + intonation + pace + timbre + accent; SFX = source/material + action + environment; BGM = soundtrack + style
- Dialogue is quoted; "No dialogue / No background music / single shot" keywords written explicitly when required
- Aesthetics and style terms cover only relevant dimensions and do not contradict each other

## Output

Return the copy-ready prompt directly, plus at most two lines: task type/formula used and version keywords. When JSON is requested, output only `{"prompt": "..."}`.

Do not invent model behavior; content follows the official "Wan AI Video Generation User Guide".

## Reference Files

- **[prompt-formulas.md](references/prompt-formulas.md)** — six formulas, subject-reference and notation rules, multi-shot and 2.7 control keywords, story-clip and storyboard structures, time-segmented camera writing.
- **[video-editing.md](references/video-editing.md)** — video-editing instructions: element add/modify/delete, environment/style/dialogue/camera changes, motion·camera·effect replication, continuation and first-last frame rules.
- **[style-vocabulary.md](references/style-vocabulary.md)** — cinematic aesthetics dictionary, motion and emotion, basic/advanced camera movement, visual styles and effects, sound types, classic mood combinations.
