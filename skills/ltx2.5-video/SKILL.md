---
name: ltx2.5-video
description: LTX 2.5 prompt optimizer. Rewrites a video idea into a production-ready LTX-2.5 generation prompt — single flowing paragraph, present tense, 4-8 descriptive sentences, physical emotion cues, quoted dialogue, single-shot or multi-shot structure with named cuts, and audio direction. Use when the user wants to write, rewrite, or optimize prompts for LTX video generation, asks about the LTX prompt enhancer, or brings a prompt written for another model such as Kling or Seedance.
tag-cn: LTX, 提示词, 视频, 优化
---

# LTX 2.5 Prompt Optimizer

Turn a video idea into a prompt written the way LTX-2.5 wants it: one flowing
paragraph that paints a complete picture from beginning to end. Do not paste
prompts written for other video models (Kling, Seedance, …) unchanged — content
carries over, but tag syntax and shot-list formatting don't and tend to
underperform; rewrite them into LTX's structure.

Full rules, vocabulary lists, and sample prompts live in
[references/prompting.md](references/prompting.md). Load it before drafting.

## Workflow

1. **Gather the brief** — subject(s) and distinguishing features, core action,
   shot scale, camera movement, lighting/atmosphere, audio (dialogue, music,
   ambience), and any pacing beats or on-screen text.
2. **Pick the structure** — single continuous shot by default; multi-shot only
   when the user wants explicit cuts (prompting.md §Multi-Shot: 2–4 shots,
   chronological, each cut names its transition, re-establishes the shot, keeps
   identity, and states audio continuity).
3. **Draft** — single flowing paragraph, present tense, ~4–8 sentences; every
   sentence carries an action verb; emotion as physical cues, not abstract
   labels; dialogue in quotation marks with language/accent when needed.
4. **Pace it** — LTX-2.5's duration predictor times the clip as written: write
   pauses and beats ("she pauses", "a beat of silence") into the prompt, or
   suggest an explicit duration.
5. **Enhancer advice** — if the prompt is short, rough, or came from another
   model, recommend the prompt enhancer (dedicated node in the official
   ComfyUI templates; `--enhance-prompt` in `ltx-pipelines`; not for direct API
   requests). Skip the advice when the prompt already follows this structure —
   the enhancer only adds latency then.
6. **Validate** against the checklist, revise once, return only the revised
   prompt.

## Checklist

- Scene focused: a few clear characters and actions, not a crowded frame
- One coherent lighting logic per shot; no mixed light sources
- Physical emotion cues instead of mood labels ("her jaw tightens", not "she is
  angry")
- Present tense, action verbs, detail matched to shot scale
- Audio described: ambient sound, music, speech; dialogue in quotes
- Multi-shot: transition named, subject re-identified, audio continuity stated
  at every cut
- On-screen text short and prominent; complex physics kept plausible
- No leftover foreign-model tag syntax or numbered shot lists

## Output

Return the prompt itself (copy-ready), plus at most two lines of notes: single-
or multi-shot choice and whether the enhancer is worth running. When the user
asks for JSON, output `{"prompt": "..."}` only.

Do not fabricate model behavior claims; stick to the rules in prompting.md and
the source guide at <https://docs.ltx.io/open-source-model/usage-guides/prompting-guide>.

## Reference

- **[prompting.md](references/prompting.md)** — key elements, structuring,
  multi-shot cuts, prompt enhancer, on-screen text/physics caveats, Dub-It
  template, helpful-term vocabulary, sample prompts.
