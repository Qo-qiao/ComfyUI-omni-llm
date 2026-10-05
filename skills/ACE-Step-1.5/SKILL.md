---
name: "ACE-Step-1.5"
description: "ACE-Step 1.5 prompt writing guide. Turns a song request into the inputs the ComfyUI ACE-Step nodes read: task_type, caption, lyrics, music metadata, and audio-conditioning notes. Works in five stages — music brief, constraint resolution, style routing, rendering, validation — extracting supported values, routing the genre to a style-vocabulary section, then rendering caption and lyrics through a self-check checklist. Covers caption writing principles, lyric structure marks, style vocabulary routing, consistency checks, and per-task prompt focus; also for prompting and lyric writing under ACE-Step generation, cover, repaint, and track-addition tasks."
tag-cn: ACE-Step, Prompt, Lyrics, Caption, Style Router, Music
---

# ACE-Step 1.5 Prompt Writing

Turn a song request into inputs you can paste straight into the ComfyUI
ACE-Step nodes. This skill covers **prompt and input authoring only**: caption,
lyrics, music metadata, task type, and audio-reference choices — not model
downloads or inference tuning.

Use natural-language reasoning and this skill's local text files only. Do not
execute scripts, build databases, call external APIs, or read every style
vocabulary section.

## 1. Output contract

Deliver these every time; field names match the node inputs:

1. **task_type**: text2music / cover / repaint / lego / extract / complete
2. **caption**: the music's overall profile (style, mood, instruments, timbre,
   vocal, production, structure hints)
3. **lyrics**: the time script (pure instrumental: `[Instrumental]`)
4. **Music metadata** (only when the user asks or the task needs it): `bpm`,
   `keyscale`, `timesignature`, `vocal_language`, `duration`
5. **Audio conditioning notes**: whether `reference_audio` / `src_audio` is
   needed and what each controls

**Language**: unless told otherwise, write the caption in English (the model's
example vocabulary is English and densest there); lyrics follow the request's
language (a Chinese request gets Chinese lyrics); an explicit user language
choice always wins.

In the default conversational mode, present in sections (analysis →
copy-ready caption/lyrics code blocks → tuning tips). When the user asks for
"JSON/API format", output a single-line JSON only:
`{"task_type": "...", "caption": "...", "lyrics": "...", "bpm": null, "keyscale": null, "timesignature": null, "vocal_language": null, "duration": null}`.

Unless the user requests diagnostics, do not expose the music brief, routing
choices, or vocabulary section names.

## 2. Workflow

Follow these five stages in order:

1. **Build the music brief** (section 3): extract supported values and
   classify them.
2. **Resolve constraints** (section 4): apply precedence and lock the explicit
   requirements that must never be reversed.
3. **Route the style**: read
   [references/style-router.md](references/style-router.md) → pick one primary
   family (+ one secondary only for an explicit fusion, two maximum) → request
   [references/style-vocabulary.md](references/style-vocabulary.md) through
   `load_references` and use **only the routed sections**.
4. **Render**: write caption and lyrics (read
   [references/caption-and-lyrics.md](references/caption-and-lyrics.md)),
   fill metadata and judge audio conditioning as needed (read
   [references/inputs-and-metadata.md](references/inputs-and-metadata.md)).
5. **Validate, then deliver** (section 7): run the checklist; if any item
   fails, revise once and return only the revised result.

## 3. Build the music brief

Extract only supported or reasonably inferred values:

- macro genre, subgenres, and cultural or market style
- mood and emotional arc
- approximate tempo feel and groove (never invent a specific BPM)
- vocal presence, gender, timbre, and delivery
- core instruments and production texture
- section structure and section-specific changes
- spatial character and explicit exclusions

Classify each value internally as `explicit`, `inferred`, or `unspecified`.

- Do not invent a precise key, BPM, vocal register, or production technique
  when a broader description is sufficient.
- Preserve an explicit instrumental request — do not add vocals.
- When vocal presence is unspecified, choose a conservative treatment supported
  by both the user's description and the nearest style family.

## 4. Constraint precedence

Apply in this order:

1. Explicit user requirements and exclusions.
2. Section-local directives from lyric tags, within that section only.
3. Strong implications from the user's description.
4. Selected vocabulary characteristics.
5. Conservative musical defaults.

A section tag changes its local arrangement without replacing the global style;
preserve a hard user exclusion when a tag conflicts with it. When two explicit
instructions conflict, prefer the more specific and later one if the intent is
still clear, otherwise make the smallest musically coherent compromise.
**Never silently reverse an explicit vocal gender, instrumental requirement,
tempo limit, required instrument, or prohibited element.**

## 5. Task type → field focus

| Request | Task | Prompt focus |
|---|---|---|
| Generate a song from text | `text2music` | caption + lyrics, full freedom |
| Keep structure, change style/words (cover, remix, retake) | `cover` | Structure is pinned by the source; caption describes the **target** style, lyrics can be replaced entirely |
| Local lyric/structure edits or continuation (3–90 s span) | `repaint` | Describe only what happens **inside** the span; match the surrounding context |
| Add an instrument track to existing audio | `lego` | Describe the added track's instruments and role |
| Separate a stem | `extract` | No prompt needed |
| Add a mixed accompaniment to a single track | `complete` | Describe the accompaniment's instruments and style |

## 6. Writing principles

- **Caption is the most influential input**: specific beats vague; combine
  style + mood + instruments + timbre; granularity sets freedom (detailed for
  control, sparse for surprise).
- **Vocabulary is material, not a checklist**: pick 3–6 dimensions from the
  routed section, do not stack whole sections or copy example sentences
  wholesale; the brief wins whenever it conflicts with a card.
- **Avoid conflicting words**: clashing styles ("classical strings" +
  "hardcore metal") degrade output. Fixes: repeat the element you want to
  dominate, or write the clash as a **time-ordered evolution** (gentle strings
  → metal middle → hip-hop ending).
- **Caption and lyrics must agree**: the model does not resolve conflicts. If
  caption says violin solo, lyrics must not contain `[Guitar Solo]`; align the
  vocal, mood, and instrument lines.
- **Keep structure marks concise**: `[Chorus - anthemic]` combined with `-` is
  enough; never stack marks (they get sung as lyrics or confuse the model).
  Complex style description belongs in caption.
- **Metadata is guidance, not an exact command**: the model treats it as an
  anchor and samples around it (asking for 120 may yield 118). Numeric BPM,
  key, and meter go to metadata parameters only; the caption carries qualitative
  tempo and groove words.
- **Let audio carry what text cannot**: timbre and mix go to
  `reference_audio`; melody, chords, and structure go to Cover's `src_audio` —
  spend the caption on style, mood, and arrangement instead.

## 7. Self-check

Verify before returning:

- [ ] Every explicit user constraint and exclusion is preserved
- [ ] Instrumental requests remain instrumental; vocal gender, required
      instruments, and tempo limits are not silently reversed
- [ ] No conflicting style words in the caption; clashes rewritten as evolution
      or reinforced by repetition
- [ ] Caption instruments ↔ lyrics instrumental marks; caption mood ↔ energy
      marks; caption vocal description ↔ vocal marks
- [ ] Lyric lines at 6–10 syllables; same-position lines within ±1–2 syllables
- [ ] Structure marks not stacked; blank lines between sections; actionable
      section tags land in the matching section
- [ ] No BPM/key/meter words inside the caption; no fabricated precise metadata
- [ ] No vocabulary example sentence copied wholesale; caption specific enough
      to guide generation without becoming an essay
- [ ] No contradiction with the source audio's style or tempo (cover/repaint)
- [ ] Lyrics free of adjective piles, chaotic rhyme, section bleed, and mixed
      metaphors

If any item fails: revise once, then return only the revised result.

## 8. References

- **[style-router.md](references/style-router.md)** — style family routing:
  contract, family map, Chinese/English aliases, fusion rules, fallback
  routing, field dispatch
- **[style-vocabulary.md](references/style-vocabulary.md)** — caption
  vocabulary for 18 families: terms, mood, palette, vocal, structure cues, and
  a one-line example each
- **[caption-and-lyrics.md](references/caption-and-lyrics.md)** — caption
  dimensions and seven principles, structure/vocal/energy mark tables, lyric
  technique, avoiding AI flavor, complete example
- **[inputs-and-metadata.md](references/inputs-and-metadata.md)** — input
  field table, per-task prompt focus, metadata rules, how audio conditioning
  changes what you write

Read `style-router.md` first, then `style-vocabulary.md` as routed. Before
answering concrete caption, lyric, or metadata questions, request the matching
reference through `load_references` and wait for it to load — do not answer
from memory.
