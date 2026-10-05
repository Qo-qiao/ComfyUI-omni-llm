# Input fields and music metadata

Everything the ACE-Step node reads as creative intent. This reference covers
prompt-side inputs only: what each field does, how to fill it, and what the
metadata parameters can and cannot do.

## Input fields

| Category | Field | Role |
|---|---|---|
| Task type | `task_type` | text2music, cover, repaint, lego, extract, complete |
| Text | `caption` | Overall music: style, instruments, mood, atmosphere, timbre, singer gender, arrangement arc |
| | `lyrics` | Time-ordered script: lyrics, structure evolution, delivery hints, instrumental passages, energy changes. Pure instrumental: `[Instrumental]` |
| Music metadata | `bpm` | Tempo (30–300) |
| | `keyscale` | Key (e.g. C Major, Am) |
| | `timesignature` | Meter (4/4, 3/4, 6/8) |
| | `vocal_language` | Vocal language |
| | `duration` | Target duration in seconds |
| Audio reference | `reference_audio` | Global acoustic reference — timbre, mixing, performance style |
| | `src_audio` | Source audio for non-text2music tasks (text2music needs none) |
| | `audio_codes` | Semantic codes in Cover mode: reuse for variants, derive new songs, splice and mix |
| Span | `repainting_start/end` | Time range for repaint or lego |

## Which prompt matters per task

| Task | What carries the intent | Prompting note |
|---|---|---|
| `text2music` | caption + lyrics + metadata | Full freedom — write both text fields well |
| `cover` | src_audio + caption + lyrics | Structure (melody/chords/arrangement) is pinned by the source; caption describes the **target** style and timbre, lyrics may be replaced entirely |
| `repaint` | src_audio + span + new lyrics/caption for the span | Describe only what should happen **inside** the span; the surrounding context is already audio |
| `lego` / `complete` | src_audio + caption for the new material | Describe the added track's instruments and role; keep it consistent with the existing track's style and tempo |
| `extract` | src_audio | No prompt needed — name the target stem |

## Music metadata: optional fine control

Usually you do **not** set metadata by hand. With `thinking` (or `use_cot_metas`)
the LM infers BPM, key, meter, and duration from caption and lyrics, and that is
usually good enough.

| Parameter | Range | Notes |
|---|---|---|
| `bpm` | 30–300 | Slow 60–80, mid 90–120, fast 130–180 |
| `keyscale` | key names | `C Major`, `Am`, `F# Minor` — pitch and emotional color |
| `timesignature` | meter | `4/4` most common, `3/4` waltz, `6/8` swing feel |
| `vocal_language` | language | LM usually detects it from the lyrics |
| `duration` | seconds | Actual output may deviate slightly |

### Guidance, not exact commands

- **BPM**: common ranges (60–180) work; extreme values (30, 280) are rare in
  training data and may be unstable.
- **Key**: common keys (C, G, D, Am, Em) are stable; obscure keys may be
  ignored or shifted.
- **Meter**: `4/4` most reliable; `3/4` and `6/8` usually fine; complex meters
  (5/4, 7/8) are advanced and style-dependent.
- **Duration**: short (30–60s) and mid-length (2–4min) are stable; very long
  generations may repeat or break structure.

The model treats `bpm=120` as an **anchor** and samples around it — you may get
118 or 122. Like telling a musician "roughly 120"; they play naturally rather
than to a metronome.

### When to set metadata manually

| Situation | Advice |
|---|---|
| Daily generation | Let the LM infer |
| A specific tempo is required | Set `bpm` |
| A specific style (e.g. waltz) | Set `timesignature=3/4` |
| Must sync with other material | Set `bpm` and `duration` |
| A specific key color is wanted | Set `keyscale` |

If manual metadata clearly does not appear in the result, check for conflicts
with caption/lyrics — "slow ballad" in caption with `bpm=160` confuses the model.

### Keep metadata out of the caption

Do not write tempo, BPM, key, or time signature into the caption. Set them with
the dedicated parameters; caption focuses on musical character (style, mood,
instruments, timbre). Metadata words in caption fight the metadata parameters
and confuse the model.

## Audio conditioning: what it means for your prompt

Text is a lossy abstraction; audio is the stronger control. Knowing which
conditioning is active tells you what to write — and what to leave to audio.

### `reference_audio` — global acoustic control

Controls **acoustic features** averaged over time: vocal and instrument timbre,
mixing style (space, dynamics, frequency balance), performance technique, and
overall feel. It does not carry melody or structure.

Prompting consequence: when a reference audio supplies timbre and mix, your
caption does not need to fight it with dense timbre words — spend the caption on
style, mood, arrangement, and structure instead.

### `src_audio` in Cover — semantic structure control

The source is quantized into melody, rhythm, chords, orchestration, and some
timbre. `audio_cover_strength` (0.0–1.0) sets how strictly the result follows
that structure: higher = closer, lower = more freedom.

Prompting consequence:

- Caption describes the **target** style/mood/genre; structure comes from the
  source.
- Lyrics may be completely rewritten — this is how you remix a song's words.
- Match the new caption to the structure you kept: do not ask for a 3/4 waltz
  caption over a 4/4 source.

### `src_audio` in Repaint — local context completion

Repaint completes or edits a **3–90 second** span using the surrounding audio as
context: change lyrics, change structure inside the span (Verse → Chorus),
continue an opening/ending, or clone the source timbre.

Prompting consequence: write the caption/lyrics fragment for the span only, and
stay consistent with the context's genre, tempo, and vocal style — the model
will blend toward the surroundings.

### `audio_codes` in Cover — advanced reuse

Reuse codes to generate variants, convert a song into codes for derivation, or
splice and mix like a DJ. Your prompt then describes the variation, not the
original structure.
