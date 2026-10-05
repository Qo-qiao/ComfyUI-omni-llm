# Writing captions and lyrics

Caption is the single most influential input. Lyrics are the time script. Both
must tell the same story — the model is bad at resolving conflicts.

## Caption

### What it is for

Caption describes the music's overall profile: style, instruments, mood,
atmosphere, timbre, singer gender, and the shape of the arrangement
(introduction-development-turn-conclusion).
It accepts simple style words, comma-separated tags, or full natural-language
prose; training covered all of these formats, so the text form itself does not
significantly change model behavior.

### Five ways to get a good caption

1. **Random dice** — click the UI's random button and study the sample captions.
   Use a normalized sample as a template and have an LLM rewrite it toward your
   direction.
2. **`format` auto-rewrite** — expand a hand-written simple caption into a rich
   description automatically.
3. **CoT rewrite** — when an LM is initialized (with or without `thinking`),
   Chain-of-Thought rewrites and expands your caption, unless you disabled it or
   no LM is loaded.
4. **Audio to caption** — the LM converts input audio into a caption. Precision
   is limited, but the direction is right — good enough as a starting point.
5. **Simple mode** — one short song description; the LM generates the full
   caption, lyrics, and metadata sample for you.

All five solve one reality: ordinary people have impoverished music vocabularies.
Prompting remains the highest-leverage option — the marginal gain and surprise
are largest there.

### Writing dimensions

| Dimension | Examples |
|---|---|
| Style/genre | pop, rock, jazz, electronic, hip-hop, R&B, folk, classical, lo-fi, synthwave |
| Mood/atmosphere | melancholic, uplifting, energetic, dreamy, dark, nostalgic, euphoric, intimate |
| Instruments | acoustic guitar, piano, synth pads, 808 drums, strings, brass, electric bass |
| Timbre texture | warm, bright, crisp, muddy, airy, punchy, lush, raw, polished |
| Era reference | 80s synth-pop, 90s grunge, 2010s EDM, vintage soul, modern trap |
| Production style | lo-fi, high-fidelity, live recording, studio-polished, bedroom pop |
| Vocal traits | female vocal, male vocal, breathy, powerful, falsetto, raspy, choir |
| Tempo/groove | slow tempo, mid-tempo, fast-paced, groovy, driving, laid-back |
| Structure hints | building intro, catchy chorus, dramatic bridge, fade-out ending |

### Principles

1. **Specific beats vague** — "sad piano ballad with female breathy vocal"
   outperforms "a sad song".
2. **Combine dimensions** — style + mood + instruments + timbre anchors the
   direction; a single dimension leaves too much room.
3. **Use references** — "in the style of 80s synthwave" or "reminiscent of Bon
   Iver" convey complex aesthetics quickly.
4. **Texture words matter** — warm, crisp, airy, punchy nudge mixing and timbre.
5. **Do not chase a perfect description** — caption is a starting point; write a
   rough direction, generate, then iterate.
6. **Granularity sets freedom** — omitted details mean more model freedom and
   more randomness; detailed descriptions constrain. Want surprise? Write less.
   Want control? Write more.
7. **Avoid conflicting words** — conflicting styles (e.g. "classical strings"
   plus "hardcore metal") degrade output; the model tries to fuse them and
   usually fails. With `thinking` on, the LM is weaker than DiT at caption
   generalization, so unreasonable prompts yield fewer pleasant surprises.

   Fixes for conflicts:
   - **Repetition reinforcement** — repeat the words of the element you want to
     dominate the mix.
   - **Conflict as evolution** — turn a style clash into a time-ordered
     progression: "gentle strings at the start, noisy dynamic metal rock in the
     middle, hip-hop at the end." The model then has explicit instructions
     instead of one impossible blend.

### Keep metadata out of caption

Do not write tempo, BPM, key, or time signature into the caption. Set them with
the dedicated metadata parameters (`bpm`, `keyscale`, `timesignature`, …) and
let caption focus on musical character. Metadata in caption conflicts with the
metadata parameters and confuses the model.

## Lyrics

Lyrics carry more than words:

- The lyric text itself
- **Structure marks** ([Verse], [Chorus], [Bridge], …)
- **Delivery hints** ([raspy vocal], [whispered], …)
- **Instrumental passages** ([guitar solo], [drum break], …)
- **Energy changes** ([building energy], [explosive drop], …)

### Structure marks (meta tags)

| Category | Mark | Meaning |
|---|---|---|
| Basic structure | `[Intro]` | Opening, sets atmosphere |
| | `[Verse]` / `[Verse 1]` | Main narrative sections |
| | `[Pre-Chorus]` | Build before the chorus |
| | `[Chorus]` | Emotional climax |
| | `[Bridge]` | Turn or transcendence |
| | `[Outro]` | Ending |
| Dynamic sections | `[Build]` | Energy climbs |
| | `[Drop]` | Electronic energy release |
| | `[Breakdown]` | Reduced arrangement, space |
| Instrumental | `[Instrumental]` | Pure instrumental, no vocals |
| | `[Guitar Solo]` | Guitar solo |
| | `[Piano Interlude]` | Piano interlude |
| Special | `[Fade Out]` | Fade out ending |
| | `[Silence]` | Silence |

### Combine marks, do not stack them

Combine with `-` for finer control:

```text
[Chorus - anthemic]
这是副歌的歌词
梦想在燃烧

[Bridge - whispered]
轻轻地说出那些话
```

This beats a bare `[Chorus]` — you say what the section is and how to perform it.

Do **not** pile marks up:

```text
❌ [Chorus - anthemic - stacked harmonies - high energy - powerful - epic]
✅ [Chorus - anthemic]
```

Stacking risks (1) the model singing the marks as lyrics and (2) the model
getting confused by too many instructions. Keep marks concise; complex style
description belongs in caption.

### Caption ↔ lyrics consistency

The model does not resolve conflicts. Contradictions between caption and lyrics
drop output quality.

```text
❌ Caption: "violin solo, classical, intimate chamber music"
   Lyrics:  [Guitar Solo - electric - distorted]

✅ Caption: "violin solo, classical, intimate chamber music"
   Lyrics:  [Violin Solo - expressive]
```

Checklist:

- Caption instruments ↔ lyrics instrumental marks
- Caption mood ↔ lyrics energy marks
- Caption vocal description ↔ lyrics vocal control marks

Caption is the overall setting; lyrics are the storyboard. Same story.

### Vocal control marks

| Mark | Effect |
|---|---|
| `[raspy vocal]` | Raspy, textured voice |
| `[whispered]` | Soft whisper |
| `[falsetto]` | Falsetto |
| `[powerful belting]` | Powerful, high-energy singing |
| `[spoken word]` | Spoken/rapped delivery |
| `[harmonies]` | Layered harmony |
| `[call and response]` | Call and response |
| `[ad-lib]` | Improvised ad-libs |

### Energy and mood marks

| Mark | Effect |
|---|---|
| `[high energy]` | High energy, rousing |
| `[low energy]` | Low energy, restrained |
| `[building energy]` | Escalating energy |
| `[explosive]` | Explosive |
| `[melancholic]` | Melancholy |
| `[euphoric]` | Euphoric |
| `[dreamy]` | Dreamy |
| `[aggressive]` | Aggressive |

## Lyric-writing technique

### 1. Control syllable count

**6–10 syllables per line** usually works best. The model aligns syllables to
the beat; a 6-syllable line followed by a 14-syllable line makes the rhythm
feel wrong.

```text
❌ 我站在窗前看着外面的世界一切都在改变（18 音节）
   你好（2 音节）

✅ 我站在窗前（5 音节）
   看着外面世界（6 音节）
   一切都在改变（6 音节）
```

Keep lines in the same position (e.g. the first line of each section) within
±1–2 syllables.

### 2. Case controls intensity

```text
[Verse]
walking through the empty streets   (normal effort)

[Chorus]
WE ARE THE CHAMPIONS!               (shouted, high intensity)
```

### 3. Parentheses are backing vocals

```text
[Chorus]
We rise together (together)
Into the light (into the light)
```

### 4. Elongate vowels by repeating them

`Feeeling so aliiive` — use sparingly; results are unstable and sometimes
ignored or mispronounced.

### 5. Separate sections with blank lines

```text
[Verse 1]
第一段的歌词
继续第一段

[Chorus]
副歌的歌词
副歌继续
```

## Avoiding "AI-flavored" lyrics

| Red flag | What it looks like |
|---|---|
| Adjective piles | "neon skies, electric hearts, endless dreams" — vague imagery stacked in one passage |
| Chaotic rhyme | Inconsistent rhyme scheme, or forced rhymes that break meaning |
| Blurry section boundaries | Verse content "bleeding" into the Chorus across a structure mark |
| No breath | Lines too long to sing in one breath |
| Mixed metaphors | Water in stanza one, fire in two, flying in three — nothing to anchor on |

**Metaphor discipline**: pick one core metaphor per song and dig into its facets.
With "water": love flows around obstacles, can be drizzle or flood, reflects the
other person, cannot be held yet is real. One image, many facets — that is
cohesion.

## Instrumental pieces

```text
[Instrumental]
```

or describe the arc with structure marks:

```text
[Intro - ambient]

[Main Theme - piano]

[Climax - powerful]

[Outro - fade out]
```

## Complete example

Caption: `female vocal, piano ballad, emotional, intimate atmosphere, strings, building to powerful chorus`

```text
[Intro - piano]

[Verse 1]
月光洒在窗台上
我听见你的呼吸
城市在远处沉睡
只有我们还醒着

[Pre-Chorus]
这一刻如此安静
却藏着汹涌的心

[Chorus - powerful]
让我们燃烧吧
像夜空中的烟火
短暂却绚烂
这就是我们的时刻

[Verse 2]
时间在指尖流过
我们抓不住什么
但至少此刻拥有
彼此眼中的火焰

[Bridge - whispered]
如果明天一切消散
至少我们曾经闪耀

[Final Chorus]
让我们燃烧吧
像夜空中的烟火
短暂却绚烂
THIS IS OUR MOMENT!

[Outro - fade out]
```

The lyrics marks (piano, powerful, whispered) match the caption (piano ballad,
building to powerful chorus, intimate). No conflicts.
