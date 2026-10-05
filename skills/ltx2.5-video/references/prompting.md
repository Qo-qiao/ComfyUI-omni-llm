# LTX 2.5 Prompting Guide

Source: <https://docs.ltx.io/open-source-model/usage-guides/prompting-guide>

Prompt-writing reference for LTX-2.5 generation prompts. Read before drafting or rewriting any prompt for LTX video generation.

The goal: paint a complete picture of the story that flows naturally from beginning to end and covers every element the model needs. Do not paste prompts written for other video models (e.g. Kling or Seedance) into LTX unchanged — content usually carries over, but tag syntax and shot-list formatting don't and tend to underperform. Rewrite into LTX's flowing-paragraph structure.

## Key Elements to Include

1. **Establish the shot** — cinematography terms matching the intended genre; shot scale or category-specific characteristics to refine the visual style.
2. **Set the scene** — lighting, color palette, surface textures, atmosphere to establish mood and tone.
3. **Describe the action** — a natural sequence flowing beginning to end. Every sentence needs a verb that *does* something (walks, turns, exhales, reaches); appearance alone gives the model little to animate.
4. **Define the character(s)** — age, hairstyle, clothing, distinguishing features. Express emotion through **physical cues**, not abstract labels.
5. **Identify camera movement(s)** — how and when the camera moves; describing how subjects appear *after* the movement helps the model complete the motion.
6. **Describe the audio** — ambient sound, music, speech, singing. Put spoken dialogue in **quotation marks**; specify language and accent if needed.

General principles:

- Keep the scene focused — a few clear characters and actions beat a crowded frame.
- Keep lighting consistent — one coherent light logic per shot; mixed light sources confuse the result.
- Start simple and layer — core shot first, add detail as you iterate.

## Structuring Your Prompt

Match the structure to what you're describing — LTX responds best to cinematic, single-subject scenes with clear camera language, consistent lighting, and well-described audio.

### Simple / Single-Shot

- Write a **single flowing paragraph**.
- **Present tense** verbs for action and movement.
- Detail level matched to shot scale (close-ups need more than wide shots).
- Camera movement described relative to the subject.
- Roughly **4–8 descriptive sentences**.

### Longer / Screenplay-Style

For dialogue, multiple beats, or precise timing: screenplay style with scene headers, character cues, and quoted dialogue (see Sample Prompts below). Same fundamentals: present tense, physical emotion cues, dialogue in quotation marks.

### Length and Pacing

- Match length to complexity; every sentence should add concrete visual or audio detail.
- **Pace the action in the prompt itself.** LTX-2.5's optional duration predictor sizes the clip to the action you describe and times it as written — it won't stretch a moment or add a pause you didn't prompt for. Write the beats you want ("she pauses", "a beat of silence") or set an explicit duration.

## Multi-Shot Prompts (LTX-2.5)

LTX-2.5 can join several distinct shots with explicit cuts inside one prompt. Write the full scene as **one chronological paragraph** (or a short sequence of sentences). Do **not** use a shot list, numbered beats, or screenplay sluglines unless you also describe the cut in prose.

|             | Single-shot                            | Multi-shot                                                                 |
| ----------- | -------------------------------------- | -------------------------------------------------------------------------- |
| Camera      | One continuous take                    | New framing after each cut                                                 |
| Transitions | Camera moves only (pan, push-in, etc.) | Name the edit: hard cut, match cut, dissolve, etc.                         |
| Continuity  | Same space / subjects throughout       | Re-identify subjects when they reappear; say what carries across the cut  |
| Audio       | One continuous soundscape              | At every cut, say whether music / dialogue / ambience continues or changes |

At every cut:

1. **Name the transition** in natural language — "A hard cut transitions to…", "A match cut connects…", "The image dissolves into…".
2. **Re-establish the new shot** — shot scale, angle, who or what is in frame, lighting if changed.
3. **Keep identity consistent** — reuse the same visual identifiers for recurring people/objects ("the woman in the red coat, earlier at the table, now…").
4. **State audio continuity** — "the piano score continues across the cut" or "the dialogue drops; only wind remains."

Tips:

- Prefer **2–4 shots** per generation; more cuts need shorter, clearer beats.
- Give each shot a clear job (establish → detail → reaction, or wide → medium → close-up).
- Keep action chronological: "Initially…", "A moment later…", "Simultaneously…".
- Same rules as single-shot: present tense, physical emotion cues, quoted dialogue, concrete camera language.
- Avoid conflicting geography or unexplained costume changes between cuts unless the cut is meant to jump time or place and you say so.

When to stay single-shot: unbroken camera motion, intimate performance, or dialogue that must stay lip-synced in one framing. For image-to-video from a first frame, prefer a single continuous take unless you intentionally describe a cut away from that opening image.

Multi-shot example:

> A wide shot frames a rainy city intersection at dusk, neon signs reflecting on wet asphalt. A young woman in a yellow raincoat walks down the sidewalk toward camera, carrying a small bag, while the rain falls and cars drive past behind her. Soft synth music and traffic noise fill the air. The shot transitions to a medium close-up of her face under the hood, raindrops catching the neon as she looks off-screen left; the synth score continues across the cut, traffic muffled. She speaks quietly to herself, "He's late." A hard cut jumps to a low-angle shot of a man's scuffed boots stepping into a puddle at the curb; the music drops to a low drone. The man she has been waiting for — short dark hair, soaked jacket — lifts his head into frame as he smiles at her off-screen. A bus rumbles past them.

## Using the Prompt Enhancer

Applies to the local open-source LTX-2.5 paths: native `ltx-pipelines` and the official ComfyUI templates. (For direct API requests, do not use `--enhance-prompt`.)

LTX pipelines include an optional **prompt enhancer**: an LLM rewrite pass (the Gemma 4 E2B model) that expands your prompt with a system prompt tuned for text-to-video or image-to-video before generation runs. Enabled with `--enhance-prompt` in `ltx-pipelines` and via a dedicated node in the official ComfyUI templates; can be turned off in both.

- **Use it** when the prompt is short, rough, or was originally written for a different model.
- **It helps least** when the prompt already follows the structure in this guide.
- It adds an extra inference pass (latency); leave it off to submit the prompt exactly as written.

## Keep in Mind

- **On-screen text** — LTX-2.5 improves short-text accuracy, but exact spelling and cross-frame consistency aren't guaranteed. Keep text short and prominent, verify throughout the clip, add critical titles/labels/logos in post.
- **Complex physics** — highly chaotic motion can introduce artifacts; simpler, plausible motion is more reliable.

## Dub-It (Speech Replacement)

The Dub-It IC-LoRA is video-to-video: replace spoken dialogue in existing video. Provide the source video and describe what the speaker should say instead — for dubbing into other languages or rephrasing in the original language.

Validated languages: English, French, Spanish, German, Russian.

Template:

```
[Speaker] is speaking [Language/Accent], saying: "[Dialogue]"
```

Example:

```
A woman speaking in Russian saying: "Сегодня отличный день, чтобы протестировать рабочие процессы ComfyUI для дубляжа с использованием LTX."
```

Emotion or delivery style can be added to the prompt.

Requirements:

- **Provide the full dialogue text** — the model follows the prompt; it does **not** translate for you.
- **Use native script** — the alphabet of the target language (Cyrillic for Russian, Chinese characters for Mandarin).
- **Single speaker** — the beta IC-LoRA does not distinguish multiple speakers.

Best practices:

- **Match audio length** — roughly the same timing and syllable length as the original dialogue; slightly longer is better than too short. Too long: the model may skip words. Too short: output sounds slow and unnatural.

## Additional Helpful Terms

### Categories

- **Animation** — Stop-motion · 2D / 3D animation · Claymation · Hand-drawn
- **Stylized** — Comic book · Cyberpunk · 8-bit pixel · Surreal · Minimalist · Painterly · Illustrated
- **Cinematic** — Period drama · Film noir · Fantasy · Epic space opera · Thriller · Modern romance · Experimental film · Arthouse · Documentary

### Visual Details

- **Lighting** — Flickering candles · Neon glow · Natural sunlight · Dramatic shadows
- **Textures** — Rough stone · Smooth metal · Worn fabric · Glossy surfaces
- **Color Palette** — Vibrant · Muted · Monochromatic · High contrast
- **Atmosphere** — Fog · Rain · Dust · Smoke · Particles

### Sound and Voice

- **Ambient Settings** — Coffeeshop noise · Wind and rain · Forest ambience with birds
- **Dialogue Style** — Energetic announcer · Resonant voice with gravitas · Distorted radio-style · Robotic monotone · Childlike curiosity
- **Volume** — Whisper · Mutter · Shout · Scream

### Technical Style Markers

- **Camera Language** — Follows · Tracks · Pans across · Circles around · Tilts upward · Pushes in / pulls back · Overhead view · Handheld movement · Over-the-shoulder · Wide establishing shot · Static frame
- **Film Characteristics** — Film grain · Lens flares · Pixelated edges · Jittery stop-motion
- **Scale Indicators** — Expansive · Epic · Intimate · Claustrophobic
- **Pacing & Temporal Effects** — Slow motion · Time-lapse · Rapid cuts · Lingering shot · Continuous shot · Freeze-frame · Fade-in / fade-out · Seamless transition · Sudden stop
- **Visual Effects** — Particle systems · Motion blur · Depth of field

## Sample Prompts

Screenplay style (news-broadcast scene):

```
EXT. TOWN STREET – MORNING – LIVE NEWS BROADCAST
The shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. The light is warm, early sun catching the camera lens. A faint hum of chatter and distant drilling fills the air. The reporter, composed but smiling nervously, looks directly into the camera, microphone in hand.
Reporter: "Thank you, Sylvia. And yes — this is a sentence I never thought I'd say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!"
He gestures toward the field behind him. "If my cameraman can pan over, you'll see what all the excitement's about." The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground and blasts upward in a violent plume.
Workers cheer and scramble as the black stream glistens in the morning light. Reporter (off-screen, shouting over the noise): "There it is, folks — a moment New Castle will never forget!" The camera catches sunlight gleaming off the oil mist, then pulls back to reveal the whole scene: a small town silhouetted against the wild fountain of oil.
```

Flowing multi-paragraph single-shot with dialogue (frog yoga studio):

```
A wide shot opens in a warm, sunlit frog yoga studio with a tactile, felt-and-fabric look. Golden morning light pours through tall wooden-framed windows, lush green foliage outside, thin wisps of incense smoke curling through the air; potted plants and wooden shelves line the softly blurred background. One housefly flies lazily around the room, flitting in and out of the smoke. A large green frog instructor sits in lotus position at the center on a woven straw mat, wearing an orange robe, eyes gently closed, a serene half-smile, hands resting on his knees. Behind him, rows of smaller green frogs sit on their own woven mats, throats swelling as they chant a deep, resonant meditative "Om" in unison, a low male vocal drone with rich harmonic overtones, no instruments, no music. Soft pond ambience underneath, and the faint buzz of the fly.

The instructor breathes in slowly, then speaks in a deep, calm voice, drawing out each word. "We are one… with the pond." The frogs answer, chanting in unison: "Om…" He smiles faintly. "We are one… with the mud." Again the frogs chant together, "Om…" a slow beat. "We are one… with the flies." A pause.

The camera pans slowly left to a small frog in the front row, who twitches, eyes darting as the fly drifts past. Suddenly its tongue snaps out, catching the buzzing housefly mid-air and pulling it back into its mouth.

The master exhales slowly, still serene, eyes still closed. "But we do not chase the flies…" Beat. "…not during class."

The guilty frog lowers its head, folding his hands back into a meditative pose as the others resume their deep, resonant chant, throats swelling: "Om…" A lingering shot holds on the guilty frog.
```
