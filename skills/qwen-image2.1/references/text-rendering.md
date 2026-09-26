# Rendering readable text inside the image

Legible in-image text is Qwen-Image-2.1's signature strength (Chinese strongest, English solid, Japanese fragile). Prompting it correctly is mostly about quoting discipline.

## The one rule that matters most

**Every string that must appear in the image goes in double quotes, exactly as it should render.**

```text
A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement
```

Measured on this model family: ~65% accuracy unquoted → ~85% quoted → ~96% quoted with raised CFG and more steps. Never paraphrase, translate, or "clean up" a user's string — the model reproduces case, punctuation, and line breaks as written.

## Specify the presentation

Font class, weight, color, size, and medium all respond:

- `a bold black headline across the top reads "SPRING SALE"`
- `a chalkboard sign reading "Qwen Coffee $2 per cup"`
- `a neon light displaying "OPEN 24 HOURS"`
- embroidered, engraved, graffiti, hand-lettered calligraphy — the rendering medium is part of the prompt.

## Multiple text blocks

Describe each block separately, in reading order, each with its own position:

```text
A movie poster. The first row is the movie title, which reads "Imagination Unleashed".
The second row is the movie subtitle, which reads "Enter a world beyond your
imagination". The third row reads "Cast: Qwen-Image". … At the bottom edge, the text
"Launching in the Cloud, August 2025" appears in bold, modern sans-serif font.
```

## Layouts, posters, infographics

Use the long observational style, and spell out *everything* readable: chart axes, tick values, legends, table cell values, step labels. A numbered infographic example from the official ComfyUI post:

```text
Clean flat vector infographic titled "FROM CHERRY TO CUP" showing five numbered
steps left to right
```

(expand each step's label and caption in quotes inside the full prompt).

## Anti-patterns

- **Implicit text** ("make a list", "show a menu") — fails; supply the literal strings.
- **Cluttered numerics** — `"#25 Jan 2026"` garbles; simplify to `"25 Jan 2026"`. Slash dates (`"9/28"`) and digit/symbol mixes are similarly fragile — still commit the user's string verbatim, but if it garbles, switch to the edit-tracing route below instead of re-rolling the whole image.
- **Invented legible signage** — if far background text should not be read, write it as "blurred, indistinct, too small to read". About a third of good images contain no text at all; don't fabricate signs.
- **Mixed scripts in one string** — keep rendered text monolingual.

## The closing clause

End text-bearing prompts with:

```text
No other text appears in the image.
```

This measurably suppresses stray gibberish text elsewhere in the frame.

## Japanese text: the important caveat

Direct text-to-image generation of Japanese is the weakest script (Chinese ≫ English > Japanese) — kanji lose strokes, kana distort. The established workaround, validated on Qwen-Image-Edit-2509 and still the recommended route:

1. Generate the exact Japanese text as a simple image (black text on white).
2. Feed it to an edit pass with an instruction like:
   `画像に書かれたテキストを忠実に再現してください。1画ごとの配置を崩さず、抜け漏れがないようにしてください` — i.e. "reproduce the text from the image faithfully, tracing every stroke, with no omissions", then place it (e.g. "place it inside the picture frame in <image1>, in a calligraphy style").

The edit path traces input text far more faithfully than the generation path invents it. Chinese text, by contrast, can be demanded directly with high fidelity.

For a poster mixing English and Japanese: try direct generation once with the Japanese string quoted (short strings sometimes survive); if it garbles, regenerate the base with the caption area described as reserved empty space ("a clean empty band beneath the headline, reserved for a caption"), then insert the Japanese via the edit-tracing route above.

## If text involves hands or the result garbles

- Raise CFG (true_cfg 6–8) and steps (35–50).
- Fix hand artifacts with a negative (`extra fingers, deformed hands`) plus an affirmative ("natural hand posture, five fingers") — reported to raise hand plausibility from ~60% to ~85%.
- Retry with simplified punctuation; if still garbled, fall back to the edit-tracing route above.
