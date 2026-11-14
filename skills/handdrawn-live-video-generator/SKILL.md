---
name: handdrawn-live-video-generator
description: |
  For creators making surreal short videos that blend rough glowing hand-drawn animation with live-action spaces. Users provide a scene idea, contact object or hand, desired mood, and optional language or style constraints. The Skill clarifies the physical contact, designs continuous morphing, escape route, and delayed handheld chase movement, then writes a reusable 15-second 16:9 video prompt in the user's language. After user confirmation it recommends MiniMax H3 generation and checks contact realism, camera delay, rough glowing stroke texture, and non-horror tone. Best for single-scene creative clips, not polished CG, horror jump scares, plush characters, or multi-scene cuts.
compatibility: Requires the MiniMax Hub agent (canvas workspace and MiniMax H3 generation); not portable to generic agent harnesses.
trigger-words: [手绘发光动画实拍融合, 15秒变形追逐视频提示词, Seedance视频prompt, H3生成视频, 视频prompt, 手绘动画接触真实物体, 蜡笔粉笔质感, 多语言视频提示词]
---

# Handdrawn Live-Action Fusion Video Generator

Use this Skill when the user wants a **finished 15-second, 16:9 live-action-and-hand-drawn fusion video**. The output must preserve the structure: a flat hand-drawn luminous animation appears in a real space, clearly contacts live-action hands or objects in the first 0-3 seconds, continuously morphs as one single entity, escapes, and a handheld phone camera follows slightly late.

This Skill **organizes the prompt in the user input language and recommends MiniMax H3 as the confirmed generation step**. Do not generate video until the user explicitly confirms H3 generation. Do not route to planner or executor for prompt-writing. The final prompt language must follow the dominant language of the user input: Chinese input produces Chinese, English input produces English, Japanese input produces Japanese; for mixed input, use the dominant language; when unclear, use the current conversation language. Only user-required proper nouns, model names, or literal parameters may remain unchanged.

## Step 1: Understand the user intent as same-language constraints

When the user provides any language, understand it as the following workflow requirements and express the final prompt in the dominant language of the user input:

- Create a brand-new 15-second video-generation prompt based on the reference prompt; do not imitate the original sentences on the surface.
- The visual structure must be preserved: a flat hand-drawn luminous animation appears in a live-action space; the animation contacts a real hand or real object; the same being continuously morphs and escapes; the camera always chases half a beat late.
- The hand-drawn animation texture must feel like crayon, chalk, colored pencil, pastel, or a rough brush; lines tremble slightly, with uneven smearing, ragged edges, and a frame-by-frame redrawn feel.
- Prohibit 3DCG, plush-toy feel, uniform vector lines, smooth neon, horror monsters, oversized eyes, slit mouths, teeth, intimidation, lunging bites, sudden black screens, and jump scares.
- 0-3 seconds must show clear contact between the live-action hand and the hand-drawn animation, for example coiling around a finger, landing in the palm, escaping when grabbed, or being born from a fingertip.
- The animation must morph continuously as one and the same entity, cycling among forms such as "lines, creatures, marks, plants, vehicles, everyday small objects", while retaining traces of the previous form.
- No completely new second character may appear out of nowhere.
- The whole film unfolds continuously within the same space or an adjacent range, without a cut jumping to another location; it should feel as if the shooter is really walking and chasing.
- Every interval — 0-3, 3-6, 6-10, 10-13, 13-15 seconds — must contain a new transformation, movement, contact, discovery, prank, or surprise.
- The shooter must also take part: reaching out, grabbing, chasing, opening a door or box, catching, backing away, being pranked, and so on.
- The tone is cute, slice-of-life, nostalgic, gentle, with a faint bittersweet edge — not horror comedy.
- 13-15 seconds must deliver a space-scale transformation: the earlier lines spread across the wall, floor, ceiling, windows, sink, or corridor, becoming giant flowers, a starry sky, sunset, clouds, ribbons, a doodled town, and so on; the ending must carry an emotional afterglow plus a small cute punchline.

## Step 2: Invent all creative content fresh

For every run, create a new combination. Do not reuse the reference prompt's dark room, PC, fridge, stars, hearts, blue vortex, butterfly, snake, octopus, hamburger, giant eye, teeth, or dark ending.

Choose new values for all of these:

1. Live-action space, but still an everyday adjacent range that can be tracked continuously;
2. Hand-drawn entity or initial form;
3. Central color;
4. Continuous morphing chain;
5. Way of contacting a real hand or real object;
6. Chase route through the space;
7. Shooter's reaction;
8. Final space-scale transformation and cute punchline.

Good example directions include: rainy-day kitchen sink, old balcony drying corner, early-morning entryway, small bookstore corridor, small table by a train window, bathroom mirror cabinet, craft table, greenhouse passage, laundromat bench, old dining table. Use only one space or adjacent connected areas.

## Step 3: Required output format

The final prompt must begin with a sentence pattern matching the dominant language of the user input. For Chinese input, use this pattern:

`15秒，16:9横版视频。将实拍的〇〇与手绘发光动画融合的影像。`

Replace the circled placeholder with the newly invented live-action space or everyday scene. For non-Chinese input, use an equivalent opening in the same language and include duration, 16:9 landscape format, live-action space, and hand-drawn glowing animation fusion.

Do not add titles, auxiliary headings such as "stage setting" or "color tone", or explanatory notes. Keep this paragraph order:

1. Opening sentence;
2. Live-action space and phone-camera texture;
3. 0-3 seconds;
4. 3-6 seconds;
5. 6-10 seconds;
6. 10-13 seconds;
7. 13-15 seconds;
8. Hand-drawn texture;
9. Camera chase style;
10. Prohibitions;
11. Ambient sound.

## Step 4: Prompt writing rules

- Keep the prompt executable as a video-generation prompt, not an essay.
- Unless the user asks for explanation, the final answer should contain the prompt text in the dominant language of the user input first, followed by one short next-step recommendation in the same language.
- The recommendation must invite the user, in the same language, to use MiniMax H3 to generate a 15-second 16:9 video from this prompt. Chinese example: `下一步建议：如果你确认这个 prompt，我可以继续用 H3 模型生成 15 秒 16:9 视频。`
- Do not generate a video, image, audio, storyboard, or intermediate asset until the user explicitly confirms the H3 generation step.
- Mention Seedance only as a target usage context when the user asks; do not add model parameters inside the creative prompt unless requested.
- The first 0-3 seconds must make real/hand-drawn fusion obvious through contact.
- The camera must not neatly center the animation. It should lag half a beat, panning, tilting, or advancing only after the entity has already left the frame edge.
- The entity should remain traceable: each new form preserves a line, tail, color smear, body curve, or motif from the previous form.
- Use soft, emotional, comic, cute beats: tiny stumbles, shy gestures, a dot falling behind, a petal sticking to lens, a star dropping into a palm, confetti sneeze, one small creature lagging behind.
- Avoid any horror-coded anatomy or threatening motion.
- The final prompt must not randomly mix in vocabulary, transition sentences, or writing systems from outside the user input language; when the user input language is Japanese, natural Japanese is allowed.

## Step 5: Delivery

Output the final prompt in the dominant language of the user input first. After the prompt, add exactly one short same-language recommendation line inviting the user to continue with MiniMax H3 video generation. Do not include a title, checklist, model call summary, filename, or canvas delivery note. If the user later confirms H3 generation, generate the 15-second 16:9 video with MiniMax H3 and check that the result preserves contact, continuous morphing, delayed camera chase, and non-horror hand-drawn texture.
