# Style router

First disclosure layer for caption writing. Route the music brief to one
primary style family; add at most one secondary family for an explicit fusion.
Then request [style-vocabulary.md](style-vocabulary.md) through
`load_references` and use only the routed sections.

Route with natural-language reasoning and these local files only. Do not read
every vocabulary section, rebuild a global catalog, or scan files outside this
skill.

## Routing contract

1. Normalize genre names and cultural-market terms with the alias table below.
2. Choose the primary family from the main genre, never from a generic mood
   adjective.
3. Add one secondary family only for an explicit fusion or a clearly requested
   contrasting palette.
4. Open at most two vocabulary sections. Keep a third style as wording inside a
   section, not a third route.
5. Treat `ballad`, `emotional`, `epic`, `modern`, `dark`, `cinematic`, `氛围`,
   `燃`, and `炸` as modifiers unless the input supplies stronger genre, groove,
   or instrumentation evidence.
6. Prefer wording that already expresses both styles of a fusion before mixing
   two sections.

## Family map

| Route | Positive cues | Disambiguation |
|---|---|---|
| `east-asian-modern` | Mandopop, C-pop, Cantopop, J-pop with electronic, R&B, hip-hop, dance, funk, rock, or metal production | Acoustic, orchestral, traditional, or conventional ballad writing → `east-asian-ballad-heritage` |
| `east-asian-ballad-heritage` | Mandopop/C-pop/Cantopop/J-pop ballad, guofeng pop, acoustic or orchestral East Asian pop | Traditional instruments central rather than decorative → add `roots-traditional-global` as secondary |
| `modern-rnb-neo-soul` | Contemporary R&B, alternative R&B, neo-soul, trap soul, atmospheric R&B | Classic soul → `soul-blues-gospel`; lo-fi beat central → `hip-hop-rap` as secondary |
| `soul-blues-gospel` | Soul, blues, blues rock, gospel, worship, soul-blues | Jazz-blues follows the user's stated groove |
| `cinematic-pop-ballad` | Cinematic pop, cinematic ballad, orchestral pop, soundtrack-like vocal ballad | Score, trailer, orchestra, or choir as main identity → `cinematic-orchestral-epic` |
| `cinematic-orchestral-epic` | Film score, orchestral, trailer, epic choral, symphonic soundtrack, contemporary classical | `Cinematic` alone is a modifier, not enough evidence |
| `electronic-synth-ambient-pop` | Synth-pop, electropop, dream pop, ambient pop, darkwave, retrowave, downtempo | Drops, club grooves, house, trance, or festival energy → `club-edm-house-trance` |
| `jazz-swing-big-band` | Vocal jazz, jazz ballad, big band, swing, bossa nova, lounge jazz | Crooner pop without strong jazz ensemble cues → `traditional-vocal-stage` |
| `traditional-vocal-stage` | Traditional pop, crooner, doo-wop, a cappella, musical theatre, cabaret | Dominant rhythm section or big-band language → `jazz-swing-big-band` |
| `hip-hop-rap` | Hip-hop, rap, trap, drill, lo-fi hip-hop, conscious and melodic rap | R&B singing over trap drums → `modern-rnb-neo-soul` primary, `hip-hop-rap` secondary |
| `metal-heavy-rock` | Metalcore, power metal, symphonic metal, nu-metal, hard rock, post-hardcore | Pop/alternative rock without heavy-metal technique → `pop-alternative-rock` |
| `pop-alternative-rock` | Pop rock, alternative rock, indie rock, arena rock, J-rock, punk, post-grunge | Country rock, blues rock, folk rock → their roots family when that identity is primary |
| `contemporary-folk-acoustic` | Indie folk, contemporary folk, folk pop, singer-songwriter, modern acoustic pop | Heritage, regional, maritime, Celtic, traditional folk → `roots-traditional-global` |
| `roots-traditional-global` | Traditional folk, Celtic, Chinese traditional, folk blues, reggae, maritime, global fusion | Pop songwriting primary → the East Asian pop family instead |
| `dance-pop-disco-funk` | Dance-pop, nu-disco, funk-pop, disco revival, groove-led pop | House, trance, hardstyle, festival drops → `club-edm-house-trance` |
| `club-edm-house-trance` | EDM, house, trance, hardstyle, dubstep, techno, festival electronic | Electronic pop without club/drop structure → `electronic-synth-ambient-pop` |
| `country-americana` | Country, Americana, bluegrass, country rock, country pop, rockabilly | Folk-country follows the user's primary label; add `contemporary-folk-acoustic` only when needed |
| `general-pop-ballad` | Pop, contemporary pop, pop ballad, broadly described emotional songs | Fallback only when no more specific family is supported |

## Common aliases

| User wording | Normalize toward |
|---|---|
| 华语流行、国语流行、中文流行 | Mandopop / C-pop |
| 粤语流行 | Cantopop |
| 国风、古风、中国风 | guofeng → `east-asian-ballad-heritage`; pop production leads → `east-asian-modern` |
| 中文说唱、说唱、Trap 说唱 | `hip-hop-rap` |
| 电子舞曲、蹦迪、打碟 | `club-edm-house-trance` |
| 复古电子 | synthwave/retrowave → `electronic-synth-ambient-pop`; disco/house cues → `dance-pop-disco-funk` or `club-edm-house-trance` |
| 氛围、ambient、环境音 | `electronic-synth-ambient-pop` |
| 电影感、史诗感、配乐感 | Modifier; `cinematic-orchestral-epic` only with score/orchestra/choir evidence |
| 燃、炸、强烈 | Energy cues, never genre evidence by themselves |
| 摇滚 | `pop-alternative-rock`; heavy technique → `metal-heavy-rock` |
| 民谣、小清新 | `contemporary-folk-acoustic`; heritage/regional → `roots-traditional-global` |
| 爵士、摇摆 | `jazz-swing-big-band` |
| R&B、节奏布鲁斯 | `modern-rnb-neo-soul`; classic soul → `soul-blues-gospel` |
| 灵魂乐、福音 | `soul-blues-gospel` |
| 乡村、乡村摇滚 | `country-americana` |
| 音乐剧、美声、阿卡贝拉 | `traditional-vocal-stage` |

Normalize spelling variants such as `Hip Hop`/`Hip-Hop`, `Dance Pop`/`Dance-Pop`,
`Synth Pop`/`Synth-Pop`, and `R&B`/`R'n'B` before routing.

## Fusion rules

- Interpret `X with Y influences` as primary `X`, secondary `Y`.
- Interpret an ordered form such as `X / Y` the same way unless the user gives
  both equal weight.
- Prefer vocabulary that already expresses both styles before combining two
  sections.
- Never let a secondary family overwrite explicit genre, tempo, vocal,
  instrument, or exclusion constraints.
- A clash (e.g. classical strings + hardcore metal) must be written as
  **time-ordered evolution** or fixed by **repetition reinforcement**, per
  [caption-and-lyrics.md](caption-and-lyrics.md) — not as one blended style.

Examples:

- 华语流行 + trap R&B → primary `east-asian-modern`, secondary `hip-hop-rap`
  or `modern-rnb-neo-soul` depending on whether singing or beat language
  dominates.
- 民谣 + 电影感 → primary `contemporary-folk-acoustic`, secondary
  `cinematic-pop-ballad` for a song; `cinematic-orchestral-epic` only for a
  score.
- 金属 + 中国乐器 → primary `metal-heavy-rock`, secondary
  `roots-traditional-global`, written as an evolution across sections.

## Fallback routing

When the user gives no genre:

1. Use groove evidence: swing, trap, four-on-the-floor, breakbeat, acoustic
   strumming.
2. Then core instrumentation and vocal delivery.
3. Then cultural or market context.
4. If only mood or imagery remains, use `general-pop-ballad` and keep the
   result conservative.

Do not route from mood alone when stronger musical evidence exists.

## Route the result to the right field

After routing, split the wording by ACE-Step input:

- Style, mood, instruments, timbre, era, production, qualitative tempo/groove
  → **caption**
- Numeric BPM, key, meter, duration → **metadata parameters**, never caption
- Section order, delivery, energy changes, solos, start/end → **lyrics
  structure marks**
- Timbre and mix already pinned by `reference_audio` → leave to audio; spend
  the caption on style, mood, and arrangement
