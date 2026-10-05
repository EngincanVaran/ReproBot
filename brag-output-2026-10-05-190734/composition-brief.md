# Hyperframes Composition Brief: ReproBot

## Objective
Create a short launch-style brag video for ReproBot.

## Output
- Composition directory: `brag-output-2026-10-05-190734/composition/`
- Rendered video: `brag-output-2026-10-05-190734/brag.mp4`
- Format: landscape — 1920x1080, 30 fps
- Duration: 22 seconds

## Source Material
- Project root: the ReproBot repo
- Primary files read: `README.md`, `TODO.md`, `CLAUDE.md`, `critic/README.md`, `report/output/2016-05-Wide-Residual-Networks.md` and its `.curve.svg`, `report/output/index.md`, `docs/handouts/generate.py`, `viewer/app.py`
- Product name: ReproBot
- Strongest claim: six papers, six model families, every Critic verdict a pass
- Key visual to recreate: the report's WRN-28-10 learning curve against the paper's 4.00% line (data taken from the report SVG, redrawn on a log axis)
- Copy that must appear verbatim: `./reproduce.sh full`; `implementation_bug`; "81 train_fit, 324 val"; the six claimed/reproduced value pairs from the README table

## Creative Direction
- Tone preset: default. Creative direction: dry lab notebook with a scoreboard.
- Interpretation: quick entrances, long holds, plain statements.
- Angle: papers print a number; ReproBot makes them prove it.
- Hook: "The paper says 4.00%." / "Did anyone rerun it?"
- Outro: "Don't trust the number. Rerun it."
- Avoid: generic SaaS language, abstract filler, any claim not in the repo.

## Visual Identity
- Background `#121211`; text `#F2F1EC`; muted `#807E77`; accent `#4FC79B`; fail `#FF5A6E`
- Display font: Archivo (variable, local woff2). Body/labels: DM Mono (local woff2).
- References: `docs/handouts/` palette and type; the report's curve.

## Storyboard
Use `brag-plan.md` as the contract.
1. Hook — 3.0s — paper table, marker, 4.00%, the question
2. Reveal — 4.0s — wordmark, seven chips, three-part tagline
3. The run — 5.0s — typed command, live curve, PASS stamp
4. The loop — 3.5s — FAIL → Critic finding → PASS
5. The table — 3.5s — six rows, six PASS badges
6. Outro — 3.0s — punchline, wordmark, URL

## Audio
- Audio role: intentional silence. `--no-music --no-sfx`. No `<audio>` element, no cue sync, no audio-reactive treatment.

## Hyperframes Instructions
One root composition, one paused GSAP timeline registered on `window.__timelines`, scenes as timed `.clip` elements, deterministic logic only, local fonts. `hyperframes check` must pass before render. Local render only.

Note: the Hyperframes domain skills are not installed in this session (installing them needs a session restart), so the composition follows the scaffold's `CLAUDE.md` key rules and `hyperframes docs` (data-attributes, compositions, gsap).
