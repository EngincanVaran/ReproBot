# Brag Plan: ReproBot

Run: `/brag --full --no-music --no-sfx` (silent video, composed in Hyperframes).

## Rubric
1. **What is the app?** A multi-agent pipeline that reads an ML paper PDF, writes and runs the training script, and checks the reproduced number against the paper's claim.
2. **Most impressive claim:** six papers across six model families, every Critic verdict a pass; WRN-28-10 on CIFAR-10 lands at 3.83% against the paper's 4.00%.
3. **Visual hook:** a paper's results table with one number highlighted, next to that number set huge.
4. **Real UI/material to show:** the real 200-epoch WRN learning curve from `report/output/`, the Critic's review of the planted-bug Wijaya run, the results table from `README.md`/`TODO.md`.
5. **Shortest satisfying video:** 22 s. The curve needs room to draw and the table needs a read.
6. **Tone:** preset `default`; direction "dry lab notebook with a scoreboard".
7. **Audio:** none. Disabled by the user (`--no-music --no-sfx`).
8. **Share caption:** see below.
9. **User flow:** paper PDF in → script written and run in Docker, curve watched live → Critic verdict against the claim (and, on a miss, a cited diagnosis and a fixed rerun).

## What is this app?
ReproBot reads an ML paper, reruns it, and checks the number.

## The angle
Papers print a number; ReproBot makes them prove it. Everything on screen is a real result from the repo.

## Hook (first 2-3 seconds)
The Wide Residual Networks results table, the WRN-28-10 row swept by a marker. "The paper says 4.00%." Then: "Did anyone rerun it?"

## Key moments
- The real WRN-28-10 test-error curve drawing toward the dashed "paper 4.00%" line, the counter landing on 3.83%, a PASS stamp.
- The planted bug: attempt 1 RMSE 4.76 FAIL, the Critic's finding (train and validation splits swapped, log line, lines 178–180, 12/12 findings verified), attempt 2 RMSE 3.40 PASS, 11 minutes.
- Six rows of "paper says" against "ReproBot", each getting a PASS badge.

## Outro / punchline
"Don't trust the number. Rerun it." Wordmark, repo URL, inzva AI Projects #10.

## User flow worth showing
`./reproduce.sh full` typed in a terminal → curve drawing with live epoch and check-up status → Critic PASS.

## Tone
- Preset: default
- Creative direction: dry lab notebook with a scoreboard
- Interpretation: confident and plain; motion is quick, holds are generous, no jokes the project did not earn. With no audio, every beat has to read on motion alone.

## Format: landscape — 1920x1080
## Duration: 22 seconds

## Visual identity (from `docs/handouts/generate.py`)
- Background: `#121211`
- Accent: `#4FC79B` (pass), `#FF5A6E` (fail)
- Text: `#F2F1EC`, muted `#807E77`
- Display font: Archivo (wide, heavy)
- Body font: DM Mono for labels and numbers
- Strongest visual element: the learning curve against the paper's dashed line

## Share copy (draft)
The paper says 4.00%. ReproBot read the PDF, wrote the training script, ran it in Docker and got 3.83%.

## Audio direction
- Role: intentional silence (user disabled music and SFX)
- Music: none. Music treatment: none. Music cue guidance: unavailable. Audio-reactive treatment: none.
- SFX posture: none
- Restraint rule: no audio track at all in the render.

## Storyboard

### Scene 1 — Hook — 3.0s
Paper card (title, authors, four table rows) tilts in. Marker sweeps the WRN-28-10 / 4.00 row. "THE PAPER SAYS" / "4.00%" / "Did anyone rerun it?"
Sequential/interaction: marker sweep, then the two lines one after the other.
Audio intent: none. Transition mood: clean → Scene 2

### Scene 2 — Reveal — 4.0s
ReproBot wordmark. Seven pipeline chips (PDF, OCR, Reader, Coder, Runner, Critic, Report) light left to right. "Reads an ML paper." "Reruns it." "Checks the number." arrive in step with the chips; full line holds 1.5 s after the last part.
Sequential/interaction: chips one by one; tagline in three parts.
Audio intent: none. Transition mood: clean → Scene 3

### Scene 3 — The run — 5.0s
"Writes the training script. Runs it in Docker." Terminal types `./reproduce.sh full`, log lines appear, epoch counts 1→200. Chart draws the real curve; counter falls to 3.83%; PASS stamp with "3.83% vs paper 4.00%". Stamp holds 1.1 s.
Sequential/interaction: simulated typing; live counter; curve draw.
Audio intent: none. Transition mood: clean → Scene 4

### Scene 4 — The loop — 3.5s
"We planted a bug. The Critic found it." Attempt 1 card (4.76, FAIL) → Critic review card → Attempt 2 card (3.40, PASS). Footer: "fail → diagnosis → fix → pass · 11 minutes".
Sequential/interaction: three cards left to right, badges pop.
Audio intent: none. Transition mood: clean → Scene 5

### Scene 5 — The table — 3.5s
"6 papers. 6 model families. All pass." Six rows arrive 0.22 s apart, a PASS badge popping on each; the full table then holds 1.6 s.
Sequential/interaction: rows one by one (fast reveal, then held as a set).
Audio intent: none. Transition mood: clean → Scene 6

### Scene 6 — Outro — 3.0s
"Don't trust the number." then "Rerun it." in green, then wordmark and `github.com/EngincanVaran/ReproBot · inzva AI Projects #10`.
Sequential/interaction: three arrivals. Audio intent: none.

Durations: 3.0 + 4.0 + 5.0 + 3.5 + 3.5 + 3.0 = 22.0 s.

**Music mood for this video:** none (disabled)
**Audio summary:** silent by request.
