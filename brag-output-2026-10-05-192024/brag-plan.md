# ReproBot brag plan — the UI, end to end

**Direction from the user:** show the real UI as an end-to-end pipeline, with audio, driven
by a real paper; accent colour `#2dd4bf`; use the robot mascot.

**What it is:** ReproBot reads an ML paper, reruns it, and checks the number. The Pipeline
Viewer is where a person does that.
**Paper used:** *A Practical Guide to Support Vector Classification* (Hsu, Chang & Lin),
the one paper in the viewer with every stage finished. Every number on screen is what the
viewer shows for it: 22 claims, target 96.9%, reproduced 96.62%, tolerance ±0.5481%, PASS,
99.3 s, review "faithful".
**How the UI was captured:** the viewer was started locally and driven in a browser: the
PDF was put into the Import uploader, then the Claims, Code, Runner, Critic and Report tabs
were opened and screenshotted at 2x. The video pans, zooms and clicks across those real
screens. No paid button was pressed (OCR, Reader, Coder and the model review call the API)
and nothing was written to the repo.

**Angle:** the mascot hands you the tool, then the tool does six things in a row.
**Tone:** `default`, friendly. **Format:** 1920×1080, 30 fps, 25 s.
**Identity:** dark teal-black `#081110`, accent `#2dd4bf`, Archivo + DM Mono; the UI itself
is shown as it is (Streamlit, light).
**Sound:** original track, 120 BPM, F major. Each UI click lands on a beat; PASS gets the chime.

## Storyboard

| # | Time | Scene | On screen |
|---|---|---|---|
| 0 | 0.0–3.0 | Hook | Mascot waves. "Hand me an ML paper." / "I'll rerun it." |
| 1 | 3.0–6.1 | Import | Viewer title and pipeline-status table, scroll to "Import a paper", click Upload, the PDF appears with "Save + run OCR extraction". |
| 2 | 6.1–9.3 | Reader | Click Claims. The 22-row claims table; row c1 (96.9%) highlighted. |
| 3 | 9.3–12.4 | Coder | Click Code. Target claim and the generated `train.py`, scrolling. |
| 4 | 12.4–15.4 | Runner | Click Runner. "Status: success", 96.62 / 96.90 / −0.28 highlighted. |
| 5 | 15.4–18.9 | Critic | Click Critic. "Verdict: pass", tolerance table, then the model review ("faithful") and its findings. |
| 6 | 18.9–22.1 | Report | Click Report. "Replication report" and its Outcome table. |
| 7 | 22.1–25.0 | Outro | Mascot. "Don't trust the number. Rerun it." Wordmark, repo URL. |

A step counter and a six-chip progress strip (Import → Report) stay above the browser the
whole way; the mascot stands at the corner and hops on each click and on PASS.
