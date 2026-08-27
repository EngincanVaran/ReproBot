"""Generate the ReproBot one-page handouts.

Shared CSS lives here once, so every sheet in the series looks like part of the same
set rather than four pages that drifted. Each handout supplies only its own content.

Two things learned the hard way and encoded here rather than rediscovered per page:

* The bullet marker is absolutely positioned, never a grid or flex sibling. Inside a
  grid li, the <strong> lead and the sentence after it become SEPARATE items and get
  dealt into different cells, wrapping the text one word per line.
* The @media print block is what makes a "one-pager" actually one page. Without it the
  card border, page background and outer padding push the content onto a second sheet.
"""

from pathlib import Path

CSS = """
  :root {
    --stock:#F4F3EF; --card:#FFFFFF; --stock2:#E9E7E1;
    --ink:#101010; --ink2:#4A4A48; --ink3:#85847F; --rule:#D3D1CA;
    --red:#C8102E; --green:#14664B;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --stock:#121211; --card:#1A1A18; --stock2:#232320;
      --ink:#F2F1EC; --ink2:#B4B2AB; --ink3:#807E77; --rule:#33332E;
      --red:#FF5A6E; --green:#4FC79B;
    }
  }
  :root[data-theme="dark"] {
    --stock:#121211; --card:#1A1A18; --stock2:#232320;
    --ink:#F2F1EC; --ink2:#B4B2AB; --ink3:#807E77; --rule:#33332E;
    --red:#FF5A6E; --green:#4FC79B;
  }

  * { box-sizing: border-box; }
  body { margin:0; background:var(--stock); color:var(--ink);
         font-family:Archivo, system-ui, sans-serif; -webkit-font-smoothing:antialiased; }

  .page { max-width:1000px; margin:0 auto; padding:clamp(20px,3vw,36px); }
  .card { background:var(--card); border:1px solid var(--rule); border-radius:8px;
          padding:clamp(22px,3vw,38px); display:flex; flex-direction:column;
          gap:clamp(16px,2vw,24px); }

  .top { display:flex; align-items:flex-start; justify-content:space-between;
         gap:20px; flex-wrap:wrap; padding-bottom:16px; border-bottom:4px solid var(--red); }
  .stagenum { font-family:"DM Mono",monospace; font-size:0.72rem; letter-spacing:0.2em;
              text-transform:uppercase; color:var(--red); font-weight:500; margin:0 0 8px; }
  h1 { font-size:clamp(1.9rem,4.4vw,3rem); font-weight:800; letter-spacing:-0.03em;
       line-height:1; margin:0; }
  h1 .path { font-family:"DM Mono",monospace; font-weight:500; font-size:0.5em;
             color:var(--ink3); letter-spacing:-0.01em; }
  .badge { font-family:"DM Mono",monospace; font-size:0.7rem; letter-spacing:0.09em;
           text-transform:uppercase; font-weight:500; padding:5px 11px; border-radius:3px;
           background:var(--green); color:var(--card); white-space:nowrap; }
  .badge.red { background:var(--red); }
  .badge.grey { background:var(--ink3); }

  .oneliner { font-size:clamp(1.08rem,1.9vw,1.32rem); color:var(--ink); line-height:1.45;
              max-width:62ch; margin:0; font-weight:500; }

  h2 { font-family:"DM Mono",monospace; font-size:0.72rem; letter-spacing:0.15em;
       text-transform:uppercase; color:var(--ink3); font-weight:500; margin:0 0 12px; }

  .flow { display:flex; align-items:stretch; gap:0; flex-wrap:wrap;
          border:1px solid var(--rule); border-radius:6px; overflow:hidden; }
  .flow > div { flex:1 1 160px; padding:14px 16px; background:var(--stock2);
                border-right:1px solid var(--rule); }
  .flow > div:last-child { border-right:none; }
  .flow .lbl { font-family:"DM Mono",monospace; font-size:0.66rem; letter-spacing:0.1em;
               text-transform:uppercase; color:var(--ink3); margin-bottom:6px; }
  .flow .val { font-size:0.98rem; font-weight:600; line-height:1.3; }
  .flow .val small { display:block; font-weight:400; font-size:0.82rem;
                     color:var(--ink2); margin-top:3px; }
  .flow .mid { background:var(--card); }

  /* Marker absolutely positioned - see module docstring. */
  ul.pts { margin:0; padding:0; list-style:none; display:flex; flex-direction:column; gap:11px; }
  ul.pts > li { position:relative; padding-left:27px; font-size:clamp(0.96rem,1.4vw,1.06rem);
                color:var(--ink2); line-height:1.55; }
  ul.pts > li::before { content:"\\25AA"; position:absolute; left:4px; top:0;
                        color:var(--red); font-size:0.9em; line-height:1.55; }
  ul.pts strong { color:var(--ink); font-weight:600; }

  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:14px; }
  .stat { border-top:3px solid var(--ink); padding-top:11px; }
  .stat.red { border-top-color:var(--red); }
  .stat.green { border-top-color:var(--green); }
  .stat .n { font-family:"DM Mono",monospace; font-size:clamp(1.5rem,3vw,2rem); font-weight:500;
             letter-spacing:-0.03em; line-height:1; font-variant-numeric:tabular-nums; }
  .stat .n.red { color:var(--red); }
  .stat .c { font-size:0.85rem; color:var(--ink2); margin-top:6px; line-height:1.35; }

  .example { background:var(--stock2); border-left:4px solid var(--red);
             padding:16px 19px; border-radius:0 5px 5px 0; }
  .example .cap { font-family:"DM Mono",monospace; font-size:0.66rem; letter-spacing:0.1em;
                  text-transform:uppercase; color:var(--ink3); margin-bottom:9px; }
  .example code { font-family:"DM Mono",monospace; font-size:0.85rem; color:var(--ink);
                  line-height:1.6; display:block; white-space:pre-wrap; }
  .example p { margin:10px 0 0; font-size:0.9rem; color:var(--ink2); line-height:1.5; }

  .note { border-left:4px solid var(--ink3); background:var(--stock2);
          padding:14px 18px; border-radius:0 5px 5px 0; }
  .note.red { border-left-color:var(--red); }
  .note p { margin:0; font-size:0.94rem; color:var(--ink2); line-height:1.5; }
  .note p + p { margin-top:9px; }
  .note strong { color:var(--ink); }

  code.inl { font-family:"DM Mono",monospace; font-size:0.87em; background:var(--stock2);
             padding:0.1em 0.36em; border-radius:2px; color:var(--ink); }

  table { border-collapse:collapse; width:100%; font-size:0.88rem; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--rule); }
  th { font-family:"DM Mono",monospace; font-size:0.64rem; letter-spacing:0.09em;
       text-transform:uppercase; color:var(--ink3); font-weight:500; }
  tbody tr:last-child td { border-bottom:none; }
  td { color:var(--ink2); }
  td.k { color:var(--ink); font-weight:600; }
  td.n { font-family:"DM Mono",monospace; font-variant-numeric:tabular-nums; color:var(--ink); }

  footer { display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;
           padding-top:18px; border-top:1px solid var(--rule);
           font-family:"DM Mono",monospace; font-size:0.72rem; color:var(--ink3); }

  .cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr));
          gap:clamp(20px,3vw,34px); }

  /* Print: what makes a one-pager one page. */
  @page { size:A4 portrait; margin:10mm; }
  @media print {
    body { background:#fff; }
    .page { padding:0; max-width:none; }
    .card { background:#fff; border:none; border-radius:0; padding:0; gap:13px; }
    .card > div, .cols > div, .example, .note, .flow { break-inside:avoid; }
    .top { padding-bottom:12px; }
    h1 { font-size:24pt; }
    .oneliner { font-size:11.6pt; }
    ul.pts { gap:8px; }
    ul.pts > li { font-size:10pt; line-height:1.42; }
    .flow .val { font-size:9.8pt; }
    .flow .val small, .example p, .note p, .stat .c, td, th { font-size:8.8pt; }
    .example code { font-size:8.2pt; line-height:1.5; }
    .stat .n { font-size:17pt; }
    footer { font-size:7.6pt; padding-top:12px; }
  }
"""

SHELL = """<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;800&family=DM+Mono:wght@400;500&display=swap">

<style>{css}</style>

<div class="page">
  <div class="card">

    <div class="top">
      <div>
        <p class="stagenum">{kicker}</p>
        <h1>{heading}{path}</h1>
      </div>
      <span class="badge {badgeclass}">{badge}</span>
    </div>

    <p class="oneliner">{oneliner}</p>

{body}

    <footer>
      <span>{footl}</span>
      <span>ReproBot &middot; inzva AI Projects #10</span>
    </footer>

  </div>
</div>
"""


def page(**kw) -> str:
    kw.setdefault("badgeclass", "")
    kw["path"] = f' <span class="path">{kw["path"]}</span>' if kw.get("path") else ""
    return SHELL.format(css=CSS, **kw)


OUT = Path("docs/handouts")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 2. reader
reader_body = """
    <div>
      <h2>What goes in, what comes out</h2>
      <div class="flow">
        <div><div class="lbl">Input</div>
          <div class="val">The paper as text<small>from stage 1</small></div></div>
        <div class="mid"><div class="lbl">What happens</div>
          <div class="val">Five separate readings<small>plus a checker that re-reads them</small></div></div>
        <div><div class="lbl">Output</div>
          <div class="val">One structured file<small>facts a program can look up</small></div></div>
      </div>
    </div>

    <div>
      <h2>The key points</h2>
      <ul class="pts">
        <li><strong>It asks five separate questions, not one big one.</strong> What problem does this paper solve? What model does it build? What results does it claim? What settings did it train with? How was the data prepared?</li>
        <li><strong>Every single fact comes with a receipt.</strong> Each extracted value records where it came from &mdash; &ldquo;Table 1, page 5&rdquo; &mdash; so any number can be traced back and checked.</li>
        <li><strong>It writes down what the paper does <em>not</em> say.</strong> Genuinely the most important part, and the reason is on the right.</li>
        <li><strong>A checker re-reads everything and flags problems.</strong> Whatever it flags gets re-done, with the complaint included so the second attempt addresses it specifically. Up to three rounds.</li>
        <li><strong>Numbers are stored as text.</strong> Sounds odd, but a setting like &ldquo;0.1, dropped by 0.2 at epochs 60, 120, 160&rdquo; is not one number &mdash; and &ldquo;not stated&rdquo; has to stay sayable.</li>
      </ul>
    </div>

    <div class="cols">
      <div>
        <h2>Why point three matters most</h2>
        <div class="example">
          <div class="cap">One of 12 gaps recorded for Network In Network</div>
          <code>"Exact number of feature maps per layer:
the paper says only 'the same number as
in the corresponding maxout network',
deferring the actual numbers elsewhere."</code>
          <p>The next stage has to write real code, and code needs a real number here. If this gap were quietly filled with a plausible guess, the result would train perfectly and report a confident number <em>for the wrong model</em>. Recording the gap keeps that decision visible instead of silent.</p>
        </div>
      </div>
      <div>
        <h2>Across the four papers read</h2>
        <table>
          <thead><tr><th>Paper</th><th>Claims</th><th>Settings</th><th>Gaps</th></tr></thead>
          <tbody>
            <tr><td class="k">Network In Network</td><td class="n">12</td><td class="n">17</td><td class="n">12</td></tr>
            <tr><td class="k">All Convolutional Net</td><td class="n">17</td><td class="n">13</td><td class="n">11</td></tr>
            <tr><td class="k">Deep Residual Learning</td><td class="n">66</td><td class="n">28</td><td class="n">7</td></tr>
            <tr><td class="k">Wide Residual Networks</td><td class="n">61</td><td class="n">20</td><td class="n">11</td></tr>
          </tbody>
        </table>
        <div class="note" style="margin-top:16px">
          <p><strong>Known rough edge.</strong> The checker never fully runs out of things to say &mdash; every paper still has a few open flags after three rounds. Looking at them, most are it raising a worry and then talking itself out of it in the same sentence. Not wrong, but it costs a re-run each time.</p>
        </div>
      </div>
    </div>
"""

(OUT / "02-reader.html").write_text(page(
    title="ReproBot Stage 2: Reader",
    kicker="ReproBot &middot; Stage 2 of 5",
    heading="Understanding the paper", path="reader/",
    badge="Built &middot; working",
    oneliner="Text is still just words. This stage turns the paper into facts a program can look things up in — what it claims, what it built, and what it never bothered to say.",
    body=reader_body,
    footl="Next stage &rarr; <code class=\"inl\">coder/</code> writes the training script",
), encoding="utf-8")


# ---------------------------------------------------------------- 3. coder
coder_body = """
    <div>
      <h2>What goes in, what comes out</h2>
      <div class="flow">
        <div><div class="lbl">Input</div>
          <div class="val">The facts + the paper<small>stage 2's file, and the original text</small></div></div>
        <div class="mid"><div class="lbl">What happens</div>
          <div class="val">One call to Claude<small>write a complete training program</small></div></div>
        <div><div class="lbl">Output</div>
          <div class="val">A runnable script<small>~340 lines, plus a launcher</small></div></div>
      </div>
    </div>

    <div>
      <h2>The key points</h2>
      <ul class="pts">
        <li><strong>It writes a whole working program in one go.</strong> Roughly 340 lines &mdash; the model, the data loading, the training loop, and the code that reports the final score.</li>
        <li><strong>It is told exactly which source to trust.</strong> The extracted facts win first; the paper's own text fills gaps; the model's own memory of a famous architecture comes <em>last</em> &mdash; and whenever it uses that memory, it has to say so in writing.</li>
        <li><strong>That last rule is the whole trick.</strong> You cannot tell a program &ldquo;never guess&rdquo; &mdash; something has to be written where a missing number goes. So the rule is <em>guess visibly</em>: every guess is listed in the output.</li>
        <li><strong>Two free checks before the script is accepted.</strong> Is it valid Python at all, and does it accept the settings the next stage will pass it? Both cost nothing and catch broken output early.</li>
        <li><strong>It also writes a small launcher script</strong> so the next stage can run it without knowing anything about how it works.</li>
      </ul>
    </div>

    <div class="cols">
      <div>
        <h2>How we know it isn't just reciting</h2>
        <div class="example">
          <div class="cap">The test we chose, and why</div>
          <p style="margin-top:0">We tested it on <strong>Network In Network</strong> rather than a famous architecture &mdash; on purpose. A well-known model would be rebuilt correctly from memory alone, proving nothing. This paper's whole contribution is an unusual layer, its numbers appear nowhere in the text, and its equations include two that belong to <em>other</em> methods it argues against.</p>
          <p>The script it produced built the unusual layer correctly, skipped both decoy equations by name, and kept the paper's distinctive ending &mdash; no standard classifier layer, which every ordinary image model has. It was following the extraction, not its own memory.</p>
        </div>
      </div>
      <div>
        <h2>The Network In Network run</h2>
        <div class="stats">
          <div class="stat green"><div class="n">343</div><div class="c">lines of working code</div></div>
          <div class="stat green"><div class="n">12</div><div class="c">guesses written down</div></div>
          <div class="stat green"><div class="n">0</div><div class="c">standard classifier layers &mdash; correct for this paper</div></div>
        </div>
        <div class="note red" style="margin-top:16px">
          <p><strong>What the checks cannot do.</strong> They confirm the script is <em>well-formed</em>, never that it <em>works</em>. Two real bugs have slipped past: code that was perfectly valid but crashed the moment training started, and a script whose own notes described a slightly different model than the one it actually built.</p>
          <p>That is exactly why the next stage exists &mdash; the only way to find those is to run it.</p>
        </div>
      </div>
    </div>
"""

(OUT / "03-coder.html").write_text(page(
    title="ReproBot Stage 3: Coder",
    kicker="ReproBot &middot; Stage 3 of 5",
    heading="Writing the code", path="coder/",
    badge="Built &middot; working",
    oneliner="This is the stage that does the actual reimplementation — turning a description of a model into a program that trains it.",
    body=coder_body,
    footl="Next stage &rarr; <code class=\"inl\">runner/</code> actually runs it",
), encoding="utf-8")


# ---------------------------------------------------------------- 4. runner
runner_body = """
    <div>
      <h2>What goes in, what comes out</h2>
      <div class="flow">
        <div><div class="lbl">Input</div>
          <div class="val">The generated script<small>from stage 3</small></div></div>
        <div class="mid"><div class="lbl">What happens</div>
          <div class="val">Runs inside a sealed container<small>cheap checks first, then longer ones</small></div></div>
        <div><div class="lbl">Output</div>
          <div class="val">Scores, logs, a verdict<small>and, on failure, a diagnosis</small></div></div>
      </div>
    </div>

    <div>
      <h2>The key points</h2>
      <ul class="pts">
        <li><strong>Nothing runs on the real computer.</strong> The script is executed inside a sealed container &mdash; code written by an AI, running unsupervised, should not have access to anything it can damage.</li>
        <li><strong>It starts cheap and escalates.</strong> First a few seconds to check the thing runs at all, then one short pass, then a longer one. A broken script fails in seconds instead of hours.</li>
        <li><strong>It knows nothing about the script it runs.</strong> It only ever says &ldquo;run this, at this size&rdquo;. That means a completely different paper needs no changes here at all.</li>
        <li><strong>When something fails, one cheap call sorts out whose fault it is</strong> &mdash; a bug in the generated code, or a problem with the container. The two need opposite responses, so the distinction matters.</li>
        <li><strong>When it succeeds, nothing is asked.</strong> A successful run needs no explanation, so no extra call is made &mdash; the common path costs nothing.</li>
      </ul>
    </div>

    <div class="cols">
      <div>
        <h2>What comes back</h2>
        <div class="example">
          <div class="cap">A real run &mdash; Network In Network</div>
          <code>[probe] PASSED in 32.2s
[smoke] PASSED in 62.1s
        Test Error = 86.7%
[loop]  VERDICT: success</code>
          <p><strong>That 86.7% is not a failure.</strong> The paper claims 10.41%, but this run trained for one pass over a few hundred images &mdash; a few seconds of work. It answers &ldquo;does this run at all&rdquo;, never &ldquo;is the number right&rdquo;. Getting a real number needs a real training run, which is the problem below.</p>
        </div>
      </div>
      <div>
        <h2>Measured</h2>
        <div class="stats">
          <div class="stat green"><div class="n">2</div><div class="c">papers run successfully</div></div>
          <div class="stat"><div class="n">1.7 GB</div><div class="c">the container image</div></div>
          <div class="stat red"><div class="n red">22 days</div><div class="c">for one real training run</div></div>
        </div>
        <div class="note red" style="margin-top:16px">
          <p><strong>The blocker, in one number.</strong> On this laptop's processor, training one of these models properly &mdash; the full run the paper actually did &mdash; would take about <strong>22 days</strong>. For one paper, for one claimed result.</p>
          <p>A graphics card would cut that to hours. Until then, every run is a short check that the machinery works, not a real reproduction.</p>
        </div>
      </div>
    </div>
"""

(OUT / "04-runner.html").write_text(page(
    title="ReproBot Stage 4: Runner",
    kicker="ReproBot &middot; Stage 4 of 5",
    heading="Running it safely", path="runner/",
    badge="Built &middot; working",
    oneliner="Code written by an AI has to actually run before anyone can believe it. This stage executes it in a sealed box and reports what happened.",
    body=runner_body,
    footl="Next &rarr; the missing piece: comparing the result to the paper",
), encoding="utf-8")


# ---------------------------------------------------------------- 5. summary
summary_body = """
    <div>
      <h2>Then and now</h2>
      <div class="flow">
        <div><div class="lbl">July &mdash; first report</div>
          <div class="val">A plan<small>research, design, one prototype. Nothing ran.</small></div></div>
        <div class="mid"><div class="lbl">August &mdash; second report</div>
          <div class="val">A working pipeline<small>5 stages, 2 papers carried all the way through</small></div></div>
        <div><div class="lbl">Still missing</div>
          <div class="val">The verdict<small>nothing checks whether the number is right</small></div></div>
      </div>
    </div>

    <div class="cols">
      <div>
        <h2>What now exists</h2>
        <ul class="pts">
          <li><strong>A paper goes in, a running program comes out.</strong> Two papers have made the whole journey &mdash; PDF, to understanding, to code, to a real training run.</li>
          <li><strong>It fixes its own mistakes.</strong> We deliberately broke a generated script; the system ran it, saw the crash, worked out what was wrong, rewrote that part, and got it working &mdash; first try.</li>
          <li><strong>It follows the paper, not its own memory.</strong> Shown on a paper whose method is unusual enough that memory would have produced something else entirely.</li>
        </ul>
      </div>
      <div>
        <h2>What is still missing</h2>
        <ul class="pts">
          <li><strong>The comparison.</strong> Nothing yet takes the number we produced and checks it against the number the paper claimed. That check is the entire point of the project, and it does not exist yet.</li>
          <li><strong>The written report</strong> that would come out at the end.</li>
          <li><strong>Enough computing power</strong> to make any of the numbers meaningful.</li>
        </ul>
      </div>
    </div>

    <div class="note red">
      <p><strong>The one thing to understand.</strong> Everything built so far proves the machinery works. None of it yet proves a paper was <em>successfully reproduced</em> &mdash; because the step that would check that has not been built, and because the runs so far are short tests rather than real training.</p>
      <p>Saying &ldquo;ReproBot reproduces papers&rdquo; today would be wrong. Saying &ldquo;ReproBot reads a paper, writes the code, and runs it&rdquo; is exactly right.</p>
    </div>

    <div class="cols">
      <div>
        <h2>Where the papers got to</h2>
        <table>
          <thead><tr><th>Step</th><th>Papers</th></tr></thead>
          <tbody>
            <tr><td class="k">Collected</td><td class="n">8</td></tr>
            <tr><td class="k">Converted to text</td><td class="n">6</td></tr>
            <tr><td class="k">Understood</td><td class="n">4</td></tr>
            <tr><td class="k">Code written</td><td class="n">2</td></tr>
            <tr><td class="k">Actually run</td><td class="n">2</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <h2>What would unblock it</h2>
        <div class="stats">
          <div class="stat red"><div class="n red">GPU</div><div class="c">a graphics card turns 22 days into hours &mdash; the single biggest blocker</div></div>
          <div class="stat"><div class="n">1</div><div class="c">component left to build: the comparison step</div></div>
        </div>
        <div class="note" style="margin-top:16px">
          <p>The comparison step can be written and tested now. It only becomes <em>believable</em> once there is enough computing power behind it to produce a real number to compare.</p>
        </div>
      </div>
    </div>
"""

(OUT / "05-summary.html").write_text(page(
    title="ReproBot Where It Stands",
    kicker="ReproBot &middot; Summary",
    heading="Where it stands", path="",
    badge="3 of 4 agents working", badgeclass="grey",
    oneliner="ReproBot reads a machine learning paper, writes the code to reproduce it, and runs that code. What it cannot yet do is tell you whether the result matched.",
    body=summary_body,
    footl="Full detail &rarr; <code class=\"inl\">TODO.md</code> in the repository",
), encoding="utf-8")

print("wrote 4 handouts to", OUT)
