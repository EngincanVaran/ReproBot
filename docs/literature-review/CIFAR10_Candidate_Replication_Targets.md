# CIFAR-10 Candidate Replication Targets

> inzva AI Projects #10
> Companion note to the main literature-review stack. The four documents in `docs/literature-review/` (`PaperAgent_LiteratureReview.md`, `Paper_Summaries.md`, `Introduction_and_Literature_Review.md`, `Evaluation_Metrics_Comparison.md`) analyze **agent-framework papers** — systems ReproBot is positioned against. This document is different in kind: it is a shortlist of actual **image-classification papers on CIFAR-10** that ReproBot could target for replication, per the pilot/eval scope described in `docs/project-plan/ReproBot_Project_Plan.md` (§0.3, §5) — canonical architectures, a single clearly-tabulated headline metric, single-GPU/short-wall-clock training, no exotic infra. The 8 PDFs are downloaded into `dataset/` (kept separate from `papers/`, which holds the 9 agent-framework papers the main literature-review stack analyzes) as a curation pass to seed the pilot set (3–5 papers first, per the plan), not a final commitment.

## Selection criteria (from the project plan)

- **Canonical architecture families only:** ResNets, plain/all-convolutional CNNs, Wide ResNets, DenseNets, small Vision Transformers — no custom CUDA kernels, no multi-GPU/distributed training.
- **Single, clearly-tabulated headline claim:** e.g. "Table X: test error on CIFAR-10" — not a paper whose main claims are qualitative or spread across dozens of ablations.
- **Cheap to train:** minutes-to-hours on one consumer/cloud GPU, fitting a 1–2 hour Runner time cap.
- **Public, standard dataset:** CIFAR-10 itself, no proprietary data or non-public code dependencies.

## Shortlist, ordered by publish date

| # | Paper | Authors | First published | Venue | Reported CIFAR-10 result |
|---|---|---|---|---|---|
| 1 | Network In Network | Lin, Chen, Yan | Dec 16, 2013 (arXiv:1312.4400) | ICLR 2014 | 8.81% test error (with augmentation) |
| 2 | Striving for Simplicity: The All Convolutional Net | Springenberg, Dosovitskiy, Brox, Riedmiller | Dec 21, 2014 (arXiv:1412.6806) | ICLR 2015 (workshop) | 7.25% test error (All-CNN-C, with augmentation) |
| 3 | Deep Residual Learning for Image Recognition | He, Zhang, Ren, Sun | Dec 10, 2015 (arXiv:1512.03385) | CVPR 2016 | ~6.4% test error (ResNet-110, CIFAR-10 analysis in §4.2) |
| 4 | Deep Networks with Stochastic Depth | Huang, Sun, Liu, Sedra, Weinberger | Mar 30, 2016 (arXiv:1603.09382) | ECCV 2016 | 4.91% test error (1202-layer ResNet w/ stochastic depth) |
| 5 | Wide Residual Networks | Zagoruyko, Komodakis | May 23, 2016 (arXiv:1605.07146) | BMVC 2016 | 4.17% test error (WRN-28-10) |
| 6 | Densely Connected Convolutional Networks | Huang, Liu, van der Maaten, Weinberger | Aug 25, 2016 (arXiv:1608.06993) | CVPR 2017 (Best Paper) | 3.74% test error (DenseNet-BC, k=24) |
| 7 | AutoAugment: Learning Augmentation Policies from Data | Cubuk, Zoph, Mané, Vasudevan, Le | May 24, 2018 (arXiv:1805.09501) | CVPR 2019 | 1.5% test error (best policy + Shake-Shake/PyramidNet backbone) |
| 8 | Escaping the Big Data Paradigm with Compact Transformers | Hassani, Walton, Shah, Abuduweili, Li, Shi | Apr 12, 2021 (arXiv:2104.05704) | — | 98.0% test accuracy (CCT-7/3x1, trained from scratch, no pretraining) |

## Brief summaries

**1. Network In Network (2013).** Replaces the linear filters of standard conv layers with small MLPs ("mlpconv") and replaces the final fully-connected classifier with global average pooling. Cheap, single-architecture, single-table result — a good "smallest possible" first pilot paper for shaking out the Reader→Coder→Runner→Critic loop before tackling deeper nets.

**2. Striving for Simplicity: The All Convolutional Net (2014).** Shows max-pooling can be replaced by a strided convolution with no accuracy loss, yielding a network built entirely from convolutions. Minimal architectural surface area (no pooling, no branching), making it an easy target for the Coder to translate directly from the paper's table into a HuggingFace `Trainer`-compatible model definition.

**3. Deep Residual Learning for Image Recognition (2015).** Introduces residual (skip) connections to train much deeper nets without degradation; the CIFAR-10 section (a small side-experiment relative to the paper's main ImageNet result) sweeps depth from 20 to 1202 layers. Useful as a "family" target — ReproBot could replicate several depths from one paper's Table 6 as multiple claims against one method summary.

**4. Deep Networks with Stochastic Depth (2016).** A training-procedure paper, not a new architecture: randomly drops residual blocks during training (identity bypass) and uses the full depth at test time, cutting training time and improving accuracy on very deep ResNets. Good test of whether the Coder can implement a *training-loop modification* rather than just a model architecture.

**5. Wide Residual Networks (2016).** Argues width, not depth, is the more efficient lever for ResNets; WRN-28-10 outperforms a 1001-layer thin ResNet with 36× fewer layers. Single dominant headline number (4.17%), well-known reference implementation, widely reproduced elsewhere — low ambiguity for the Critic's numeric-tolerance check.

**6. Densely Connected Convolutional Networks (2016).** Connects each layer to every other layer in a feed-forward fashion (feature reuse instead of summation), improving parameter efficiency versus ResNet/WRN at matched accuracy. CVPR 2017 Best Paper; reports a clean parameter-count-vs-error table on CIFAR-10, ideal for testing multiple claims (different DenseNet-BC configurations) from a single paper.

**7. AutoAugment (2018).** Learns a data-augmentation policy via RL search on a small proxy task, then applies the found policy to train standard backbones (Wide-ResNet, Shake-Shake, PyramidNet). For ReproBot, the target claim is the *downstream* training run with the paper's published policy (Table 1's CIFAR-10 result), not the RL search itself — this tests whether the Reader can correctly scope "what to reproduce" out of a paper with more than one moving part.

**8. Escaping the Big Data Paradigm with Compact Transformers (2021).** Adapts Vision Transformers to small datasets by adding a convolutional tokenizer and sequence pooling, letting a ViT-style model train from scratch on CIFAR-10 in under 30 minutes on a single GPU and reach ~98% accuracy without any large-scale pretraining. The one non-CNN entry on this list — useful once the pilot set is validated on CNN/ResNet-family papers, to test the Coder/Runner against a small-ViT architecture as called out as in-scope in the project plan.

## Suggested pilot ordering

Per the project plan's "build and validate against 3–5 papers first" guidance (§0.3): start with **#1 (NIN)** and **#2 (All-CNN)** — smallest architectural surface area, fastest training, easiest for the Critic to score unambiguously — then **#3 (ResNet)** and **#5 (WRN)** to exercise a "family of depths/widths from one paper" claim structure, before attempting **#7 (AutoAugment)**, which has the added complexity of a non-architectural, augmentation-policy claim, or **#8 (CCT)**, which departs from the CNN family entirely.

## Next steps

- Confirm this shortlist (or swap entries) — PDFs are already downloaded into `dataset/`, renamed to full titles for readability, matching the naming convention used in `papers/`.
- Add per-paper entries to `Paper_Summaries.md` if a full deep-dive (not just this brief note) is wanted for the pilot papers, once the pilot set is finalized.
