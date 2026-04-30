# 02 Optimization And Innovation Roadmap

Date: 2026-04-30

Scope: no new training by default. This roadmap starts from the completed experiment table in `/Users/yusuf/Desktop/finalround/project/experiment_table_completed.csv` and the current C03/TSViT organized bundle.

## Current Decision

C03 remains the safest operational baseline because it is simple, validated, and already ran successfully on the platform.

C20 Fourier is the strongest prior innovation candidate in the experiment history:

- C20 validation score: `1.000000`
- C20 crop F1: `1.000000`
- C20 rice-stage F1: `1.000000`
- C20 validation loss: `0.001425`
- C03 validation loss: `0.061087`

This means the next optimization/innovation work should not repeat dropout, generic date augmentation, monotonic decoding, or auxiliary feature branches. Those axes were already tested and mostly underperformed.

Additional audit on 2026-04-30 compared the existing parent C20 checkpoint against C03 using the same validation query-shift slices. C20 remains attractive as an innovation story because its normal validation loss is much lower, but it is not automatically safer for submission because its `+7d` query-shift score is lower than C03.

| Model | Shift | Val Score | Rice-Stage F1 | Loss |
| --- | ---: | ---: | ---: | ---: |
| C03 | `0d` | 1.000000 | 1.000000 | 0.040946 |
| C03 | `-3d` | 1.000000 | 1.000000 | 0.049386 |
| C03 | `+3d` | 1.000000 | 1.000000 | 0.063523 |
| C03 | `-7d` | 0.983757 | 0.972929 | 0.118476 |
| C03 | `+7d` | 0.941365 | 0.902275 | 0.161090 |
| C20 parent | `0d` | 1.000000 | 1.000000 | 0.002719 |
| C20 parent | `-3d` | 1.000000 | 1.000000 | 0.002288 |
| C20 parent | `+3d` | 0.990730 | 0.984550 | 0.022822 |
| C20 parent | `-7d` | 0.988389 | 0.980649 | 0.041446 |
| C20 parent | `+7d` | 0.896779 | 0.827966 | 0.227997 |

Platform evidence later on 2026-04-30 confirmed that caution: C20 Fourier with `transition_viterbi` stage postprocess scored `0.907497` on the platform with `0.0106 H` card consumption time. It was faster than the C03 full-data baseline by card time (`0.0106 H` vs `0.0119 H`) but much worse in score (`0.907497` vs `0.994132`).

Interpretation for paper:

- C20 Fourier is a useful innovation/ablation result because it improves validation confidence and encodes temporal periodicity explicitly.
- C20 Fourier is not the final operational model because hidden/platform generalization was weaker.
- The C20 result supports the decision to separate innovation discussion from final submission reliability.

## Completed Experiment Lessons

| Area | Evidence | Decision |
| --- | --- | --- |
| C03 baseline | C00/E1 and C03 both reached perfect validation; C03 had lower loss than C00. | Keep C03 as baseline and safe submission path. |
| Auxiliary features | C04/C05/C06 improved loss in one narrow case but did not beat C03/C20 score reliability. | Do not make aux features the next main path. |
| Mask channels | C06 removing masks hurt performance. | Keep valid-mask channels. |
| Query DOY | C07 and C18 collapsed rice-stage prediction without query date. | Query date is essential. |
| Acquisition DOY | C08 stayed near-perfect but with worse confidence. | Keep acquisition date. |
| Leakage sanity | C09 shuffle-label sanity failed as expected. | No obvious leakage signal from this check. |
| Monotonic decode | C13 and C15 hurt badly. | Do not use hard monotonic decoding. |
| Ordinal/structured stage | C14/C16 helped some variants but did not beat C20; C17 underperformed. | Keep as paper discussion or secondary ablation, not next main step. |
| Date shift/dropout | C19 was worse than C03/C20. | Do not repeat temporal dropout training now. |
| Fourier temporal encoding | C20 reached perfect validation with the lowest recorded loss. | Treat C20 as the main innovation candidate to audit and compare. |
| Time2Vec / cross-series | C21 and C27 underperformed C20. | Do not prioritize unless needed for paper discussion. |
| Spectral indices | C23 close but not best; C24 broken. | Do not prioritize spectral-index training now. |

## Relevant Literature Direction

- TSViT supports the value of explicit temporal-spatial modeling for satellite image time series and remains our comparison architecture. Source: <https://arxiv.org/abs/2301.04944>
- Lightweight temporal attention and pixel-set encoders support future efficiency work, but they are larger architecture changes than we need right now. Sources: <https://arxiv.org/abs/2007.00586>, <https://arxiv.org/abs/1911.07757>
- TempCNN supports temporal convolution as a possible speed-oriented future variant, not an immediate replacement. Source: <https://arxiv.org/abs/1811.10166>
- Distillation is useful only after selecting the teacher checkpoint. Source: <https://arxiv.org/abs/2106.05237>

## Roadmap

### Step 1: Audit Existing C20 Before Any New Training

Goal: verify whether C20 can be safely represented as our innovation/optimization result.

Actions:

- Inspect C20 config and checkpoint compatibility. Status: done.
- Run the same evaluator/checks on C20 if the checkpoint is available. Status: done using `configs/evaluate_c20_parent.json`.
- Compare C03 vs C20 on validation and robust query-shift diagnostics. Status: done.
- Compare parameter count, checkpoint size, and inference timing. Status: pending.
- Record C20 platform result. Status: done for the `864547fe` run: score `0.907497`, card time `0.0106 H`, C20 Fourier epoch `77`, `transition_viterbi` stage postprocess.

Decision gate:

- C20 has better normal validation loss and the same normal validation score.
- C03 is more robust under the simple `+7d` query-shift diagnostic.
- Platform C20 Fourier score was substantially lower than C03.
- Current decision: keep C03 as the safe operational model; use C20 Fourier as an innovation/ablation candidate, not as a replacement.

### Step 2: Runtime Optimization Without Changing Model Weights

Goal: earn optimization points through deployment and pipeline efficiency.

Actions:

- Use the timing instrumentation already added to split patch extraction, model load, forward pass, postprocess, and JSON write.
- Reduce submission package clutter so only C03 and TSViT/C20-comparison artifacts are present.
- Cache or streamline TIFF inventory/patch extraction only if it does not change outputs.
- Benchmark CPU proxy and GPU platform timing separately.

Decision gate:

- Same predictions, faster runtime, cleaner package.

### Step 3: Paper-Ready Diagnostics

Goal: turn the safety checks into article evidence.

Actions:

- Record C03 platform run: GPU, rows, duplicate handling, valid-pixel ratio, file failures, extraction failures.
- Record scoring formula and leakage checks.
- Add experiment table summary: why C03 is baseline, why TSViT is comparison, why C20 is innovation candidate.

Decision gate:

- The paper can explain both performance and reliability without implying hidden-test tuning.

### Step 4: Only Then Consider New Training

New training should happen only if the audit reveals a specific gap not already tested.

Allowed examples:

- Distill C20 into C03-size model if C20 is better but slower.
- Train a strictly smaller C03 if runtime becomes the bottleneck.
- Do a tiny ablation only if it answers a paper question.

Rejected for now:

- More dropout/date-shift training.
- More monotonic decoding.
- More aux feature variants.
- More spectral-index variants.

## Immediate Next Action

Benchmark existing C20 against C03 for size and inference delay, then decide whether C20 is worth copying into the organized comparison set or should stay as a parent-folder paper result. No new training is needed for this step.
