# C03 Optimization And Innovation Plan

Date: 2026-04-30

## Baseline Lock

Frozen submitted baseline:

- Job ID: `jb-aitrain-155893052943654656`
- Submission time: `2026-04-30 08:27:55`
- Status: `Succeed`
- Score: `0.994132`
- Visible secondary metric: `0.938218`
- Card consumption time: `0.0119 H`
- Frozen files: `frozen_baselines/c03_submitted_20260430_082755/`
- Working duplicate checkpoint: `checkpoints/optimization/c03_working_model.pt`
- Working duplicate config: `configs/optimization/submission_c03_working.json`

The frozen C03 submitted baseline is the reference. Every optimization must report:

- output-contract status
- local smoke status
- parameter count and checkpoint size
- runtime delta
- validation or robustness delta when labels are involved
- platform score only after a clean local diagnostic pass

## Status Summary

Done:

- Frozen submitted C03 baseline.
- Created numbered data-prep pipeline versions `P0` through `P4`.
- Added clean experiment area with fresh per-run work/output folders.
- Added fine-grained data-prep timing.
- Optimized selected-file index construction.
- Added opt-in GDAL settings and in-memory patch transfer.
- Added inference-only fast TIFF inventory.
- Added P4 raster-window batching for points sharing the same raster file.
- Verified P1/P2/P3 produce identical local smoke output to P0.
- Recorded C20 Fourier platform comparison: score `0.907497`, card time `0.0106 H`, worse score but slightly lower card time than C03 baseline.
- Submitted C03 P3 optimized pre-model pipeline and recorded platform timing for job `jb-aitrain-155924679797274304`.

Current optimized pre-model version:

- `P3`: `configs/optimization/versions/P3_submission_c03_fast_inventory_gdal_inmemory.json`
- Current working config: `configs/optimization/submission_c03_working.json`
- Median local smoke total: `0.5713 s`
- P0 median local smoke total in same run: `0.8149 s`
- Median local smoke data-prep time: `0.4942 s`
- P0 median local smoke data-prep time in same run: `0.7251 s`
- Output JSON match vs P0: `true`
- Missing band cells: `0`
- File read failures: `0`
- Patch extraction errors: `0`
- Platform P3 internal total runtime: `35.3711 s`
- Platform P3 patch extraction time: `33.4366 s`
- Platform P3 model forward time: `0.4281 s`
- P4 local clean patch-array test: arrays match P3 exactly; raster read calls dropped from `540` to `180` on the smoke input.
- Ported P4 into `project_organized` as `configs/optimization/versions/P4_submission_c03_batched_raster_windows.json`.
- `project_organized` P4 verification: P4 output JSON matches P0, direct P4 arrays match P3 exactly, median local total is `0.5491 s`, and P4 read calls are `180` with `0` fallback reads.
- Added P5 adaptive block-aware raster cache as `configs/optimization/versions/P5_submission_c03_block_aware_raster_cache.json`.
- P5 increases optimized `GDAL_CACHEMAX` to `2048`, attempts internal-block reads when bounded, and falls back to P4/P3 when block overread is too high.
- `project_organized` P5 verification: output JSON matches P0, local adaptive mode fell back to P4 on the smoke input, median local total is `0.5494 s`, and forced-block mode was exact but slower (`900` block reads vs `540` P3 patch reads).

Not done yet:

- Band importance analysis.
- Dynamic quantization.
- Static/export quantization.
- Distillation or teacher-student training.
- Pruning.
- C20 replacement or retraining.

Current comparison note:

- C03 baseline platform score: `0.994132`, card time `0.0119 H`.
- C20 Fourier + `transition_viterbi` platform score: `0.907497`, card time `0.0106 H`.
- C20 was about `4.68 s` lower in card time but lost `0.086635` score, so it should not replace C03.
- The pasted C20 log did not include fine-grained `timing_seconds`; it only supports outer timestamp timing.
- C03 P3 platform job `jb-aitrain-155924679797274304` activated `fast_inventory`, GDAL cache/read-dir options, and in-memory patches. Its runtime config did not include `batch_raster_reads`, confirming it was P3 rather than P4. It reported `35.3711 s` internal total runtime, but `33.1392 s` of the patch report was still spent in actual raster window reads. Therefore the card-consumption time did not visibly change: P3 removed small bookkeeping/NPZ overhead, not the dominant raster I/O.

## Priority Order

### 1. Data-Prep And Inference Pipeline Optimization

This comes before model compression because the local smoke shows patch extraction dominates wall time. The model forward pass is already a small fraction of local end-to-end time.

Actions:

- Profile TIFF inventory, filename parsing, point-to-region mapping, patch extraction, normalization, and JSON writing separately. Status: done.
- Avoid repeated scans/expensive grouping over all TIFF names. Status: done with P1/P3.
- Use GDAL open/cache options for the inference container. Status: done in P2/P3:
  - `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
  - `GDAL_READDIR_LIMIT_ON_OPEN=0`
  - `GDAL_CACHEMAX=256`
  - `VSI_CACHE=TRUE`
  - `VSI_CACHE_SIZE=33554432`
- Use in-memory patch arrays for inference so the pipeline does not write a compressed NPZ and immediately read it back. Status: done in P2/P3.
- Add inference-only fast TIFF inventory. Status: done in P3.
- Preserve exact input tensor values before claiming a speedup. Status: local output equivalence verified.
- Compare C03 frozen config vs C03 working config on the same smoke input. Status: done.

Decision gate:

- Same result keys and predictions, same failure counts, lower total runtime. Status: passed locally for P3.

Remaining:

- Submit P4 only if we accept the risk/reward tradeoff: it reduces read-call count by grouping points per raster, but may read more pixels per call when points are spread out.
- Submit P5 adaptive instead of plain P4 if we want one platform test that includes P4 behavior, larger GDAL cache, block-read counters, and safe block-aware fallback.

### 2. Band Importance Analysis

This is the first model-behavior analysis step because it can inform pruning, quantization risk, and paper explanation.

Actions:

- Run leave-one-band-out or mask-one-band validation on C03.
- Run grouped ablations for visible, red-edge/NIR, water-vapor, and SWIR bands.
- Measure crop macro-F1, rice-stage macro-F1, loss, and confidence shift.
- Prefer validation-time ablation before retraining.

Decision gate:

- Identify removable or low-sensitivity bands only if score and rice-stage F1 are stable.

Status: not started.

### 3. Dynamic Quantization

This is low-risk for CPU deployment and can be tested without retraining.

Actions:

- Apply PyTorch dynamic quantization to linear layers.
- Benchmark checkpoint/runtime on CPU.
- Validate output drift on smoke and validation sets.

Decision gate:

- Smaller or faster CPU model with negligible validation drift.
- Treat GPU benefit as uncertain unless platform tests confirm it.

Status: not started.

### 4. Static / Export-Oriented Quantization

Higher risk than dynamic quantization because CNN layers and activations are involved.

Actions:

- Test post-training quantization only after dynamic quantization is measured.
- If accuracy drops, consider quantization-aware training on the working duplicate.

Decision gate:

- Must preserve rice-stage F1; size improvement alone is not enough.

Status: not started.

### 5. Teacher-Student Distillation

Main path: use C03 as teacher for a smaller C03-style student. Secondary path: use C20 or TSViT as teacher only if they improve robustness or calibration.

Actions:

- Design smaller student by reducing embedding/transformer width or layers.
- Train with hard labels plus teacher logits.
- Measure validation score, shift robustness, params, checkpoint size, and runtime.

Decision gate:

- Student must approach submitted C03 score while materially reducing size or runtime.

Status: not started.

### 6. Pruning

Use after band importance and distillation baselines.

Actions:

- Start with structured pruning of channels/hidden dimensions, not unstructured sparsity.
- Fine-tune pruned student or working C03.
- Keep deployment simple; avoid sparse kernels unless the runtime stack benefits from them.

Decision gate:

- Actual measured inference speedup, not only parameter sparsity.

Status: not started.

## Innovation Candidates For Paper

- Compact query-conditioned CNN + temporal transformer baseline.
- Fourier temporal/query encoding from C20 as the main architecture innovation candidate.
- Band importance as explainability and deployment optimization.
- Teacher-student compression for edge/on-orbit deployment.
- Quantization/pruning as deployment engineering, reported only if measured.

## Do Not Do Yet

- Do not tune from one hidden score without local evidence.
- Do not replace C03 with C20 until C20 passes runtime and robustness gates.
- Do not prune bands before band importance confirms low sensitivity.
- Do not treat dynamic quantization as a GPU improvement without platform evidence.
