# Final Phase Scoring Brief

This note is structured to map directly onto the final judging rubric.

## Part I: Overall Task Implementation And Architecture

### End-to-end logic

The full pipeline is:

1. ingest point/date task rows and Sentinel-2 TIFF tiles
2. build a normalized 15x15 multi-temporal patch tensor per point
3. run crop-stage inference with a query-aware model
4. apply consistency logic
5. downlink a compact `result.json`

### Architecture decisions

- Modularity:
  - preprocessing, training, validation, and submission inference are separated
  - checkpoints, configs, and normalization stats are versioned independently
- Heterogeneous scheduling:
  - preprocessing can run on CPU
  - inference can run on CPU as a safe fallback or GPU when available
  - C03 and C29 naturally support a two-tier deployment strategy
- Fault tolerance:
  - explicit config files make reruns deterministic
  - validation script checks checkpoint compatibility before submission
  - fallback to C03 reduces operational risk if heavier models fail timing or memory checks

### Space-environment fit

- C03 is the resource-constrained deployment choice.
- C29 TSViT is the performance-oriented path for stronger on-orbit compute nodes.
- The same preprocessing and output contract is shared by both, so switching models does not require changing the surrounding system.

## Part II: On-Orbit Feasibility

### Uplink size and packaging

- C03 checkpoint: `12.55 MB`
- C29 TSViT checkpoint: `88.51 MB`
- Suggested uplink package:
  - checkpoint
  - submission config
  - normalization JSON
  - inference code

Recommended strategy:

- uplink C03 as the default flight package
- uplink TSViT only when bandwidth and storage margins are acceptable
- compress code/config package as `tar.gz`; the `.pt` checkpoints are already binary and gain limited extra compression

### Resource considerations

- C03 parameters: `3.28 M`
- C29 TSViT parameters: `23.17 M`
- Input tensor footprint scales with:
  - `batch_size x time_steps x channels x 15 x 15`
- For a representative inference batch of 64 with 29 timesteps and 24 channels, raw patch tensor memory is about `40 MB` before model activations.

Operational interpretation:

- C03 is the better choice for CPU-bound or low-memory execution.
- C29 is more suitable when GPU or higher-memory edge compute is available.

### Dependent data

Required inputs:

- point/date query CSV
- Sentinel-2 TIFF time series
- normalization statistics JSON
- selected model checkpoint

Management approach:

- keep raster ingestion separate from inference
- cache the prebuilt NPZ dataset when repeated execution is expected
- keep normalization stats immutable across inference runs for reproducibility

### Latency and execution risk

- Single execution is bounded by:
  - TIFF discovery and patch extraction
  - batch inference
  - JSON writeout
- Worst-case risk is dominated by raster I/O and large-model inference.
- C03 is the safer low-latency path for worst-case timing control.

## Part III: Innovation Of Implementation Path

### Technical innovation

- query-aware phenology classification from multi-temporal satellite patches
- valid-mask channel design to make missing-observation structure explicit
- TSViT path introduces stronger spatial-temporal token mixing for richer context modeling

### Difference from ground-only solutions

- design is oriented around moving only compact models and compact outputs, not raw imagery
- supports staged inference and selective recomputation instead of full ground retraining loops
- architecture allows a low-risk model and a high-capability model to share the same mission interface

### Space-specific design points

- lightweight fallback model (`C03`) for power and memory constraints
- explicit validation and deterministic configs for autonomous operation
- candidate separation enables mission planning by compute budget, not by rewriting the pipeline

## Part IV: Value And Application Scenarios

### Value path

- agricultural condition awareness from orbit to ground users
- compact result downlink rather than full-scene raw data transfer
- lower latency for crop-type and phenophase status products

### Application scenarios

- seasonal crop monitoring
- food-security analysis
- regional growth-stage tracking for rice, corn, and soybean
- rapid agricultural reporting in bandwidth-limited environments

### Satellite-terrestrial coordination

- orbit side:
  - preprocessing and candidate inference
- ground side:
  - model updates
  - archive analytics
  - dashboarding and business integration

## Part V: Future Roadmap

### Near term

- benchmark C03 and C29 on fixed resource budgets
- reduce TSViT memory and package size
- quantify latency on target hardware instead of desktop proxies

### Mid term

- distill TSViT behavior into a smaller student model
- add model quantization and structured pruning
- improve autonomous failure recovery around corrupted or partial raster inputs

### Long term

- adaptive scheduling across multiple on-orbit compute nodes
- model refresh by mission phase and region
- tighter satellite-terrestrial coordination where orbit handles filtering and ground handles strategic retraining

## Part VI: Team Introduction

The team slide should emphasize:

- remote-sensing pipeline ownership from raw raster ingestion to result packaging
- ability to compare operational baselines against more advanced transformer designs
- disciplined validation, checkpoint management, and submission engineering

## Presentation And Logic

Recommended PPT structure:

1. problem and mission framing
2. end-to-end architecture
3. C03 baseline and why it is flight-safe
4. C29 TSViT and why it is innovative
5. on-orbit feasibility tradeoff table
6. value and application scenarios
7. roadmap
8. team capability

Recommended contrast slide:

- `C03`: smaller, simpler, safer
- `C29 TSViT`: richer, heavier, forward-looking

That contrast is likely the cleanest way to score well on both feasibility and innovation without pretending one model solves every constraint equally well.
