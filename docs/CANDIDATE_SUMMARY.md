# Candidate Summary

This organized bundle keeps only two finalists.

## C03

- Role: safest final operational baseline.
- Checkpoint: `checkpoints/c03_full_data_model.pt`
- Model type: `query_cnn_transformer`
- Size: `12.55 MB`
- Parameters: `3,275,146`
- Fixed full-data epoch: `75`
- Strength:
  - compact checkpoint
  - simpler architecture
  - easiest candidate to justify for constrained on-orbit deployment
- Weakness:
  - less architecturally novel than TSViT

## C29 TSViT

- Role: stronger comparison model for final-phase architecture discussion.
- Checkpoint: `checkpoints/c29_tsvit_model.pt`
- Model type: `query_tsvit`
- Size: `88.51 MB`
- Parameters: `23,170,666`
- Fixed full-data epoch: `42`
- Strength:
  - richer spatial-temporal transformer design
  - better story for innovation, heterogeneous compute, and future roadmap
- Weakness:
  - much heavier uplink and memory footprint than C03
  - harder to justify for CPU-only or low-power fallback execution

## Decision Rule

- For a conservative final submission package, prefer `C03`.
- For architecture comparison and final-presentation discussion, keep `C29 TSViT` as the high-capability path.
- In the final PPT, present the pair as:
  - `C03` = flight-safe baseline
  - `C29 TSViT` = scaled intelligent variant for stronger future on-orbit hardware
