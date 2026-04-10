# FG-CLIP2 Optimization Log

Append one entry for every runtime/model optimization. Record both the benefit
and the accuracy cost. Do not accept an optimization only because it is faster or
smaller.

## Acceptance Policy

Default gates for fp32 baseline replacements:

| Check | Gate |
| --- | ---: |
| text feature cosine vs current fp32 ONNX | `>= 0.9999` |
| image feature cosine vs current fp32 ONNX | `>= 0.999` |
| fixture text max abs diff | record; investigate if `> 1e-4` |
| fixture image max abs diff | record; investigate if `> 1e-3` |
| real search Top-K overlap | record when a test corpus exists |

An optimization can be accepted below these gates only if the product explicitly
uses it as an optional "fast / low-memory" mode.

## Baseline - Split Text ONNX + Image ONNX, FP32

Date: 2026-04-09

Status: accepted as the current correctness baseline.

What changed:

- Exported full text-feature ONNX for short text.
- Exported image-feature ONNX that takes `pos_embed` as input.
- Moved SigLIP2 image patchify and FG-CLIP2 position-embedding resize to Rust preprocessing.
- Added Rust verification/runtime prototype.

Artifacts:

| Artifact | Size |
| --- | ---: |
| `fgclip2_text_short_b1_s64.onnx` | `~1077 MiB` |
| `fgclip2_image_core_posin_dynamic.onnx` | `~354 MiB` |
| `vision_pos_embedding_16x16x768_f32.bin` | `~0.75 MiB` |

Accuracy:

| Scenario | Metric | Value |
| --- | --- | ---: |
| Rust ONNX text vs PyTorch fixture | max abs diff | `~1e-7` |
| Rust ONNX image vs PyTorch fixture | max abs diff | `~2e-6` |
| Rust end-to-end JPEG image vs Python/PIL path | image-feature cosine | `0.999913` |
| Rust end-to-end JPEG score vs Python/PIL path | score relative diff | `~0.57%` |
| Rust reading PIL-decoded PNG vs Python/PIL path | image-feature max abs diff | `~1.6e-6` |

CPU speed:

| Runtime | Text | Image 256 | Image 1024 |
| --- | ---: | ---: | ---: |
| PyTorch CPU | `~53-56 ms` | `~162 ms` | `~768 ms` |
| Python ORT CPU | `~34-41 ms` | `~124 ms` | `~711 ms` |
| Rust ORT CPU | `~40 ms` | `~130-141 ms` | `~670-724 ms` |

Memory:

| Runtime Point | Memory |
| --- | ---: |
| Rust text + image end-to-end peak Working Set | `~1621 MiB` |
| Rust text + image end-to-end peak Private | `~1922 MiB` |
| Rust image-only B=1 peak Working Set | `525.2 MiB` |
| Rust image-only B=1 peak Private | `551.9 MiB` |

Decision:

- Keep this as the baseline.
- Use CPU first.
- Use `1024` patches by default for素材 management quality.
- Accept the Rust JPEG decoder difference for now.

## Optimization - Dynamic Batch Image ONNX

Date: 2026-04-09

Status: accepted, but CPU batch size should remain small.

What changed:

- Rewrote the image pooling attention in the export wrapper without
  `nn.MultiheadAttention.forward`, so ONNX does not freeze the traced batch or
  patch length.
- Image ONNX accepts `pixel_values [B,N,768]`, `pixel_attention_mask [B,N]`,
  `pos_embed [B,N,768]`.
- Rust prototype added `run-batch`.

Accuracy:

| Check | Value |
| --- | ---: |
| PyTorch wrapper B=1 vs original model | `0.0` max abs diff in test |
| PyTorch wrapper B=2 row vs original model | `~2.1e-7` max abs diff |
| ORT B=1/B=2/256/128 vs wrapper | `~1.7e-6` to `~2.2e-6` max abs diff |

CPU image-only benchmark, 1024 patches:

| Batch | Total Time | Per Image | Peak WS | Peak Private |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `723.53 ms` | `723.53 ms` | `525.2 MiB` | `551.9 MiB` |
| `2` | `1.46 s` | `729.47 ms` | `662.2 MiB` | `729.3 MiB` |
| `4` | `2.73 s` | `683.02 ms` | `936.3 MiB` | `1083.9 MiB` |

Benefit:

- Enables future batching and GPU experiments.
- CPU B=4 improved per-image time by only about `5.6%` in this sample.

Cost:

- Higher batch sizes increase memory substantially.
- CPU throughput gain is small.

Decision:

- Keep the dynamic-batch ONNX because B=1 remains valid.
- Default CPU `max_batch_size` should be `1`; try `2` only if benchmarked on the user's machine.
- Do not force waiting for full batches during normal one-by-one imports.

## Optimization Experiment - Dynamic Int8 ONNX

Date: 2026-04-09

Status: optional conservative profile accepted for experiments; naive all-linear
dynamic int8 rejected.

What changed:

- Added `quantize_fgclip2_onnx.py`.
- Added Rust runtime path selection:
  - `FGCLIP2_RUNTIME_VARIANT=dynamic-int8`
  - `FGCLIP2_TEXT_ONNX=<path>`
  - `FGCLIP2_IMAGE_ONNX=<path>`
- Added a quantized verifier manifest at
  `artifacts/fgclip2/fixtures/manifest_dynamic_int8.json`.

Rejected profile:

| Profile | Text Cosine vs fp32 | Image Cosine vs fp32 | Decision |
| --- | ---: | ---: | --- |
| all weight-backed `MatMul` / `Gemm` | `0.946112` | `0.218806` | rejected |

Accepted optional profile:

| Model | Quantized Nodes | Size Before | Size After |
| --- | --- | ---: | ---: |
| text | all text encoder self-attention projections | `1077.0 MiB` | `996.3 MiB` |
| image | vision encoder self-attention projections in layers `6..11` | `353.9 MiB` | `313.6 MiB` |

Python ORT fixture verification, 256 patches:

| Path | Max Abs Diff vs fp32 ONNX | Cosine vs fp32 ONNX | fp32 Hot Time | Quant Hot Time |
| --- | ---: | ---: | ---: | ---: |
| text | `0.00439942` | `0.999537` | `33.07 ms` | `28.08 ms` |
| image | `0.00584997` | `0.999110` | `132.31 ms` | `109.24 ms` |

Rust verifier with `artifacts/fgclip2/fixtures/manifest_dynamic_int8.json`:

| Path | Max Abs Diff vs PyTorch Fixture | Cosine vs PyTorch Fixture |
| --- | ---: | ---: |
| text | `0.004399398` | `0.999537230` |
| image | `0.005849747` | `0.999110699` |

Image 1024 check, using existing Rust dump tensors:

| Check | Value |
| --- | ---: |
| image cosine vs fp32 ONNX | `0.999560654` |
| image max abs diff vs fp32 ONNX | `0.00415317` |
| Python ORT hot image time | `670.42 ms fp32 -> 649.64 ms quantized` |

Rust image-only process, 1024 patches:

| Model | Image Inference | Peak WS | Peak Private |
| --- | ---: | ---: | ---: |
| fp32 baseline | `723.53 ms` | `525.2 MiB` | `551.9 MiB` |
| conservative dynamic-int8 | `650.01 ms` | `488.0 MiB` | `513.3 MiB` |

Cost / limitation:

- Text ONNX is still large because dynamic MatMul quantization does not shrink
  the token embedding table.
- Quantizing early vision layers, image MLPs, or all text+image linears did not
  satisfy the current cosine gate on the single fixture.

## Optimization - Split Text Token Embedding Out Of ONNX

Date: 2026-04-09

Status: accepted as the recommended low-memory text path.

What changed:

- Added `split_fgclip2_text_embedding.py`.
- Removed `model.text_model.embeddings.token_embedding.weight` from the text
  ONNX.
- Split text ONNX input is now `token_embeds [1,64,768]`.
- Wrote the token table as
  `artifacts/fgclip2/runtime/assets/text_token_embedding_256000x768_f16.bin`.
- Rust detects `*_token_embeds.onnx`, reads only requested token rows from the
  external token table, converts them to f32, and feeds `token_embeds`.
- Added `run-text` to the Rust prototype.
- Added runtime variants:
  - `FGCLIP2_RUNTIME_VARIANT=split-text`: split text + fp32 image.
  - `FGCLIP2_RUNTIME_VARIANT=lowmem`: split text + conservative dynamic-int8 image.

Why this works:

- The full text ONNX had a `750.0 MiB` initializer:
  `model.text_model.embeddings.token_embedding.weight [256000,768]`.
- A query only needs the rows named by its `64` token ids.

Artifacts:

| Artifact | Size |
| --- | ---: |
| full text ONNX | `1077.0 MiB` |
| split text ONNX | `327.0 MiB` |
| external token embedding, fp16 | `375.0 MiB` |

Text accuracy:

| Scenario | Value |
| --- | ---: |
| Rust split-text fixture max abs diff vs PyTorch fixture | `8.9e-8` |
| Rust split-text fixture cosine vs PyTorch fixture | `1.0` |
| 10 real-query split-text min cosine vs full text ONNX | `0.999999881` |
| 10 real-query max abs diff | `0` printed at current precision |

Rust memory:

| Workload | Variant | Peak WS | Peak Private |
| --- | --- | ---: | ---: |
| text-only `run-text 山` | full fp32 text ONNX | `1818.6 MiB` | `1913.8 MiB` |
| text-only `run-text 山` | split-text | `372.0 MiB` | `412.3 MiB` |
| end-to-end 1024 `run ... 山 1024` | split-text + fp32 image | `520.6 MiB` | `548.2 MiB` |
| end-to-end 1024 `run ... 山 1024` | lowmem | `498.2 MiB` | `523.8 MiB` |

Rust same-image score comparison, 1024 patches:

| Variant | Score Cosine | Text Feature vs fp32 | Image Feature vs fp32 |
| --- | ---: | ---: | ---: |
| full fp32 text + fp32 image | `0.02757593` | `1.0` | `1.0` |
| split-text + fp32 image | `0.02757593` | `1.0` | `1.0` |
| lowmem | `0.02819987` | `1.0` | `0.999570668` |

Decision:

- Prefer `split-text` for the application path; it gives the large memory win
  without changing the image model.
- Keep `lowmem` as optional. It saved only another `~22 MiB` in the measured
  end-to-end run and shifted the single-image score by `0.000623945`.
