# FG-CLIP2 Rust Runtime Plan

This document tracks the current FG-CLIP2 ONNX/Rust baseline and the planned
optimization order. Keep benchmark numbers here when changing export format,
execution provider, quantization, batch size, or preprocessing.

For the append-only record of completed experiments and accepted/rejected
changes, see `docs/fgclip2-optimization-log.md`.

## Current Runtime Shape

- Model: `qihoo360/fg-clip2-base`
- Embedding output: `768` float values, L2-normalized
- Text ONNX input: `input_ids [1, 64]`
- Image ONNX input: `pixel_values [B, N, 768]`, `pixel_attention_mask [B, N]`, `pos_embed [B, N, 768]`
- Runtime assets:
  - `artifacts/fgclip2/runtime/fgclip2_text_short_b1_s64.onnx`
  - `artifacts/fgclip2/runtime/fgclip2_image_core_posin_dynamic.onnx`
  - `artifacts/fgclip2/runtime/assets/vision_pos_embedding_16x16x768_f32.bin`
  - `artifacts/fgclip2/runtime/assets/logit_params.json`
  - `models/source/fg-clip2-base/tokenizer.json`
- Optional conservative dynamic-int8 experiment assets:
  - `artifacts/fgclip2/runtime/quantized/fgclip2_text_short_b1_s64_dynamic_int8.onnx`
  - `artifacts/fgclip2/runtime/quantized/fgclip2_image_core_posin_dynamic_int8.onnx`
  - `artifacts/fgclip2/fixtures/manifest_dynamic_int8.json`
  - `artifacts/fgclip2/reports/dynamic_int8_report.json`
- Optional split-text assets:
  - `artifacts/fgclip2/runtime/split/fgclip2_text_short_b1_s64_token_embeds.onnx`
  - `artifacts/fgclip2/runtime/assets/text_token_embedding_256000x768_f16.bin`
  - `artifacts/fgclip2/fixtures/manifest_split_text.json`
  - `artifacts/fgclip2/reports/split_text_embedding_report.json`

## Current Accuracy Baseline

Environment: Windows, CPU, ONNX Runtime through `ort`, fp32 ONNX.

When Rust consumes Python-generated fixture tensors:

| Path | Max Abs Diff vs PyTorch | Cosine vs PyTorch |
| --- | ---: | ---: |
| text feature | `8.9e-8` to `1.8e-7` | `1.0` |
| image feature | `~2.0e-6` | `~1.0` |

When Rust runs the full JPEG path with `image` crate decoding:

| Item | Value |
| --- | ---: |
| Python original-model score cosine | `0.02773380` |
| Rust end-to-end score cosine | `0.02757590` |
| Score cosine absolute diff | `0.00015791` |
| Score relative diff | `~0.57%` |
| Image-feature cosine vs Python | `0.999913` |

When the JPEG is first decoded by PIL and saved as PNG, then read by Rust:

| Item | Value |
| --- | ---: |
| pixel_values max abs diff | `~5.9e-8` |
| image feature max abs diff | `~1.6e-6` |
| Rust score cosine | `0.02773416` |

Conclusion: text, ONNX execution, patchify, resize-to-patches, and position
embedding resize are aligned. The remaining JPEG-only end-to-end difference is
caused by JPEG decoder differences.

## Current Size And Memory Baseline

Static files:

| File | Size |
| --- | ---: |
| text ONNX | `~1077 MiB` |
| image ONNX | `~354 MiB` |
| original `model.safetensors` | `~1464 MiB` |
| tokenizer JSON | `~33 MiB` |
| runtime position embedding asset | `~0.75 MiB` |

Single image input tensors:

| Max Patches | pixel_values | pos_embed | Total ONNX inputs approx |
| ---: | ---: | ---: | ---: |
| `256` | `0.75 MiB` | `0.75 MiB` | `~1.5 MiB` |
| `576` | `1.69 MiB` | `1.69 MiB` | `~3.4 MiB` |
| `1024` | `3.0 MiB` | `3.0 MiB` | `~6.0 MiB` |

Rust end-to-end text + image run, batch 1, 1024 patches:

| Metric | Value |
| --- | ---: |
| Peak Working Set | `~1621 MiB` |
| Peak Private Memory | `~1922 MiB` |

Image-only Rust batch run, 1024 patches:

| Batch | Image Inference Total | Per Image | Peak Working Set | Peak Private Memory |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `723.53 ms` | `723.53 ms` | `525.2 MiB` | `551.9 MiB` |
| `2` | `1.46 s` | `729.47 ms` | `662.2 MiB` | `729.3 MiB` |
| `4` | `2.73 s` | `683.02 ms` | `936.3 MiB` | `1083.9 MiB` |

Clean Python ORT process, with input arrays already loaded:

| Point | RSS |
| --- | ---: |
| start | `~53 MiB` |
| after text session | `~1141 MiB` |
| after image session | `~1501 MiB` |
| after one 1024 image run | `~1630 MiB` |
| after second 1024 image run | `~1745 MiB` |

## Current CPU Speed Baseline

CPU, 4 intra-op threads, batch 1 unless noted.

| Path | Text Encode | Image 256 | Image 1024 |
| --- | ---: | ---: | ---: |
| PyTorch CPU | `~53-56 ms` | `~162 ms` | `~768 ms` |
| Python ORT CPU | `~34-41 ms` | `~124 ms` | `~711 ms` |
| Rust ORT CPU | `~40 ms` | `~130-141 ms` | `~670-724 ms` |

## Current Product Direction

- Current phase is CPU-only.
- Do not spend engineering time on CUDA / DirectML / CoreML yet.
- Default quality target: `1024` patches.
- Accept the current JPEG decoder difference for now; it is small enough for
  the素材管理 prototype.
- Optimize memory by loading only the model half needed for the current job.
- Every optimization must record accuracy, speed, and memory in
  `docs/fgclip2-optimization-log.md`.

## Optimization Plan

### P0 - Keep Text And Image Sessions Separate

Goal: avoid loading the 1+ GiB text ONNX while indexing images.

Update: the low-memory text path removes the 750 MiB token embedding initializer
from the text ONNX. In Rust use:

| Variant | Text ONNX | Image ONNX | Intended Use |
| --- | --- | --- | --- |
| unset | split text encoder + external token embedding | fp32 image | recommended low-memory accurate path |
| `FGCLIP2_RUNTIME_VARIANT=split-text` | split text encoder + external token embedding | fp32 image | explicit low-memory accurate path |
| `FGCLIP2_RUNTIME_VARIANT=baseline` or `full-text` | full fp32 text | fp32 image | correctness baseline |
| `FGCLIP2_RUNTIME_VARIANT=lowmem` | split text encoder + external token embedding | conservative dynamic-int8 image | optional lower-memory experiment |

Expected effect:

| Workload | Before | Target |
| --- | ---: | ---: |
| image indexing worker peak | `~1.6-1.9 GiB` if text and image are both loaded | `~0.5-1.0 GiB` image-only |

Implementation notes:

- Build separate `TextEmbedder` and `ImageEmbedder`.
- Lazy-load `TextEmbedder` on search-box focus or first query.
- Let `ImageEmbedder` run in the background import queue.
- Drop text session after an idle timeout, for example 1-3 minutes.

Metrics to record:

| Metric | Before | After |
| --- | ---: | ---: |
| Rust image-only peak WS | `525.2 MiB` fp32 / `488.0 MiB` quantized | depends on selected image ONNX |
| Rust fp32 text-only peak WS | `1818.6 MiB` | `372.0 MiB` with split-text |
| Rust end-to-end 1024 peak WS | `~1621-1819 MiB` if full text ONNX is loaded | `520.6 MiB` with split-text; `498.2 MiB` with lowmem |
| text-session reload latency | `~1.3-1.7 s` full text | `~0.47-0.57 s` split-text observed |

### P1 - Package A Dynamic-Batch Image ONNX, But Use Small CPU Batches

Goal: keep a single image ONNX that can run `B=1..max_batch_size`.

Current status:

- The image ONNX export supports dynamic `B` and dynamic `N`.
- Rust `run-batch` already feeds a stacked `[B, N, 768]` tensor.

Recommendation:

| Backend | Initial max_batch_size |
| --- | ---: |
| CPU | `1` or `2` |
| GPU | deferred |

Reason: CPU batch 4 improved per-image time by only about `5.6%` in the current
test, while memory nearly doubled versus batch 1.

Metrics to record:

| Backend | Batch | Total Time | Per Image | Peak WS | Peak Private |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | `1` | `723.53 ms` | `723.53 ms` | `525.2 MiB` | `551.9 MiB` |
| CPU | `2` | `1.46 s` | `729.47 ms` | `662.2 MiB` | `729.3 MiB` |
| CPU | `4` | `2.73 s` | `683.02 ms` | `936.3 MiB` | `1083.9 MiB` |

### P2 - Build The Image-Indexing Worker Around Image-Only Runtime

Goal: make the real application path avoid text-session memory during imports.

Recommended CPU worker behavior:

- Lazy-load the image ONNX on the first pending image job.
- Keep image ONNX session warm while the import/index queue is active.
- Encode images at `1024` patches.
- Start with `max_batch_size = 1`; raise to `2` only if a local benchmark says
  throughput improves on the target machine.
- Persist only the `768`-dimensional normalized image vector.
- Optionally unload the image session after a long idle timeout.

Application-facing API target:

```text
embed_image(path) -> [f32; 768]
embed_images(paths, max_batch_size) -> Vec<[f32; 768]>
```

Metrics to record:

| Metric | Baseline | After worker integration |
| --- | ---: | ---: |
| first image end-to-end latency | TBD in app | TBD |
| warm image encode latency | `~670-724 ms` standalone Rust run | TBD |
| image-indexing worker peak memory | `~525 MiB` image-only B=1 process | TBD |

### P3 - Quantization Experiment

Goal: reduce ONNX size and resident memory if CPU-only memory is too high.

Current status:

- Implemented `quantize_fgclip2_onnx.py`.
- Rust prototype can load the optional profile with
  `FGCLIP2_RUNTIME_VARIANT=dynamic-int8`.
- Rust prototype also supports exact overrides:
  `FGCLIP2_TEXT_ONNX=<path>` and `FGCLIP2_IMAGE_ONNX=<path>`.
- Rejected naive "all weight-backed linear MatMul/Gemm" dynamic int8 for now:
  fixture cosine was too low.
- Accepted only as an optional experiment: the current default
  `conservative` int8 profile.
  - text: quantize text encoder self-attention projection MatMuls.
  - image: quantize vision encoder self-attention projection MatMuls for
    layers `6..11`.
- This does not solve text memory by itself; most of the text ONNX size is still
  the token embedding table.

Try in this order:

1. Dynamic int8 for linear layers.
2. Dynamic int8 only for text model, if search memory is the pain point.
3. Optional text-token-embedding split or mmap-friendly asset.

Acceptance gates:

| Metric | Required |
| --- | --- |
| text feature cosine vs fp32 | `>= 0.999` initially |
| image feature cosine vs fp32 | `>= 0.999` initially |
| retrieval top-10 overlap | measure on a real素材 test set |
| memory reduction | record peak WS/private |
| speed | record hot encode time |

Current conservative dynamic-int8 numbers:

| Check | Value |
| --- | ---: |
| text ONNX size | `1077.0 MiB -> 996.3 MiB` |
| image ONNX size | `353.9 MiB -> 313.6 MiB` |
| fixture text cosine vs fp32 ONNX | `0.999537` |
| fixture image cosine vs fp32 ONNX | `0.999110` |
| 1024-patch image cosine vs fp32 ONNX | `0.999561` |
| Rust image-only 1024 peak WS | `488.0 MiB` |
| Rust image-only 1024 peak Private | `513.3 MiB` |
| 1024-patch image hot ORT time | `~650 ms quantized vs ~670 ms fp32 in one-process comparison` |

Text-token-embedding split numbers:

| Check | Value |
| --- | ---: |
| full text ONNX size | `1077.0 MiB` |
| split text encoder ONNX size | `327.0 MiB` |
| external token embedding asset | `375.0 MiB fp16` |
| Rust text-only peak WS | `1818.6 MiB -> 372.0 MiB` |
| Rust split-text end-to-end 1024 peak WS | `520.6 MiB` |
| text feature cosine on fixture | `~1.0` |
| 10-query text feature min cosine | `0.999999881` |

### P4 - Future GPU Execution-Provider Selection

Status: deferred. Do not implement during the CPU-only phase.

Do not select, benchmark, package, or expose GPU providers until the CPU
indexing/search path is integrated into the app. When this is reopened, keep CPU
as the fallback and add a fresh benchmark table here.

### P5 - JPEG Decoder Policy

Goal: choose stable product behavior rather than matching PIL at all costs.

Current recommendation:

- Keep Rust `image` crate decoding.
- Accept the tiny JPEG-vs-PIL difference for now.
- Do not add a native JPEG dependency unless retrieval regression tests show a
  real ranking problem.

If exact parity is needed, evaluate a `libjpeg-turbo` binding and record:

| Decoder | pixel max diff vs PIL | image-feature cosine vs PIL | Packaging impact |
| --- | ---: | ---: | --- |
| image crate JPEG | `~0.047` on sample pixel_values | `0.999913` | pure Rust dependency already used |
| PIL-decoded PNG | `~5.9e-8` | `~1.0` | diagnostic only |
| libjpeg-turbo | TBD | TBD | native library/toolchain needed |
