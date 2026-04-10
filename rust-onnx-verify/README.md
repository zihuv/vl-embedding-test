# FG-CLIP2 ONNX Rust verifier

This is a small runtime/verification binary. It can:

- run exported FG-CLIP2 ONNX files from Rust;
- tokenize text with `models/source/fg-clip2-base/tokenizer.json`;
- preprocess an image into SigLIP2-style patches;
- compare ONNX Runtime outputs with PyTorch reference fixtures.

For the full integration guide, see
[`docs/fgclip2-onnx-usage.md`](../docs/fgclip2-onnx-usage.md).

Expected files, generated outside this crate:

- `../artifacts/fgclip2/runtime/fgclip2_text_short_b1_s64.onnx`
- `../artifacts/fgclip2/runtime/fgclip2_image_core_posin_dynamic.onnx`
- `../artifacts/fgclip2/runtime/assets/vision_pos_embedding_16x16x768_f32.bin`
- `../artifacts/fgclip2/runtime/assets/logit_params.json`
- `../artifacts/fgclip2/fixtures/manifest.json`

Optional conservative dynamic-int8 files:

- `../artifacts/fgclip2/runtime/quantized/fgclip2_text_short_b1_s64_dynamic_int8.onnx`
- `../artifacts/fgclip2/runtime/quantized/fgclip2_image_core_posin_dynamic_int8.onnx`
- `../artifacts/fgclip2/fixtures/manifest_dynamic_int8.json`

Legacy layout paths are still accepted as a fallback.

End-to-end Rust run:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run .\images\browser_20260409_124726_272943500.jpg "山" 1024
```

Text-only encoding:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run-text "山"
```

Batch image encoding:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run-batch 1024 .\images\a.jpg .\images\b.jpg
```

Fixture verification:

From the repository root:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- .\artifacts\fgclip2\fixtures\manifest.json
```

Quantized fixture verification:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- .\artifacts\fgclip2\fixtures\manifest_dynamic_int8.json
```

Split-text fixture verification:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- .\artifacts\fgclip2\fixtures\manifest_split_text.json
```

By default, `run` / `run-text` use the recommended low-memory text runtime with
the fp32 image model.

Explicitly use that same profile:

```powershell
$env:FGCLIP2_RUNTIME_VARIANT = "split-text"
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run .\images\browser_20260409_124726_272943500.jpg "山" 1024
```

Use the legacy full-text fp32 path for a correctness/memory comparison:

```powershell
$env:FGCLIP2_RUNTIME_VARIANT = "baseline"
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run .\images\browser_20260409_124726_272943500.jpg "山" 1024
```

Use the optional quantized runtime for `run` / `run-batch`:

```powershell
$env:FGCLIP2_RUNTIME_VARIANT = "dynamic-int8"
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run .\images\browser_20260409_124726_272943500.jpg "山" 1024
```

Use the lowest-memory experimental runtime:

```powershell
$env:FGCLIP2_RUNTIME_VARIANT = "lowmem"
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run .\images\browser_20260409_124726_272943500.jpg "山" 1024
```

Or override one model path explicitly:

```powershell
$env:FGCLIP2_IMAGE_ONNX = ".\artifacts\fgclip2\runtime\quantized\fgclip2_image_core_posin_dynamic_int8.onnx"
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run-batch 1024 .\images\a.jpg
```
