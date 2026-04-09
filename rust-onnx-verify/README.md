# FG-CLIP2 ONNX Rust verifier

This is a small runtime/verification binary. It can:

- run exported FG-CLIP2 ONNX files from Rust;
- tokenize text with `models/fg-clip2-base/tokenizer.json`;
- preprocess an image into SigLIP2-style patches;
- compare ONNX Runtime outputs with PyTorch reference fixtures.

Expected files, generated outside this crate:

- `../.onnx-wrapper-test/fgclip2_text_short_b1_s64.onnx`
- `../.onnx-wrapper-test/fgclip2_image_core_posin_dynamic.onnx`
- `../.onnx-wrapper-test/assets/vision_pos_embedding_16x16x768_f32.bin`
- `../.onnx-wrapper-test/assets/logit_params.json`
- `../.onnx-wrapper-test/fixtures/manifest.json`

Optional conservative dynamic-int8 files:

- `../.onnx-wrapper-test/quantized/fgclip2_text_short_b1_s64_dynamic_int8.onnx`
- `../.onnx-wrapper-test/quantized/fgclip2_image_core_posin_dynamic_int8.onnx`
- `../.onnx-wrapper-test/fixtures/manifest_dynamic_int8.json`

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
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- .\.onnx-wrapper-test\fixtures\manifest.json
```

Quantized fixture verification:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- .\.onnx-wrapper-test\fixtures\manifest_dynamic_int8.json
```

Split-text fixture verification:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- .\.onnx-wrapper-test\fixtures\manifest_split_text.json
```

Use the recommended low-memory text runtime with the fp32 image model:

```powershell
$env:FGCLIP2_RUNTIME_VARIANT = "split-text"
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
$env:FGCLIP2_IMAGE_ONNX = ".\.onnx-wrapper-test\quantized\fgclip2_image_core_posin_dynamic_int8.onnx"
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run-batch 1024 .\images\a.jpg
```
