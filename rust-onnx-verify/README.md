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

End-to-end Rust run:

```powershell
cargo run --release --manifest-path .\rust-onnx-verify\Cargo.toml -- run .\images\browser_20260409_124726_272943500.jpg "山" 1024
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
