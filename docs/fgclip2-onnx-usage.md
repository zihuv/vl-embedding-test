# FG-CLIP2 ONNX Usage

This document explains how to load and use the exported FG-CLIP2 ONNX models in
this repository.

Current preferred layout is `artifacts/fgclip2/...`, with source model files
under `models/source/...`.

The short version:

- default runtime: `split-text + fp32 image`
- text output: normalized `768`-dim vector
- image output: normalized `768`-dim vector
- retrieval: use cosine similarity or dot product on normalized vectors
- if you need the original model score, apply `logit_scale_exp` and
  `logit_bias` from `logit_params.json`

## Files You Need

Recommended default runtime files:

- `artifacts/fgclip2/runtime/split/fgclip2_text_short_b1_s64_token_embeds.onnx`
- `artifacts/fgclip2/runtime/assets/text_token_embedding_256000x768_f16.bin`
- `artifacts/fgclip2/runtime/fgclip2_image_core_posin_dynamic.onnx`
- `artifacts/fgclip2/runtime/assets/vision_pos_embedding_16x16x768_f32.bin`
- `artifacts/fgclip2/runtime/assets/logit_params.json`
- `models/source/fg-clip2-base/tokenizer.json`

Optional comparison / experiment files:

- full text fp32:
  `artifacts/fgclip2/runtime/fgclip2_text_short_b1_s64.onnx`
- dynamic-int8:
  `artifacts/fgclip2/runtime/quantized/fgclip2_text_short_b1_s64_dynamic_int8.onnx`
  and
  `artifacts/fgclip2/runtime/quantized/fgclip2_image_core_posin_dynamic_int8.onnx`

## Recommended Runtime Modes

`FGCLIP2_RUNTIME_VARIANT` controls which artifacts are used.

| Variant | Text Model | Image Model | Use |
| --- | --- | --- | --- |
| unset | split-text | fp32 image | recommended default |
| `split-text` | split-text | fp32 image | same as default, explicit |
| `baseline` | full fp32 text | fp32 image | correctness / memory comparison |
| `dynamic-int8` | quantized text | quantized image | optional experiment |
| `lowmem` | split-text | quantized image | lowest-memory experiment |

## Fastest Way To Use It

Build the Rust runtime:

```powershell
cargo build --release --manifest-path .\rust-onnx-verify\Cargo.toml
```

Run text-to-image scoring with the recommended default runtime:

```powershell
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe run .\artifacts\fgclip2\diagnostics\pil-decoded-source.png "山" 1024
```

Text-only embedding:

```powershell
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe run-text "山"
```

Image-only embedding:

```powershell
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe run-batch 1024 .\artifacts\fgclip2\diagnostics\pil-decoded-source.png
```

Use the full fp32 text model for comparison:

```powershell
$env:FGCLIP2_RUNTIME_VARIANT = "baseline"
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe run .\artifacts\fgclip2\diagnostics\pil-decoded-source.png "山" 1024
```

## What The Models Output

Both text and image models output normalized `768`-dimensional float vectors.

- text output name: `text_features`
- image output name: `image_features`

Because the vectors are L2-normalized:

- cosine similarity and dot product are equivalent for retrieval
- you can store only the `768` floats per item in your vector index

If you want the raw FG-CLIP2 score instead of just cosine:

```text
logit = cosine(text_feature, image_feature) * logit_scale_exp + logit_bias
```

Read `logit_scale_exp` and `logit_bias` from
`artifacts/fgclip2/runtime/assets/logit_params.json`.

For most retrieval systems, cosine is enough.

## Text Model Loading

There are two text-model formats in this repo.

### 1. Full Text ONNX

File:

- `artifacts/fgclip2/runtime/fgclip2_text_short_b1_s64.onnx`

Input:

- `input_ids [1, 64]`

Use this only when you want the simplest integration or need a baseline
comparison. It uses much more memory because the token embedding table is
inside the ONNX file.

### 2. Split-Text ONNX

Files:

- `artifacts/fgclip2/runtime/split/fgclip2_text_short_b1_s64_token_embeds.onnx`
- `artifacts/fgclip2/runtime/assets/text_token_embedding_256000x768_f16.bin`

Input:

- `token_embeds [1, 64, 768]`

Important: this model does **not** take `input_ids` directly.

You must:

1. tokenize the query with `models/source/fg-clip2-base/tokenizer.json`
2. pad / truncate to length `64`
3. read the corresponding rows from
   `text_token_embedding_256000x768_f16.bin`
4. convert those rows to `f32`
5. stack them into `token_embeds [1, 64, 768]`
6. feed that tensor into the ONNX model

This is the recommended path because it keeps memory much lower while matching
the fp32 baseline closely.

## Text Tokenization Rules Used By This Repo

The Rust runtime does the following before text inference:

- lowercases the query
- uses `models/source/fg-clip2-base/tokenizer.json`
- enables special tokens
- truncates to `64`
- pads to fixed length `64`
- pad token id is `0`

If you want exact parity with this repo, follow the same rules.

## Image Model Loading

File:

- `artifacts/fgclip2/runtime/fgclip2_image_core_posin_dynamic.onnx`

Inputs:

- `pixel_values [B, N, 768]`
- `pixel_attention_mask [B, N]`
- `pos_embed [B, N, 768]`

Output:

- `image_features [B, 768]`

The image model expects pre-patchified image tokens, not raw pixels.

### Image Preprocessing Used By This Repo

The Rust runtime does the following:

1. decode the image to RGB
2. choose a target size that fits within `max_patches`
   with patch size `16`
3. resize the image with triangle filtering
4. split the resized image into `16x16` patches
5. flatten each patch into `768 = 16 * 16 * 3` floats
6. normalize pixel values with:

```text
value = pixel / 127.5 - 1.0
```

7. build `pixel_attention_mask` with `1` for valid patches and `0` for padded
   patches
8. resize the base vision position embedding from
   `vision_pos_embedding_16x16x768_f32.bin` to the actual patch grid
9. feed `pixel_values`, `pixel_attention_mask`, and `pos_embed` into the ONNX
   model

### Patch Count

Common settings:

- `256`: lower memory, faster
- `576`: balanced
- `1024`: best quality in the current prototype

Current project recommendation: start with `1024`.

## Minimal Integration Pattern

### Text-to-Image Search

If your images are already indexed:

1. load the text model
2. encode the user query to a `768`-dim text vector
3. compare it with stored image vectors

You do **not** need the image ONNX model for ordinary text-to-image search over
precomputed image vectors.

### Image Indexing

When importing images:

1. load the image model
2. preprocess each image to patches
3. encode to `768`-dim image vectors
4. store those vectors in your index

You do **not** need the text model while indexing images.

### Single Runtime Recommendation

If memory matters, keep text and image sessions separate and avoid loading both
at once unless the request really needs both.

A practical pattern is:

1. default: load no ONNX session
2. text request: load text session, encode, keep briefly or unload
3. image import request: unload text session, load image session, encode images

## Override Paths Explicitly

You can override model paths with environment variables:

```powershell
$env:FGCLIP2_TEXT_ONNX = ".\artifacts\fgclip2\runtime\split\fgclip2_text_short_b1_s64_token_embeds.onnx"
$env:FGCLIP2_IMAGE_ONNX = ".\artifacts\fgclip2\runtime\fgclip2_image_core_posin_dynamic.onnx"
```

If the split-text token table is not at the default asset path, override it:

```powershell
$env:FGCLIP2_TEXT_TOKEN_EMBEDDING = ".\artifacts\fgclip2\runtime\assets\text_token_embedding_256000x768_f16.bin"
```

If the token table is stored as `f32` instead of `f16`, set:

```powershell
$env:FGCLIP2_TEXT_TOKEN_EMBEDDING_DTYPE = "f32"
```

## Verification Commands

Verify the full fp32 baseline:

```powershell
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe .\artifacts\fgclip2\fixtures\manifest.json
```

Verify the split-text path:

```powershell
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe .\artifacts\fgclip2\fixtures\manifest_split_text.json
```

Verify the quantized experiment:

```powershell
.\rust-onnx-verify\target\release\fgclip2-onnx-verify.exe .\artifacts\fgclip2\fixtures\manifest_dynamic_int8.json
```

## Current Practical Numbers

Measured on Windows CPU in this repo:

- default unset runtime: about `521.7 MiB` peak working set for
  `run image + text 1024`
- split-text text-only: about `369 MiB`
- image-only 1024: about `525 MiB`
- baseline full-text end-to-end: about `1.8 GiB`

That is why the default is now `split-text`.

## Common Mistakes

1. Feeding `input_ids` into the split-text ONNX.
   It expects `token_embeds`, not token ids.

2. Forgetting `text_token_embedding_256000x768_f16.bin`.
   The split-text ONNX cannot run without that external token table.

3. Treating the image ONNX as a raw-image model.
   It expects patch tokens, attention mask, and resized position embeddings.

4. Comparing unnormalized vectors with the wrong metric.
   These outputs are already normalized; use cosine or dot product directly.

5. Loading text and image sessions together when you only need one.
   This increases memory without helping the common retrieval path.
