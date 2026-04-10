# Model Layout

Current preferred layout:

```text
models/
  source/
    chinese-clip-vit-base-patch16/
    fg-clip2-base/
    qwen3-vl-embedding-2b/

artifacts/
  fgclip2/
    runtime/
      fgclip2_text_short_b1_s64.onnx
      fgclip2_image_core_posin_dynamic.onnx
      assets/
        text_token_embedding_256000x768_f16.bin
        vision_pos_embedding_16x16x768_f32.bin
        logit_params.json
      split/
        fgclip2_text_short_b1_s64_token_embeds.onnx
      quantized/
        fgclip2_text_short_b1_s64_dynamic_int8.onnx
        fgclip2_image_core_posin_dynamic_int8.onnx
    fixtures/
      manifest.json
      manifest_split_text.json
      manifest_dynamic_int8.json
      *.bin
    reports/
      split_text_embedding_report.json
      dynamic_int8_report.json
    diagnostics/
      compare-*/
      rust-*/
      mem-*.txt
      quant-*.txt
```

Compatibility:

- Python and Rust runtime defaults now prefer `artifacts/fgclip2/...`
- Legacy runtime output layout is still accepted as a fallback
- Downloaded source models now prefer `models/source/...`
- Legacy source-model layout is still accepted as a fallback

Migration:

```powershell
uv run python .\scripts\migrate_fgclip2_layout.py
uv run python .\scripts\migrate_fgclip2_layout.py --apply
```
