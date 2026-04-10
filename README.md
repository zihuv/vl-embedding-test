chinese clip vs fg-clip2 你知道吗？

## VL embedding demos

三个脚本都会在第一次运行时把模型下载到项目内的 `models/`，然后用自然语言检索 `images/` 里的本地图片。

当前推荐目录布局见 `docs/model-layout.md`。

- 源模型下载到 `models/source/`
- FG-CLIP2 导出的 ONNX / 量化 / split-text / fixture / report 放到 `artifacts/fgclip2/`
- 旧目录布局仍然兼容

```powershell
uv run python demo_qwen3_vl_embedding.py "一张有人和动物的照片"
uv run python demo_fg_clip2.py "一张有人和动物的照片"
uv run python demo_chinese_clip.py "一张有人和动物的照片"
```

Smoke test:

```powershell
uv run python demo_chinese_clip.py "山" --top-k 5
uv run python demo_fg_clip2.py "山" --top-k 5
uv run python demo_qwen3_vl_embedding.py "山" --top-k 3 --batch-size 1
```

当前 demo 是单模型、终端排序版。后面可以把三份排序结果接到同一个 Gradio 页面里。

Gradio 对照页面：

```powershell
uv run python app_compare_clip.py
```

默认会在启动时加载 Chinese-CLIP 和 FG-CLIP2，并把 `images/` 预编码到内存。打开终端里打印的本地地址，然后输入自然语言搜索。

如果已经导出 FG-CLIP2 ONNX / split-text 资产，可以加上 ONNX Runtime 依赖。页面会自动多出第三列 `FG-CLIP2 ONNX / split-text`：

```powershell
uv run --with onnxruntime python app_compare_clip.py
```

现在 `FG-CLIP2 ONNX / split-text` 支持配置 ORT provider 顺序：

```powershell
uv run --with onnxruntime-gpu python app_compare_clip.py --fg-onnx-mode split-text --fg-onnx-providers cuda,cpu
uv run --with onnxruntime python app_compare_clip.py --fg-onnx-mode split-text --fg-onnx-providers cpu
```

也支持更简单的 backend profile：

```powershell
uv run --with onnxruntime-gpu python app_compare_clip.py --fg-onnx-mode split-text --fg-onnx-backend auto
uv run --with onnxruntime-gpu python app_compare_clip.py --fg-onnx-mode split-text --fg-onnx-backend cuda
uv run --with onnxruntime-gpu python app_compare_clip.py --fg-onnx-mode split-text --fg-onnx-backend cpu
```

优先级规则：

- 如果设置了 `--fg-onnx-providers` 或环境变量 `FGCLIP2_ORT_PROVIDERS`，就按显式 provider 顺序走
- 否则看 `--fg-onnx-backend` 或环境变量 `FGCLIP2_ORT_BACKEND`
- 如果两者都没指定，默认是 `auto`
- `auto` 在 Windows 上默认等于 `cuda,cpu`
- `auto` 在 macOS 上默认等于 `coreml,cpu`
- `auto` 在其他平台上默认等于 `cpu`

强制打开第三列，或测试最低内存实验版：

```powershell
uv run --with onnxruntime python app_compare_clip.py --fg-onnx-mode split-text
uv run --with onnxruntime python app_compare_clip.py --fg-onnx-mode lowmem
```

FG-CLIP2 ONNX 的文件组成、输入输出约定、推荐运行模式和命令示例见
`docs/fgclip2-onnx-usage.md`。

如果你已经有一批旧目录产物，先看迁移计划：

```powershell
uv run python .\scripts\migrate_fgclip2_layout.py
uv run python .\scripts\migrate_fgclip2_layout.py --apply
```

如果启动后又往 `images/` 放了新图，点页面上的“刷新图片”。它会增量编码新文件；已有图片会复用内存里的向量。

CPU 机器上可以先降低 FG-CLIP2 的图片 patch 数：

```powershell
uv run python app_compare_clip.py --fg-max-image-patches 256 --fg-batch-size 1
```

一键脚本：

```powershell
.\scripts\run_app_onnx_cuda.ps1
.\scripts\run_app_onnx_cuda.ps1 -Backend cuda
.\scripts\run_app_onnx_cuda.ps1 -Backend cpu
.\scripts\run_app_onnx_cuda.ps1 -OnnxProviders cuda,cpu -ExtraArgs @("--fg-onnx-max-image-patches", "576")
```

```bash
chmod +x ./scripts/run_app_onnx_macos.sh
./scripts/run_app_onnx_macos.sh
FGCLIP2_ORT_BACKEND=coreml ./scripts/run_app_onnx_macos.sh
FGCLIP2_ORT_PROVIDERS=coreml,cpu ./scripts/run_app_onnx_macos.sh --fg-onnx-max-image-patches 576
```

macOS 脚本会优先尝试 `CoreMLExecutionProvider`，找不到时自动回退到 `CPUExecutionProvider`。如果你要真正跑 CoreML，需要使用暴露了 `CoreMLExecutionProvider` 的 ONNX Runtime Python 环境；默认 `uv run --with onnxruntime` 这条路径更适合作为 CPU fallback。

Windows 上如果要走 NVIDIA CUDA，建议直接用：

```powershell
$env:FGCLIP2_ORT_BACKEND = "cuda"
uv run --with onnxruntime-gpu python app_compare_clip.py --fg-onnx-mode split-text
```

如果 `CUDAExecutionProvider` 能看到但 session 还是回退到 CPU，通常是 CUDA / cuDNN DLL 搜索路径问题。可以显式指定：

```powershell
$env:FGCLIP2_ORT_DLL_PATHS = "C:\Users\10413\miniconda3\Lib\site-packages\torch\lib"
.\scripts\run_app_onnx_cuda.ps1 -Backend cuda
```

Windows 一键脚本也会自动尝试这些常见目录：

- `%USERPROFILE%\miniconda3\Lib\site-packages\torch\lib`
- `%USERPROFILE%\anaconda3\Lib\site-packages\torch\lib`

当前这台机器上已经验证到 `uv run --with onnxruntime-gpu` 可见 `CUDAExecutionProvider`，适合 RTX 3050 这类 NVIDIA 卡。

> 注意：当前 Windows 环境里通过 PyPI 安装到的是 CPU 版 `torch==2.8.0`。这三份 demo 都能在 CPU 上跑通，但 Qwen3-VL-Embedding-2B 明显更慢；正式对比建议换成可用 CUDA 的 PyTorch 环境。
