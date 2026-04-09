chinese clip vs fg-clip2 你知道吗？

## VL embedding demos

三个脚本都会在第一次运行时把模型下载到项目内的 `models/`，然后用自然语言检索 `images/` 里的本地图片。

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

如果启动后又往 `images/` 放了新图，点页面上的“刷新图片”。它会增量编码新文件；已有图片会复用内存里的向量。

CPU 机器上可以先降低 FG-CLIP2 的图片 patch 数：

```powershell
uv run python app_compare_clip.py --fg-max-image-patches 256 --fg-batch-size 1
```

> 注意：当前 Windows 环境里通过 PyPI 安装到的是 CPU 版 `torch==2.8.0`。这三份 demo 都能在 CPU 上跑通，但 Qwen3-VL-Embedding-2B 明显更慢；正式对比建议换成可用 CUDA 的 PyTorch 环境。
