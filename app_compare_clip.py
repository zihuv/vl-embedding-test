from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

import demo_chinese_clip
import demo_fg_clip2
import ort_runtime
import project_layout

FG_LAYOUT = project_layout.FGCLIP2_LAYOUT
FG_ONNX_SPLIT_TEXT = FG_LAYOUT.split_text_onnx_resolved
FG_ONNX_FP32_IMAGE = FG_LAYOUT.baseline_image_onnx_resolved
FG_ONNX_INT8_IMAGE = FG_LAYOUT.quantized_image_onnx_resolved
FG_TOKEN_EMBEDDING_F16 = FG_LAYOUT.token_embedding_f16_resolved
FG_VISION_POS_EMBEDDING = FG_LAYOUT.vision_pos_embedding_resolved
FG_LOGIT_PARAMS = FG_LAYOUT.logit_params_resolved
FG_ORT_BACKEND = os.environ.get("FGCLIP2_ORT_BACKEND", "auto")
FG_ORT_PROVIDERS = os.environ.get("FGCLIP2_ORT_PROVIDERS")
FG_ORT_COREML_CACHE_DIR = (
    Path(os.environ["FGCLIP2_ORT_COREML_CACHE_DIR"])
    if "FGCLIP2_ORT_COREML_CACHE_DIR" in os.environ
    else None
)


@dataclass
class SearchResult:
    path: Path
    score: float


class ChineseCLIPSearcher:
    name = "Chinese-CLIP"

    def __init__(
        self,
        image_paths: list[Path],
        model_dir: Path,
        batch_size: int,
        device: torch.device,
    ) -> None:
        print(f"Loading {self.name}: {model_dir}")
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.device = device
        self.model = ChineseCLIPModel.from_pretrained(model_dir).to(device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(model_dir)

        print(f"Pre-encoding images with {self.name}")
        self.image_features = demo_chinese_clip.encode_images(
            self.model,
            self.processor,
            image_paths,
            batch_size,
            device,
        )

    @torch.no_grad()
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        text_features = demo_chinese_clip.encode_text(self.model, self.processor, query, self.device)
        scores = (self.image_features @ text_features.T).squeeze(1).cpu()
        return rank_results(self.image_paths, scores, top_k)

    @torch.no_grad()
    def refresh_image_index(self, image_paths: list[Path]) -> int:
        new_paths = [path for path in image_paths if path not in self.image_paths]
        if new_paths:
            print(f"Encoding {len(new_paths)} new images with {self.name}")
            new_features = demo_chinese_clip.encode_images(
                self.model,
                self.processor,
                new_paths,
                self.batch_size,
                self.device,
            )
        else:
            new_features = None

        self.image_paths, self.image_features = merge_image_index(
            self.image_paths,
            self.image_features,
            image_paths,
            new_paths,
            new_features,
            self.device,
        )
        return len(new_paths)


class FgCLIP2Searcher:
    name = "FG-CLIP2"

    def __init__(
        self,
        image_paths: list[Path],
        model_dir: Path,
        batch_size: int,
        max_image_patches: int | None,
        max_length: int,
        walk_type: str,
        device: torch.device,
    ) -> None:
        print(f"Loading {self.name}: {model_dir}")
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.max_image_patches = max_image_patches
        self.max_length = max_length
        self.walk_type = walk_type
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
        demo_fg_clip2.repair_fgclip2_text_buffers(self.model, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)

        print(f"Pre-encoding images with {self.name}")
        self.image_features = demo_fg_clip2.encode_images(
            self.model,
            self.image_processor,
            image_paths,
            batch_size,
            max_image_patches,
            device,
        )

    @torch.no_grad()
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        text_features = demo_fg_clip2.encode_text(
            self.model,
            self.tokenizer,
            query,
            self.max_length,
            self.walk_type,
            self.device,
        )
        scores = (self.image_features @ text_features.T).squeeze(1)
        scores = scores * self.model.logit_scale.to(self.device).exp() + self.model.logit_bias.to(self.device)
        return rank_results(self.image_paths, scores.cpu(), top_k)

    @torch.no_grad()
    def refresh_image_index(self, image_paths: list[Path]) -> int:
        new_paths = [path for path in image_paths if path not in self.image_paths]
        if new_paths:
            print(f"Encoding {len(new_paths)} new images with {self.name}")
            new_features = demo_fg_clip2.encode_images(
                self.model,
                self.image_processor,
                new_paths,
                self.batch_size,
                self.max_image_patches,
                self.device,
            )
        else:
            new_features = None

        self.image_paths, self.image_features = merge_image_index(
            self.image_paths,
            self.image_features,
            image_paths,
            new_paths,
            new_features,
            self.device,
        )
        return len(new_paths)


class FgCLIP2OnnxSearcher:
    name = "FG-CLIP2 ONNX split-text"

    def __init__(
        self,
        image_paths: list[Path],
        model_dir: Path,
        batch_size: int,
        max_image_patches: int | None,
        max_length: int,
        text_onnx: Path,
        image_onnx: Path,
        token_embedding: Path,
        vision_pos_embedding: Path,
        logit_params: Path,
        provider_request: str,
        coreml_cache_dir: Path | None,
    ) -> None:
        try:
            added_dll_dirs = ort_runtime.prepare_ort_environment(provider_request)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit(
                "onnxruntime is required for FG-CLIP2 ONNX comparison. "
                "Run: uv run --with onnxruntime python app_compare_clip.py "
                "or uv run --with onnxruntime-gpu python app_compare_clip.py"
            ) from exc

        try:
            provider_selection = ort_runtime.resolve_ort_providers(
                ort,
                provider_request,
                coreml_cache_dir=coreml_cache_dir,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

        print(f"Loading {self.name}:")
        print(f"  text ONNX:  {text_onnx}")
        print(f"  image ONNX: {image_onnx}")
        print(f"  requested providers: {provider_selection.requested}")
        print(f"  available providers: {', '.join(provider_selection.available_names)}")
        print(f"  selected providers: {', '.join(provider_selection.selected_names)}")
        if provider_selection.unavailable_names:
            print(f"  unavailable requested providers: {', '.join(provider_selection.unavailable_names)}")
        if added_dll_dirs:
            print(f"  added CUDA DLL dirs: {', '.join(str(path) for path in added_dll_dirs)}")
        if coreml_cache_dir is not None and "CoreMLExecutionProvider" in provider_selection.selected_names:
            print(f"  CoreML cache dir: {coreml_cache_dir}")
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.max_image_patches = max_image_patches
        self.max_length = max_length
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.token_embedding = np.memmap(token_embedding, dtype=np.float16, mode="r", shape=(256000, 768))
        self.base_pos_embedding = np.fromfile(vision_pos_embedding, dtype=np.float32).reshape(16, 16, 768)

        params = json.loads(logit_params.read_text(encoding="utf-8"))
        self.logit_scale_exp = float(params["logit_scale_exp"])
        self.logit_bias = float(params["logit_bias"])

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4
        self.text_session = ort.InferenceSession(
            str(text_onnx),
            sess_options=options,
            providers=provider_selection.provider_args,
        )
        self.image_session = ort.InferenceSession(
            str(image_onnx),
            sess_options=options,
            providers=provider_selection.provider_args,
        )
        print(f"  text session providers: {', '.join(self.text_session.get_providers())}")
        print(f"  image session providers: {', '.join(self.image_session.get_providers())}")

        print(f"Pre-encoding images with {self.name}")
        self.image_features = self.encode_images(image_paths)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        text_features = self.encode_text(query)
        scores = (self.image_features @ text_features[0]).astype(np.float32)
        scores = scores * self.logit_scale_exp + self.logit_bias
        return rank_numpy_results(self.image_paths, scores, top_k)

    def refresh_image_index(self, image_paths: list[Path]) -> int:
        new_paths = [path for path in image_paths if path not in self.image_paths]
        if new_paths:
            print(f"Encoding {len(new_paths)} new images with {self.name}")
            new_features = self.encode_images(new_paths)
        else:
            new_features = None

        self.image_paths, self.image_features = merge_numpy_image_index(
            self.image_paths,
            self.image_features,
            image_paths,
            new_paths,
            new_features,
        )
        return len(new_paths)

    def encode_text(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(
            [query.lower()],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = inputs["input_ids"].astype(np.int64, copy=False)
        token_embeds = self.token_embedding[input_ids].astype(np.float32, copy=False)
        features = self.text_session.run(["text_features"], {"token_embeds": token_embeds})[0].astype(np.float32)
        return normalize_numpy(features)

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        all_features: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                max_num_patches = max(demo_fg_clip2.determine_max_patches(image) for image in images)
                if self.max_image_patches is not None:
                    max_num_patches = min(max_num_patches, self.max_image_patches)
                inputs = self.image_processor(
                    images=images,
                    max_num_patches=max_num_patches,
                    return_tensors="np",
                )
            finally:
                for image in images:
                    image.close()

            pixel_values = inputs["pixel_values"].astype(np.float32, copy=False)
            pixel_attention_mask = inputs["pixel_attention_mask"].astype(np.int32, copy=False)
            spatial_shapes = inputs["spatial_shapes"]
            pos_embed = make_onnx_pos_embed(self.base_pos_embedding, spatial_shapes, pixel_values.shape[1])
            features = self.image_session.run(
                ["image_features"],
                {
                    "pixel_values": pixel_values,
                    "pixel_attention_mask": pixel_attention_mask,
                    "pos_embed": pos_embed,
                },
            )[0].astype(np.float32)
            all_features.append(normalize_numpy(features))
            print(f"Encoded ONNX images {min(start + self.batch_size, len(image_paths))}/{len(image_paths)}")

        return np.concatenate(all_features, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Chinese-CLIP and FG-CLIP2 with Gradio.")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "images")
    parser.add_argument("--chinese-model-dir", type=Path, default=demo_chinese_clip.DEFAULT_MODEL_DIR)
    parser.add_argument("--fg-model-dir", type=Path, default=demo_fg_clip2.DEFAULT_MODEL_DIR)
    parser.add_argument("--chinese-batch-size", type=int, default=8)
    parser.add_argument("--fg-batch-size", type=int, default=4)
    parser.add_argument("--fg-max-image-patches", type=int, default=None)
    parser.add_argument("--fg-max-length", type=int, default=64)
    parser.add_argument("--fg-walk-type", choices=["short", "long"], default="short")
    parser.add_argument("--fg-onnx-mode", choices=["auto", "disabled", "split-text", "lowmem"], default="auto")
    parser.add_argument("--fg-onnx-batch-size", type=int, default=1)
    parser.add_argument("--fg-onnx-max-image-patches", type=int, default=None)
    parser.add_argument("--fg-onnx-text", type=Path, default=FG_ONNX_SPLIT_TEXT)
    parser.add_argument("--fg-onnx-image", type=Path, default=None)
    parser.add_argument(
        "--fg-onnx-backend",
        default=FG_ORT_BACKEND,
        help="Backend profile: auto, cuda, cpu, coreml. Ignored when --fg-onnx-providers is set.",
    )
    parser.add_argument(
        "--fg-onnx-providers",
        default=FG_ORT_PROVIDERS,
        help="Optional explicit ORT provider order. Examples: cuda,cpu or coreml,cpu. Overrides --fg-onnx-backend.",
    )
    parser.add_argument(
        "--fg-onnx-coreml-cache-dir",
        type=Path,
        default=FG_ORT_COREML_CACHE_DIR,
        help="Optional cache directory passed to CoreMLExecutionProvider when CoreML is selected.",
    )
    parser.add_argument("--fg-token-embedding", type=Path, default=FG_TOKEN_EMBEDDING_F16)
    parser.add_argument("--fg-vision-pos-embedding", type=Path, default=FG_VISION_POS_EMBEDDING)
    parser.add_argument("--fg-logit-params", type=Path, default=FG_LOGIT_PARAMS)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    return parser.parse_args()


def rank_results(image_paths: list[Path], scores: torch.Tensor, top_k: int) -> list[SearchResult]:
    top_k = min(top_k, len(image_paths))
    ranking = torch.topk(scores, k=top_k)
    results = []
    for index in ranking.indices.tolist():
        results.append(SearchResult(path=image_paths[index], score=scores[index].item()))
    return results


def rank_numpy_results(image_paths: list[Path], scores: np.ndarray, top_k: int) -> list[SearchResult]:
    scores = scores.reshape(-1)
    top_k = min(top_k, len(image_paths))
    ranking = np.argsort(-scores)[:top_k]
    return [SearchResult(path=image_paths[int(index)], score=float(scores[int(index)])) for index in ranking]


def merge_image_index(
    old_paths: list[Path],
    old_features: torch.Tensor,
    target_paths: list[Path],
    new_paths: list[Path],
    new_features: torch.Tensor | None,
    device: torch.device,
) -> tuple[list[Path], torch.Tensor]:
    old_features_cpu = old_features.detach().cpu()
    features_by_path = {path: old_features_cpu[index] for index, path in enumerate(old_paths)}

    if new_features is not None:
        new_features_cpu = new_features.detach().cpu()
        for index, path in enumerate(new_paths):
            features_by_path[path] = new_features_cpu[index]

    merged_features = torch.stack([features_by_path[path] for path in target_paths], dim=0).to(device)
    return list(target_paths), merged_features


def merge_numpy_image_index(
    old_paths: list[Path],
    old_features: np.ndarray,
    target_paths: list[Path],
    new_paths: list[Path],
    new_features: np.ndarray | None,
) -> tuple[list[Path], np.ndarray]:
    features_by_path = {path: old_features[index] for index, path in enumerate(old_paths)}
    if new_features is not None:
        for index, path in enumerate(new_paths):
            features_by_path[path] = new_features[index]

    merged_features = np.stack([features_by_path[path] for path in target_paths], axis=0).astype(np.float32)
    return list(target_paths), merged_features


def normalize_numpy(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, np.finfo(np.float32).tiny)


def make_onnx_pos_embed(
    base_pos_embedding: np.ndarray,
    spatial_shapes: np.ndarray,
    max_patches: int,
) -> np.ndarray:
    batch = int(spatial_shapes.shape[0])
    pos_embed = np.zeros((batch, max_patches, 768), dtype=np.float32)
    for index, shape in enumerate(spatial_shapes.tolist()):
        height, width = int(shape[0]), int(shape[1])
        resized = resize_pos_embedding_no_antialias(base_pos_embedding, height, width)
        patch_count = min(height * width, max_patches)
        pos_embed[index, :patch_count, :] = resized.reshape(-1, 768)[:patch_count]
    return pos_embed


def resize_pos_embedding_no_antialias(base_pos_embedding: np.ndarray, height: int, width: int) -> np.ndarray:
    output = np.empty((height, width, 768), dtype=np.float32)
    source_height, source_width = base_pos_embedding.shape[:2]
    for y in range(height):
        source_y = linear_source_coordinate(y, height, source_height)
        y0 = int(np.floor(source_y))
        y1 = min(y0 + 1, source_height - 1)
        wy = source_y - y0
        for x in range(width):
            source_x = linear_source_coordinate(x, width, source_width)
            x0 = int(np.floor(source_x))
            x1 = min(x0 + 1, source_width - 1)
            wx = source_x - x0
            top = base_pos_embedding[y0, x0] + (base_pos_embedding[y0, x1] - base_pos_embedding[y0, x0]) * wx
            bottom = base_pos_embedding[y1, x0] + (base_pos_embedding[y1, x1] - base_pos_embedding[y1, x0]) * wx
            output[y, x] = top + (bottom - top) * wy
    return output


def linear_source_coordinate(output_index: int, output_size: int, input_size: int) -> float:
    source = (output_index + 0.5) * input_size / output_size - 0.5
    return min(max(source, 0.0), input_size - 1.0)


def gallery_items(results: list[SearchResult]) -> list[tuple[str, str]]:
    return [
        (str(result.path), f"{rank}. {result.score:.4f} | {result.path.name}")
        for rank, result in enumerate(results, start=1)
    ]


def table_rows(results: list[SearchResult]) -> list[list[str | float | int]]:
    return [
        [rank, result.path.name, round(result.score, 6), str(result.path)]
        for rank, result in enumerate(results, start=1)
    ]


def build_demo(
    chinese_searcher: ChineseCLIPSearcher,
    fg_searcher: FgCLIP2Searcher,
    fg_onnx_searcher: FgCLIP2OnnxSearcher | None,
    image_dir: Path,
    default_top_k: int,
    device: torch.device,
):
    index_lock = threading.RLock()

    def search(query: str, top_k: int):
        top_k = int(top_k)
        query = query.strip()
        if not query:
            return [], [], [], [], "请输入搜索词。"

        with index_lock:
            chinese_results = chinese_searcher.search(query, top_k)
            fg_results = fg_searcher.search(query, top_k)
            fg_onnx_results = fg_onnx_searcher.search(query, top_k) if fg_onnx_searcher is not None else None
            status = f"query: {query} | images: {len(chinese_searcher.image_paths)} | device: {device}"

        outputs: tuple[Any, ...] = (
            gallery_items(chinese_results),
            table_rows(chinese_results),
            gallery_items(fg_results),
            table_rows(fg_results),
        )
        if fg_onnx_results is not None:
            outputs += (gallery_items(fg_onnx_results), table_rows(fg_onnx_results))
        return (*outputs, status)

    def refresh_and_search(query: str, top_k: int):
        query = query.strip() or "山"
        with index_lock:
            image_paths = demo_chinese_clip.list_images(image_dir)
            fg_image_paths = demo_fg_clip2.list_images(image_dir)
            if image_paths != fg_image_paths:
                raise gr.Error("两套索引扫描到的图片顺序不一致。")

            chinese_added = chinese_searcher.refresh_image_index(image_paths)
            fg_added = fg_searcher.refresh_image_index(image_paths)
            fg_onnx_added = (
                fg_onnx_searcher.refresh_image_index(image_paths) if fg_onnx_searcher is not None else None
            )
            outputs = search(query, int(top_k))

        status = (
            f"{outputs[-1]} | refreshed: Chinese-CLIP +{chinese_added}, "
            f"FG-CLIP2 +{fg_added}"
        )
        if fg_onnx_added is not None:
            status += f", FG-CLIP2 ONNX +{fg_onnx_added}"
        return (*outputs[:-1], status)

    image_count = len(chinese_searcher.image_paths)
    with gr.Blocks(title="CLIP 对照检索") as demo:
        gr.Markdown("# Chinese-CLIP / FG-CLIP2 图片检索对照")
        status = gr.Markdown(f"已预编码 {image_count} 张图片。device: `{device}`")

        with gr.Row():
            query = gr.Textbox(label="自然语言搜索", value="山", scale=4)
            top_k = gr.Slider(1, min(20, image_count), value=min(default_top_k, image_count), step=1, label="Top K")
            search_button = gr.Button("搜索", variant="primary")
            refresh_button = gr.Button("刷新图片")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Chinese-CLIP")
                chinese_gallery = gr.Gallery(label="排序结果", columns=4, height=520, object_fit="cover")
                chinese_table = gr.Dataframe(
                    headers=["rank", "filename", "score", "path"],
                    datatype=["number", "str", "number", "str"],
                    label="排名",
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("## FG-CLIP2")
                fg_gallery = gr.Gallery(label="排序结果", columns=4, height=520, object_fit="cover")
                fg_table = gr.Dataframe(
                    headers=["rank", "filename", "score", "path"],
                    datatype=["number", "str", "number", "str"],
                    label="排名",
                    interactive=False,
                )

            if fg_onnx_searcher is not None:
                with gr.Column():
                    gr.Markdown("## FG-CLIP2 ONNX / split-text")
                    fg_onnx_gallery = gr.Gallery(label="排序结果", columns=4, height=520, object_fit="cover")
                    fg_onnx_table = gr.Dataframe(
                        headers=["rank", "filename", "score", "path"],
                        datatype=["number", "str", "number", "str"],
                        label="排名",
                        interactive=False,
                    )

        outputs = [chinese_gallery, chinese_table, fg_gallery, fg_table]
        if fg_onnx_searcher is not None:
            outputs.extend([fg_onnx_gallery, fg_onnx_table])
        outputs.append(status)
        demo.load(search, inputs=[query, top_k], outputs=outputs)
        search_button.click(search, inputs=[query, top_k], outputs=outputs)
        refresh_button.click(refresh_and_search, inputs=[query, top_k], outputs=outputs)
        query.submit(search, inputs=[query, top_k], outputs=outputs)

        gr.Examples(
            examples=["山", "雪山", "湖边", "一只猫", "房间里的床", "蓝天和草地"],
            inputs=query,
        )

    return demo


def build_fg_onnx_searcher(
    args: argparse.Namespace,
    image_paths: list[Path],
) -> FgCLIP2OnnxSearcher | None:
    mode = args.fg_onnx_mode
    if mode == "disabled":
        return None

    image_onnx = args.fg_onnx_image
    if image_onnx is None:
        image_onnx = FG_ONNX_INT8_IMAGE if mode == "lowmem" else FG_ONNX_FP32_IMAGE

    required_assets = [
        args.fg_onnx_text,
        image_onnx,
        args.fg_token_embedding,
        args.fg_vision_pos_embedding,
        args.fg_logit_params,
    ]
    missing_assets = [path for path in required_assets if not path.exists()]
    if missing_assets:
        if mode == "auto":
            print("Skipping FG-CLIP2 ONNX comparison; missing assets:")
            for path in missing_assets:
                print(f"  {path}")
            return None
        raise SystemExit("Missing FG-CLIP2 ONNX assets:\n" + "\n".join(f"  {path}" for path in missing_assets))

    if mode == "auto" and not python_package_available("onnxruntime"):
        print("Skipping FG-CLIP2 ONNX comparison; onnxruntime is not installed.")
        print("Run with: uv run --with onnxruntime python app_compare_clip.py")
        print("For NVIDIA CUDA on Windows, use: uv run --with onnxruntime-gpu python app_compare_clip.py")
        return None

    provider_request = args.fg_onnx_providers or args.fg_onnx_backend

    return FgCLIP2OnnxSearcher(
        image_paths=image_paths,
        model_dir=args.fg_model_dir,
        batch_size=args.fg_onnx_batch_size,
        max_image_patches=(
            args.fg_onnx_max_image_patches
            if args.fg_onnx_max_image_patches is not None
            else args.fg_max_image_patches
        ),
        max_length=args.fg_max_length,
        text_onnx=args.fg_onnx_text,
        image_onnx=image_onnx,
        token_embedding=args.fg_token_embedding,
        vision_pos_embedding=args.fg_vision_pos_embedding,
        logit_params=args.fg_logit_params,
        provider_request=provider_request,
        coreml_cache_dir=args.fg_onnx_coreml_cache_dir,
    )


def python_package_available(package: str) -> bool:
    try:
        __import__(package)
    except ImportError:
        return False
    return True


def main() -> None:
    args = parse_args()
    demo_chinese_clip.ensure_model(args.chinese_model_dir, args.local_only)
    demo_fg_clip2.ensure_model(args.fg_model_dir, args.local_only)
    image_paths = demo_chinese_clip.list_images(args.image_dir)
    fg_image_paths = demo_fg_clip2.list_images(args.image_dir)
    if image_paths != fg_image_paths:
        raise SystemExit("Image path order differs between the two searchers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} images under {args.image_dir}")

    chinese_searcher = ChineseCLIPSearcher(
        image_paths=image_paths,
        model_dir=args.chinese_model_dir,
        batch_size=args.chinese_batch_size,
        device=device,
    )
    fg_searcher = FgCLIP2Searcher(
        image_paths=image_paths,
        model_dir=args.fg_model_dir,
        batch_size=args.fg_batch_size,
        max_image_patches=args.fg_max_image_patches,
        max_length=args.fg_max_length,
        walk_type=args.fg_walk_type,
        device=device,
    )
    fg_onnx_searcher = build_fg_onnx_searcher(args, image_paths)

    demo = build_demo(chinese_searcher, fg_searcher, fg_onnx_searcher, args.image_dir, args.top_k, device)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
