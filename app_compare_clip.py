from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, ChineseCLIPProcessor

import demo_chinese_clip
import demo_fg_clip2
import ort_runtime
import project_layout

CHINESE_LAYOUT = project_layout.CHINESE_CLIP_LAYOUT
FG_LAYOUT = project_layout.FGCLIP2_LAYOUT

CHINESE_ONNX_MANIFEST = CHINESE_LAYOUT.export_manifest_resolved
CHINESE_ONNX_TEXT = CHINESE_LAYOUT.text_onnx_resolved
CHINESE_ONNX_IMAGE = CHINESE_LAYOUT.image_onnx_resolved

FG_ONNX_SPLIT_TEXT = FG_LAYOUT.split_text_onnx_resolved
FG_ONNX_FP32_IMAGE = FG_LAYOUT.baseline_image_onnx_resolved
FG_ONNX_INT8_IMAGE = FG_LAYOUT.quantized_image_onnx_resolved
FG_TOKEN_EMBEDDING_F16 = FG_LAYOUT.token_embedding_f16_resolved
FG_VISION_POS_EMBEDDING = FG_LAYOUT.vision_pos_embedding_resolved

FG_ORT_BACKEND = os.environ.get("FGCLIP2_ORT_BACKEND", "auto")
FG_ORT_PROVIDERS = os.environ.get("FGCLIP2_ORT_PROVIDERS")
FG_ORT_COREML_CACHE_DIR = (
    Path(os.environ["FGCLIP2_ORT_COREML_CACHE_DIR"])
    if "FGCLIP2_ORT_COREML_CACHE_DIR" in os.environ
    else None
)

SUPPORTED_FG_PATCH_VARIANTS = (128, 256, 576, 784, 1024)
DEFAULT_FG_PATCH_VARIANTS = "256,128"


@dataclass(frozen=True)
class SearchResult:
    path: Path
    score: float


@dataclass(frozen=True)
class OrtRuntimeContext:
    ort: Any
    provider_request: str
    provider_selection: ort_runtime.OrtProviderSelection
    intra_threads: int | None


class Searcher(Protocol):
    name: str
    image_paths: list[Path]

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        ...

    def refresh_image_index(self, image_paths: list[Path]) -> int:
        ...


class BaseOnnxSearcher:
    name: str

    def __init__(self, image_paths: list[Path], batch_size: int) -> None:
        self.image_paths = list(image_paths)
        self.batch_size = batch_size
        print(f"Pre-encoding images with {self.name}")
        self.image_features = self.encode_images(image_paths)

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        raise NotImplementedError

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


class ChineseCLIPOnnxSearcher(BaseOnnxSearcher):
    name = "Chinese-CLIP ONNX"

    def __init__(
        self,
        image_paths: list[Path],
        model_dir: Path,
        batch_size: int,
        text_onnx: Path,
        image_onnx: Path,
        export_manifest: Path,
        runtime: OrtRuntimeContext,
    ) -> None:
        print(f"Loading {self.name}:")
        print(f"  text ONNX:  {text_onnx}")
        print(f"  image ONNX: {image_onnx}")
        self.processor = ChineseCLIPProcessor.from_pretrained(model_dir)
        self.text_max_length = load_chinese_text_max_length(export_manifest)
        self.text_session = build_ort_session(runtime, text_onnx)
        self.image_session = build_ort_session(runtime, image_onnx)
        print(f"  text session providers: {', '.join(self.text_session.get_providers())}")
        print(f"  image session providers: {', '.join(self.image_session.get_providers())}")
        super().__init__(image_paths=image_paths, batch_size=batch_size)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        text_features = self.encode_text(query)
        scores = (self.image_features @ text_features[0]).astype(np.float32, copy=False)
        return rank_numpy_results(self.image_paths, scores, top_k)

    def encode_text(self, query: str) -> np.ndarray:
        inputs = self.processor(
            text=[query],
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = np.ascontiguousarray(inputs["input_ids"].astype(np.int64, copy=False))
        attention_mask = np.ascontiguousarray(inputs["attention_mask"].astype(np.int64, copy=False))
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        token_type_ids = np.ascontiguousarray(token_type_ids.astype(np.int64, copy=False))

        features = self.text_session.run(
            ["text_features"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0].astype(np.float32, copy=False)
        return normalize_numpy(features)

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        all_features: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                inputs = self.processor(images=images, return_tensors="np")
            finally:
                for image in images:
                    image.close()

            pixel_values = np.ascontiguousarray(inputs["pixel_values"].astype(np.float32, copy=False))
            features = self.image_session.run(["image_features"], {"pixel_values": pixel_values})[0]
            all_features.append(normalize_numpy(features.astype(np.float32, copy=False)))
            print(f"Encoded ONNX images {min(start + self.batch_size, len(image_paths))}/{len(image_paths)}")

        return np.concatenate(all_features, axis=0)


class FgCLIP2OnnxSearcher(BaseOnnxSearcher):
    def __init__(
        self,
        image_paths: list[Path],
        model_dir: Path,
        batch_size: int,
        max_image_patches: int,
        max_length: int,
        text_onnx: Path,
        image_onnx: Path,
        token_embedding: Path,
        vision_pos_embedding: Path,
        runtime: OrtRuntimeContext,
    ) -> None:
        self.name = f"FG-CLIP2 ONNX p{max_image_patches}"
        print(f"Loading {self.name}:")
        print(f"  text ONNX:  {text_onnx}")
        print(f"  image ONNX: {image_onnx}")
        self.max_image_patches = max_image_patches
        self.max_length = max_length
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.token_embedding = np.memmap(token_embedding, dtype=np.float16, mode="r", shape=(256000, 768))
        base_pos_embedding = np.fromfile(vision_pos_embedding, dtype=np.float32)
        token_count = base_pos_embedding.size // 768
        side = int(np.sqrt(token_count))
        if side == 0 or side * side != token_count:
            raise SystemExit(
                f"Unexpected FG positional embedding size: {vision_pos_embedding} ({base_pos_embedding.size})"
            )
        self.base_pos_embedding = base_pos_embedding.reshape(side, side, 768)
        self.text_session = build_ort_session(runtime, text_onnx)
        self.image_session = build_ort_session(runtime, image_onnx)
        print(f"  text session providers: {', '.join(self.text_session.get_providers())}")
        print(f"  image session providers: {', '.join(self.image_session.get_providers())}")
        super().__init__(image_paths=image_paths, batch_size=batch_size)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        text_features = self.encode_text(query)
        scores = (self.image_features @ text_features[0]).astype(np.float32, copy=False)
        return rank_numpy_results(self.image_paths, scores, top_k)

    def encode_text(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(
            [query.lower()],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = np.ascontiguousarray(inputs["input_ids"].astype(np.int64, copy=False))
        token_embeds = np.ascontiguousarray(self.token_embedding[input_ids].astype(np.float32, copy=False))
        features = self.text_session.run(["text_features"], {"token_embeds": token_embeds})[0]
        return normalize_numpy(features.astype(np.float32, copy=False))

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        all_features: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                max_num_patches = max(demo_fg_clip2.determine_max_patches(image) for image in images)
                max_num_patches = min(max_num_patches, self.max_image_patches)
                inputs = self.image_processor(
                    images=images,
                    max_num_patches=max_num_patches,
                    return_tensors="np",
                )
            finally:
                for image in images:
                    image.close()

            pixel_values = np.ascontiguousarray(inputs["pixel_values"].astype(np.float32, copy=False))
            pixel_attention_mask = np.ascontiguousarray(inputs["pixel_attention_mask"].astype(np.int32, copy=False))
            spatial_shapes = np.ascontiguousarray(inputs["spatial_shapes"])
            pos_embed = make_onnx_pos_embed(self.base_pos_embedding, spatial_shapes, pixel_values.shape[1])
            features = self.image_session.run(
                ["image_features"],
                {
                    "pixel_values": pixel_values,
                    "pixel_attention_mask": pixel_attention_mask,
                    "pos_embed": pos_embed,
                },
            )[0]
            all_features.append(normalize_numpy(features.astype(np.float32, copy=False)))
            print(f"Encoded ONNX images {min(start + self.batch_size, len(image_paths))}/{len(image_paths)}")

        return np.concatenate(all_features, axis=0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Chinese-CLIP and FG-CLIP2 ONNX retrieval with Gradio.")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "images")
    parser.add_argument("--chinese-model-dir", type=Path, default=demo_chinese_clip.DEFAULT_MODEL_DIR)
    parser.add_argument("--fg-model-dir", type=Path, default=demo_fg_clip2.DEFAULT_MODEL_DIR)
    parser.add_argument("--chinese-batch-size", type=int, default=8)
    parser.add_argument("--fg-batch-size", type=int, default=1)
    parser.add_argument("--chinese-onnx-manifest", type=Path, default=CHINESE_ONNX_MANIFEST)
    parser.add_argument("--chinese-onnx-text", type=Path, default=CHINESE_ONNX_TEXT)
    parser.add_argument("--chinese-onnx-image", type=Path, default=CHINESE_ONNX_IMAGE)
    parser.add_argument("--fg-max-length", type=int, default=64)
    parser.add_argument("--fg-patch-variants", default=DEFAULT_FG_PATCH_VARIANTS)
    parser.add_argument("--fg-onnx-text", type=Path, default=FG_ONNX_SPLIT_TEXT)
    parser.add_argument("--fg-onnx-image", type=Path, default=None)
    parser.add_argument(
        "--onnx-backend",
        "--fg-onnx-backend",
        dest="onnx_backend",
        default=FG_ORT_BACKEND,
        help="Backend profile: auto, cuda, cpu, coreml. Ignored when --onnx-providers is set.",
    )
    parser.add_argument(
        "--onnx-providers",
        "--fg-onnx-providers",
        dest="onnx_providers",
        default=FG_ORT_PROVIDERS,
        help="Optional explicit ORT provider order. Examples: cuda,cpu or coreml,cpu.",
    )
    parser.add_argument(
        "--onnx-coreml-cache-dir",
        "--fg-onnx-coreml-cache-dir",
        dest="onnx_coreml_cache_dir",
        type=Path,
        default=FG_ORT_COREML_CACHE_DIR,
        help="Optional cache directory passed to CoreMLExecutionProvider when CoreML is selected.",
    )
    parser.add_argument("--ort-intra-threads", type=int, default=4)
    parser.add_argument("--fg-token-embedding", type=Path, default=FG_TOKEN_EMBEDDING_F16)
    parser.add_argument("--fg-vision-pos-embedding", type=Path, default=FG_VISION_POS_EMBEDDING)
    parser.add_argument("--fg-onnx-mode", choices=["auto", "disabled", "split-text", "lowmem"], default="split-text")
    parser.add_argument("--fg-max-image-patches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fg-onnx-max-image-patches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fg-logit-params", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)
    normalize_args(args)
    return args


def normalize_args(args: argparse.Namespace) -> None:
    if args.fg_onnx_mode == "disabled":
        raise SystemExit("app_compare_clip.py is now ONNX-only. Remove --fg-onnx-mode disabled.")

    deprecated_single_variant = args.fg_onnx_max_image_patches or args.fg_max_image_patches
    raw_variants = str(deprecated_single_variant) if deprecated_single_variant is not None else args.fg_patch_variants
    args.fg_patch_variants = parse_patch_variants(raw_variants)

    if args.fg_onnx_image is None:
        args.fg_onnx_image = FG_ONNX_INT8_IMAGE if args.fg_onnx_mode == "lowmem" else FG_ONNX_FP32_IMAGE

    if args.ort_intra_threads is not None and args.ort_intra_threads <= 0:
        raise SystemExit("--ort-intra-threads must be greater than 0.")


def parse_patch_variants(value: str) -> list[int]:
    variants: list[int] = []
    seen: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            patch_budget = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid FG patch budget: {token!r}") from exc
        if patch_budget not in SUPPORTED_FG_PATCH_VARIANTS:
            allowed = ", ".join(str(item) for item in SUPPORTED_FG_PATCH_VARIANTS)
            raise SystemExit(f"Unsupported FG patch budget {patch_budget}. Supported values: {allowed}.")
        if patch_budget in seen:
            continue
        seen.add(patch_budget)
        variants.append(patch_budget)
    if not variants:
        raise SystemExit("At least one FG patch budget is required.")
    return variants


def load_chinese_text_max_length(export_manifest: Path) -> int:
    default_length = 52
    if not export_manifest.exists():
        return default_length
    try:
        manifest = json.loads(export_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_length
    value = manifest.get("sample_inputs", {}).get("text_max_length")
    return int(value) if isinstance(value, int) and value > 0 else default_length


def create_ort_runtime_context(
    provider_request: str,
    coreml_cache_dir: Path | None,
    intra_threads: int | None,
) -> OrtRuntimeContext:
    try:
        added_dll_dirs = ort_runtime.prepare_ort_environment(provider_request)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for app_compare_clip.py. "
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

    print("Using ONNX Runtime:")
    print(f"  requested providers: {provider_selection.requested}")
    print(f"  available providers: {', '.join(provider_selection.available_names)}")
    print(f"  selected providers: {', '.join(provider_selection.selected_names)}")
    if provider_selection.unavailable_names:
        print(f"  unavailable requested providers: {', '.join(provider_selection.unavailable_names)}")
    if added_dll_dirs:
        print(f"  added CUDA DLL dirs: {', '.join(str(path) for path in added_dll_dirs)}")
    if coreml_cache_dir is not None and "CoreMLExecutionProvider" in provider_selection.selected_names:
        print(f"  CoreML cache dir: {coreml_cache_dir}")

    return OrtRuntimeContext(
        ort=ort,
        provider_request=provider_request,
        provider_selection=provider_selection,
        intra_threads=intra_threads,
    )


def build_session_options(ort: Any, intra_threads: int | None) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads is not None:
        options.intra_op_num_threads = intra_threads
    return options


def build_ort_session(runtime: OrtRuntimeContext, onnx_path: Path) -> Any:
    return runtime.ort.InferenceSession(
        str(onnx_path),
        sess_options=build_session_options(runtime.ort, runtime.intra_threads),
        providers=runtime.provider_selection.provider_args,
    )


def ensure_assets_exist(paths: Sequence[Path]) -> None:
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f"  {path}" for path in missing_paths)
        raise SystemExit(f"Missing ONNX assets:\n{formatted}")


def rank_numpy_results(image_paths: list[Path], scores: np.ndarray, top_k: int) -> list[SearchResult]:
    scores = scores.reshape(-1)
    top_k = min(top_k, len(image_paths))
    ranking = np.argsort(-scores)[:top_k]
    return [SearchResult(path=image_paths[int(index)], score=float(scores[int(index)])) for index in ranking]


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

    merged_features = np.stack([features_by_path[path] for path in target_paths], axis=0)
    return list(target_paths), merged_features.astype(np.float32, copy=False)


def normalize_numpy(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
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
    return np.ascontiguousarray(pos_embed)


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


def list_images(image_dir: Path) -> list[Path]:
    return demo_chinese_clip.list_images(image_dir)


def build_searchers(args: argparse.Namespace, image_paths: list[Path]) -> tuple[list[Searcher], str]:
    ensure_assets_exist(
        [
            args.chinese_onnx_text,
            args.chinese_onnx_image,
            args.fg_onnx_text,
            args.fg_onnx_image,
            args.fg_token_embedding,
            args.fg_vision_pos_embedding,
        ]
    )
    provider_request = args.onnx_providers or args.onnx_backend
    runtime = create_ort_runtime_context(
        provider_request=provider_request,
        coreml_cache_dir=args.onnx_coreml_cache_dir,
        intra_threads=args.ort_intra_threads,
    )

    searchers: list[Searcher] = [
        ChineseCLIPOnnxSearcher(
            image_paths=image_paths,
            model_dir=args.chinese_model_dir,
            batch_size=args.chinese_batch_size,
            text_onnx=args.chinese_onnx_text,
            image_onnx=args.chinese_onnx_image,
            export_manifest=args.chinese_onnx_manifest,
            runtime=runtime,
        )
    ]
    for patch_budget in args.fg_patch_variants:
        searchers.append(
            FgCLIP2OnnxSearcher(
                image_paths=image_paths,
                model_dir=args.fg_model_dir,
                batch_size=args.fg_batch_size,
                max_image_patches=patch_budget,
                max_length=args.fg_max_length,
                text_onnx=args.fg_onnx_text,
                image_onnx=args.fg_onnx_image,
                token_embedding=args.fg_token_embedding,
                vision_pos_embedding=args.fg_vision_pos_embedding,
                runtime=runtime,
            )
        )
    provider_label = ", ".join(runtime.provider_selection.selected_names)
    return searchers, provider_label


def build_demo(
    searchers: list[Searcher],
    image_dir: Path,
    default_top_k: int,
    provider_label: str,
):
    index_lock = threading.RLock()

    def empty_outputs(message: str) -> tuple[Any, ...]:
        outputs: list[Any] = []
        for _ in searchers:
            outputs.extend([[], []])
        outputs.append(message)
        return tuple(outputs)

    def search(query: str, top_k: int):
        top_k = int(top_k)
        query = query.strip()
        if not query:
            return empty_outputs("请输入搜索词。")

        with index_lock:
            ranked_results = [searcher.search(query, top_k) for searcher in searchers]
            image_count = len(searchers[0].image_paths)
            status = f"query: {query} | images: {image_count} | providers: {provider_label}"

        outputs: list[Any] = []
        for results in ranked_results:
            outputs.extend([gallery_items(results), table_rows(results)])
        outputs.append(status)
        return tuple(outputs)

    def refresh_and_search(query: str, top_k: int):
        query = query.strip() or "山"
        with index_lock:
            image_paths = list_images(image_dir)
            added_parts = [
                f"{searcher.name} +{searcher.refresh_image_index(image_paths)}"
                for searcher in searchers
            ]
            outputs = search(query, int(top_k))

        status = f"{outputs[-1]} | refreshed: {', '.join(added_parts)}"
        return (*outputs[:-1], status)

    image_count = len(searchers[0].image_paths)
    with gr.Blocks(title="CLIP ONNX 对照检索") as demo:
        gr.Markdown("# Chinese-CLIP / FG-CLIP2 ONNX 图片检索对照")
        gr.Markdown(
            "分数是归一化 embedding 的 cosine。FG-CLIP2 的不同列共用同一套 ONNX，"
            "只是在预处理阶段使用不同 `max_patches`。"
        )
        status = gr.Markdown(f"已预编码 {image_count} 张图片。providers: `{provider_label}`")

        with gr.Row():
            query = gr.Textbox(label="自然语言搜索", value="山", scale=4)
            top_k = gr.Slider(1, min(20, image_count), value=min(default_top_k, image_count), step=1, label="Top K")
            search_button = gr.Button("搜索", variant="primary")
            refresh_button = gr.Button("刷新图片")

        outputs: list[Any] = []
        with gr.Row():
            for searcher in searchers:
                with gr.Column(min_width=320):
                    gr.Markdown(f"## {searcher.name}")
                    gallery = gr.Gallery(label="排序结果", columns=4, height=520, object_fit="cover")
                    table = gr.Dataframe(
                        headers=["rank", "filename", "cosine", "path"],
                        datatype=["number", "str", "number", "str"],
                        label="排名",
                        interactive=False,
                    )
                    outputs.extend([gallery, table])

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


def main() -> None:
    args = parse_args()
    demo_chinese_clip.ensure_model(args.chinese_model_dir, args.local_only)
    demo_fg_clip2.ensure_model(args.fg_model_dir, args.local_only)
    image_paths = list_images(args.image_dir)

    print(f"Found {len(image_paths)} images under {args.image_dir}")
    print(f"FG patch variants: {', '.join(f'p{value}' for value in args.fg_patch_variants)}")
    searchers, provider_label = build_searchers(args, image_paths)
    demo = build_demo(searchers, args.image_dir, args.top_k, provider_label)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
