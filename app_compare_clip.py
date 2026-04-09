from __future__ import annotations

import argparse
import os
import threading
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))

import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

import demo_chinese_clip
import demo_fg_clip2


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
            status = f"query: {query} | images: {len(chinese_searcher.image_paths)} | device: {device}"

        return (
            gallery_items(chinese_results),
            table_rows(chinese_results),
            gallery_items(fg_results),
            table_rows(fg_results),
            status,
        )

    def refresh_and_search(query: str, top_k: int):
        query = query.strip() or "山"
        with index_lock:
            image_paths = demo_chinese_clip.list_images(image_dir)
            fg_image_paths = demo_fg_clip2.list_images(image_dir)
            if image_paths != fg_image_paths:
                raise gr.Error("两套索引扫描到的图片顺序不一致。")

            chinese_added = chinese_searcher.refresh_image_index(image_paths)
            fg_added = fg_searcher.refresh_image_index(image_paths)
            outputs = search(query, int(top_k))

        status = (
            f"{outputs[-1]} | refreshed: Chinese-CLIP +{chinese_added}, "
            f"FG-CLIP2 +{fg_added}"
        )
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

        outputs = [chinese_gallery, chinese_table, fg_gallery, fg_table, status]
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

    demo = build_demo(chinese_searcher, fg_searcher, args.image_dir, args.top_k, device)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
