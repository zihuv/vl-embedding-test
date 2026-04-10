from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import project_layout

REPO_ID = "Qwen/Qwen3-VL-Embedding-2B"
PROJECT_ROOT = project_layout.PROJECT_ROOT
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images"
DEFAULT_MODEL_DIR = project_layout.source_model_dir("qwen3-vl-embedding-2b")
DEFAULT_INSTRUCTION = "Represent the user's input for retrieving relevant images."
HF_CACHE_DIR = project_layout.HF_CACHE_DIR

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))

import torch
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local images with Qwen/Qwen3-VL-Embedding-2B.")
    parser.add_argument("query", nargs="?", default="一张有人和动物的照片")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--local-only", action="store_true", help="Do not download missing model files.")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")
    return image_paths


def ensure_model(model_dir: Path, local_only: bool) -> Path:
    has_config = (model_dir / "config.json").exists()
    has_weights = any(model_dir.glob("*.bin")) or any(model_dir.glob("*.safetensors"))
    has_embedding_script = (model_dir / "scripts" / "qwen3_vl_embedding.py").exists()
    if has_config and has_weights and has_embedding_script:
        return model_dir
    if local_only:
        raise SystemExit(f"Model is incomplete under {model_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {model_dir}")
    snapshot_download(repo_id=REPO_ID, local_dir=str(model_dir), cache_dir=str(HF_CACHE_DIR))
    return model_dir


def load_qwen_embedder_class(model_dir: Path):
    script_path = model_dir / "scripts" / "qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("local_qwen3_vl_embedding", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import Qwen embedding helper from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def choose_model_kwargs() -> dict:
    if torch.cuda.is_available():
        return {"torch_dtype": torch.bfloat16}
    return {"torch_dtype": torch.float32}


@torch.no_grad()
def encode_text(embedder, query: str, instruction: str) -> torch.Tensor:
    embeddings = embedder.process([{"text": query, "instruction": instruction}])
    return embeddings.cpu()


@torch.no_grad()
def encode_images(embedder, image_paths: list[Path], instruction: str, batch_size: int) -> torch.Tensor:
    all_features: list[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            inputs = [{"image": image, "instruction": instruction} for image in images]
            all_features.append(embedder.process(inputs).cpu())
        finally:
            for image in images:
                image.close()
        print(f"Encoded images {min(start + batch_size, len(image_paths))}/{len(image_paths)}")
    return torch.cat(all_features, dim=0)


def main() -> None:
    args = parse_args()
    model_dir = ensure_model(args.model_dir, args.local_only)
    image_paths = list_images(args.image_dir)

    print(f"Loading {REPO_ID} from {model_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    Qwen3VLEmbedder = load_qwen_embedder_class(model_dir)
    embedder = Qwen3VLEmbedder(
        model_name_or_path=str(model_dir),
        default_instruction=args.instruction,
        **choose_model_kwargs(),
    )

    text_features = encode_text(embedder, args.query, args.instruction)
    image_features = encode_images(embedder, image_paths, args.instruction, args.batch_size)
    scores = (image_features @ text_features.T).squeeze(1)

    top_k = min(args.top_k, len(image_paths))
    ranking = torch.topk(scores, k=top_k)

    print(f"\nQuery: {args.query}")
    print(f"Instruction: {args.instruction}")
    print(f"Model: {REPO_ID}")
    print("Top results:")
    for rank, index in enumerate(ranking.indices.tolist(), start=1):
        score = scores[index].item()
        print(f"{rank:02d}. score={score:.4f}  {image_paths[index]}")


if __name__ == "__main__":
    main()
