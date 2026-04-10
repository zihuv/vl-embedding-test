from __future__ import annotations

import argparse
import os
from pathlib import Path

import project_layout

PROJECT_ROOT = project_layout.PROJECT_ROOT
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images"
DEFAULT_MODEL_DIR = project_layout.source_model_dir("fg-clip2-base")
HF_CACHE_DIR = project_layout.HF_CACHE_DIR

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer


REPO_ID = "qihoo360/fg-clip2-base"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local images with qihoo360/fg-clip2-base.")
    parser.add_argument("query", nargs="?", default="一张有人和动物的照片")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-image-patches", type=int, default=None)
    parser.add_argument("--walk-type", default="short", choices=["short", "long"])
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
    has_custom_code = (model_dir / "modeling_fgclip2.py").exists()
    if has_config and has_weights and has_custom_code:
        return model_dir
    if local_only:
        raise SystemExit(f"Model is incomplete under {model_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {model_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(model_dir),
        cache_dir=str(HF_CACHE_DIR),
        ignore_patterns=["*.jpg", "*.png", "*.jpeg"],
    )
    return model_dir


def determine_max_patches(image: Image.Image) -> int:
    width, height = image.size
    max_value = (width // 16) * (height // 16)
    if max_value > 784:
        return 1024
    if max_value > 576:
        return 784
    if max_value > 256:
        return 576
    if max_value > 128:
        return 256
    return 128


def to_device(inputs, device: torch.device):
    return {key: value.to(device) for key, value in inputs.items()}


def repair_fgclip2_text_buffers(model, device: torch.device) -> None:
    embeddings = model.text_model.embeddings
    longtext_len = embeddings.position_embedding_res.num_embeddings
    keep_len = model.config.text_config.keep_len

    embeddings.position_ids = torch.arange(longtext_len, dtype=torch.long, device=device).expand((1, -1))
    embeddings.mask1 = torch.zeros((longtext_len, 1), device=device)
    embeddings.mask1[:keep_len, :] = 1
    embeddings.mask2 = torch.zeros((longtext_len, 1), device=device)
    embeddings.mask2[keep_len:, :] = 1


@torch.no_grad()
def encode_text(model, tokenizer, query: str, max_length: int, walk_type: str, device: torch.device):
    inputs = tokenizer(
        [query.lower()],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_features = model.get_text_features(**to_device(inputs, device), walk_type=walk_type)
    return F.normalize(text_features, p=2, dim=-1)


@torch.no_grad()
def encode_images(
    model,
    image_processor,
    image_paths: list[Path],
    batch_size: int,
    max_image_patches: int | None,
    device: torch.device,
) -> torch.Tensor:
    all_features: list[torch.Tensor] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            max_num_patches = max(determine_max_patches(image) for image in images)
            if max_image_patches is not None:
                max_num_patches = min(max_num_patches, max_image_patches)
            inputs = image_processor(
                images=images,
                max_num_patches=max_num_patches,
                return_tensors="pt",
            )
            image_features = model.get_image_features(**to_device(inputs, device))
            all_features.append(F.normalize(image_features, p=2, dim=-1).cpu())
        finally:
            for image in images:
                image.close()
        print(f"Encoded images {min(start + batch_size, len(image_paths))}/{len(image_paths)}")
    return torch.cat(all_features, dim=0).to(device)


def main() -> None:
    args = parse_args()
    model_dir = ensure_model(args.model_dir, args.local_only)
    image_paths = list_images(args.image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {REPO_ID} from {model_dir}")
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
    repair_fgclip2_text_buffers(model, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)

    text_features = encode_text(model, tokenizer, args.query, args.max_length, args.walk_type, device)
    image_features = encode_images(
        model,
        image_processor,
        image_paths,
        args.batch_size,
        args.max_image_patches,
        device,
    )

    scores = (image_features @ text_features.T).squeeze(1)
    scores = scores * model.logit_scale.to(device).exp() + model.logit_bias.to(device)
    scores = scores.cpu()

    top_k = min(args.top_k, len(image_paths))
    ranking = torch.topk(scores, k=top_k)

    print(f"\nQuery: {args.query}")
    print(f"Model: {REPO_ID}")
    print("Top results:")
    for rank, index in enumerate(ranking.indices.tolist(), start=1):
        score = scores[index].item()
        print(f"{rank:02d}. score={score:.4f}  {image_paths[index]}")


if __name__ == "__main__":
    main()
