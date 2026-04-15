# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "huggingface_hub[hf_xet]==0.36.0",
#   "open-clip-torch==3.2.0",
#   "torch==2.8.0",
# ]
# ///
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import open_clip
import torch
from huggingface_hub import hf_hub_download
from timm.utils import reparameterize_model
from torch import nn

from export_flat_common import MODELS_ROOT, prepare_output_dir, write_json

OPTIONAL_HF_FILES = [
    ".gitattributes",
    "README.md",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


class VisualWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(pixel_values, normalize=True)


class TextWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(input_ids, normalize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an OpenCLIP-compatible Hugging Face repo to omni_search flat format."
    )
    parser.add_argument("--id", required=True, help="Hugging Face repo id, e.g. timm/MobileCLIP2-S2-OpenCLIP")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Flat output directory. Defaults to ./models/<repo-leaf>.",
    )
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_dir(
        args.output.resolve() if args.output is not None else default_output_dir(args.id),
        force=args.force,
    )

    open_clip_config = load_open_clip_config(args.id)
    model = load_model(args.id)

    tokenizer_path = download_required_hf_file(args.id, "tokenizer.json")
    copy_file(tokenizer_path, output_dir / "tokenizer.json")
    for name in OPTIONAL_HF_FILES:
        maybe_copy_hf_file(args.id, name, output_dir / name)

    model_config = build_model_config(
        repo_id=args.id,
        model_revision=args.model_revision,
        open_clip_config=open_clip_config,
        model=model,
    )
    write_json(output_dir / "model_config.json", model_config)

    image_size = int(model_config["image"]["preprocess"]["image_size"])
    context_length = int(model_config["text"]["context_length"])
    vocab_size = infer_vocab_size(model)

    export_onnx(
        VisualWrapper(model),
        torch.randn(args.batch_size, 3, image_size, image_size),
        output_dir / "visual.onnx",
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        opset=args.opset,
    )
    export_onnx(
        TextWrapper(model),
        torch.randint(0, vocab_size, (args.batch_size, context_length), dtype=torch.long),
        output_dir / "text.onnx",
        input_names=["input_ids"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"},
        },
        opset=args.opset,
    )

    print(f"Flat OpenCLIP export ready -> {output_dir}")


def default_output_dir(repo_id: str) -> Path:
    leaf = repo_id.rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._-") or "openclip"
    return MODELS_ROOT / slug


def load_model(repo_id: str) -> nn.Module:
    model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:{repo_id}")
    model.eval()
    try:
        model = reparameterize_model(model)
        model.eval()
    except Exception:
        pass
    return model


def load_open_clip_config(repo_id: str) -> dict[str, Any]:
    path = download_required_hf_file(repo_id, "open_clip_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_config(
    *,
    repo_id: str,
    model_revision: str,
    open_clip_config: dict[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    model_cfg = open_clip_config["model_cfg"]
    preprocess_cfg = open_clip_config["preprocess_cfg"]
    text_cfg = model_cfg["text_cfg"]

    resize_mode = preprocess_cfg.get("resize_mode", "shortest")
    crop = {"shortest": "center", "squash": "none"}.get(resize_mode)
    if crop is None:
        raise SystemExit(f"unsupported preprocess_cfg.resize_mode: {resize_mode}")

    is_siglip = "siglip" in repo_id.lower() or "init_logit_bias" in model_cfg
    is_siglip2 = "siglip2" in repo_id.lower()

    return {
        "format": "omni_flat_v1",
        "schema_version": 1,
        "family": "open_clip",
        "model_id": repo_id,
        "model_revision": model_revision,
        "embedding_dim": int(model_cfg["embed_dim"]),
        "normalize_output": True,
        "text": {
            "onnx": "text.onnx",
            "output_name": "text_embeddings",
            "tokenizer": "tokenizer.json",
            "context_length": int(text_cfg["context_length"]),
            "input": {
                "kind": "input_ids",
                "input_ids_name": "input_ids",
                "lower_case": is_siglip,
                "pad_id": 1 if (is_siglip and not is_siglip2) else infer_pad_id(model),
            },
        },
        "image": {
            "onnx": "visual.onnx",
            "output_name": "image_embeddings",
            "preprocess": {
                "kind": "clip_image",
                "image_size": int(model_cfg["vision_cfg"]["image_size"]),
                "resize_shortest_edge": int(model_cfg["vision_cfg"]["image_size"]),
                "crop": crop,
                "mean": list(map(float, preprocess_cfg["mean"])),
                "std": list(map(float, preprocess_cfg["std"])),
            },
        },
    }


def infer_vocab_size(model: nn.Module) -> int:
    if hasattr(model, "vocab_size"):
        return int(model.vocab_size)
    token_embedding = getattr(model, "token_embedding", None)
    if token_embedding is not None:
        return int(token_embedding.weight.shape[0])
    transformer = getattr(model, "transformer", None)
    config = getattr(transformer, "config", None)
    if config is not None and hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    raise SystemExit("could not determine OpenCLIP vocab_size")


def infer_pad_id(model: nn.Module) -> int:
    pad_id = getattr(model, "pad_id", None)
    if isinstance(pad_id, int):
        return pad_id
    return 0


def download_required_hf_file(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def maybe_copy_hf_file(repo_id: str, filename: str, dest: Path) -> None:
    try:
        path = download_required_hf_file(repo_id, filename)
    except Exception:
        return
    copy_file(path, dest)


def copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())


def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
