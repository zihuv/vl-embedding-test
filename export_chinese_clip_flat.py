# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "pillow==11.3.0",
#   "torch==2.8.0",
#   "transformers>=4.57.0",
# ]
# ///
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from export_flat_common import MODELS_ROOT, copy_optional_files, copy_required_file, prepare_output_dir, write_json

OPTIONAL_SOURCE_FILES = [
    "README.md",
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class TextEncoder(nn.Module):
    def __init__(self, model: ChineseCLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )


class VisionEncoder(nn.Module):
    def __init__(self, model: ChineseCLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=pixel_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local Chinese-CLIP model directory to omni_search flat format."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_ROOT / "chinese_clip_flat",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default="local")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")

    output_dir = prepare_output_dir(args.output.resolve(), force=args.force)
    model = load_model(model_dir)
    processor = ChineseCLIPProcessor.from_pretrained(model_dir, local_files_only=True)

    context_length = infer_context_length(model)
    embedding_dim = infer_embedding_dim(model)
    image_size = infer_image_size(processor.image_processor)
    tokenizer_name = materialize_tokenizer_assets(model_dir, output_dir, processor)
    copy_optional_files(model_dir, output_dir, OPTIONAL_SOURCE_FILES)

    write_json(
        output_dir / "model_config.json",
        {
            "format": "omni_flat_v1",
            "schema_version": 1,
            "family": "chinese_clip",
            "model_id": args.model_id or model_dir.name,
            "model_revision": args.model_revision,
            "embedding_dim": embedding_dim,
            "normalize_output": True,
            "text": {
                "onnx": "text.onnx",
                "output_name": "text_features",
                "tokenizer": tokenizer_name,
                "context_length": context_length,
                "input": {
                    "kind": "bert_like",
                    "input_ids_name": "input_ids",
                    "attention_mask_name": "attention_mask",
                    "token_type_ids_name": "token_type_ids",
                },
            },
            "image": {
                "onnx": "visual.onnx",
                "output_name": "image_features",
                "preprocess": {
                    "kind": "clip_image",
                    "image_size": image_size,
                    "resize_shortest_edge": image_size,
                    "crop": "none",
                    "mean": infer_float_list(getattr(processor.image_processor, "image_mean", None), CLIP_MEAN),
                    "std": infer_float_list(getattr(processor.image_processor, "image_std", None), CLIP_STD),
                },
            },
        },
    )

    export_text_onnx(model, output_dir / "text.onnx", args.batch_size, context_length, args.opset)
    export_visual_onnx(model, output_dir / "visual.onnx", args.batch_size, image_size, args.opset)
    print(f"Flat Chinese-CLIP export ready -> {output_dir}")


def load_model(model_dir: Path) -> ChineseCLIPModel:
    model = ChineseCLIPModel.from_pretrained(
        model_dir,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    return model


def infer_context_length(model: ChineseCLIPModel) -> int:
    return int(model.config.text_config.max_position_embeddings)


def infer_embedding_dim(model: ChineseCLIPModel) -> int:
    projection_dim = getattr(model.config, "projection_dim", None)
    if projection_dim is not None:
        return int(projection_dim)
    return int(model.text_projection.weight.shape[0])


def infer_image_size(image_processor: Any) -> int:
    for candidate in (getattr(image_processor, "size", None), getattr(image_processor, "crop_size", None)):
        if isinstance(candidate, int):
            return int(candidate)
        if isinstance(candidate, dict):
            if "shortest_edge" in candidate:
                return int(candidate["shortest_edge"])
            if "height" in candidate and "width" in candidate:
                return int(min(candidate["height"], candidate["width"]))
    return 224


def infer_float_list(value: Any, fallback: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [float(item) for item in value]
    return fallback


def materialize_tokenizer_assets(
    model_dir: Path,
    output_dir: Path,
    processor: ChineseCLIPProcessor,
) -> str:
    vocab_path = model_dir / "vocab.txt"
    if vocab_path.is_file():
        copy_required_file(vocab_path, output_dir / "vocab.txt")
        return "vocab.txt"

    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        copy_required_file(tokenizer_json, output_dir / "tokenizer.json")
        return "tokenizer.json"

    processor.tokenizer.save_pretrained(output_dir)
    if (output_dir / "vocab.txt").is_file():
        return "vocab.txt"
    return "tokenizer.json"


def export_text_onnx(
    model: ChineseCLIPModel,
    output_path: Path,
    batch_size: int,
    context_length: int,
    opset: int,
) -> None:
    wrapper = TextEncoder(model).eval()
    dummy_input_ids = torch.ones(batch_size, context_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, context_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(batch_size, context_length, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        str(output_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["text_features"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "text_features": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def export_visual_onnx(
    model: ChineseCLIPModel,
    output_path: Path,
    batch_size: int,
    image_size: int,
    opset: int,
) -> None:
    wrapper = VisionEncoder(model).eval()
    dummy_pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    torch.onnx.export(
        wrapper,
        (dummy_pixel_values,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
