from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE_DIR / "xet"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

DEFAULT_MODEL_DIR = PROJECT_ROOT / "chinese-clip-vit-base-patch16"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "chinese-clip-vit-base-patch16" / "onnx"
DEFAULT_ONNX_PREFIX = DEFAULT_OUTPUT_DIR / "vit-b-16"
DEFAULT_SAMPLE_IMAGE = DEFAULT_MODEL_DIR / "festival.jpg"
DEFAULT_SAMPLE_TEXT = "test"


def feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is None:
        raise TypeError(f"Cannot extract embedding tensor from {type(output).__name__}")
    return pooler_output


class TextEncoder(nn.Module):
    def __init__(self, model: ChineseCLIPModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        return feature_tensor(
            self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        )


class VisionEncoder(nn.Module):
    def __init__(self, model: ChineseCLIPModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return feature_tensor(self.model.get_image_features(pixel_values=pixel_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a local Hugging Face Chinese-CLIP model directory into separate ONNX text "
            "and vision encoders."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--save-onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PREFIX,
        help="Output file prefix in official Chinese-CLIP style, e.g. .../vit-b-16",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Deprecated compatibility option. If set, outputs use <output-dir>/vit-b-16 as prefix.",
    )
    parser.add_argument("--sample-image", type=Path, default=DEFAULT_SAMPLE_IMAGE)
    parser.add_argument("--sample-text", default=DEFAULT_SAMPLE_TEXT)
    parser.add_argument(
        "--text-max-length",
        type=int,
        default=52,
        help="Dummy text length used during export tracing. Runtime sequence length stays dynamic.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    onnx_prefix = resolve_onnx_prefix(args)
    output_dir = onnx_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    print(f"Loading model from {model_dir}")
    model = load_model(model_dir)
    processor = ChineseCLIPProcessor.from_pretrained(model_dir, local_files_only=True)

    text_inputs = build_text_inputs(processor, args.sample_text, args.text_max_length)
    image_inputs = build_image_inputs(processor, args.sample_image)

    text_wrapper = TextEncoder(model).eval()
    vision_wrapper = VisionEncoder(model).eval()

    text_onnx = output_path_from_prefix(onnx_prefix, "txt")
    vision_onnx = output_path_from_prefix(onnx_prefix, "img")

    with torch.no_grad():
        text_ref = text_wrapper(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            text_inputs["token_type_ids"],
        )
        image_ref = vision_wrapper(image_inputs["pixel_values"])

    export_text(text_wrapper, text_inputs, text_onnx, args.opset)
    export_vision(vision_wrapper, image_inputs, vision_onnx, args.opset)
    write_manifest(
        output_dir=output_dir,
        model_dir=model_dir,
        text_onnx=text_onnx,
        vision_onnx=vision_onnx,
        text_inputs=text_inputs,
        image_inputs=image_inputs,
        text_ref=text_ref,
        image_ref=image_ref,
        model=model,
    )

    if not args.skip_verify:
        verify_export(text_onnx, vision_onnx, text_inputs, image_inputs, text_ref, image_ref)

    print("Finished ONNX export.")
    print(f"Text ONNX:   {text_onnx}")
    print(f"Vision ONNX: {vision_onnx}")
    print(f"Manifest:    {output_dir / 'export_manifest.json'}")


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


def resolve_onnx_prefix(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve() / DEFAULT_ONNX_PREFIX.name
    return args.save_onnx_path.resolve()


def output_path_from_prefix(prefix: Path, tower: str) -> Path:
    return prefix.parent / f"{prefix.name}.{tower}.fp32.onnx"


def build_text_inputs(
    processor: ChineseCLIPProcessor,
    sample_text: str,
    text_max_length: int,
) -> dict[str, torch.Tensor]:
    text_inputs = processor(
        text=[sample_text],
        padding="max_length",
        truncation=True,
        max_length=text_max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": text_inputs["input_ids"].to(torch.int64),
        "attention_mask": text_inputs["attention_mask"].to(torch.int64),
        "token_type_ids": text_inputs["token_type_ids"].to(torch.int64),
    }


def build_image_inputs(
    processor: ChineseCLIPProcessor,
    sample_image: Path,
) -> dict[str, torch.Tensor]:
    image_path = sample_image.resolve()
    if not image_path.exists():
        raise SystemExit(f"Sample image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    try:
        image_inputs = processor(images=image, return_tensors="pt")
    finally:
        image.close()
    return {"pixel_values": image_inputs["pixel_values"].to(torch.float32)}


def export_text(
    wrapper: nn.Module,
    text_inputs: dict[str, torch.Tensor],
    output_path: Path,
    opset: int,
) -> None:
    print(f"Exporting text encoder -> {output_path}")
    torch.onnx.export(
        wrapper,
        (
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            text_inputs["token_type_ids"],
        ),
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
        external_data=True,
    )
    print_size(output_path)


def export_vision(
    wrapper: nn.Module,
    image_inputs: dict[str, torch.Tensor],
    output_path: Path,
    opset: int,
) -> None:
    print(f"Exporting vision encoder -> {output_path}")
    torch.onnx.export(
        wrapper,
        (image_inputs["pixel_values"],),
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
        external_data=True,
    )
    print_size(output_path)


def write_manifest(
    output_dir: Path,
    model_dir: Path,
    text_onnx: Path,
    vision_onnx: Path,
    text_inputs: dict[str, torch.Tensor],
    image_inputs: dict[str, torch.Tensor],
    text_ref: torch.Tensor,
    image_ref: torch.Tensor,
    model: ChineseCLIPModel,
) -> None:
    manifest = {
        "model_dir": str(model_dir),
        "text_onnx": str(text_onnx),
        "vision_onnx": str(vision_onnx),
        "sample_inputs": {
            "text_max_length": int(text_inputs["input_ids"].shape[1]),
            "image_shape": list(image_inputs["pixel_values"].shape),
        },
        "outputs": {
            "text_feature_shape": list(text_ref.shape),
            "image_feature_shape": list(image_ref.shape),
        },
        "logit_scale": float(model.logit_scale.detach().cpu().item()),
        "logit_scale_exp": float(model.logit_scale.detach().cpu().exp().item()),
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_export(
    text_onnx: Path,
    vision_onnx: Path,
    text_inputs: dict[str, torch.Tensor],
    image_inputs: dict[str, torch.Tensor],
    text_ref: torch.Tensor,
    image_ref: torch.Tensor,
) -> None:
    text_outputs = run_ort(
        text_onnx,
        {
            "input_ids": text_inputs["input_ids"].cpu().numpy(),
            "attention_mask": text_inputs["attention_mask"].cpu().numpy(),
            "token_type_ids": text_inputs["token_type_ids"].cpu().numpy(),
        },
    )
    image_outputs = run_ort(
        vision_onnx,
        {"pixel_values": image_inputs["pixel_values"].cpu().numpy()},
    )

    report_diff("text", text_ref.cpu().numpy(), text_outputs["text_features"])
    report_diff("vision", image_ref.cpu().numpy(), image_outputs["image_features"])


def run_ort(onnx_path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for verification. Install it or rerun with --skip-verify."
        ) from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, feeds)
    output_names = [output.name for output in session.get_outputs()]
    return dict(zip(output_names, outputs, strict=True))


def report_diff(label: str, expected: np.ndarray, actual: np.ndarray) -> None:
    max_abs_diff = float(np.max(np.abs(expected - actual)))
    cosine = float(np.sum(expected * actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    print(f"{label}: max_abs_diff={max_abs_diff:.9g} cosine_vs_pytorch={cosine:.9g}")


def print_size(path: Path) -> None:
    print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
