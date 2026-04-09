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
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

import demo_fg_clip2


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "fg-clip2-base"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".onnx-wrapper-test"
DEFAULT_IMAGE = PROJECT_ROOT / "images" / "browser_20260409_124726_272943500.jpg"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class TextFeatures(nn.Module):
    def __init__(self, model, walk_type: str):
        super().__init__()
        self.model = model
        self.walk_type = walk_type

    def forward(self, input_ids):
        features = self.model.get_text_features(input_ids=input_ids, walk_type=self.walk_type)
        return F.normalize(features, p=2, dim=-1)


class ImageFeaturesWithPosEmbedInput(nn.Module):
    """FG-CLIP2 image tower, with dynamic position-resize moved outside ONNX."""

    def __init__(self, model):
        super().__init__()
        vision_model = model.vision_model
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        head = vision_model.head
        self.probe = head.probe
        self.attention = head.attention
        self.layernorm = head.layernorm
        self.mlp = head.mlp

    def forward(self, pixel_values, pixel_attention_mask, pos_embed):
        hidden = self.patch_embedding(pixel_values) + pos_embed
        key_mask = pixel_attention_mask[:, None, None, :].to(dtype=hidden.dtype)
        attention_mask = (1.0 - key_mask) * torch.finfo(hidden.dtype).min
        last_hidden_state = self.encoder(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
        ).last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        batch_size = last_hidden_state.shape[0]
        probe = self.probe.expand(batch_size, -1, -1)
        features = self.pool(probe, last_hidden_state, last_hidden_state, pixel_attention_mask)
        features = features + self.mlp(self.layernorm(features))
        features = features[:, 0]
        return F.normalize(features, p=2, dim=-1)

    def pool(self, query, key, value, pixel_attention_mask):
        # Equivalent to nn.MultiheadAttention(batch_first=True) in the original pooling head,
        # but written out so ONNX keeps both batch size and patch count dynamic.
        attention = self.attention
        batch_size = query.shape[0]
        target_len = query.shape[1]
        source_len = key.shape[1]
        num_heads = attention.num_heads
        head_dim = attention.head_dim
        embed_dim = attention.embed_dim

        q = F.linear(query, attention.in_proj_weight[:embed_dim], attention.in_proj_bias[:embed_dim])
        k = F.linear(
            key,
            attention.in_proj_weight[embed_dim : 2 * embed_dim],
            attention.in_proj_bias[embed_dim : 2 * embed_dim],
        )
        v = F.linear(value, attention.in_proj_weight[2 * embed_dim :], attention.in_proj_bias[2 * embed_dim :])

        q = q.view(batch_size, target_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, source_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, source_len, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
        mask = (1.0 - pixel_attention_mask[:, None, None, :].to(dtype=scores.dtype))
        scores = scores + mask * torch.finfo(scores.dtype).min
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).reshape(batch_size, target_len, num_heads * head_dim)
        return F.linear(output, attention.out_proj.weight, attention.out_proj.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export qihoo360/fg-clip2-base to ONNX and verify ONNX Runtime output. "
            "The image ONNX intentionally takes pos_embed as an input; generate it in preprocessing."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--query", default="山")
    parser.add_argument("--walk-type", default="short", choices=["short", "long"])
    parser.add_argument("--text-max-length", type=int, default=None)
    parser.add_argument("--image-max-patches", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--fixture-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_max_length = args.text_max_length
    if text_max_length is None:
        text_max_length = 64 if args.walk_type == "short" else 196

    text_onnx = output_dir / f"fgclip2_text_{args.walk_type}_b1_s{text_max_length}.onnx"
    image_onnx = output_dir / "fgclip2_image_core_posin_dynamic.onnx"

    print(f"Loading FG-CLIP2 from {model_dir}")
    model = load_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir)

    text_inputs = tokenizer(
        [args.query.lower()],
        padding="max_length",
        max_length=text_max_length,
        truncation=True,
        return_tensors="pt",
    )
    image = load_image(args.image)
    try:
        image_inputs = image_processor(
            images=image,
            max_num_patches=args.image_max_patches,
            return_tensors="pt",
        )
    finally:
        image.close()
    pos_embed = make_pos_embed(model, image_inputs)

    text_wrapper = TextFeatures(model, args.walk_type).eval()
    image_wrapper = ImageFeaturesWithPosEmbedInput(model).eval()

    with torch.no_grad():
        text_ref = text_wrapper(text_inputs["input_ids"])
        image_ref = image_wrapper(
            image_inputs["pixel_values"],
            image_inputs["pixel_attention_mask"],
            pos_embed,
        )

    if not args.skip_export:
        export_text(text_wrapper, text_inputs["input_ids"], text_onnx, args.opset)
        export_image_core(image_wrapper, image_inputs, pos_embed, image_onnx, args.opset)

    if not args.skip_verify:
        verify_with_onnxruntime(
            text_onnx,
            image_onnx,
            text_inputs["input_ids"],
            text_ref,
            image_inputs,
            pos_embed,
            image_ref,
        )

        alternate = make_alternate_image_inputs(
            image_processor=image_processor,
            image_path=args.image,
            current_max_patches=args.image_max_patches,
        )
        if alternate is not None:
            alternate_pos_embed = make_pos_embed(model, alternate)
            with torch.no_grad():
                alternate_image_ref = image_wrapper(
                    alternate["pixel_values"],
                    alternate["pixel_attention_mask"],
                    alternate_pos_embed,
                )
            verify_image_only(
                image_onnx,
                alternate,
                alternate_pos_embed,
                alternate_image_ref,
                label=f"image_alt_p{alternate['pixel_values'].shape[1]}",
            )

    if not args.skip_assets:
        write_runtime_assets(model, output_dir)

    if args.write_fixture:
        fixture_dir = args.fixture_dir or output_dir / "fixtures"
        write_fixture(
            output_dir=output_dir,
            fixture_dir=fixture_dir.resolve(),
            text_onnx=text_onnx,
            image_onnx=image_onnx,
            query=args.query,
            image_path=args.image,
            text_inputs=text_inputs,
            text_ref=text_ref,
            image_inputs=image_inputs,
            pos_embed=pos_embed,
            image_ref=image_ref,
            text_image_cosine=float((text_ref * image_ref).sum()),
        )

    print("Done.")
    print(f"Text ONNX:  {text_onnx}")
    print(f"Image ONNX: {image_onnx}")
    print("Image ONNX inputs: pixel_values [B,N,768], pixel_attention_mask [B,N], pos_embed [B,N,768]")


def load_model(model_dir: Path):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    demo_fg_clip2.repair_fgclip2_text_buffers(model, torch.device("cpu"))
    return model


def load_image(image_path: Path) -> Image.Image:
    if image_path.exists():
        return Image.open(image_path).convert("RGB")
    for candidate in sorted((PROJECT_ROOT / "images").rglob("*")):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            print(f"Image not found: {image_path}; using {candidate}")
            return Image.open(candidate).convert("RGB")
    raise SystemExit(f"Image not found: {image_path}")


def make_pos_embed(model, image_inputs) -> torch.Tensor:
    embeddings = model.vision_model.embeddings
    positional_embeddings = embeddings.position_embedding.weight.reshape(
        embeddings.position_embedding_size,
        embeddings.position_embedding_size,
        -1,
    )
    return embeddings.resize_positional_embeddings(
        positional_embeddings,
        image_inputs["spatial_shapes"],
        max_length=image_inputs["pixel_values"].shape[1],
    ).detach()


def export_text(wrapper: nn.Module, input_ids: torch.Tensor, output_path: Path, opset: int) -> None:
    print(f"Exporting text ONNX -> {output_path}")
    torch.onnx.export(
        wrapper,
        (input_ids,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["text_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print_size(output_path)


def export_image_core(
    wrapper: nn.Module,
    image_inputs,
    pos_embed: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    print(f"Exporting image ONNX -> {output_path}")
    torch.onnx.export(
        wrapper,
        (
            image_inputs["pixel_values"],
            image_inputs["pixel_attention_mask"],
            pos_embed,
        ),
        str(output_path),
        input_names=["pixel_values", "pixel_attention_mask", "pos_embed"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 1: "num_patches"},
            "pixel_attention_mask": {0: "batch_size", 1: "num_patches"},
            "pos_embed": {0: "batch_size", 1: "num_patches"},
            "image_features": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print_size(output_path)


def verify_with_onnxruntime(
    text_onnx: Path,
    image_onnx: Path,
    input_ids: torch.Tensor,
    text_ref: torch.Tensor,
    image_inputs,
    pos_embed: torch.Tensor,
    image_ref: torch.Tensor,
) -> None:
    text_out = run_ort(text_onnx, {"input_ids": input_ids.cpu().numpy()})["text_features"]
    report_diff("text", text_ref.cpu().numpy(), text_out)
    verify_image_only(image_onnx, image_inputs, pos_embed, image_ref, label="image")


def verify_image_only(
    image_onnx: Path,
    image_inputs,
    pos_embed: torch.Tensor,
    image_ref: torch.Tensor,
    label: str,
) -> None:
    image_out = run_ort(
        image_onnx,
        {
            "pixel_values": image_inputs["pixel_values"].cpu().numpy(),
            "pixel_attention_mask": image_inputs["pixel_attention_mask"].cpu().numpy(),
            "pos_embed": pos_embed.cpu().numpy(),
        },
    )["image_features"]
    report_diff(label, image_ref.cpu().numpy(), image_out)


def run_ort(onnx_path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for --verify. "
            "Example: uv run --with onnx --with onnxruntime python export_fgclip2_onnx.py --write-fixture"
        ) from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, feeds)
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, outputs, strict=True))


def report_diff(label: str, expected: np.ndarray, actual: np.ndarray) -> None:
    max_abs_diff = float(np.max(np.abs(expected - actual)))
    cosine = float(np.sum(expected * actual) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    print(f"{label}: max_abs_diff={max_abs_diff:.9g} cosine_vs_pytorch={cosine:.9g}")


def make_alternate_image_inputs(image_processor, image_path: Path, current_max_patches: int):
    alternate_max_patches = 128 if current_max_patches != 128 else 256
    image = load_image(image_path)
    try:
        alternate = image_processor(
            images=image,
            max_num_patches=alternate_max_patches,
            return_tensors="pt",
        )
    finally:
        image.close()
    if alternate["pixel_values"].shape[1] == current_max_patches:
        return None
    return alternate


def write_fixture(
    output_dir: Path,
    fixture_dir: Path,
    text_onnx: Path,
    image_onnx: Path,
    query: str,
    image_path: Path,
    text_inputs,
    text_ref: torch.Tensor,
    image_inputs,
    pos_embed: torch.Tensor,
    image_ref: torch.Tensor,
    text_image_cosine: float,
) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)

    def tensor(name: str, value: torch.Tensor) -> dict:
        array = value.detach().cpu().contiguous().numpy()
        file_path = fixture_dir / f"{name}.bin"
        file_path.write_bytes(array.tobytes())
        return {
            "file": relative_to_root(file_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": file_path.stat().st_size,
        }

    manifest = {
        "query": query,
        "image_path": str(image_path),
        "output_dir": relative_to_root(output_dir),
        "spatial_shapes": image_inputs["spatial_shapes"].tolist(),
        "tensors": {
            "input_ids": tensor("input_ids_i64", text_inputs["input_ids"].to(torch.int64)),
            "text_ref": tensor("text_ref_f32", text_ref.to(torch.float32)),
            "pixel_values": tensor("pixel_values_f32", image_inputs["pixel_values"].to(torch.float32)),
            "pixel_attention_mask": tensor(
                "pixel_attention_mask_i32",
                image_inputs["pixel_attention_mask"].to(torch.int32),
            ),
            "pos_embed": tensor("pos_embed_f32", pos_embed.to(torch.float32)),
            "image_ref": tensor("image_ref_f32", image_ref.to(torch.float32)),
        },
        "onnx": {
            "text": relative_to_root(text_onnx),
            "image": relative_to_root(image_onnx),
        },
        "expected": {
            "text_norm": float(torch.linalg.vector_norm(text_ref)),
            "image_norm": float(torch.linalg.vector_norm(image_ref)),
            "text_image_cosine": text_image_cosine,
        },
    }

    manifest_path = fixture_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote fixture manifest -> {manifest_path}")


def write_runtime_assets(model, output_dir: Path) -> None:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    embeddings = model.vision_model.embeddings
    position_embedding = embeddings.position_embedding.weight.reshape(
        embeddings.position_embedding_size,
        embeddings.position_embedding_size,
        -1,
    )
    write_array(assets_dir / "vision_pos_embedding_16x16x768_f32.bin", position_embedding)

    logit_params = {
        "logit_scale": float(model.logit_scale.detach().cpu()[0]),
        "logit_scale_exp": float(model.logit_scale.detach().cpu().exp()[0]),
        "logit_bias": float(model.logit_bias.detach().cpu()[0]),
    }
    (assets_dir / "logit_params.json").write_text(
        json.dumps(logit_params, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote runtime assets -> {assets_dir}")


def write_array(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    path.write_bytes(array.tobytes())


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def print_size(path: Path) -> None:
    print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
