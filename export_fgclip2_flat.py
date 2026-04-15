# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.3.3",
#   "onnx==1.19.0",
#   "pillow==11.3.0",
#   "torch==2.8.0",
#   "transformers>=4.57.0",
# ]
# ///
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from export_flat_common import MODELS_ROOT, copy_optional_files, copy_required_file, prepare_output_dir, write_json

OPTIONAL_SOURCE_FILES = [
    "README.md",
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]

TOKEN_EMBEDDING_INIT = "model.text_model.embeddings.token_embedding.weight"
TOKEN_GATHER_NODE = "/text_model/embeddings/token_embedding/Gather"
TOKEN_RESHAPE_NODE = "/text_model/Reshape"
TOKEN_GATHER_OUTPUT = "/text_model/embeddings/token_embedding/Gather_output_0"
INPUT_IDS_INPUT = "input_ids"
TOKEN_EMBEDS_INPUT = "token_embeds"


class TextFeatures(nn.Module):
    def __init__(self, model: Any, walk_type: str) -> None:
        super().__init__()
        self.model = model
        self.walk_type = walk_type

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.model.get_text_features(input_ids=input_ids, walk_type=self.walk_type)
        return F.normalize(features, p=2, dim=-1)


class ImageFeaturesWithPosEmbedInput(nn.Module):
    def __init__(self, model: Any) -> None:
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

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
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

    def pool(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        mask = 1.0 - pixel_attention_mask[:, None, None, :].to(dtype=scores.dtype)
        scores = scores + mask * torch.finfo(scores.dtype).min
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).reshape(batch_size, target_len, num_heads * head_dim)
        return F.linear(output, attention.out_proj.weight, attention.out_proj.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export qihoo360/fg-clip2-base to omni_search flat format."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_ROOT / "fgclip2_flat",
    )
    parser.add_argument("--model-id", default="fgclip2-base")
    parser.add_argument("--model-revision", default="local")
    parser.add_argument("--query", default="test")
    parser.add_argument("--walk-type", choices=["short", "long"], default="short")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--default-max-patches", type=int, default=1024)
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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)

    input_ids = build_text_inputs(tokenizer, args.query, args.text_max_length)
    image_inputs = build_image_inputs(image_processor, args.default_max_patches)
    pos_embed = make_pos_embed(model, image_inputs)

    with tempfile.TemporaryDirectory(prefix="fgclip2-export-") as temp_dir:
        raw_text_onnx = Path(temp_dir) / "text_input_ids.onnx"
        export_text_onnx(model, raw_text_onnx, input_ids, args.walk_type, args.opset)
        split_text_embedding(
            source_path=raw_text_onnx,
            output_path=output_dir / "text.onnx",
            embedding_output=output_dir / "text_token_embedding.bin",
            text_max_length=args.text_max_length,
            hidden_size=infer_hidden_size(model),
        )

    export_image_onnx(
        model=model,
        output_path=output_dir / "visual.onnx",
        pixel_values=image_inputs["pixel_values"],
        pixel_attention_mask=image_inputs["pixel_attention_mask"],
        pos_embed=pos_embed,
        opset=args.opset,
    )
    write_vision_pos_embedding(model, output_dir / "vision_pos_embedding.bin")
    materialize_tokenizer_assets(model_dir, output_dir, tokenizer)
    copy_optional_files(model_dir, output_dir, OPTIONAL_SOURCE_FILES)

    write_json(
        output_dir / "model_config.json",
        {
            "format": "omni_flat_v1",
            "schema_version": 1,
            "family": "fg_clip",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "embedding_dim": infer_embedding_dim(model),
            "normalize_output": True,
            "text": {
                "onnx": "text.onnx",
                "output_name": "text_features",
                "tokenizer": "tokenizer.json",
                "context_length": args.text_max_length,
                "input": {"kind": "token_embeds"},
                "token_embedding": {
                    "file": "text_token_embedding.bin",
                    "dtype": "f16",
                    "embedding_dim": infer_hidden_size(model),
                },
            },
            "image": {
                "onnx": "visual.onnx",
                "output_name": "image_features",
                "preprocess": {
                    "kind": "fgclip_patch_tokens",
                    "patch_size": infer_patch_size(model),
                    "default_max_patches": args.default_max_patches,
                    "vision_pos_embedding": "vision_pos_embedding.bin",
                },
            },
        },
    )

    print(f"Flat FG-CLIP2 export ready -> {output_dir}")


def load_model(model_dir: Path) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
    repair_fgclip2_text_buffers(model, torch.device("cpu"))
    return model


def repair_fgclip2_text_buffers(model: Any, device: torch.device) -> None:
    embeddings = model.text_model.embeddings
    longtext_len = embeddings.position_embedding_res.num_embeddings
    keep_len = model.config.text_config.keep_len

    embeddings.position_ids = torch.arange(longtext_len, dtype=torch.long, device=device).expand((1, -1))
    embeddings.mask1 = torch.zeros((longtext_len, 1), device=device)
    embeddings.mask1[:keep_len, :] = 1
    embeddings.mask2 = torch.zeros((longtext_len, 1), device=device)
    embeddings.mask2[keep_len:, :] = 1


def build_text_inputs(tokenizer: Any, query: str, text_max_length: int) -> torch.Tensor:
    return tokenizer(
        [query.lower()],
        padding="max_length",
        truncation=True,
        max_length=text_max_length,
        return_tensors="pt",
    )["input_ids"].to(torch.int64)


def build_image_inputs(image_processor: Any, max_num_patches: int) -> dict[str, torch.Tensor]:
    image = Image.new("RGB", (2048, 2048), color="white")
    return image_processor(images=image, max_num_patches=max_num_patches, return_tensors="pt")


def make_pos_embed(model: Any, image_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
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


def infer_hidden_size(model: Any) -> int:
    return int(model.text_model.embeddings.token_embedding.weight.shape[1])


def infer_embedding_dim(model: Any) -> int:
    projection_dim = getattr(model.config, "projection_dim", None)
    if projection_dim is not None:
        return int(projection_dim)
    return int(model.text_projection.weight.shape[0])


def infer_patch_size(model: Any) -> int:
    patch_size = model.vision_model.embeddings.patch_size
    if isinstance(patch_size, tuple):
        return int(patch_size[0])
    return int(patch_size)


def materialize_tokenizer_assets(model_dir: Path, output_dir: Path, tokenizer: Any) -> None:
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        copy_required_file(tokenizer_json, output_dir / "tokenizer.json")
        return
    tokenizer.save_pretrained(output_dir)
    if not (output_dir / "tokenizer.json").is_file():
        raise SystemExit("failed to materialize tokenizer.json for FG-CLIP2 export")


def export_text_onnx(
    model: Any,
    output_path: Path,
    input_ids: torch.Tensor,
    walk_type: str,
    opset: int,
) -> None:
    wrapper = TextFeatures(model, walk_type).eval()
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


def export_image_onnx(
    *,
    model: Any,
    output_path: Path,
    pixel_values: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    pos_embed: torch.Tensor,
    opset: int,
) -> None:
    wrapper = ImageFeaturesWithPosEmbedInput(model).eval()
    torch.onnx.export(
        wrapper,
        (pixel_values, pixel_attention_mask, pos_embed),
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


def write_vision_pos_embedding(model: Any, output_path: Path) -> None:
    embeddings = model.vision_model.embeddings
    position_embedding = embeddings.position_embedding.weight.reshape(
        embeddings.position_embedding_size,
        embeddings.position_embedding_size,
        -1,
    )
    array = position_embedding.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    output_path.write_bytes(array.tobytes())


def split_text_embedding(
    *,
    source_path: Path,
    output_path: Path,
    embedding_output: Path,
    text_max_length: int,
    hidden_size: int,
) -> None:
    model = onnx.load(str(source_path), load_external_data=False)
    graph = model.graph

    embedding_init = find_initializer(graph, TOKEN_EMBEDDING_INIT)
    embedding = numpy_helper.to_array(embedding_init)
    if embedding.shape[1] != hidden_size:
        raise SystemExit(f"unexpected FG-CLIP2 token embedding width: {embedding.shape}")

    embedding_output.write_bytes(embedding.astype(np.float16, copy=False).tobytes())

    replace_node_inputs(graph, TOKEN_GATHER_OUTPUT, TOKEN_EMBEDS_INPUT)
    remove_node_by_name(graph, TOKEN_GATHER_NODE)
    remove_node_by_name(graph, TOKEN_RESHAPE_NODE)
    remove_initializer(graph, TOKEN_EMBEDDING_INIT)
    remove_graph_input(graph, INPUT_IDS_INPUT)

    graph.input.append(
        helper.make_tensor_value_info(
            TOKEN_EMBEDS_INPUT,
            TensorProto.FLOAT,
            [1, text_max_length, hidden_size],
        )
    )
    model.producer_name = "export_fgclip2_flat.py"
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def find_initializer(graph: Any, name: str) -> Any:
    for initializer in graph.initializer:
        if initializer.name == name:
            return initializer
    raise SystemExit(f"initializer not found: {name}")


def remove_initializer(graph: Any, name: str) -> None:
    graph.initializer.remove(find_initializer(graph, name))


def remove_node_by_name(graph: Any, name: str) -> None:
    for node in graph.node:
        if node.name == name:
            graph.node.remove(node)
            return
    raise SystemExit(f"node not found: {name}")


def remove_graph_input(graph: Any, name: str) -> None:
    for graph_input in graph.input:
        if graph_input.name == name:
            graph.input.remove(graph_input)
            return
    raise SystemExit(f"graph input not found: {name}")


def replace_node_inputs(graph: Any, old_name: str, new_name: str) -> None:
    replacement_count = 0
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
                replacement_count += 1
    if replacement_count == 0:
        raise SystemExit(f"no node inputs referenced {old_name}")


if __name__ == "__main__":
    main()
