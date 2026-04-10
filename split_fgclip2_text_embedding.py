from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import project_layout


PROJECT_ROOT = Path(__file__).resolve().parent
FG_LAYOUT = project_layout.FGCLIP2_LAYOUT
DEFAULT_SOURCE_TEXT = FG_LAYOUT.baseline_text_onnx_resolved
DEFAULT_SPLIT_TEXT = FG_LAYOUT.split_text_onnx
DEFAULT_ASSET = FG_LAYOUT.token_embedding_f16
DEFAULT_BASE_MANIFEST = FG_LAYOUT.base_manifest_resolved
DEFAULT_SPLIT_MANIFEST = FG_LAYOUT.split_manifest
DEFAULT_REPORT = FG_LAYOUT.split_report

TOKEN_EMBEDDING_INIT = "model.text_model.embeddings.token_embedding.weight"
TOKEN_GATHER_NODE = "/text_model/embeddings/token_embedding/Gather"
TOKEN_RESHAPE_NODE = "/text_model/Reshape"
TOKEN_GATHER_OUTPUT = "/text_model/embeddings/token_embedding/Gather_output_0"
INPUT_IDS_INPUT = "input_ids"
TOKEN_EMBEDS_INPUT = "token_embeds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove FG-CLIP2's large text token embedding table from the text ONNX. "
            "The split ONNX accepts token_embeds [1,S,768]."
        )
    )
    parser.add_argument("--source-text-onnx", type=Path, default=DEFAULT_SOURCE_TEXT)
    parser.add_argument("--output-text-onnx", type=Path, default=DEFAULT_SPLIT_TEXT)
    parser.add_argument("--embedding-output", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--embedding-dtype", choices=["f16", "f32"], default="f16")
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--intra-threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--cosine-gate", type=float, default=0.9999)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_export:
        split_text_embedding(args)
    write_split_manifest(args)

    report: dict[str, Any] = {
        "variant": "split-text-token-embedding",
        "source_text_onnx": relative_to_project(args.source_text_onnx),
        "split_text_onnx": relative_to_project(args.output_text_onnx),
        "token_embedding": {
            "file": relative_to_project(args.embedding_output),
            "dtype": args.embedding_dtype,
            "shape": [256000, args.hidden_size],
        },
        "source_text_mib": size_mib(args.source_text_onnx),
        "split_text_mib": size_mib(args.output_text_onnx) if args.output_text_onnx.exists() else None,
        "embedding_asset_mib": size_mib(args.embedding_output) if args.embedding_output.exists() else None,
        "base_manifest": relative_to_project(args.base_manifest),
        "split_manifest": relative_to_project(args.split_manifest),
        "verification": {},
    }

    if not args.skip_verify:
        metrics = verify_split_text(args)
        report["verification"]["text"] = metrics
        if metrics["cosine_vs_source"] < args.cosine_gate:
            message = f"text cosine {metrics['cosine_vs_source']:.9g} < gate {args.cosine_gate}"
            if args.enforce_gates:
                raise SystemExit(message)
            print(message)
        else:
            print("Split text ONNX verification passed configured cosine gate.")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote split-text report -> {args.report_json}")


def split_text_embedding(args: argparse.Namespace) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise SystemExit(
            "onnx is required. Example:\n"
            "uv run --with onnx --with onnxruntime python split_fgclip2_text_embedding.py"
        ) from exc

    source = args.source_text_onnx
    print(f"Loading text ONNX -> {source}")
    model = onnx.load(str(source), load_external_data=False)
    graph = model.graph

    embedding_init = find_initializer(graph, TOKEN_EMBEDDING_INIT)
    embedding = numpy_helper.to_array(embedding_init)
    if embedding.shape != (256000, args.hidden_size):
        raise SystemExit(f"Unexpected token embedding shape: {embedding.shape}")
    write_embedding_asset(args.embedding_output, embedding, args.embedding_dtype)

    replace_node_inputs(graph, TOKEN_GATHER_OUTPUT, TOKEN_EMBEDS_INPUT)
    remove_node_by_name(graph, TOKEN_GATHER_NODE)
    remove_node_by_name(graph, TOKEN_RESHAPE_NODE)
    remove_initializer(graph, TOKEN_EMBEDDING_INIT)
    remove_graph_input(graph, INPUT_IDS_INPUT)

    graph.input.append(
        helper.make_tensor_value_info(
            TOKEN_EMBEDS_INPUT,
            TensorProto.FLOAT,
            [1, args.text_max_length, args.hidden_size],
        )
    )
    model.producer_name = "split_fgclip2_text_embedding.py"
    onnx.checker.check_model(model)

    args.output_text_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving split text ONNX -> {args.output_text_onnx}")
    onnx.save(model, str(args.output_text_onnx))
    print(
        f"Text ONNX size: {size_mib(source):.1f} MiB -> "
        f"{size_mib(args.output_text_onnx):.1f} MiB"
    )


def verify_split_text(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.base_manifest)
    input_ids = load_tensor(manifest["tensors"]["input_ids"])
    embedding = load_embedding_asset(args.embedding_output, args.embedding_dtype, args.hidden_size)
    token_embeds = embedding[input_ids].astype(np.float32, copy=False)

    source_out, source_ms = run_ort(
        args.source_text_onnx,
        {"input_ids": input_ids},
        output_name="text_features",
        args=args,
    )
    split_out, split_ms = run_ort(
        args.output_text_onnx,
        {TOKEN_EMBEDS_INPUT: token_embeds},
        output_name="text_features",
        args=args,
    )
    max_abs_diff = float(np.max(np.abs(source_out - split_out)))
    cosine = cosine_similarity(source_out, split_out)
    metrics = {
        "max_abs_diff_vs_source": max_abs_diff,
        "cosine_vs_source": cosine,
        "source_avg_ms": source_ms,
        "split_avg_ms": split_ms,
    }
    print(
        f"text split: max_abs_diff_vs_source={max_abs_diff:.9g} "
        f"cosine_vs_source={cosine:.9g}"
    )
    print(f"text split: source_avg_ms={source_ms:.2f} split_avg_ms={split_ms:.2f}")
    return metrics


def run_ort(
    onnx_path: Path,
    feeds: dict[str, np.ndarray],
    *,
    output_name: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for verification. Example:\n"
            "uv run --with onnx --with onnxruntime python split_fgclip2_text_embedding.py"
        ) from exc

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.intra_threads
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])

    output = None
    for _ in range(args.warmups):
        output = session.run([output_name], feeds)[0]
    start = time.perf_counter()
    for _ in range(args.runs):
        output = session.run([output_name], feeds)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.runs
    if output is None:
        output = session.run([output_name], feeds)[0]
    return output, elapsed_ms


def write_embedding_asset(path: Path, embedding: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "f16":
        output = embedding.astype(np.float16)
    elif dtype == "f32":
        output = embedding.astype(np.float32)
    else:
        raise ValueError(dtype)
    print(f"Writing token embedding asset -> {path}")
    output.tofile(path)
    print(f"Token embedding asset size: {size_mib(path):.1f} MiB")


def load_embedding_asset(path: Path, dtype: str, hidden_size: int) -> np.ndarray:
    np_dtype = {"f16": np.float16, "f32": np.float32}[dtype]
    embedding = np.fromfile(path, dtype=np_dtype)
    return embedding.reshape((-1, hidden_size))


def write_split_manifest(args: argparse.Namespace) -> None:
    manifest = load_json(args.base_manifest)
    manifest["onnx"]["text"] = relative_to_project(args.output_text_onnx)
    manifest["split_text_embedding"] = {
        "token_embedding": {
            "file": relative_to_project(args.embedding_output),
            "dtype": args.embedding_dtype,
            "shape": [256000, args.hidden_size],
        },
        "source_manifest": relative_to_project(args.base_manifest),
    }
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote split-text manifest -> {args.split_manifest}")


def find_initializer(graph, name: str):
    for initializer in graph.initializer:
        if initializer.name == name:
            return initializer
    raise SystemExit(f"Initializer not found: {name}")


def remove_initializer(graph, name: str) -> None:
    initializer = find_initializer(graph, name)
    graph.initializer.remove(initializer)


def remove_node_by_name(graph, name: str) -> None:
    for node in graph.node:
        if node.name == name:
            graph.node.remove(node)
            return
    raise SystemExit(f"Node not found: {name}")


def remove_graph_input(graph, name: str) -> None:
    for graph_input in graph.input:
        if graph_input.name == name:
            graph.input.remove(graph_input)
            return
    raise SystemExit(f"Graph input not found: {name}")


def replace_node_inputs(graph, old_name: str, new_name: str) -> None:
    replacement_count = 0
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
                replacement_count += 1
    if replacement_count == 0:
        raise SystemExit(f"No node inputs referenced {old_name}")


def load_tensor(info: dict[str, Any]) -> np.ndarray:
    dtype = {
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
    }[info["dtype"]]
    array = np.fromfile(PROJECT_ROOT / info["file"], dtype=dtype)
    return array.reshape(info["shape"])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / max(denom, np.finfo(np.float32).tiny))


def relative_to_project(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def size_mib(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


if __name__ == "__main__":
    main()
