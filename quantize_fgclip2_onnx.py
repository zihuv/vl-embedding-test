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
DEFAULT_QUANT_DIR = FG_LAYOUT.runtime_quantized_dir
DEFAULT_FIXTURE_MANIFEST = FG_LAYOUT.base_manifest_resolved
DEFAULT_QUANT_MANIFEST = FG_LAYOUT.quant_manifest

DEFAULT_TEXT_ONNX = FG_LAYOUT.baseline_text_onnx_resolved
DEFAULT_IMAGE_ONNX = FG_LAYOUT.baseline_image_onnx_resolved
DEFAULT_TEXT_QUANT_ONNX = FG_LAYOUT.quantized_text_onnx
DEFAULT_IMAGE_QUANT_ONNX = FG_LAYOUT.quantized_image_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create dynamic-int8 ONNX variants for the exported FG-CLIP2 "
            "text/image models, then compare them against the fp32 ONNX baseline."
        )
    )
    parser.add_argument("--text-onnx", type=Path, default=DEFAULT_TEXT_ONNX)
    parser.add_argument("--image-onnx", type=Path, default=DEFAULT_IMAGE_ONNX)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_QUANT_ONNX)
    parser.add_argument("--image-output", type=Path, default=DEFAULT_IMAGE_QUANT_ONNX)
    parser.add_argument("--fixture-manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--quant-manifest", type=Path, default=DEFAULT_QUANT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=FG_LAYOUT.quant_report)
    parser.add_argument("--op-type", action="append", dest="op_types", default=None)
    parser.add_argument(
        "--profile",
        choices=["conservative", "all-linear", "attention-only", "late-attention"],
        default="conservative",
        help=(
            "Node-selection profile. conservative currently uses all text attention projections "
            "and image attention projections in encoder layers 6..11."
        ),
    )
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--per-tensor", action="store_true", help="Disable per-channel weight quantization.")
    parser.add_argument(
        "--quantize-all-matmuls",
        action="store_true",
        help="Also quantize activation-by-activation MatMul nodes. This is usually much less accurate.",
    )
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--intra-threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--text-cosine-gate", type=float, default=0.999)
    parser.add_argument("--image-cosine-gate", type=float, default=0.999)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    op_types = args.op_types or ["MatMul", "Gemm"]
    text_quantized = not args.skip_text
    image_quantized = not args.skip_image

    if not args.skip_quantize:
        if text_quantized:
            quantize_model(
                args.text_onnx,
                args.text_output,
                op_types=op_types,
                model_kind="text",
                profile=args.profile,
                per_channel=not args.per_tensor,
                reduce_range=args.reduce_range,
                weight_only_nodes=not args.quantize_all_matmuls,
            )
        if image_quantized:
            quantize_model(
                args.image_onnx,
                args.image_output,
                op_types=op_types,
                model_kind="image",
                profile=args.profile,
                per_channel=not args.per_tensor,
                reduce_range=args.reduce_range,
                weight_only_nodes=not args.quantize_all_matmuls,
            )

    quant_paths = {
        "text": args.text_output if text_quantized else args.text_onnx,
        "image": args.image_output if image_quantized else args.image_onnx,
    }
    write_quant_manifest(args.fixture_manifest, args.quant_manifest, quant_paths)

    report: dict[str, Any] = {
        "variant": "dynamic-int8-linear",
        "op_types": op_types,
        "profile": args.profile,
        "per_channel": not args.per_tensor,
        "weight_only_nodes": not args.quantize_all_matmuls,
        "reduce_range": args.reduce_range,
        "models": model_report(args, quant_paths),
        "fixture_manifest": relative_to_project(args.fixture_manifest),
        "quant_manifest": relative_to_project(args.quant_manifest),
        "verification": {},
    }

    if not args.skip_verify:
        manifest = load_json(args.fixture_manifest)
        if text_quantized:
            report["verification"]["text"] = verify_text(args, manifest, quant_paths["text"])
        if image_quantized:
            report["verification"]["image"] = verify_image(args, manifest, quant_paths["image"])
        check_gates(args, report["verification"])

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote quantization report -> {args.report_json}")


def quantize_model(
    source: Path,
    target: Path,
    *,
    op_types: list[str],
    model_kind: str,
    profile: str,
    per_channel: bool,
    reduce_range: bool,
    weight_only_nodes: bool,
) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for quantization. Example:\n"
            "uv run --with onnx --with onnxruntime python quantize_fgclip2_onnx.py"
        ) from exc

    if not source.exists():
        raise SystemExit(f"ONNX source does not exist: {source}")

    nodes_to_quantize = None
    if weight_only_nodes:
        nodes_to_quantize = select_quant_nodes(source, op_types, model_kind, profile)
        print(f"Selected {len(nodes_to_quantize)} {profile} quantization nodes in {source.name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f"Quantizing {source} -> {target}")
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        op_types_to_quantize=op_types,
        nodes_to_quantize=nodes_to_quantize,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
    )
    elapsed = time.perf_counter() - start
    print(
        f"Quantized in {elapsed:.2f}s; "
        f"size {size_mib(source):.1f} MiB -> {size_mib(target):.1f} MiB"
    )


def select_quant_nodes(onnx_path: Path, op_types: list[str], model_kind: str, profile: str) -> list[str]:
    nodes = find_weight_matmul_nodes(onnx_path, op_types)
    if profile == "all-linear":
        return nodes
    if profile == "attention-only":
        return [node for node in nodes if "/self_attn/" in node]
    if profile == "late-attention":
        return [
            node
            for node in nodes
            if "/self_attn/" in node and (layer_index(node, model_kind) or -1) >= 6
        ]
    if profile == "conservative":
        if model_kind == "text":
            return [node for node in nodes if "/self_attn/" in node]
        if model_kind == "image":
            return [
                node
                for node in nodes
                if "/self_attn/" in node and (layer_index(node, model_kind) or -1) >= 6
            ]
    raise SystemExit(f"Unsupported quantization profile {profile!r} for {model_kind!r}")


def layer_index(node_name: str, model_kind: str) -> int | None:
    import re

    prefix = "/text_model/encoder/layers." if model_kind == "text" else "/encoder/layers."
    match = re.search(re.escape(prefix) + r"(\d+)/", node_name)
    if match is None:
        return None
    return int(match.group(1))


def find_weight_matmul_nodes(onnx_path: Path, op_types: list[str]) -> list[str]:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "onnx is required to select linear ONNX nodes. Example:\n"
            "uv run --with onnx --with onnxruntime python quantize_fgclip2_onnx.py"
        ) from exc

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    wanted_ops = set(op_types)
    nodes = []
    unnamed_index = 0
    for node in model.graph.node:
        if node.op_type not in wanted_ops:
            continue
        if not any(input_name in initializer_names for input_name in node.input):
            continue
        if node.name:
            nodes.append(node.name)
            continue
        # ONNX Runtime quantization addresses nodes by name. The exporter normally
        # names these nodes, but fail loudly if that ever changes.
        unnamed_index += 1

    if unnamed_index:
        raise SystemExit(f"{onnx_path} has {unnamed_index} unnamed weight-backed {sorted(wanted_ops)} nodes")
    return nodes


def verify_text(args: argparse.Namespace, manifest: dict[str, Any], quant_onnx: Path) -> dict[str, Any]:
    input_ids = load_tensor(manifest["tensors"]["input_ids"])
    pytorch_ref = load_tensor(manifest["tensors"]["text_ref"])
    feeds = {"input_ids": input_ids}

    fp32_out, fp32_ms = run_ort(args.text_onnx, feeds, args)
    quant_out, quant_ms = run_ort(quant_onnx, feeds, args)
    metrics = verification_report(
        fp32_out["text_features"],
        quant_out["text_features"],
        pytorch_ref,
        fp32_ms,
        quant_ms,
    )
    print_metrics("text", metrics)
    return metrics


def verify_image(args: argparse.Namespace, manifest: dict[str, Any], quant_onnx: Path) -> dict[str, Any]:
    pixel_values = load_tensor(manifest["tensors"]["pixel_values"])
    pixel_attention_mask = load_tensor(manifest["tensors"]["pixel_attention_mask"])
    pos_embed = load_tensor(manifest["tensors"]["pos_embed"])
    pytorch_ref = load_tensor(manifest["tensors"]["image_ref"])
    feeds = {
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "pos_embed": pos_embed,
    }

    fp32_out, fp32_ms = run_ort(args.image_onnx, feeds, args)
    quant_out, quant_ms = run_ort(quant_onnx, feeds, args)
    metrics = verification_report(
        fp32_out["image_features"],
        quant_out["image_features"],
        pytorch_ref,
        fp32_ms,
        quant_ms,
    )
    print_metrics("image", metrics)
    return metrics


def run_ort(
    onnx_path: Path,
    feeds: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for verification. Example:\n"
            "uv run --with onnx --with onnxruntime python quantize_fgclip2_onnx.py"
        ) from exc

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.intra_threads
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]

    outputs = None
    for _ in range(args.warmups):
        outputs = session.run(output_names, feeds)

    start = time.perf_counter()
    for _ in range(args.runs):
        outputs = session.run(output_names, feeds)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.runs

    if outputs is None:
        outputs = session.run(output_names, feeds)
    return dict(zip(output_names, outputs, strict=True)), elapsed_ms


def verification_report(
    fp32: np.ndarray,
    quant: np.ndarray,
    pytorch_ref: np.ndarray,
    fp32_ms: float,
    quant_ms: float,
) -> dict[str, Any]:
    vs_fp32 = diff_metrics(fp32, quant)
    vs_pytorch = diff_metrics(pytorch_ref, quant)
    return {
        "quant_vs_fp32": vs_fp32,
        "quant_vs_pytorch_ref": vs_pytorch,
        "fp32_avg_ms": fp32_ms,
        "quant_avg_ms": quant_ms,
        "speed_ratio_fp32_over_quant": fp32_ms / max(quant_ms, np.finfo(np.float64).tiny),
    }


def diff_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    max_abs_diff = float(np.max(np.abs(reference - actual)))
    denom = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    cosine = float(np.sum(reference * actual) / max(denom, np.finfo(np.float32).tiny))
    return {"max_abs_diff": max_abs_diff, "cosine": cosine}


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    vs_fp32 = metrics["quant_vs_fp32"]
    vs_pt = metrics["quant_vs_pytorch_ref"]
    print(
        f"{name}: quant_vs_fp32 max_abs_diff={vs_fp32['max_abs_diff']:.9g} "
        f"cosine={vs_fp32['cosine']:.9g}"
    )
    print(
        f"{name}: quant_vs_pytorch max_abs_diff={vs_pt['max_abs_diff']:.9g} "
        f"cosine={vs_pt['cosine']:.9g}"
    )
    print(
        f"{name}: fp32_avg_ms={metrics['fp32_avg_ms']:.2f} "
        f"quant_avg_ms={metrics['quant_avg_ms']:.2f} "
        f"speed_ratio={metrics['speed_ratio_fp32_over_quant']:.3f}"
    )


def check_gates(args: argparse.Namespace, verification: dict[str, Any]) -> None:
    failures = []
    if "text" in verification:
        cosine = verification["text"]["quant_vs_fp32"]["cosine"]
        if cosine < args.text_cosine_gate:
            failures.append(f"text cosine {cosine:.9g} < {args.text_cosine_gate}")
    if "image" in verification:
        cosine = verification["image"]["quant_vs_fp32"]["cosine"]
        if cosine < args.image_cosine_gate:
            failures.append(f"image cosine {cosine:.9g} < {args.image_cosine_gate}")

    if not failures:
        print("Quantized model verification passed configured cosine gates.")
        return

    message = "Quantized model verification failed configured gates: " + "; ".join(failures)
    if args.enforce_gates:
        raise SystemExit(message)
    print(message)


def load_tensor(info: dict[str, Any]) -> np.ndarray:
    dtype = {
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
    }[info["dtype"]]
    path = PROJECT_ROOT / info["file"]
    array = np.fromfile(path, dtype=dtype)
    return array.reshape(info["shape"])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_quant_manifest(base_manifest_path: Path, target_manifest_path: Path, onnx_paths: dict[str, Path]) -> None:
    manifest = load_json(base_manifest_path)
    manifest["onnx"]["text"] = relative_to_project(onnx_paths["text"])
    manifest["onnx"]["image"] = relative_to_project(onnx_paths["image"])
    manifest["quantization"] = {
        "variant": "dynamic-int8-linear",
        "source_manifest": relative_to_project(base_manifest_path),
    }

    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote quantized verifier manifest -> {target_manifest_path}")


def model_report(args: argparse.Namespace, quant_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "text": one_model_report(args.text_onnx, quant_paths["text"]),
        "image": one_model_report(args.image_onnx, quant_paths["image"]),
    }


def one_model_report(fp32: Path, quant: Path) -> dict[str, Any]:
    report = {
        "fp32": relative_to_project(fp32),
        "quantized": relative_to_project(quant),
        "fp32_mib": size_mib(fp32),
    }
    if quant.exists():
        report["quantized_mib"] = size_mib(quant)
        report["size_ratio_fp32_over_quant"] = size_mib(fp32) / max(size_mib(quant), 1e-9)
    return report


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
