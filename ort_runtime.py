from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ProviderArg = str | tuple[str, dict[str, str]]

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
}

GPU_PROVIDER_NAMES = {
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
}

CUDA_DLL_ENV = "FGCLIP2_ORT_DLL_PATHS"

BACKEND_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "cuda": "cuda",
    "nvidia": "cuda",
    "coreml": "coreml",
    "apple": "coreml",
    "dml": "dml",
    "directml": "dml",
}


@dataclass(frozen=True)
class OrtProviderSelection:
    requested: str
    desired_names: list[str]
    selected_names: list[str]
    unavailable_names: list[str]
    available_names: list[str]
    provider_args: list[ProviderArg]

    @property
    def uses_accelerator(self) -> bool:
        return any(name in GPU_PROVIDER_NAMES for name in self.selected_names)


def resolve_ort_providers(
    ort: Any,
    provider_request: str,
    *,
    coreml_cache_dir: Path | None = None,
) -> OrtProviderSelection:
    requested = provider_request.strip() or "auto"
    desired_names = expand_provider_request(requested)
    available_names = list(ort.get_available_providers())
    selected_names = [name for name in desired_names if name in available_names]
    unavailable_names = [name for name in desired_names if name not in available_names]
    if not selected_names:
        guidance = install_guidance(desired_names)
        raise RuntimeError(
            "No requested ONNX Runtime providers are available. "
            f"requested={desired_names}, available={available_names}.{guidance}"
        )

    provider_args = [build_provider_arg(name, coreml_cache_dir=coreml_cache_dir) for name in selected_names]
    return OrtProviderSelection(
        requested=requested,
        desired_names=desired_names,
        selected_names=selected_names,
        unavailable_names=unavailable_names,
        available_names=available_names,
        provider_args=provider_args,
    )


def expand_provider_request(provider_request: str) -> list[str]:
    request = provider_request.lower()
    if request in BACKEND_ALIASES:
        provider_names = backend_provider_names(BACKEND_ALIASES[request])
    else:
        provider_names = [normalize_provider_name(token) for token in provider_request.split(",")]
    return dedupe_preserve_order(provider_names)


def auto_provider_names() -> list[str]:
    if sys.platform == "win32":
        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if sys.platform == "darwin":
        return [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def backend_provider_names(backend: str) -> list[str]:
    if backend == "auto":
        return auto_provider_names()
    if backend == "cpu":
        return ["CPUExecutionProvider"]
    if backend == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if backend == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if backend == "dml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"Unsupported backend {backend!r}.")


def normalize_provider_name(token: str) -> str:
    normalized = token.strip()
    if not normalized:
        raise ValueError("Empty provider name in request.")
    if normalized in PROVIDER_ALIASES.values():
        return normalized
    alias = PROVIDER_ALIASES.get(normalized.lower())
    if alias is None:
        supported = ", ".join(sorted(PROVIDER_ALIASES))
        raise ValueError(f"Unsupported provider {normalized!r}. Supported aliases: {supported}.")
    return alias


def build_provider_arg(name: str, *, coreml_cache_dir: Path | None) -> ProviderArg:
    if name == "CoreMLExecutionProvider" and coreml_cache_dir is not None:
        coreml_cache_dir.mkdir(parents=True, exist_ok=True)
        return (name, {"ModelCacheDirectory": str(coreml_cache_dir)})
    return name


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def install_guidance(desired_names: list[str]) -> str:
    if "CUDAExecutionProvider" in desired_names:
        return (
            " Install an ONNX Runtime build with CUDA support, such as "
            "`onnxruntime-gpu`, and ensure CUDA/cuDNN are on PATH."
        )
    if "CoreMLExecutionProvider" in desired_names:
        machine = platform.machine().lower()
        extra = " on Apple Silicon" if machine in {"arm64", "aarch64"} else ""
        return (
            " Use an ONNX Runtime build that exposes `CoreMLExecutionProvider`"
            f"{extra}; the default Python CPU package may not include it."
        )
    if "DmlExecutionProvider" in desired_names:
        return " Use an ONNX Runtime build that exposes `DmlExecutionProvider`."
    return ""


def prepare_ort_environment(provider_request: str) -> list[Path]:
    desired_names = expand_provider_request(provider_request)
    if sys.platform != "win32" or "CUDAExecutionProvider" not in desired_names:
        return []

    added_dirs: list[Path] = []
    current_path = os_path_entries()
    for dll_dir in discover_windows_cuda_dll_dirs():
        dll_dir_str = str(dll_dir)
        if dll_dir_str not in current_path:
            os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
            current_path.add(dll_dir_str)
            added_dirs.append(dll_dir)
    return added_dirs


def discover_windows_cuda_dll_dirs() -> list[Path]:
    candidates: list[Path] = []

    explicit = os.environ.get(CUDA_DLL_ENV, "")
    for raw in explicit.split(os.pathsep):
        raw = raw.strip()
        if raw:
            candidates.append(Path(raw))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Lib" / "site-packages" / "torch" / "lib")

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        home = Path(user_profile)
        candidates.extend(
            [
                home / "miniconda3" / "Lib" / "site-packages" / "torch" / "lib",
                home / "anaconda3" / "Lib" / "site-packages" / "torch" / "lib",
            ]
        )

    return [
        path
        for path in dedupe_preserve_order([str(path) for path in candidates])
        if valid_windows_cuda_dll_dir(Path(path))
    ]


def valid_windows_cuda_dll_dir(path: Path) -> bool:
    return path.is_dir() and (path / "cudnn64_9.dll").exists()


def os_path_entries() -> set[str]:
    return {entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry}
