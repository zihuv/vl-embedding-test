from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = PROJECT_ROOT / ".hf-cache"

MODELS_ROOT = PROJECT_ROOT / "models"
PREFERRED_SOURCE_MODELS_ROOT = MODELS_ROOT / "source"
LEGACY_SOURCE_MODELS_ROOT = MODELS_ROOT

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def source_model_dir(model_name: str) -> Path:
    return PREFERRED_SOURCE_MODELS_ROOT / model_name


@dataclass(frozen=True)
class FgClip2Layout:
    preferred_root: Path = ARTIFACTS_ROOT / "fgclip2"

    @property
    def runtime_root(self) -> Path:
        return self.preferred_root / "runtime"

    @property
    def runtime_assets_dir(self) -> Path:
        return self.runtime_root / "assets"

    @property
    def runtime_split_dir(self) -> Path:
        return self.runtime_root / "split"

    @property
    def runtime_quantized_dir(self) -> Path:
        return self.runtime_root / "quantized"

    @property
    def fixtures_root(self) -> Path:
        return self.preferred_root / "fixtures"

    @property
    def reports_root(self) -> Path:
        return self.preferred_root / "reports"

    @property
    def diagnostics_root(self) -> Path:
        return self.preferred_root / "diagnostics"

    def ensure_preferred_dirs(self) -> None:
        for path in (
            self.runtime_root,
            self.runtime_assets_dir,
            self.runtime_split_dir,
            self.runtime_quantized_dir,
            self.fixtures_root,
            self.reports_root,
            self.diagnostics_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def runtime_file(self, *parts: str) -> Path:
        return self.runtime_root.joinpath(*parts)

    def fixture_file(self, name: str) -> Path:
        return self.fixtures_root / name

    def report_file(self, name: str) -> Path:
        return self.reports_root / name

    @property
    def baseline_text_onnx(self) -> Path:
        return self.runtime_file("fgclip2_text_short_b1_s64.onnx")

    @property
    def baseline_text_onnx_resolved(self) -> Path:
        return self.baseline_text_onnx

    @property
    def baseline_image_onnx(self) -> Path:
        return self.runtime_file("fgclip2_image_core_posin_dynamic.onnx")

    @property
    def baseline_image_onnx_resolved(self) -> Path:
        return self.baseline_image_onnx

    @property
    def split_text_onnx(self) -> Path:
        return self.runtime_split_dir / "fgclip2_text_short_b1_s64_token_embeds.onnx"

    @property
    def split_text_onnx_resolved(self) -> Path:
        return self.split_text_onnx

    @property
    def token_embedding_f16(self) -> Path:
        return self.runtime_assets_dir / "text_token_embedding_256000x768_f16.bin"

    @property
    def token_embedding_f16_resolved(self) -> Path:
        return self.token_embedding_f16

    @property
    def vision_pos_embedding(self) -> Path:
        return self.runtime_assets_dir / "vision_pos_embedding_16x16x768_f32.bin"

    @property
    def vision_pos_embedding_resolved(self) -> Path:
        return self.vision_pos_embedding

    @property
    def logit_params(self) -> Path:
        return self.runtime_assets_dir / "logit_params.json"

    @property
    def logit_params_resolved(self) -> Path:
        return self.logit_params

    @property
    def quantized_text_onnx(self) -> Path:
        return self.runtime_quantized_dir / "fgclip2_text_short_b1_s64_dynamic_int8.onnx"

    @property
    def quantized_text_onnx_resolved(self) -> Path:
        return self.quantized_text_onnx

    @property
    def quantized_image_onnx(self) -> Path:
        return self.runtime_quantized_dir / "fgclip2_image_core_posin_dynamic_int8.onnx"

    @property
    def quantized_image_onnx_resolved(self) -> Path:
        return self.quantized_image_onnx

    @property
    def base_manifest(self) -> Path:
        return self.fixture_file("manifest.json")

    @property
    def base_manifest_resolved(self) -> Path:
        return self.base_manifest

    @property
    def split_manifest(self) -> Path:
        return self.fixture_file("manifest_split_text.json")

    @property
    def split_manifest_resolved(self) -> Path:
        return self.split_manifest

    @property
    def quant_manifest(self) -> Path:
        return self.fixture_file("manifest_dynamic_int8.json")

    @property
    def quant_manifest_resolved(self) -> Path:
        return self.quant_manifest

    @property
    def split_report(self) -> Path:
        return self.report_file("split_text_embedding_report.json")

    @property
    def split_report_resolved(self) -> Path:
        return self.split_report

    @property
    def quant_report(self) -> Path:
        return self.report_file("dynamic_int8_report.json")

    @property
    def quant_report_resolved(self) -> Path:
        return self.quant_report


FGCLIP2_LAYOUT = FgClip2Layout()
