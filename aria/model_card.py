"""
aria.model_card — Model Card generation for AI transparency (EU AI Act Art. 13).

Generates structured model cards following the Hugging Face / Google Model Card
format, enriched with ARIA audit metrics.

Usage::

    from aria.model_card import ModelCardGenerator, ModelCardConfig

    gen = ModelCardGenerator(storage, ab_runner=runner)

    config = ModelCardConfig(
        model_id="sentiment-v2",
        model_name="Sentiment Classifier v2",
        intended_use="Customer feedback analysis",
        limitations="May underperform on non-English text",
    )

    card = gen.generate(config, epoch_ids=["epoch-prod-123"])
    print(card.to_markdown())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelCardConfig:
    """Configuration for model card generation."""
    model_id:           str
    model_name:         str = ""
    version:            str = "1.0.0"
    model_type:         str = ""          # e.g. "text-classification"
    language:           list[str] = field(default_factory=lambda: ["en"])
    license:            str = "proprietary"
    intended_use:       str = ""
    out_of_scope:       str = ""
    limitations:        str = ""
    bias_considerations: str = ""
    training_data:      str = ""
    evaluation_data:    str = ""
    contact:            str = ""
    extra_sections:     dict[str, str] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    """Aggregated metrics for a set of inference epochs."""
    n_records:        int = 0
    mean_confidence:  float | None = None
    std_confidence:   float | None = None
    mean_latency_ms:  float | None = None
    std_latency_ms:   float | None = None
    p95_latency_ms:   float | None = None
    error_rate:       float | None = None  # fraction of records with confidence < 0.5


@dataclass
class ModelCard:
    """Generated model card document."""
    config:           ModelCardConfig
    metrics:          EpochMetrics
    epoch_ids:        list[str]
    generated_at:     str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_markdown(self) -> str:
        """Render the model card as Markdown."""
        cfg = self.config
        m = self.metrics
        lines = [
            f"# Model Card: {cfg.model_name or cfg.model_id}",
            f"",
            f"**Model ID**: `{cfg.model_id}`  ",
            f"**Version**: {cfg.version}  ",
            f"**Generated**: {self.generated_at}  ",
        ]

        if cfg.model_type:
            lines.append(f"**Type**: {cfg.model_type}  ")
        if cfg.language:
            lines.append(f"**Languages**: {', '.join(cfg.language)}  ")
        if cfg.license:
            lines.append(f"**License**: {cfg.license}  ")

        lines += ["", "---", ""]

        if cfg.intended_use:
            lines += ["## Intended Use", "", cfg.intended_use, ""]
        if cfg.out_of_scope:
            lines += ["## Out-of-Scope Use", "", cfg.out_of_scope, ""]
        if cfg.limitations:
            lines += ["## Limitations", "", cfg.limitations, ""]
        if cfg.bias_considerations:
            lines += ["## Bias Considerations", "", cfg.bias_considerations, ""]

        lines += ["## Performance Metrics", ""]
        if m.n_records > 0:
            lines.append(f"*Based on {m.n_records} inference records from epochs: "
                         f"{', '.join(self.epoch_ids)}*")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            if m.mean_confidence is not None:
                lines.append(f"| Mean confidence | {m.mean_confidence:.4f} |")
            if m.std_confidence is not None:
                lines.append(f"| Std confidence  | {m.std_confidence:.4f} |")
            if m.mean_latency_ms is not None:
                lines.append(f"| Mean latency    | {m.mean_latency_ms:.1f} ms |")
            if m.p95_latency_ms is not None:
                lines.append(f"| P95 latency     | {m.p95_latency_ms:.1f} ms |")
            if m.error_rate is not None:
                lines.append(f"| Low-conf rate   | {m.error_rate:.2%} |")
        else:
            lines.append("*No metrics available for the specified epochs.*")

        lines.append("")

        if cfg.training_data:
            lines += ["## Training Data", "", cfg.training_data, ""]
        if cfg.evaluation_data:
            lines += ["## Evaluation Data", "", cfg.evaluation_data, ""]

        for section_title, content in cfg.extra_sections.items():
            lines += [f"## {section_title}", "", content, ""]

        if cfg.contact:
            lines += ["## Contact", "", cfg.contact, ""]

        lines += [
            "---",
            "*Generated by [ARIA BSV](https://github.com/ARIA-BSV) — "
            "Auditable AI on Bitcoin SV.*",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to plain dict."""
        return {
            "model_id":       self.config.model_id,
            "model_name":     self.config.model_name,
            "version":        self.config.version,
            "generated_at":   self.generated_at,
            "epoch_ids":      self.epoch_ids,
            "metrics": {
                "n_records":       self.metrics.n_records,
                "mean_confidence": self.metrics.mean_confidence,
                "mean_latency_ms": self.metrics.mean_latency_ms,
                "p95_latency_ms":  self.metrics.p95_latency_ms,
                "error_rate":      self.metrics.error_rate,
            },
            "intended_use":   self.config.intended_use,
            "limitations":    self.config.limitations,
        }


# ---------------------------------------------------------------------------
# ModelCardGenerator
# ---------------------------------------------------------------------------

class ModelCardGenerator:
    """Generates model cards from ARIA storage metrics.

    Args:
        storage:       StorageInterface implementation.
        conf_threshold: Confidence below which a record is counted as "error".
    """

    def __init__(
        self,
        storage: "StorageInterface",
        conf_threshold: float = 0.5,
    ) -> None:
        self._storage = storage
        self._conf_threshold = conf_threshold

    def generate(
        self,
        config: ModelCardConfig,
        epoch_ids: list[str] | None = None,
    ) -> ModelCard:
        """Generate a model card for the given epoch(s).

        Args:
            config:     ModelCardConfig with metadata.
            epoch_ids:  Epoch IDs to aggregate metrics from.

        Returns:
            ModelCard instance.
        """
        epoch_ids = epoch_ids or []
        records = []
        for eid in epoch_ids:
            try:
                records.extend(self._storage.list_records_by_epoch(eid))
            except Exception as exc:
                _log.warning("ModelCard: could not load epoch %s: %s", eid, exc)

        metrics = self._compute_metrics(records)
        return ModelCard(config=config, metrics=metrics, epoch_ids=epoch_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_metrics(self, records: list[Any]) -> EpochMetrics:
        if not records:
            return EpochMetrics()

        confidences = []
        latencies   = []

        for r in records:
            c = getattr(r, "confidence", None)
            if c is not None and float(c) > 0:
                confidences.append(float(c))
            l = getattr(r, "latency_ms", None)
            if l is not None and float(l) > 0:
                latencies.append(float(l))

        def mean(vals):
            return sum(vals) / len(vals) if vals else None

        def std(vals):
            import math
            if len(vals) < 2:
                return None
            m = mean(vals)
            return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

        def p95(vals):
            if not vals:
                return None
            sv = sorted(vals)
            idx = int(0.95 * len(sv))
            return sv[min(idx, len(sv) - 1)]

        error_rate = None
        if confidences:
            low = sum(1 for c in confidences if c < self._conf_threshold)
            error_rate = low / len(confidences)

        return EpochMetrics(
            n_records=len(records),
            mean_confidence=mean(confidences),
            std_confidence=std(confidences),
            mean_latency_ms=mean(latencies),
            std_latency_ms=std(latencies),
            p95_latency_ms=p95(latencies),
            error_rate=error_rate,
        )
