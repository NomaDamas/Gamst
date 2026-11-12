"""Offline empirical calibration for explicitly labeled interaction outcomes.

This module calibrates an existing scorer; it does not learn a psychological
target from a prompt. Training and held-out samples must come from distinct
datasets with disjoint IDs. Callers must additionally prevent participant,
session, and temporal leakage appropriate to their intended deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score must be finite and between 0 and 1")
    return float(value)


def _count(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{name} must be an {'positive' if positive else 'non-negative'} integer")
    return value


@dataclass(frozen=True)
class LabeledOutcome:
    """One labeled target event, identified before the dataset split."""
    id: str
    score: float
    outcome: bool

    def __post_init__(self):
        _text(self.id, "sample id")
        _score(self.score)
        if not isinstance(self.outcome, bool):
            raise ValueError("outcome must be boolean, not an unverified model label")


@dataclass(frozen=True)
class CalibrationBin:
    index: int
    training_count: int
    training_positives: int
    validation_count: int
    validation_positives: int

    def __post_init__(self):
        for name in ("index", "training_count", "training_positives", "validation_count", "validation_positives"):
            _count(getattr(self, name), name)
        if self.training_positives > self.training_count or self.validation_positives > self.validation_count:
            raise ValueError("positive count exceeds sample count")

    @property
    def probability(self):
        return self.training_positives / self.training_count if self.training_count else None


@dataclass(frozen=True)
class CalibrationArtifact:
    target: str
    training_dataset: str
    validation_dataset: str
    scorer_model_id: str
    scorer_model_version: str
    bins: tuple[CalibrationBin, ...]
    min_train_per_bin: int = 20
    min_validation_per_bin: int = 20
    max_ece: float = .1
    max_brier: float = .25
    min_coverage: float = .5

    def __post_init__(self):
        for name in ("target", "training_dataset", "validation_dataset", "scorer_model_id", "scorer_model_version"):
            _text(getattr(self, name), name)
        if self.training_dataset == self.validation_dataset:
            raise ValueError("training and held-out dataset identifiers must differ")
        if not isinstance(self.bins, tuple) or not 1 <= len(self.bins) <= 100:
            raise ValueError("bins must be an immutable tuple with 1 to 100 bins")
        if any(not isinstance(item, CalibrationBin) or item.index != i for i, item in enumerate(self.bins)):
            raise ValueError("bins must have consecutive ordered indexes")
        _count(self.min_train_per_bin, "min_train_per_bin", positive=True)
        _count(self.min_validation_per_bin, "min_validation_per_bin", positive=True)
        for name in ("max_ece", "max_brier", "min_coverage"):
            _score(getattr(self, name))
        if self.min_coverage <= 0:
            raise ValueError("min_coverage must be positive")

    def _eligible(self, item):
        return item.training_count >= self.min_train_per_bin and item.validation_count >= self.min_validation_per_bin

    @property
    def evaluation(self):
        total = sum(item.validation_count for item in self.bins)
        eligible = [item for item in self.bins if self._eligible(item)]
        count = sum(item.validation_count for item in eligible)
        ece = (sum(abs(item.probability-item.validation_positives/item.validation_count)*item.validation_count
                   for item in eligible)/count) if count else None
        # For a bin's constant training prediction, the binary Brier sum is
        # positives*(1-p)^2 + negatives*p^2.
        brier = (sum(item.validation_positives*(1-item.probability)**2 +
                     (item.validation_count-item.validation_positives)*item.probability**2
                     for item in eligible)/count) if count else None
        coverage = count/total if total else 0.0
        passed = count > 0 and coverage >= self.min_coverage and ece <= self.max_ece and brier <= self.max_brier
        return {"method": "held_out_equal_width_histogram", "validated": passed,
                "validation_sample_count": total, "evaluated_sample_count": count,
                "eligible_bin_count": len(eligible), "coverage": coverage,
                "ece": ece, "brier_score": brier,
                "thresholds": {"max_ece": self.max_ece, "max_brier": self.max_brier,
                               "min_coverage": self.min_coverage,
                               "min_train_per_bin": self.min_train_per_bin,
                               "min_validation_per_bin": self.min_validation_per_bin}}

    @property
    def validated(self):
        return self.evaluation["validated"]

    def predict(self, score, *, model_id=None, model_version=None):
        """Return an empirical probability or abstain; never fill missing bins."""
        score = _score(score)
        item = self.bins[min(int(score*len(self.bins)), len(self.bins)-1)]
        reason = None
        if (model_id is not None and model_id != self.scorer_model_id) or (model_version is not None and model_version != self.scorer_model_version):
            reason = "scorer_mismatch"
        elif not self._eligible(item):
            reason = "insufficient_bin_samples"
        elif not self.validated:
            reason = "held_out_validation_failed"
        elif abs(item.probability-item.validation_positives/item.validation_count) > self.max_ece:
            reason = "held_out_bin_validation_failed"
        result = {"target": self.target, "model_score": score, "probability": None,
                  "calibrated": False, "calibration": None,
                  "status": "abstained", "reason": reason}
        if reason is None:
            result.update({"probability": item.probability, "calibrated": True,
                "status": "available", "calibration": {
                    "dataset": self.validation_dataset, "training_dataset": self.training_dataset,
                    "model_version": "equal-width-histogram-v1", "target": self.target,
                    "scorer_model_id": self.scorer_model_id, "scorer_model_version": self.scorer_model_version,
                    "evaluation": self.evaluation,
                    "bin": {"index": item.index, "lower": item.index/len(self.bins),
                            "upper": (item.index+1)/len(self.bins),
                            "training_count": item.training_count, "validation_count": item.validation_count}}})
        return result

    def to_dict(self):
        return {"schema_version": "0.1", "method": "equal-width-histogram-v1",
                "target": self.target, "training_dataset": self.training_dataset,
                "validation_dataset": self.validation_dataset,
                "scorer_model_id": self.scorer_model_id, "scorer_model_version": self.scorer_model_version,
                "bins": [{"index": item.index, "training_count": item.training_count,
                          "training_positives": item.training_positives, "validation_count": item.validation_count,
                          "validation_positives": item.validation_positives} for item in self.bins],
                "min_train_per_bin": self.min_train_per_bin,
                "min_validation_per_bin": self.min_validation_per_bin,
                "max_ece": self.max_ece, "max_brier": self.max_brier, "min_coverage": self.min_coverage}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, payload):
        """Load a trusted artifact, revalidating counts and recomputing metrics.

        Serialization is not an authenticity signature. Dataset provenance must
        be verified before accepting an artifact supplied by another party.
        """
        data = json.loads(payload)
        if not isinstance(data, dict) or data.pop("schema_version", None) != "0.1" or data.pop("method", None) != "equal-width-histogram-v1":
            raise ValueError("unsupported calibration artifact")
        try:
            data["bins"] = tuple(CalibrationBin(**item) for item in data["bins"])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid calibration artifact") from exc


def fit_calibration(*, target: str, training: Iterable[LabeledOutcome], validation: Iterable[LabeledOutcome],
                    training_dataset: str, validation_dataset: str, scorer_model_id: str,
                    scorer_model_version: str, bins: int = 5, min_train_per_bin: int = 20,
                    min_validation_per_bin: int = 20, max_ece: float = .1,
                    max_brier: float = .25, min_coverage: float = .5) -> CalibrationArtifact:
    """Fit train-set bin frequencies; assess them on held-out labeled events.

    Validation labels never alter probabilities. Thresholds are acceptance
    criteria, not a guarantee of calibration on new users or distributions.
    """
    _count(bins, "bins", positive=True)
    if bins > 100:
        raise ValueError("at most 100 bins are supported")
    stats = [[0, 0, 0, 0] for _ in range(bins)]
    seen = set()
    for dataset, samples in enumerate((training, validation)):
        for sample in samples:
            if not isinstance(sample, LabeledOutcome):
                raise TypeError("samples must be LabeledOutcome instances")
            if sample.id in seen:
                raise ValueError("duplicate sample ID or training/validation leakage")
            seen.add(sample.id)
            item = stats[min(int(sample.score*bins), bins-1)]
            item[dataset*2] += 1
            item[dataset*2+1] += int(sample.outcome)
    return CalibrationArtifact(target=target, training_dataset=training_dataset,
        validation_dataset=validation_dataset, scorer_model_id=scorer_model_id,
        scorer_model_version=scorer_model_version,
        bins=tuple(CalibrationBin(i, *item) for i, item in enumerate(stats)),
        min_train_per_bin=min_train_per_bin, min_validation_per_bin=min_validation_per_bin,
        max_ece=max_ece, max_brier=max_brier, min_coverage=min_coverage)
