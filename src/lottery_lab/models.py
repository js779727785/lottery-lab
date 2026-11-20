"""Leakage-free online models for 双色球 rankings."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .metrics import BLUE_NUMBERS, RED_NUMBERS
from .schema import Draw


@dataclass(frozen=True)
class Prediction:
    red_scores: Mapping[int, float]
    blue_scores: Mapping[int, float]
    red_probs: Mapping[int, float]
    blue_probs: Mapping[int, float]
    red_ranking: Tuple[int, ...]
    blue_ranking: Tuple[int, ...]


def _rank(scores: Mapping[int, float], numbers: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(numbers, key=lambda number: (-scores.get(number, 0.0), number)))


def _zscore(scores: Mapping[int, float], numbers: Sequence[int]) -> Dict[int, float]:
    values = [float(scores.get(number, 0.0)) for number in numbers]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-12:
        return {number: 0.0 for number in numbers}
    return {number: (float(scores.get(number, 0.0)) - mean) / std for number in numbers}


def _bounded_probabilities(
    scores: Mapping[int, float],
    numbers: Sequence[int],
    total_mass: float,
    strength: float,
    minimum: float,
    maximum: float,
) -> Dict[int, float]:
    """Calibrate a ranking into bounded marginal probabilities.

    The cap prevents a single ball from being presented as a near-certainty.
    The sum constraint keeps the expected number of selected balls realistic.
    """

    z_scores = _zscore(scores, numbers)
    baseline = total_mass / len(numbers)
    probabilities = {
        number: min(maximum, max(minimum, baseline + strength * z_scores[number]))
        for number in numbers
    }
    for _ in range(24):
        difference = total_mass - sum(probabilities.values())
        if abs(difference) < 1e-10:
            break
        if difference > 0:
            adjustable = [number for number in numbers if probabilities[number] < maximum - 1e-9]
        else:
            adjustable = [number for number in numbers if probabilities[number] > minimum + 1e-9]
        if not adjustable:
            break
        share = difference / len(adjustable)
        for number in adjustable:
            probabilities[number] = min(maximum, max(minimum, probabilities[number] + share))
    return probabilities


def _make_prediction(
    red_scores: Mapping[int, float],
    blue_scores: Mapping[int, float],
) -> Prediction:
    red_probs = _bounded_probabilities(
        red_scores, RED_NUMBERS, 6.0, strength=0.055, minimum=0.02, maximum=0.45
    )
    blue_probs = _bounded_probabilities(
        blue_scores, BLUE_NUMBERS, 1.0, strength=0.025, minimum=0.005, maximum=0.30
    )
    return Prediction(
        red_scores=dict(red_scores),
        blue_scores=dict(blue_scores),
        red_probs=red_probs,
        blue_probs=blue_probs,
        red_ranking=_rank(red_scores, RED_NUMBERS),
        blue_ranking=_rank(blue_scores, BLUE_NUMBERS),
    )


class Model:
    key = "base"
    label = "Base"

    def reset(self) -> None:
        pass

    def observe(self, draw: Draw) -> None:
        pass

    def predict(self, target_issue: str) -> Prediction:
        raise NotImplementedError


class RandomModel(Model):
    """A fair baseline: uniform marginal probabilities, random ranking."""

    key = "random"
    label = "随机基线"

    def predict(self, target_issue: str) -> Prediction:
        digest = hashlib.sha256(f"ssq-random-baseline:{target_issue}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        red_ranking = list(RED_NUMBERS)
        blue_ranking = list(BLUE_NUMBERS)
        rng.shuffle(red_ranking)
        rng.shuffle(blue_ranking)
        red_probs = {number: 6.0 / 33.0 for number in RED_NUMBERS}
        blue_probs = {number: 1.0 / 16.0 for number in BLUE_NUMBERS}
        return Prediction(
            red_scores=dict(red_probs),
            blue_scores=dict(blue_probs),
            red_probs=red_probs,
            blue_probs=blue_probs,
            red_ranking=tuple(red_ranking),
            blue_ranking=tuple(blue_ranking),
        )


class FrequencyModel(Model):
    key = "frequency"
    label = "历史频率"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.red_counts = {number: 0 for number in RED_NUMBERS}
        self.blue_counts = {number: 0 for number in BLUE_NUMBERS}

    def reset(self) -> None:
        self.red_counts = {number: 0 for number in RED_NUMBERS}
        self.blue_counts = {number: 0 for number in BLUE_NUMBERS}

    def observe(self, draw: Draw) -> None:
        for number in draw.reds:
            self.red_counts[number] += 1
        self.blue_counts[draw.blue] += 1

    def red_scores(self) -> Dict[int, float]:
        return {number: self.red_counts[number] + self.alpha for number in RED_NUMBERS}

    def blue_scores(self) -> Dict[int, float]:
        return {number: self.blue_counts[number] + self.alpha for number in BLUE_NUMBERS}

    def predict(self, target_issue: str) -> Prediction:
        return _make_prediction(self.red_scores(), self.blue_scores())


class RecencyModel(Model):
    key = "recency"
    label = "近因加权"

    def __init__(self, half_life: float = 52.0) -> None:
        self.half_life = half_life
        self.decay = 0.5 ** (1.0 / half_life)
        self.red_weights = {number: 0.0 for number in RED_NUMBERS}
        self.blue_weights = {number: 0.0 for number in BLUE_NUMBERS}

    def reset(self) -> None:
        self.red_weights = {number: 0.0 for number in RED_NUMBERS}
        self.blue_weights = {number: 0.0 for number in BLUE_NUMBERS}

    def observe(self, draw: Draw) -> None:
        for number in RED_NUMBERS:
            self.red_weights[number] *= self.decay
        for number in BLUE_NUMBERS:
            self.blue_weights[number] *= self.decay
        for number in draw.reds:
            self.red_weights[number] += 1.0
        self.blue_weights[draw.blue] += 1.0

    def red_scores(self) -> Dict[int, float]:
        return dict(self.red_weights)

    def blue_scores(self) -> Dict[int, float]:
        return dict(self.blue_weights)

    def predict(self, target_issue: str) -> Prediction:
        return _make_prediction(self.red_scores(), self.blue_scores())


class GapModel(Model):
    key = "gap"
    label = "遗漏回补"

    def __init__(self) -> None:
        self.step = 0
        self.red_last_seen = {number: 0 for number in RED_NUMBERS}
        self.blue_last_seen = {number: 0 for number in BLUE_NUMBERS}

    def reset(self) -> None:
        self.step = 0
        self.red_last_seen = {number: 0 for number in RED_NUMBERS}
        self.blue_last_seen = {number: 0 for number in BLUE_NUMBERS}

    def observe(self, draw: Draw) -> None:
        self.step += 1
        for number in draw.reds:
            self.red_last_seen[number] = self.step
        self.blue_last_seen[draw.blue] = self.step

    def red_scores(self) -> Dict[int, float]:
        return {
            number: float(self.step - self.red_last_seen[number])
            if self.red_last_seen[number]
            else float(self.step + 1)
            for number in RED_NUMBERS
        }

    def blue_scores(self) -> Dict[int, float]:
        return {
            number: float(self.step - self.blue_last_seen[number])
            if self.blue_last_seen[number]
            else float(self.step + 1)
            for number in BLUE_NUMBERS
        }

    def predict(self, target_issue: str) -> Prediction:
        return _make_prediction(self.red_scores(), self.blue_scores())


class MarkovModel(Model):
    key = "markov"
    label = "一阶转移"

    def __init__(self, smoothing: float = 0.5) -> None:
        self.smoothing = smoothing
        self.red_transition = {
            left: {right: 0 for right in RED_NUMBERS} for left in RED_NUMBERS
        }
        self.blue_transition = {
            left: {right: 0 for right in BLUE_NUMBERS} for left in BLUE_NUMBERS
        }
        self.red_counts = {number: 0 for number in RED_NUMBERS}
        self.blue_counts = {number: 0 for number in BLUE_NUMBERS}
        self.last_draw = None  # type: ignore[assignment]
        self.n = 0

    def reset(self) -> None:
        self.red_transition = {
            left: {right: 0 for right in RED_NUMBERS} for left in RED_NUMBERS
        }
        self.blue_transition = {
            left: {right: 0 for right in BLUE_NUMBERS} for left in BLUE_NUMBERS
        }
        self.red_counts = {number: 0 for number in RED_NUMBERS}
        self.blue_counts = {number: 0 for number in BLUE_NUMBERS}
        self.last_draw = None
        self.n = 0

    def observe(self, draw: Draw) -> None:
        if self.last_draw is not None:
            for left in self.last_draw.reds:
                for right in draw.reds:
                    self.red_transition[left][right] += 1
            self.blue_transition[self.last_draw.blue][draw.blue] += 1
        for number in draw.reds:
            self.red_counts[number] += 1
        self.blue_counts[draw.blue] += 1
        self.last_draw = draw
        self.n += 1

    def red_scores(self) -> Dict[int, float]:
        if self.last_draw is None or self.n == 0:
            return {number: 1.0 for number in RED_NUMBERS}
        return {
            number: sum(self.red_transition[left][number] for left in self.last_draw.reds)
            + self.smoothing * self.red_counts[number] / self.n
            for number in RED_NUMBERS
        }

    def blue_scores(self) -> Dict[int, float]:
        if self.last_draw is None or self.n == 0:
            return {number: 1.0 for number in BLUE_NUMBERS}
        return {
            number: self.blue_transition[self.last_draw.blue][number]
            + self.smoothing * self.blue_counts[number] / self.n
            for number in BLUE_NUMBERS
        }

    def predict(self, target_issue: str) -> Prediction:
        return _make_prediction(self.red_scores(), self.blue_scores())


class EnsembleModel(Model):
    key = "ensemble"
    label = "多因子集成"

    def __init__(self) -> None:
        self.frequency = FrequencyModel()
        self.recency = RecencyModel()
        self.gap = GapModel()
        self.markov = MarkovModel()
        self.red_weights = {"frequency": 0.18, "recency": 0.30, "gap": 0.10, "markov": 0.42}
        self.blue_weights = {"frequency": 0.22, "recency": 0.24, "gap": 0.14, "markov": 0.40}

    def reset(self) -> None:
        for model in (self.frequency, self.recency, self.gap, self.markov):
            model.reset()

    def observe(self, draw: Draw) -> None:
        for model in (self.frequency, self.recency, self.gap, self.markov):
            model.observe(draw)

    def _combined(
        self,
        components: Mapping[str, Model],
        weights: Mapping[str, float],
        numbers: Sequence[int],
        kind: str,
    ) -> Dict[int, float]:
        combined = {number: 0.0 for number in numbers}
        for name, model in components.items():
            if kind == "red":
                raw = model.red_scores()  # type: ignore[attr-defined]
            else:
                raw = model.blue_scores()  # type: ignore[attr-defined]
            normalized = _zscore(raw, numbers)
            weight = weights[name]
            for number in numbers:
                combined[number] += weight * normalized[number]
        return combined

    def red_scores(self) -> Dict[int, float]:
        return self._combined(
            {
                "frequency": self.frequency,
                "recency": self.recency,
                "gap": self.gap,
                "markov": self.markov,
            },
            self.red_weights,
            RED_NUMBERS,
            "red",
        )

    def blue_scores(self) -> Dict[int, float]:
        return self._combined(
            {
                "frequency": self.frequency,
                "recency": self.recency,
                "gap": self.gap,
                "markov": self.markov,
            },
            self.blue_weights,
            BLUE_NUMBERS,
            "blue",
        )

    def predict(self, target_issue: str) -> Prediction:
        return _make_prediction(self.red_scores(), self.blue_scores())


class OnlineLogisticModel(Model):
    """Online logistic regression trained only on past draws."""

    label = "在线逻辑回归"

    def __init__(self, key: str = "logistic", learning_rate: float = 0.03, l2: float = 0.0005) -> None:
        self.key = key
        self.learning_rate = learning_rate
        self.l2 = l2
        self.frequency = FrequencyModel()
        self.recency = RecencyModel()
        self.gap = GapModel()
        self.markov = MarkovModel()
        self.red_weights = [0.0] * 7
        self.blue_weights = [0.0] * 7
        self._pending = None  # type: ignore[assignment]

    def reset(self) -> None:
        for model in (self.frequency, self.recency, self.gap, self.markov):
            model.reset()
        self.red_weights = [0.0] * 7
        self.blue_weights = [0.0] * 7
        self._pending = None

    def observe(self, draw: Draw) -> None:
        features = self._pending if self._pending is not None else self._build_features()
        red_features, blue_features = features
        self._update_weights(self.red_weights, red_features, set(draw.reds), self.learning_rate)
        self._update_weights(self.blue_weights, blue_features, {draw.blue}, self.learning_rate)
        self._pending = None
        for model in (self.frequency, self.recency, self.gap, self.markov):
            model.observe(draw)

    def _build_features(self):
        freq_red = _zscore(self.frequency.red_scores(), RED_NUMBERS)
        rec_red = _zscore(self.recency.red_scores(), RED_NUMBERS)
        gap_red = _zscore(self.gap.red_scores(), RED_NUMBERS)
        markov_red = _zscore(self.markov.red_scores(), RED_NUMBERS)
        last_draw = self.markov.last_draw
        red_features = {
            number: (
                1.0,
                freq_red[number],
                rec_red[number],
                gap_red[number],
                markov_red[number],
                1.0 if last_draw is not None and number in last_draw.reds else 0.0,
                (number - 17.0) / 16.0,
            )
            for number in RED_NUMBERS
        }

        freq_blue = _zscore(self.frequency.blue_scores(), BLUE_NUMBERS)
        rec_blue = _zscore(self.recency.blue_scores(), BLUE_NUMBERS)
        gap_blue = _zscore(self.gap.blue_scores(), BLUE_NUMBERS)
        markov_blue = _zscore(self.markov.blue_scores(), BLUE_NUMBERS)
        blue_features = {
            number: (
                1.0,
                freq_blue[number],
                rec_blue[number],
                gap_blue[number],
                markov_blue[number],
                1.0 if last_draw is not None and last_draw.blue == number else 0.0,
                (number - 8.5) / 7.5,
            )
            for number in BLUE_NUMBERS
        }
        return red_features, blue_features

    def _scores(self, features, weights):
        return {
            number: sum(weight * value for weight, value in zip(weights, vector))
            for number, vector in features.items()
        }

    def _update_weights(self, weights, features, positives, learning_rate):
        for number, vector in features.items():
            score = sum(weight * value for weight, value in zip(weights, vector))
            probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score))))
            error = probability - (1.0 if number in positives else 0.0)
            for index, value in enumerate(vector):
                weights[index] += learning_rate * error * value - self.l2 * weights[index]

    def predict(self, target_issue: str) -> Prediction:
        features = self._build_features()
        self._pending = features
        red_features, blue_features = features
        red_scores = self._scores(red_features, self.red_weights)
        blue_scores = self._scores(blue_features, self.blue_weights)
        return _make_prediction(red_scores, blue_scores)


def model_registry() -> List[Model]:
    return [
        RandomModel(),
        FrequencyModel(),
        RecencyModel(),
        GapModel(),
        MarkovModel(),
        EnsembleModel(),
        OnlineLogisticModel("logistic", 0.03, 0.0005),
        OnlineLogisticModel("logistic_fast", 0.08, 0.0005),
    ]


def model_by_key(key: str) -> Model:
    for model in model_registry():
        if model.key == key:
            return model
    raise KeyError(key)
