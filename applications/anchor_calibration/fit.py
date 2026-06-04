"""Per-system Bradley-Terry intercept fitting for TuneJury.

The released TuneJury produces a scalar reward r(x) per (prompt, audio).
On generators not represented in the training mix, the reward exhibits a
systematic per-generator additive bias (see paper §A.D). We model:

    P(a wins) = sigmoid( (r(a) - beta_{system(a)}) - (r(b) - beta_{system(b)}) )

and estimate {beta_s} by L-BFGS maximum likelihood with TuneJury's margin
as offset. Fitting from K anchor pairs is ~ms; no neural training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


@dataclass
class AnchorCalibrator:
    """Per-system additive bias for TuneJury, fit on K anchor preference pairs."""

    l2: float = 1.0
    beta: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        anchor_records: Iterable[tuple[float, float, str, str, int]],
        anchor_system: str | None = None,
    ) -> "AnchorCalibrator":
        """Estimate beta_s by regularized MLE.

        Each anchor record is `(score_a, score_b, system_a, system_b, y)`
        where `score_*` are TuneJury reward scalars, `system_*` are generator
        identifiers, and `y` is `1` if human voters preferred A, else `0`.
        Ties should be excluded by the caller.

        If `anchor_system` is provided, that generator is treated as the
        in-distribution anchor and pinned at beta=0 (identifiability
        convention used in paper §A.D and §6); its column is removed from
        the L-BFGS optimization and its bias is recorded as 0.0 in the
        output dictionary. If `anchor_system` is None, beta=0 anchor is
        not pinned; identifiability is data-dependent under L2 regularization.
        """
        records = list(anchor_records)
        if not records:
            self.beta = {}
            return self

        systems = sorted({r[2] for r in records} | {r[3] for r in records})
        n = len(records)

        if anchor_system is not None and anchor_system in systems:
            free_systems = [s for s in systems if s != anchor_system]
        else:
            free_systems = list(systems)
        idx = {s: i for i, s in enumerate(free_systems)}
        k = len(free_systems)

        x_mat = np.zeros((n, k))
        offset = np.zeros(n)
        y = np.zeros(n, dtype=np.float64)
        for i, (sa, sb, sys_a, sys_b, yi) in enumerate(records):
            if sys_a in idx:
                x_mat[i, idx[sys_a]] = 1.0
            if sys_b in idx:
                x_mat[i, idx[sys_b]] = -1.0
            offset[i] = sa - sb
            y[i] = float(yi)

        def nll(beta_vec: np.ndarray) -> float:
            logit = offset - x_mat @ beta_vec
            loss = np.where(y == 1, np.log1p(np.exp(-logit)), np.log1p(np.exp(logit)))
            return float(loss.sum() + self.l2 * (beta_vec @ beta_vec))

        if k > 0:
            result = minimize(nll, np.zeros(k), method="L-BFGS-B")
            self.beta = {s: float(result.x[idx[s]]) for s in free_systems}
        else:
            self.beta = {}
        if anchor_system is not None and anchor_system in systems:
            self.beta[anchor_system] = 0.0
        return self

    def correct(self, score: float, system: str) -> float:
        """Apply the per-system correction: r(x) - beta_s."""
        return score - self.beta.get(system, 0.0)

    def predict_pairwise(
        self, score_a: float, score_b: float, system_a: str, system_b: str,
    ) -> float:
        """Return P(a wins) under the calibrated Bradley-Terry model."""
        margin = self.correct(score_a, system_a) - self.correct(score_b, system_b)
        return float(1.0 / (1.0 + np.exp(-margin)))

    def state_dict(self) -> dict:
        return {"l2": self.l2, "beta": dict(self.beta)}

    def load_state_dict(self, state: dict) -> None:
        self.l2 = float(state.get("l2", 1.0))
        self.beta = dict(state.get("beta", {}))
