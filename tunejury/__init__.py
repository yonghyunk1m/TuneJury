"""TuneJury: open instance-level pairwise reward model for text-to-music."""

from .model import TuneJury  # noqa: F401
from .dataset import TuneJuryDataset  # noqa: F401

# Inference-only entry points; pull in the optional laion_clap / transformers
# inference stack. Skip silently if those deps are unavailable, so the
# training / pre-extracted-feature pipeline still imports.
try:
    from .score import Scorer  # noqa: F401
    from .differentiable import DifferentiableScorer  # noqa: F401
except ImportError:
    Scorer = None  # type: ignore[assignment]
    DifferentiableScorer = None  # type: ignore[assignment]

__version__ = "1.0.0"
__all__ = [
    "TuneJury",
    "TuneJuryDataset",
    "Scorer",
    "DifferentiableScorer",
]
