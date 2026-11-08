"""parallelogram — strict validator for fine-tuning datasets."""
from __future__ import annotations

__version__ = "1.6.0"

from .api import ParallelogramError, check, fix, rules

__all__ = ["ParallelogramError", "__version__", "check", "fix", "rules"]
