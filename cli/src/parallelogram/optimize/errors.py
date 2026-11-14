"""Failure types for `parallelogram optimize`.

The CLI maps these onto the existing exit-code contract instead of adding new
codes: a contract the user wrote wrong is a usage error (exit 2), and
something the machine could not do (missing extras, missing model, out of
memory) is an operational failure (exit 3). Neither is ever a data finding.
"""
from __future__ import annotations

from ..api import ParallelogramError


class ConfigError(ValueError):
    """The experiment contract is invalid. CLI exit 2."""


class UnsupportedError(ConfigError):
    """The contract asks for a format or objective this release does not train.

    Raised instead of guessing: silently training tool-calling or preference
    records as plain chat would produce a model that looks trained and is not.
    """


class LeakageError(ConfigError):
    """Splits overlap in a way the configured policy refuses to proceed with."""


class SplitIsolationError(RuntimeError):
    """Code tried to read final test data before the candidate was frozen.

    This is a programming error, not a user error: it exists so that a future
    change which routes test examples into selection fails loudly in tests.
    """


class OptimizeError(ParallelogramError):
    """The experiment could not be carried out. CLI exit 3."""


class MissingDependencyError(OptimizeError):
    """The optional ML extras are not installed."""


class OutOfMemory(OptimizeError):
    """Training or scoring ran out of memory even at the smallest batch."""


class SelectionInfeasible(ConfigError):
    """Coverage minimums cannot be met inside the token budget or the pool."""
