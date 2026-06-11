"""Token-counting strategies for the context-window rule.

Different model families tokenize differently, so we pick a *model-specific*
tokenizer wherever one is available offline:

  * OpenAI models (gpt-4o, gpt-4, gpt-3.5, o-series, …) → tiktoken, the exact
    BPE tokenizer OpenAI ships. Offline (after a one-time ranks download),
    no auth.
  * Open-weight HuggingFace models (Llama, Mistral, Qwen, Gemma, Phi, …) →
    the `tokenizers` package, loaded from the exact repo or a short alias.
    These are the usual fine-tuning targets. Loads tokenizer.json from the
    Hub on first use; gated repos need `huggingface-cli login` + license
    acceptance.
  * Claude / Anthropic models → no official offline tokenizer exists. Exact
    counts require Anthropic's count_tokens API (network + key), which is out
    of scope for a local pre-flight, so we estimate.

When no name is given, the named tokenizer can't be loaded (missing package,
gated repo, network failure), or the family has no offline tokenizer, we fall
back to a length-based approximation (~4 chars/token). The check still runs —
it's just advisory rather than authoritative, which the rule reflects by
downgrading its findings from errors to warnings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


# Chars per token for the length-based fallback. 4 is the usual rule of thumb
# for English; it over-counts dense/code text slightly, which is the safe
# direction for a check that guards against silent truncation.
_CHARS_PER_TOKEN = 4


@dataclass
class TokenCounter:
    """A resolved token-counting strategy.

    `encode_len` maps a string to a token count. `exact` is True only when a
    real model tokenizer is in use; when False the count is a heuristic and
    callers should present findings as approximate/advisory. `note` is a
    one-line, user-facing explanation, always set when not exact.
    """
    encode_len: Callable[[str], int]
    method: str
    exact: bool
    note: Optional[str] = None


def _approx_counter(reason: str, hint: Optional[str] = None) -> TokenCounter:
    detail = reason if not hint else f"{reason} — {hint}"
    return TokenCounter(
        encode_len=lambda text: -(-len(text) // _CHARS_PER_TOKEN),  # ceil div
        method=f"approximate (~{_CHARS_PER_TOKEN} chars/token)",
        exact=False,
        note=detail,
    )


# ── OpenAI: tiktoken ──────────────────────────────────────────────────────
#
# Model-name prefix → tiktoken encoding. Order matters: more specific
# prefixes first so "gpt-4o" wins over "gpt-4".
_OPENAI_ENCODINGS: tuple[tuple[str, str], ...] = (
    ("gpt-4o", "o200k_base"),
    ("gpt-4.1", "o200k_base"),
    ("gpt-4.5", "o200k_base"),
    ("gpt-5", "o200k_base"),
    ("chatgpt-4o", "o200k_base"),
    ("o1", "o200k_base"),
    ("o3", "o200k_base"),
    ("o4-mini", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("gpt-35", "cl100k_base"),  # Azure-style name
    ("text-embedding-3", "cl100k_base"),
    ("text-embedding-ada", "cl100k_base"),
)


def _openai_encoding(low: str) -> Optional[str]:
    for prefix, enc in _OPENAI_ENCODINGS:
        if low.startswith(prefix):
            return enc
    return None


def _tiktoken_counter(encoding_name: str, model_label: str) -> Optional[TokenCounter]:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:  # noqa: BLE001 — network/ranks load failure → fall back
        return None
    # disallowed_special=() so encode never raises on literal control strings
    # like "<|endoftext|>" that can legitimately appear in training data.
    return TokenCounter(
        encode_len=lambda text: len(enc.encode(text, disallowed_special=())),
        method=f"tiktoken {encoding_name} ({model_label})",
        exact=True,
    )


# ── Open-weight models: HuggingFace tokenizers ────────────────────────────
#
# Friendly short names → an HF repo with a published tokenizer.json. Anything
# not in this map is treated as a literal repo id and passed straight to
# from_pretrained, so users can always name the exact repo. Gated repos
# (Llama, Gemma) need `huggingface-cli login` + license acceptance; if the
# load fails we fall back to approximate counting with an explanatory note.
_HF_ALIASES: dict[str, str] = {
    "llama-3": "meta-llama/Meta-Llama-3-8B",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama-3.1": "meta-llama/Llama-3.1-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b",
    "gemma-2": "google/gemma-2-9b",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
}


def _hf_counter(repo: str) -> tuple[Optional[TokenCounter], Optional[str]]:
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        return None, ("tokenizers package not installed "
                      "(pip install 'parallelogram[tokenizer]')")
    try:
        tok = Tokenizer.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001 — surface any HF load failure
        return None, f"failed to load {repo!r}: {e}"
    return TokenCounter(
        encode_len=lambda text: len(tok.encode(text).ids),
        method=f"HuggingFace tokenizer ({repo})",
        exact=True,
    ), None


def resolve_counter(name: Optional[str]) -> TokenCounter:
    """Pick the best available token counter for `name`.

    Never raises and never returns None: when nothing model-specific is
    available it returns the length-based approximate counter, so the
    context-window check can always run.
    """
    if name is None or not name.strip():
        return _approx_counter(
            "no tokenizer specified",
            "pass --tokenizer <model> (e.g. gpt-4o, Qwen/Qwen2.5-7B) for an exact count",
        )

    raw = name.strip()
    low = raw.lower()

    # Claude / Anthropic — no official offline tokenizer.
    if low.startswith(("claude", "anthropic")):
        return _approx_counter(
            f"no offline tokenizer available for {raw!r}",
            "Claude token counts require Anthropic's count_tokens API; using a length estimate",
        )

    # OpenAI families — tiktoken.
    enc = _openai_encoding(low)
    if enc is not None:
        counter = _tiktoken_counter(enc, raw)
        if counter is not None:
            return counter
        return _approx_counter(
            f"tiktoken unavailable, can't tokenize {raw!r} exactly",
            "pip install tiktoken (or 'parallelogram[tokenizer]') for exact OpenAI counts",
        )

    # Everything else: treat as an HF repo, via alias or literal id.
    repo = _HF_ALIASES.get(low, raw)
    counter, err = _hf_counter(repo)
    if counter is not None:
        return counter
    return _approx_counter(
        err or f"could not load tokenizer for {raw!r}",
        "using a length estimate; pass a loadable tokenizer for an exact count",
    )
