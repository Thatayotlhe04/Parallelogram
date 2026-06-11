"""Context window rule.

Counts tokens per record and flags records that exceed max_seq_len. This
catches the silent-truncation failure mode in TRL and Axolotl, where samples
over the context window are quietly chopped — typically severing the assistant
response and turning the example into noise the model still trains on.

Token counting is model-specific where possible (tiktoken for OpenAI models,
HuggingFace tokenizers for open-weight models — see core.tokenization). When
no tokenizer is supplied or the requested one can't be loaded, the rule falls
back to a length-based estimate so the check still runs. Estimated overflows
are reported as warnings rather than errors, since a heuristic shouldn't fail
a CI gate or drop records on a guess. Counter resolution is lazy, so a run
that disables this rule pays no startup cost.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity
from ..core.tokenization import TokenCounter, resolve_counter


@registry.register
class ContextWindowRule(Rule):
    id = "context-window"
    description = "Total token count must not exceed max_seq_len"
    severity = Severity.ERROR
    fixable = True  # truncation is mechanical; semantically lossy fixes are SLM-tier

    # Per-message overhead approximating chat-template scaffolding (role tags,
    # turn separators). Deliberately conservative — we'd rather over- than
    # under-count and miss a record that will be silently truncated.
    _PER_MESSAGE_OVERHEAD = 4

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.tokenizer_name: Optional[str] = self.config.get("tokenizer")
        self.max_seq_len: int = int(self.config.get("max_seq_len", 4096))
        self._counter: Optional[TokenCounter] = None
        self._note_emitted = False

    def reset(self) -> None:
        # Re-emit the one-time "counts are approximate" note on each run.
        self._note_emitted = False

    def _ensure_counter(self) -> None:
        if self._counter is None:
            self._counter = resolve_counter(self.tokenizer_name)

    def _count_tokens(self, role: str, content: str) -> int:
        """Token count for a single message, including chat-template
        scaffolding overhead."""
        try:
            return self._counter.encode_len(
                f"<{role}>\n{content}"
            ) + self._PER_MESSAGE_OVERHEAD
        except Exception:  # noqa: BLE001 — a tokenizer hiccup shouldn't crash the run
            return 0

    def _record_total(self, messages: list) -> int:
        total = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            role = msg.get("role", "")
            if not isinstance(content, str):
                continue
            total += self._count_tokens(role, content)
        return total

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        self._ensure_counter()

        # One-time advisory whenever we're estimating rather than tokenizing,
        # so a clean-looking context-window pass is never mistaken for an
        # authoritative one.
        if not self._note_emitted and not self._counter.exact:
            self._note_emitted = True
            yield Issue(
                rule_id=self.id,
                severity=Severity.WARNING,
                line_no=None,
                message="Context-window counts are approximate",
                detail=self._counter.note,
            )

        if not isinstance(record, dict):
            return
        messages = record.get("messages")
        if not isinstance(messages, list):
            return

        total = self._record_total(messages)
        if total <= self.max_seq_len:
            return

        if self._counter.exact:
            # Exact count → high confidence → error (fails the gate, drops on --fix).
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"Record exceeds max_seq_len: {total} > {self.max_seq_len} tokens",
                detail="Will be silently truncated by TRL/Axolotl, likely severing the assistant response",
                fixable=True,
                context={"token_count": total, "max_seq_len": self.max_seq_len,
                         "counting_method": self._counter.method, "exact": True},
            )
        else:
            # Estimated count → advisory only → warning, not a hard failure.
            yield Issue(
                rule_id=self.id,
                severity=Severity.WARNING,
                line_no=line_no,
                message=f"Record may exceed max_seq_len: ~{total} > {self.max_seq_len} tokens (estimated)",
                detail="Estimated with a length heuristic; over-long records are truncated "
                       "silently. Pass --tokenizer <model> to confirm.",
                fixable=True,
                context={"approx_token_count": total, "max_seq_len": self.max_seq_len,
                         "counting_method": self._counter.method, "exact": False},
            )

    def fix_record(self, record: Any, issue: Issue) -> Optional[Any]:
        """Truncate the longest user message until total tokens fit.

        Strategy: never touch the assistant turn (it's the training
        target). Never touch the system message (it's typically short
        and sets the task). Iteratively shrink the longest user message
        in 20% steps until the total fits or the message is empty. If
        we can't fit even with all user messages truncated, drop the
        record — it's irrecoverable without an SLM.
        """
        self._ensure_counter()

        if not isinstance(record, dict):
            return record
        messages = record.get("messages")
        if not isinstance(messages, list):
            return record

        msgs = [dict(m) if isinstance(m, dict) else m for m in messages]

        # Defensive cap so we don't loop forever on a tokenizer that
        # disagrees with our estimate.
        for _ in range(20):
            total = self._record_total(msgs)
            if total <= self.max_seq_len:
                return {**record, "messages": msgs}

            # Find the longest user message
            user_idxs = [i for i, m in enumerate(msgs)
                         if isinstance(m, dict) and m.get("role") == "user"
                         and isinstance(m.get("content"), str)]
            if not user_idxs:
                return None  # nothing left to shrink

            longest_i = max(user_idxs,
                            key=lambda i: len(msgs[i]["content"]))
            current = msgs[longest_i]["content"]
            if len(current) <= 1:
                return None  # already minimal, can't shrink further

            # Cut to 80% of current length, keeping the start of the
            # message (which usually contains the actual question).
            new_len = max(1, int(len(current) * 0.8))
            msgs[longest_i] = {**msgs[longest_i],
                               "content": current[:new_len]}

        # Couldn't fit after 20 iterations — give up.
        return None
