"""Context window rule.

Tokenizes each record using the user-supplied HuggingFace tokenizer and flags
records that exceed max_seq_len. This catches the silent-truncation failure
mode in TRL and Axolotl, where samples over the context window are quietly
chopped — typically severing the assistant response and turning the example
into noise the model still trains on.

The rule self-disables if no tokenizer is supplied. Tokenizer load is lazy
so users who pass --disable context-window pay no startup cost.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity


@registry.register
class ContextWindowRule(Rule):
    id = "context-window"
    description = "Total token count must not exceed max_seq_len"
    severity = Severity.ERROR
    fixable = True  # truncation is mechanical; semantically lossy fixes are SLM-tier

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.tokenizer_name: Optional[str] = self.config.get("tokenizer")
        self.max_seq_len: int = int(self.config.get("max_seq_len", 4096))
        self._tokenizer = None
        self._disabled = self.tokenizer_name is None
        self._load_error: Optional[str] = None
        self._warned_disabled = False

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None or self._disabled:
            return
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError:
            self._load_error = (
                "tokenizers package not installed. "
                "Install with: pip install 'parallelogram[tokenizer]'"
            )
            self._disabled = True
            return
        try:
            self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:  # noqa: BLE001 — surface any HF load failure
            self._load_error = f"Failed to load tokenizer {self.tokenizer_name!r}: {e}"
            self._disabled = True

    def _count_tokens(self, role: str, content: str) -> int:
        """Approximate token count for a single message including chat
        template scaffolding. +4 covers role tags and turn separators —
        deliberately conservative so we over- not under-estimate.
        """
        if self._tokenizer is None:
            return 0
        try:
            return len(self._tokenizer.encode(f"<{role}>\n{content}").ids) + 4
        except Exception:  # noqa: BLE001
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
        if self._disabled and self._load_error and not self._warned_disabled:
            self._warned_disabled = True
            yield Issue(
                rule_id=self.id,
                severity=Severity.WARNING,
                line_no=None,
                message="Context window check disabled",
                detail=self._load_error,
            )
            return
        if self._disabled:
            return

        self._ensure_tokenizer()
        if self._tokenizer is None:
            return
        if not isinstance(record, dict):
            return
        messages = record.get("messages")
        if not isinstance(messages, list):
            return

        total = self._record_total(messages)

        if total > self.max_seq_len:
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"Record exceeds max_seq_len: ~{total} > {self.max_seq_len} tokens",
                detail="Will be silently truncated by TRL/Axolotl, likely severing the assistant response",
                fixable=True,
                context={"approx_token_count": total, "max_seq_len": self.max_seq_len},
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
        self._ensure_tokenizer()
        if self._tokenizer is None or self._disabled:
            return record

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
