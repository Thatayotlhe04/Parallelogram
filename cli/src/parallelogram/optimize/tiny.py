"""Build a very small local causal LM for CPU validation.

`parallelogram optimize` never downloads a model unless asked, and CI has no
GPU, so the integration experiment and the tests need a model that can be
created from nothing in seconds. This writes a character-level GPT-2 of a few
tens of thousands of parameters, with a fast tokenizer and a chat template,
as an ordinary Hugging Face model directory.

It is a test fixture, not a model anyone should draw conclusions about:
results measured on it are pilots by construction.
"""
from __future__ import annotations

import random
import string
from pathlib import Path
from typing import Sequence

from .backend import require_ml

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
_SPECIALS = ["<pad>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_CHARS = sorted(set(string.ascii_lowercase + string.ascii_uppercase + string.digits + " :,.+-=?!'\n"))


def build_tiny_model(path: str | Path, *, seed: int = 0, n_embd: int = 64, n_layer: int = 2, n_head: int = 2,
                     n_positions: int = 128, pretrain: Sequence[Sequence[dict]] = (),
                     pretrain_steps: int = 0, batch_size: int = 32, learning_rate: float = 3e-3) -> Path:
    """Create the model directory at `path` and return it.

    With `pretrain` conversations and `pretrain_steps`, all weights are first
    trained as a plain language model on the rendered conversations, so that
    later LoRA runs adapt a model that already knows the chat format instead
    of random weights. Deterministic for a given seed and torch build.
    """
    ml = require_ml()
    torch = ml.torch
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(_SPECIALS + _CHARS)}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    core.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="<pad>", unk_token="<unk>", eos_token="<|end|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"])
    tokenizer.chat_template = CHAT_TEMPLATE

    torch.manual_seed(seed)
    config = GPT2Config(vocab_size=len(vocab), n_positions=n_positions, n_embd=n_embd, n_layer=n_layer,
                        n_head=n_head, bos_token_id=vocab["<|end|>"], eos_token_id=vocab["<|end|>"],
                        pad_token_id=vocab["<pad>"], resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(config)

    if pretrain and pretrain_steps > 0:
        rows = []
        for messages in pretrain:
            text = tokenizer.apply_chat_template(list(messages), tokenize=False)
            rows.append(tokenizer(text, add_special_tokens=False)["input_ids"][:n_positions])
        rng = random.Random(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()
        for step in range(pretrain_steps):
            batch = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
            width = max(len(r) for r in batch)
            ids = torch.full((len(batch), width), vocab["<pad>"], dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for i, r in enumerate(batch):
                ids[i, :len(r)] = torch.tensor(r)
                labels[i, :len(r)] = torch.tensor(r)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * (1 - step / pretrain_steps)
            loss = model(input_ids=ids, attention_mask=(labels != -100).long(), labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path
