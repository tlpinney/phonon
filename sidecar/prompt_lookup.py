"""Prompt-lookup speculative decoding for the prompted correction model (Gemma 4 E2B on MLX).

A correction is mostly the transcript copied with small edits, so the transcript itself is a good
draft: stream_draft.DraftProposer follows the model's output through the request's tokens and
proposes the tokens that come next there. Each forward pass checks [last token] + draft and keeps
the longest prefix the model agrees with, plus the model's own next token, so the output is the
plain greedy output whatever the draft says (up to the floating point difference between a
batched and a one-token forward).

Rejected draft tokens are trimmed back out of the cache, which mlx-lm's RotatingKVCache cannot do
once it has rotated (Gemma 4's sliding-window layers, window 512, rotate inside the system
prompt). Gemma 4 passes its window to the attention mask itself, so a plain KVCache per layer
with that windowed mask computes the same attention and trims; it keeps every position, which at
dictation lengths is a few MB.
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

try:  # imported as sidecar.prompt_lookup (tests) or as a top-level module (the sidecar)
    from .stream_draft import AdaptiveProposer
except ImportError:
    from stream_draft import AdaptiveProposer

# Draft lengths per pass (stream_draft.AdaptiveProposer): 3 after a rejection, 31 after a draft
# accepted in full. On a base M3, Gemma 4 E2B 4-bit passes of 1, 4 and 32 tokens took 20, 25 and
# 71 ms, and passes of 8 and 16 were nearly as dear as 32.
LADDER = (3, 31)
DRAFT_TOKENS = max(LADDER)

# Transcripts shorter than this decode token by token: their output is a few passes either way,
# and on the measured inputs drafts there saved nothing.
MIN_SOURCE_TOKENS = 64

# Models whose sliding-window layers give the attention mask their window themselves, so a
# plain KVCache in place of their RotatingKVCache is exact.
WINDOW_IN_MASK_MODEL_TYPES = {"gemma4", "gemma4_text"}


def supported(model) -> bool:
    """Whether every layer cache can be a trimmable KVCache without changing attention."""
    try:
        caches = model.make_cache()
    except Exception:  # noqa: BLE001 - no make_cache: unknown layout, do not guess
        return False
    if not caches:
        return False
    if all(type(c) is KVCache for c in caches):
        return True
    rotating_ok = getattr(model, "model_type", "") in WINDOW_IN_MASK_MODEL_TYPES
    return rotating_ok and all(
        type(c) is KVCache or (type(c) is RotatingKVCache and c.keep == 0) for c in caches
    )


class WindowKVCache(KVCache):
    """A trimmable KVCache for a sliding-window layer. A one-token step attends to the last
    `window` positions, so it gets just those and no mask, as from RotatingKVCache; a longer
    pass gets every position and the model's windowed mask."""

    def __init__(self, window: int):
        super().__init__()
        self.window = window

    def update_and_fetch(self, keys, values):
        k, v = super().update_and_fetch(keys, values)
        if keys.shape[2] == 1 and self.offset > self.window:
            return k[..., -self.window :, :], v[..., -self.window :, :]
        return k, v

    def make_mask(self, N, window_size=None, return_array=False):
        if N == 1 and window_size == self.window:
            return None
        return super().make_mask(N, window_size=window_size, return_array=return_array)


def trimmable_cache(model) -> list:
    """The model's cache layout with a trimmable cache for every layer."""
    return [
        WindowKVCache(c.max_size) if type(c) is RotatingKVCache else KVCache()
        for c in model.make_cache()
    ]


def trim(cache: list, n: int) -> None:
    if n:
        for layer in cache:
            layer.trim(n)


def decode(
    model,
    prompt: list[int],
    cache: list,
    source: list[int],
    max_tokens: int,
    eos: set[int],
    draft_tokens: int = DRAFT_TOKENS,
) -> dict:
    """Greedy-decode `prompt` (the tokens after whatever `cache` already holds) with drafts from
    `source`. Returns the output tokens (no end-of-sequence token), the number of forward passes
    after the prefill, and the time to the first token."""
    started = time.perf_counter()
    # As mlx-lm's generate_step: the prompt but its last token fills the cache without the
    # vocabulary projection (262,144 rows for Gemma), then the last token alone gives the logits.
    if len(prompt) > 1:
        model(mx.array(prompt[:-1])[None], cache=cache)
        mx.eval([layer.state for layer in cache])
    logits = model(mx.array(prompt[-1:])[None], cache=cache)
    y = mx.argmax(logits[0, -1]).item()
    ttft = time.perf_counter() - started
    propose = AdaptiveProposer(
        [source], primary=False, cap_lookup=draft_tokens, ladder=LADDER, substitute=False
    )
    out: list[int] = []
    passes = 0
    while y not in eos and len(out) < max_tokens:
        out.append(y)
        draft = propose(out, min(draft_tokens, max_tokens - len(out)))
        logits = model(mx.array([y] + draft)[None], cache=cache)
        passes += 1
        pred = mx.argmax(logits[0], axis=-1).tolist()
        accepted = 0
        while (
            accepted < len(draft)
            and draft[accepted] == pred[accepted]
            and draft[accepted] not in eos
        ):
            accepted += 1
        out.extend(draft[:accepted])
        trim(cache, len(draft) - accepted)
        y = pred[accepted]
    return {"tokens": out[:max_tokens], "passes": passes, "ttft": ttft}
