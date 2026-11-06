"""prompt_lookup.decode on a fake model: the output is the plain greedy output whatever the draft,
rejected draft tokens are trimmed out of the cache, and a draft the model agrees with saves
forward passes."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from sidecar import prompt_lookup

EOS = 99
VOCAB = 100


class FakeCache:
    """Holds the token sequence the fake model has seen; trim drops the newest tokens."""

    def __init__(self):
        self.tokens: list[int] = []
        self.state = mx.array([0])

    def trim(self, n):
        del self.tokens[len(self.tokens) - n :]
        return n


class FakeModel:
    """Greedy choice after a sequence: the user's words (20 <= t < 90) after the header token 2,
    in order, with every word above 60 rewritten to w - 30, then EOS."""

    def __init__(self):
        self.passes = 0

    @staticmethod
    def next_token(seq):
        if 2 not in seq:
            return 3
        head = seq.index(2)
        words = [t for t in seq[:head] if 20 <= t < 90]
        i = len(seq) - head - 1
        if i >= len(words):
            return EOS
        return words[i] - 30 if words[i] > 60 else words[i]

    def __call__(self, x, cache):
        self.passes += 1
        layer = cache[0]
        rows = []
        for t in x[0].tolist():
            layer.tokens.append(t)
            row = [0.0] * VOCAB
            row[self.next_token(layer.tokens)] = 1.0
            rows.append(row)
        return mx.array([rows])


def greedy(model, prompt, max_tokens):
    seq, out = list(prompt), []
    while len(out) < max_tokens:
        y = model.next_token(seq)
        if y == EOS:
            break
        out.append(y)
        seq.append(y)
    return out


WORDS = [21, 22, 23, 70, 24, 25, 26, 80, 27, 28, 29, 30, 31, 32, 33, 34]
PROMPT = [1] + WORDS + [2]


def decode(source, max_tokens=64, draft_tokens=16):
    model = FakeModel()
    result = prompt_lookup.decode(
        model, PROMPT, [FakeCache()], source, max_tokens, {EOS}, draft_tokens
    )
    return model, result


def test_output_is_the_greedy_output_with_the_transcript_as_draft():
    model, result = decode(WORDS)
    assert result["tokens"] == greedy(FakeModel(), PROMPT, 64)
    # two rewritten words: far fewer passes than one per token
    assert result["passes"] < len(result["tokens"]) // 2


def test_a_useless_draft_still_gives_the_greedy_output():
    _, result = decode([5, 6, 7, 8])
    assert result["tokens"] == greedy(FakeModel(), PROMPT, 64)
    assert result["passes"] == len(result["tokens"])


def test_rejected_draft_tokens_are_trimmed_from_the_cache():
    model = FakeModel()
    cache = [FakeCache()]
    result = prompt_lookup.decode(model, PROMPT, cache, WORDS, 64, {EOS})
    # the cache holds the prompt and the accepted output, plus the end-of-sequence check
    assert cache[0].tokens[: len(PROMPT) + len(result["tokens"])] == PROMPT + result["tokens"]
    assert len(cache[0].tokens) == len(PROMPT) + len(result["tokens"])


def test_budget_caps_the_output():
    _, result = decode(WORDS, max_tokens=5)
    assert result["tokens"] == greedy(FakeModel(), PROMPT, 5)


def test_supported_layouts():
    class Plain:
        def make_cache(self):
            return [KVCache(), KVCache()]

    class Gemma:
        model_type = "gemma4"

        def make_cache(self):
            return [RotatingKVCache(max_size=512, keep=0), KVCache()]

    class OtherSliding:
        model_type = "mistral"

        def make_cache(self):
            return [RotatingKVCache(max_size=512, keep=0)]

    class NoCache:
        pass

    assert prompt_lookup.supported(Plain())
    assert prompt_lookup.supported(Gemma())
    assert not prompt_lookup.supported(OtherSliding())
    assert not prompt_lookup.supported(NoCache())
    caches = prompt_lookup.trimmable_cache(Gemma())
    assert [type(c) for c in caches] == [prompt_lookup.WindowKVCache, KVCache]
    assert caches[0].window == 512
