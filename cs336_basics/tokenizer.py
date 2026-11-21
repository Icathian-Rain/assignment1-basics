from pathlib import Path
from typing import Iterable, Iterator
import regex as re
import json
from functools import lru_cache

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token2id = {token: idx for idx, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        if self.special_tokens:
            tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(t) for t in tokens)
            self.special_pattern = re.compile(pattern)
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        with Path(vocab_filepath).open("r", encoding="utf-8") as f:
            data = json.load(f)
            vocab = {int(i): bytes.fromhex(token) for i, token in data.items()}

        with Path(merges_filepath).open("r", encoding="utf-8") as f:
            data = json.load(f)
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data]

        return cls(vocab, merges, special_tokens)

    # 优化1: LRU缓存
    @lru_cache(maxsize=100_000)
    def encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        tokens = [bytes([b]) for b in pretoken.encode("utf-8")]
        while len(tokens) > 1:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break
            tokens[best_idx : best_idx + 2] = [tokens[best_idx] + tokens[best_idx + 1]]
        return tuple(self.token2id[token] for token in tokens)


    def _iter_encode_normal(self, text: str) -> Iterator[int]:
        for match in PAT.finditer(text):
            yield from self.encode_pretoken(match.group())

    def iter_encode(self, text: str) -> Iterator[int]:
        if not self.special_tokens:
            yield from self._iter_encode_normal(text)
            return

        start = 0
        assert self.special_pattern is not None
        for match in self.special_pattern.finditer(text):
            yield from self._iter_encode_normal(text[start:match.start()])
            yield self.token2id[match.group().encode("utf-8")]
            start = match.end()

        yield from self._iter_encode_normal(text[start:])

    def encode(self, text: str) -> list[int]:
        return list(self.iter_encode(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.iter_encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')
        return text