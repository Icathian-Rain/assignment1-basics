from pathlib import Path
from typing import Iterable, Iterator
import regex as re
import json

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
        self.special_tokens = special_tokens
        self.token2id = {token: idx for idx, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        with Path(vocab_filepath).open("r", encoding="utf-8") as f:
            data = json.load(f)
            vocab = {int(i): bytes.fromhex(token) for i, token in data.items()}

        with Path(merges_filepath).open("r", encoding="utf-8") as f:
            data = json.load(f)
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data]

        return cls(vocab, merges, special_tokens)

    def encode_pretoken(self, pretoken: str) -> list[int]:
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
        return [self.token2id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_tokens:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        special_set = set(self.special_tokens) if self.special_tokens else set()
        for chunk in chunks:
            if chunk in special_set:
                ids.append(self.token2id[chunk.encode("utf-8")])
            else:
                for match in PAT.finditer(chunk):
                    ids.extend(self.encode_pretoken(match.group()))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join([self.vocab[id] for id in ids]).decode('utf-8', errors='replace')
        return text