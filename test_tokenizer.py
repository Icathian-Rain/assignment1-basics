import argparse
from itertools import islice
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from icecream import ic
import numpy as np

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.utils import MemoryMonitor, Timer


@dataclass
class TestConfig:
    input_path: str
    vocab_filepath: str
    merges_filepath: str
    special_tokens: tuple[str, ...] = ("<|endoftext|>",)


CONFIGS = {
    "tinystories_train": TestConfig(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_filepath="output/tinystories/vocab.json",
        merges_filepath="output/tinystories/merges.json",
    ),
    "tinystories_valid": TestConfig(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_filepath="output/tinystories/vocab.json",
        merges_filepath="output/tinystories/merges.json",
    ),
    "owt": TestConfig(
        input_path="data/owt_train.txt",
        vocab_filepath="output/tinystories/vocab.json",
        merges_filepath="output/tinystories/merges.json",
    ),
    "testtiny": TestConfig(
        input_path="data/TinyStoriesV2-GPT4-sample.txt",
        vocab_filepath="output/tinystories/vocab.json",
        merges_filepath="output/tinystories/merges.json",
    ),
}

def get_compression_ratio(string: str, indices: list[int]) -> float:  
    """Given `string` that has been tokenized into `indices`, return the number of UTF-8 bytes per token.."""
    num_bytes = len(bytes(string, encoding="utf-8"))  
    num_tokens = len(indices)                       
    return num_bytes / num_tokens

def save_tokens(token_ids, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(token_ids, dtype=np.uint16)
    tokens.tofile(output_dir/"tokens.bin")

def iter_documents(
    filepath: str,
    separator: str = "<|endoftext|>",
    chunk_size: int = 1024 * 1024,
) -> Iterator[str]:
    buffer = ""

    with open(filepath, "r", encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            buffer += chunk

            while separator in buffer:
                document, buffer = buffer.split(separator, 1)
                if document:
                    yield document
                yield separator

        if buffer:
            yield buffer

def save_token_ids(
    token_ids: Iterable[int],
    output_path: str,
    batch_size: int = 100_000,
) -> int:
    iterator = iter(token_ids)
    total = 0

    with open(output_path, "wb") as f:
        while True:
            batch = np.fromiter(
                islice(iterator, batch_size),
                dtype=np.dtype("<u2"),
            )

            if batch.size == 0:
                break

            batch.tofile(f)
            total += batch.size

    return total

from time import perf_counter

def test(name: str):
    config = CONFIGS[name]
    input_path = Path(config.input_path)
    output_path = Path("output") / name / "tokens.bin"

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=config.vocab_filepath,
        merges_filepath=config.merges_filepath,
        special_tokens=config.special_tokens,
    )

    if any(not 0 <= idx <= 65535 for idx in tokenizer.vocab):
        raise ValueError("Vocabulary contains token IDs outside the uint16 range")

    input_bytes = input_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    documents = iter_documents(input_path) # type: ignore
    token_ids = tokenizer.encode_iterable(documents)
    total = save_token_ids(token_ids, output_path) # type: ignore
    elapsed = perf_counter() - start

    bytes_per_second = input_bytes / elapsed
    print(f"Time: {elapsed:.4f} s")
    print(f"Tokens: {total:,}")
    print(f"Tokens/s: {total / elapsed:,.0f}")
    print(f"Throughput: {bytes_per_second / 2**20:.2f} MiB/s")

    if total:
        print(f"Compression ratio: {input_bytes / total:.4f} bytes/token")
    if input_bytes:
        print(f"Estimated time for 825 GB Pile: {825e9 / bytes_per_second / 3600:.2f} hours")

    ic(tokenizer.encode_pretoken.cache_info())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIGS, nargs="?", default="tinystories")
    args = parser.parse_args()

    test(args.dataset)
