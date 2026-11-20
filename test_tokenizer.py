import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
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


def test(name: str):
    config = CONFIGS[name]
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=config.vocab_filepath,
        merges_filepath=config.merges_filepath,
        special_tokens=config.special_tokens,
    )

    text = Path(config.input_path).read_text()
    data_size = len(text.encode('utf-8'))

    timer = Timer("Encode")
    timer.start()
    token_ids = tokenizer.encode(text)
    timer.stop()

    ic(len(token_ids))
    output_dir = Path("output") / name
    save_tokens(token_ids, output_dir)
    

    # 计算压缩比
    # ic(get_compression_ratio(text, token_ids))

    # throughput = data_size / timer.elapsed / 1024**2
    # estimated_hours = 825 * 1000 / throughput / 3600
    # print(f"Throughput: {throughput:.2f} MiB/s")
    # print(f"Tokens/s: {len(token_ids) / timer.elapsed:.0f}")
    # print(f"Estimated time for 825 GB Pile: {estimated_hours:.2f} hours")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIGS, nargs="?", default="tinystories")
    args = parser.parse_args()

    test(args.dataset)
