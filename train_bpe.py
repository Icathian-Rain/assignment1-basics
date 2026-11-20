import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.utils import MemoryMonitor


@dataclass
class TrainConfig:
    input_path: str
    vocab_size: int = 10000
    num_processes: int = 8
    profile_memory: bool = False
    profile_time: bool = False
    special_tokens: tuple[str, ...] = ("<|endoftext|>",)


CONFIGS = {
    "tinystories": TrainConfig(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        num_processes=8,
        profile_time=True,
    ),
    "owt": TrainConfig(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        num_processes=16,
        profile_memory=False,
        profile_time=True,
    ),
    "test": TrainConfig(
        input_path="data/sample.txt",
        vocab_size=263,
        num_processes=1,
    ),
}


def save_results(vocab, merges, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            {str(i): token.hex() for i, token in vocab.items()},
            f,
            indent=2,
        )

    with (output_dir / "merges.json").open("w", encoding="utf-8") as f:
        json.dump(
            [[a.hex(), b.hex()] for a, b in merges],
            f,
            indent=2,
        )


def train(name: str):
    config = CONFIGS[name]
    output_dir = Path("output") / name
    monitor = MemoryMonitor(interval=0.05) if config.profile_memory else nullcontext()

    with monitor:
        vocab, merges = train_bpe(
            input_path=config.input_path,
            vocab_size=config.vocab_size,
            special_tokens=list(config.special_tokens),
            num_processes=config.num_processes,
            profile_time=config.profile_time,
        )

    max_token = max(vocab.values(), key=len, default=b"")

    print(f"Max token length: {len(max_token)}")
    print(f"Max token: {max_token!r}")

    save_results(vocab, merges, output_dir)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Merge rules: {len(merges)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIGS, nargs="?", default="tinystories")
    args = parser.parse_args()

    train(args.dataset)