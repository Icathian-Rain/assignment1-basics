import json
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.utils import Timer, timeit, MemoryMonitor


def test():
    # train_bpe('./data/TinyStoriesV2-GPT4-sample.txt', 1000, ['<|endoftext|>'])
    # 1 + 256 + 6 = 263
    train_bpe("./data/sample.txt", 263, ["<|endoftext|>"])


def train_tiny_stories(num_processes=8, profile_time=False, profile_memory=False):
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = Path("output/tinystories")
    output_dir.mkdir(parents=True, exist_ok=True)

    if profile_memory:
        with MemoryMonitor(interval=0.05):
            vocab, merges = train_bpe(
                input_path=input_path,
                vocab_size=10000,
                special_tokens=["<|endoftext|>"],
                num_processes=num_processes,
                profile_time=profile_time,
            )
    else:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
            num_processes=num_processes,
            profile_time=profile_time,
        )

    # 寻找bytes最长的条目
    max_len = 0
    max_token = b""
    for token_id, token in vocab.items():
        if len(token) > max_len:
            max_len = len(token)
            max_token = token
    print(f"max Len: {max_len}")
    print(max_token)

    # 保存词表，使用 hex 保证任意 bytes 可无损序列化
    vocab_json = {str(token_id): token.hex() for token_id, token in vocab.items()}

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # 保存 BPE 合并规则，保留原始顺序
    merges_json = [[a.hex(), b.hex()] for a, b in merges]

    with open(output_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Merge rules: {len(merges)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    # test()
    num_processes = 8
    train_tiny_stories(
        num_processes=num_processes, profile_memory=False, profile_time=False
    )
