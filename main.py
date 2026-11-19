import json
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe

def train_tiny_stories():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = Path("output/tinystories")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )

    # 保存词表，使用 hex 保证任意 bytes 可无损序列化
    vocab_json = {
        str(token_id): token.hex()
        for token_id, token in vocab.items()
    }

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # 保存 BPE 合并规则，保留原始顺序
    merges_json = [
        [a.hex(), b.hex()]
        for a, b in merges
    ]

    with open(output_dir / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Merge rules: {len(merges)}")
    print(f"Training time: {elapsed:.2f}s")
    print(f"Saved to: {output_dir}")

if __name__ == '__main__':
    # train_bpe('./data/TinyStoriesV2-GPT4-sample.txt', 1000, ['<|endoftext|>'])
    # 1 + 256 + 6 = 263
    # train_bpe('./data/sample.txt', 263, ['<|endoftext|>'])
    train_tiny_stories()