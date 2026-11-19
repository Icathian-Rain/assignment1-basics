import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from icecream import ic
from multiprocessing import Pool

from .utils import Timer

ic.configureOutput(includeContext=True)
ic.disable()

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    # 1 删除specail token
    pattern = "|".join(re.escape(token) for token in special_tokens)
    chunks = re.split(pattern, chunk) if pattern else [chunk]

    # 2 进行pre-tokenization, 划分pre token, 统计每个pre token的出现次数
    pretoken_freq = Counter()

    for chunk in chunks:
        for match in PAT.finditer(chunk):
            token = match.group()
            pretoken_freq[token.encode("utf-8")] += 1
    return pretoken_freq


def merge(tokens, best_pair, merged_token):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == best_pair[0]
            and tokens[i + 1] == best_pair[1]
        ):
            new_tokens.append(merged_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


from .utils import Timer


def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    num_processes=1,
    profile_time=False
):
    total_timer = Timer("Total Training", enabled=profile_time).start()

    # 1. 初始化词表
    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    merges = []

    # 2. 文件分块
    with Timer("Chunk Boundaries", enabled=profile_time):
        num_processes = min(num_processes, os.cpu_count() or 1)

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if start < end
        ]

    # 3. 多进程预分词
    with Timer("Pretokenization", enabled=profile_time):
        pretoken_freq = Counter()

        with Pool(processes=num_processes) as pool:
            for freq in pool.imap_unordered(process_chunk, tasks):
                pretoken_freq.update(freq)

    # 4. 初始化 Pair 统计
    with Timer("Pair Initialization", enabled=profile_time):
        word_freq = {}
        pair2word = defaultdict(set)
        pair_freq = Counter()

        for word_id, (word, freq) in enumerate(pretoken_freq.items()):
            tokens = tuple(bytes([b]) for b in word)
            word_freq[word_id] = (tokens, freq)

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair2word[pair].add(word_id)
                pair_freq[pair] += freq

    # 5. BPE Merge
    merge_timer = Timer("BPE Merge", enabled=profile_time).start()

    find_timer = Timer("Find Best Pair", auto_print=False, enabled=profile_time)
    update_timer = Timer("Update Pairs", auto_print=False, enabled=profile_time)
    while len(vocab) < vocab_size:
        if not pair_freq:
            break

        # 5.1 寻找best_pair
        # TODO: 改用堆实现
        find_timer.start()
        best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        find_timer.stop()

        update_timer.start()
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        affected_word_ids = pair2word[best_pair].copy()
        changed_pairs = set()

        for word_id in affected_word_ids:
            tokens, freq = word_freq[word_id]

            # 5.2 删除受影响的token pair
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freq[pair] -= freq
                pair2word[pair].discard(word_id)
                changed_pairs.add(pair)

            # 5.3 merge
            new_tokens = merge(tokens, best_pair, merged_token)

            # 5.4 添加新的token pair
            for i in range(len(new_tokens) - 1):
                pair = (new_tokens[i], new_tokens[i + 1])
                pair_freq[pair] += freq
                pair2word[pair].add(word_id)
                changed_pairs.add(pair)

            word_freq[word_id] = (new_tokens, freq)

        # 5.5 删除为0的pair
        for pair in changed_pairs:
            if pair_freq[pair] <= 0:
                pair_freq.pop(pair, None)
                pair2word.pop(pair, None)
        
        if profile_time and len(merges) % 1000 == 0:
            merge_timer.print(
                f"Merge {len(merges)}/{vocab_size - len(special_tokens) - 256}"
            )
        update_timer.stop()

    ic(vocab, merges)
    find_timer.print()
    update_timer.print()
    merge_timer.stop()
    total_timer.stop()

    return vocab, merges
