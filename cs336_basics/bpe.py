import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from icecream import ic

ic.configureOutput(includeContext=True)
ic.disable()
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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始化

    vocab: dict[int, bytes] = dict()
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    merges: list[tuple[bytes, bytes]] = list()

    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # 1 删除specail token
            pattern = "|".join(re.escape(token) for token in special_tokens)
            chunks = re.split(pattern, chunk)

            # 2 进行pre-tokenization, 划分pre token, 统计每个pre token的出现次数
            pretoken_freq = Counter()
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for chunk in chunks:
                for match in re.finditer(PAT, chunk):
                    token = match.group()
                    pretoken_freq[token.encode("utf-8")] += 1
            ic(pretoken_freq)

            # 3. word_freq, 初始化按单个字符token分组的情况
            word_freq = {}
            # 优化1: 反向查找, 看哪些word需要修改
            pair2word = defaultdict(set)
            # 优化2: pair_freq 增删而不是每次重建
            pair_freq = Counter()
            for word_id, (word, freq) in enumerate(pretoken_freq.items()):
                tokens = tuple(bytes([b]) for b in word)
                word_freq[word_id] = (tokens, freq)
                for i in range(len(tokens)-1):
                    pair = (tokens[i], tokens[i+1])
                    pair2word[pair].add(word_id)
                    pair_freq[pair] += freq

            ic(word_freq)

            # 4. BPE merge
            while len(vocab) < vocab_size:
                ic(pair_freq)
                if not pair_freq:
                    break

                # 4. 寻找best_pair
                best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
                ic(best_pair)

                # 5. 更新vocab, merges
                merged_token = best_pair[0] + best_pair[1]
                vocab[len(vocab)] = merged_token
                merges.append(best_pair)

                # 6. 更新受影响的word和pair
                affected_word_ids = pair2word[best_pair].copy()
                changed_pairs = set()
                for word_id in affected_word_ids:
                    tokens, freq = word_freq[word_id]
                    # 6.1 删除旧pair
                    for i in range(len(tokens)-1):
                        pair = (tokens[i], tokens[i+1])
                        pair_freq[pair] -= freq
                        pair2word[pair].discard(word_id)
                        changed_pairs.add(pair)

                    # 6.2 执行merge
                    new_tokens = []
                    i = 0
                    while i<len(tokens):
                        if(i+1 < len(tokens) and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]):
                            new_tokens.append(merged_token)
                            i+=2
                        else:
                            new_tokens.append(tokens[i])
                            i+=1

                    # 6.3 更新pair的贡献与map
                    for i in range(len(new_tokens)-1):
                        pair = (new_tokens[i], new_tokens[i+1])
                        pair_freq[pair]+=freq
                        pair2word[pair].add(word_id)
                        changed_pairs.add(pair)

                    # 6.4 更新word
                    word_freq[word_id] = (new_tokens, freq)
                for pair in changed_pairs:
                    if pair_freq[pair] <= 0:
                        pair_freq.pop(pair, None)
                        pair2word.pop(pair, None)

    ic(vocab, merges)
    return vocab, merges
