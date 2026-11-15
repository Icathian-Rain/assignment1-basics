from cs336_basics.bpe import train_bpe


if __name__ == '__main__':
    # train_bpe('./data/TinyStoriesV2-GPT4-sample.txt', 1000, ['<|endoftext|>'])
    # 1 + 256 + 6 = 263
    train_bpe('./data/sample.txt', 263, ['<|endoftext|>'])