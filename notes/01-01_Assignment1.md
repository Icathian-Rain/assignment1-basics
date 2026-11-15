# CS336 Assignment 1：中文答题模板

> **Building a Transformer LM · Version 26.0.3 · Spring 2026**  
> 来源：用户上传的 `cs336_assignment1_basics(1).pdf`，共 47 页。  
> 本模板按原文顺序收录 **38 道 Problem，共 109 分**。保留英文题干、原题标识、全部字母子问、分值、交付物、接口、测试命令及答题必需的原文公式和代码。页码指 PDF 的页面编号。  
> **本文件不含新增解答或实验结论。** 公式、伪代码、SGD 示例和 staff 参考结果均明确标为原文给定内容；“待填写”区域供自行完成。题目没有单独标明子问分值时，不自行分配。

## 使用说明与提交信息
- 原文题号使用 `Problem (标识)`；本模板的 01–38 仅为导航序号。
- 每题先列英文要求，再列必要的原文上下文，最后提供中文填写区域。带 (a)、(b) 等子问的题目有逐项答案位。
- 每题英文要求之后附有 `#### 中文翻译` 小节，仅供理解参考；如与原文有歧义，以英文原文为准。
- 编程题提供“实现思路 / 代码文件 / 测试结果”模板；实验题另留配置、曲线和日志位置。代码区域不填入实现。
- 公式使用 Markdown LaTeX；建议使用支持数学公式的阅读器。填写后，原文要求提交排版的 `writeup.pdf` 和 `code.zip`；本文件是写作底稿。
- 原文 §1：代码写在 `cs336_basics/*`；`adapters.py` 只用于调用自己的实现，不放实质逻辑；不要修改测试文件。用 `make_submission.sh` 构造提交包，排除不需提交的大数据和 checkpoint。
- 原文 §1 的从零实现约束：除 `torch.nn.Parameter`、`torch.nn` 容器类（如 `Module`、`ModuleList`、`Sequential`）和 `torch.optim.Optimizer` 基类外，不使用 `torch.nn`、`torch.nn.functional`、`torch.optim` 的现成组件。具体题目另明确要求使用 `torch.nn.init.trunc_normal_`，并在 SwiGLU 题中允许 `torch.sigmoid`，均在题目处保留。
姓名：Qihang Fan            
学号 / 标识：M202472149  
代码仓库 / 提交版本：https://github.com/Icathian-Rain/assignment1-basics  
实验环境：

| 项目 | 信息 |
|---|---|
| 操作系统 | Debian GNU/Linux trixie/sid |
| 内核 | 6.5.0-5-amd64 |
| 架构 | x86_64 |
| CPU | 13th Gen Intel(R) Core(TM) i9-13900KF |
| 逻辑核心数 | 32 |
| GPU | NVIDIA GeForce RTX 4080, 16376 MiB;NVIDIA GeForce RTX 4080, 16376 MiB |
| 内存 (已用/总量) | 24Gi / 125Gi |
| 根分区 (已用/总量) | 1.5T / 1.8T (87%) |
## 题目总览

| 序号 | 原题标识与英文标题 | 分值 | 原文页码 |
| --- | --- | ---: | --- |
| 01 | [unicode1 · Understanding Unicode](#problem-unicode1) | 1 | 3–4 |
| 02 | [unicode2 · Unicode Encodings](#problem-unicode2) | 3 | 4–5 |
| 03 | [train_bpe · BPE Tokenizer Training](#problem-train_bpe) | 15 | 9 |
| 04 | [train_bpe_tinystories · BPE Training on TinyStories](#problem-train_bpe_tinystories) | 2 | 9–10 |
| 05 | [train_bpe_expts_owt · BPE Training on OpenWebText](#problem-train_bpe_expts_owt) | 2 | 10 |
| 06 | [tokenizer · Implementing the tokenizer](#problem-tokenizer) | 15 | 11–12 |
| 07 | [tokenizer_experiments · Experiments with tokenizers](#problem-tokenizer_experiments) | 4 | 12–13 |
| 08 | [linear · Implementing the linear module](#problem-linear) | 1 | 18 |
| 09 | [embedding · Implement the embedding module](#problem-embedding) | 1 | 18–19 |
| 10 | [rmsnorm · Root Mean Square Layer Normalization](#problem-rmsnorm) | 1 | 20 |
| 11 | [positionwise_feedforward · Implement the position-wise feed-forward network](#problem-positionwise_feedforward) | 2 | 22 |
| 12 | [rope · Implement RoPE](#problem-rope) | 2 | 23 |
| 13 | [softmax · Implement softmax](#problem-softmax) | 1 | 24 |
| 14 | [scaled_dot_product_attention · Implement scaled dot-product attention](#problem-scaled_dot_product_attention) | 5 | 24–25 |
| 15 | [multihead_self_attention · Implement causal multi-head self-attention](#problem-multihead_self_attention) | 5 | 25–26 |
| 16 | [transformer_block · Implement the Transformer block](#problem-transformer_block) | 3 | 26 |
| 17 | [transformer_lm · Implementing the Transformer LM](#problem-transformer_lm) | 3 | 26–27 |
| 18 | [transformer_accounting · Transformer LM resource accounting](#problem-transformer_accounting) | 5 | 27–28 |
| 19 | [cross_entropy · Implement cross-entropy](#problem-cross_entropy) | 1 | 29 |
| 20 | [learning_rate_tuning · Tuning the learning rate](#problem-learning_rate_tuning) | 1 | 31 |
| 21 | [adamw · Implement AdamW](#problem-adamw) | 2 | 32 |
| 22 | [adamw_accounting · Resource accounting for training with AdamW](#problem-adamw_accounting) | 2 | 32–33 |
| 23 | [learning_rate_schedule · Implement cosine learning rate schedule with warmup](#problem-learning_rate_schedule) | 1 | 33–34 |
| 24 | [gradient_clipping · Implement gradient clipping](#problem-gradient_clipping) | 1 | 34 |
| 25 | [data_loading · Implement data loading](#problem-data_loading) | 2 | 34–35 |
| 26 | [checkpointing · Implement model checkpointing](#problem-checkpointing) | 1 | 36 |
| 27 | [training_together · Put it together](#problem-training_together) | 4 | 36–37 |
| 28 | [decoding · Decoding](#problem-decoding) | 3 | 38 |
| 29 | [experiment_log · Experiment logging](#problem-experiment_log) | 3 | 38–39 |
| 30 | [learning_rate · Tune the learning rate (2 B200 hrs)](#problem-learning_rate) | 3 | 40–41 |
| 31 | [batch_size_experiment · Batch size variations (1 B200 hr)](#problem-batch_size_experiment) | 1 | 41 |
| 32 | [generate · Generate text](#problem-generate) | 1 | 42 |
| 33 | [layer_norm_ablation · Remove RMSNorm and train (0.5 B200 hrs)](#problem-layer_norm_ablation) | 1 | 42 |
| 34 | [pre_norm_ablation · Implement post-norm and train (0.5 B200 hrs)](#problem-pre_norm_ablation) | 1 | 43 |
| 35 | [no_pos_emb · Implement NoPE (0.5 B200 hrs)](#problem-no_pos_emb) | 1 | 43 |
| 36 | [swiglu_ablation · SwiGLU vs. SiLU (0.5 B200 hrs)](#problem-swiglu_ablation) | 1 | 43–44 |
| 37 | [main_experiment · Experiment on OWT (2 B200 hrs)](#problem-main_experiment) | 2 | 45 |
| 38 | [leaderboard · Leaderboard (10 B200 hrs)](#problem-leaderboard) | 6 | 45–46 |

---

## 2. BPE 分词器
<a id="problem-unicode1"></a>
### 01. Unicode 基础（1 分）
**原题：Problem (unicode1): Understanding Unicode (1 point)**  
**来源：§2.1，PDF 第 3–4 页。**
#### 英文题目要求与交付物
**(a)** What Unicode character does chr(0) return?
**Deliverable:** A one-sentence response.
**(b)** How does this character’s string representation (`__repr__()`) differ from its printed representation?
**Deliverable:** A one-sentence response.
**(c)** What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**Deliverable:** A one-sentence response.

#### 中文翻译
**(a)** `chr(0)` 返回的是哪个 Unicode 字符？
**交付物：** 一句话回答。
**(b)** 该字符的字符串表示（`__repr__()`）与它的打印表示有何不同？
**交付物：** 一句话回答。
**(c)** 当这个字符出现在文本中时会发生什么？可以在 Python 解释器里试试下面的代码，看看结果是否符合预期：
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
**交付物：** 一句话回答。

#### 答题区域（待填写）
##### (a)
一句话答案：'', 空字符
##### (b)
一句话答案：字符串表示为"'\\x00'", 打印出来为空
##### (c)
一句话答案：
'this is a test\x00string'
this is a teststring
直接输出chr(0)时不可见,但该字符确实为\x00

- [x] 已按本题原文要求完成全部交付物。

#### 评审意见（2026-11-14）
**评分：0.5 / 1**
- **(a) 事实性错误**：`''` 是空字符串，`chr(0) == ''` 为 False。应写 NUL 字符（U+0000），避免与空字符串/空格混淆。
- **(b) 未说透区别**：应点明 `__repr__()` 给出转义形式 `'\x00'`（无歧义、可复制），print 给出字面字符（终端不可见），并解释为什么不可见。
- **(c) 超出句数**：交付物要求一句话，目前是三行输出加解释，需压缩成一句完整的话。

---

<a id="problem-unicode2"></a>
### 02. Unicode 编码（3 分）
**原题：Problem (unicode2): Unicode Encodings (3 points)**  
**来源：§2.2，PDF 第 4–5 页。**
#### 英文题目要求与交付物
**(a)** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
**Deliverable:** A one-to-two sentence response.
**(b)** Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**Deliverable:** An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect.
**(c)** Give a two-byte sequence that does not decode to any Unicode character(s).
**Deliverable:** An example, with a one-sentence explanation.

#### 中文翻译
**(a)** 有哪些原因让我们更倾向于在 UTF-8 编码的字节（而不是 UTF-16 或 UTF-32）上训练分词器？比较这些编码对各种输入字符串的输出可能会有帮助。
**交付物：** 一至两句话回答。

**(b)** 考虑下面这个（错误的）函数，其意图是将 UTF-8 字节串解码为 Unicode 字符串。这个函数为什么是错的？给出一个会产生错误结果的输入字节串示例。
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**交付物：** 一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，并附一句话解释该函数错误的原因。

**(c)** 给出一个不能解码为任何 Unicode 字符的两字节序列。
**交付物：** 一个示例，并附一句话解释。

#### 答题区域（待填写）
##### (a)
1–2 句话回答：
```
utf-8: b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
utf-16: b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
utf-32: b'\xff\xfe\x00\x00h\x00\x00\x00e\x00\x00\x00l\x00\x00\x00l\x00\x00\x00o\x00\x00\x00!\x00\x00\x00 \x00\x00\x00S0\x00\x00\x930\x00\x00k0\x00\x00a0\x00\x00o0\x00\x00!\x00\x00\x00'
```

utf-16和utf-32输出时会有大量\x00字节, 合并过程中会出现不必要的噪声,训练时会增加计算量, 而utf-8是变长编码, 只会在必要时使用多个字节表示一个字符, 因此更节省空间和计算量
##### (b)
输入字节串：你好

一句话解释：
```python
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
Cell In[22], line 1
----> 1 decode_utf8_bytes_to_str_wrong(wrong_bytes)

Cell In[19], line 2
      1 def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
----> 2     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```
bytes\[b\]将每个字节拆分出来, 单独做decode , 而有些字符是由多个字节组成的, 因此decode的时候会报错

##### (c)
两字节序列：0xDF 0xFF

一句话解释：因为二字节的utf-8编码, 第一个字节为110开头会被识别为二字节字符, 因此第二个字节必须以10 开头, 而0xFF以11开头

- [x] 已按本题原文要求完成全部交付物。

#### 评审意见（2026-11-14）
**评分：2 / 3**
- **(a) 满分**：对比数据 + `\x00` 噪声 + 变长编码的结论完整。可选加强：说明 `\x00` 会涌入 pair 统计浪费词表额度；提 UTF-8 对 ASCII 的向后兼容。
- **(b) 扣分**：① 题目要求 byte string，`你好` 是 str，应写编码后的 `b'\xe4\xbd\xa0\xe5\xa5\xbd'`（同时展示三个字节各自 decode 失败）；② 解释中"encode"应为"decode"（traceback 里就是 can't decode）。
- **(c) 满分**：`0xDF 0xFF` 验证过确实解码失败（`invalid continuation byte`），解释正确。注意上一版 `0xDF 0xBF` 实际可解码为 U+07FF，是错的，现已改对。

---

<a id="problem-train_bpe"></a>
### 03. 训练 BPE 分词器（15 分）
**原题：Problem (train_bpe): BPE Tokenizer Training (15 points)**  
**来源：§2.5，PDF 第 9 页。**
#### 英文题目要求与交付物
**Deliverable:** Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:
Input
input_path: str   Path to a text file with BPE tokenizer training data.
vocab_size: int    A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str]   A list of strings to add to the vocabulary. During training, treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.
Your BPE training function should return the resulting vocabulary and merges:
Output
vocab: dict[int, bytes]   The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]]    A list of BPE merges produced from training. Each list item is a tuple of bytes (`<token1>`, `<token2>`), representing that `<token1>` was merged with `<token2>`. The merges should be ordered by order of creation.
To test your BPE training function against our provided tests, you will first need to implement the test adapter at `adapters.run_train_bpe`. Then, run `uv run pytest tests/test_train_bpe.py`. Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider cppyy or nanobind) or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.

#### 中文翻译
**交付物：** 编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：
输入
`input_path: str`   BPE 分词器训练数据的文本文件路径。
`vocab_size: int`    定义最终词表最大大小的正整数（包括初始字节词表、由合并产生的词表项，以及任何特殊 token）。
`special_tokens: list[str]`   要加入词表的字符串列表。训练期间，将它们视为阻止其跨度内合并的硬边界，但在计算合并统计时不要包含它们。
你的 BPE 训练函数应返回最终的词表和合并规则：
输出
`vocab: dict[int, bytes]`   分词器词表，从 int（词表中的 token ID）到 bytes（token 字节）的映射。
`merges: list[tuple[bytes, bytes]]`    训练产生的 BPE 合并规则列表。每个列表项是一个字节元组（`<token1>`、`<token2>`），表示 `<token1>` 与 `<token2>` 合并。合并规则按创建顺序排列。
要用我们提供的测试来测试你的 BPE 训练函数，首先需要实现 `adapters.run_train_bpe` 测试适配器。然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过全部测试。可选地（这可能需要大量时间投入），你可以用某种系统级语言实现训练方法的关键部分，例如 C++（考虑 cppyy 或 nanobind）或 Rust（使用 PyO3）。如果这样做，注意哪些操作需要复制、哪些直接读取 Python 内存，并确保留下构建说明，或确保仅用 pyproject.toml 即可构建。另请注意，GPT-2 正则在大多数正则引擎中支持不佳，在支持它的引擎中大多也太慢。我们已验证 Oniguruma 相当快且支持负向前瞻，但 Python 的 regex 包甚至更快。
#### 原文给定的公式、代码或关联条件
**原文相关约束（§2.4–2.5，PDF pp. 5–8）：** 初始词表包含 256 个单字节；不能跨 pre-token 边界合并。最高频 pair 并列时，选择字典序最大的 pair。特殊 token 加入词表，作为硬分界，但不参与 merge 计数；预分词前需按特殊 token 分割。原文要求使用 `regex` 包，并建议用 `finditer` 避免保存整份预分词列表。
原文给定的 GPT-2 预分词正则：
```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```
原文允许直接使用的分块示例：[pretokenization_example.py](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py)。分块边界应落在特殊 token 开始处；题目允许假设语料含 `<|endoftext|>`。
#### 答题区域（待填写）
**实现思路：**
1. 首先初始化词表与merges列表, 初始词表 [sepcial tokens] + [256] 
2. 目前暂时选择单进程操作, num_processes=1
3. 删除special token, sepcial token不进行BPE
4. 进行pre-tokenization, 划分每个pre token
5. 初始化构建word_freq, 记录word_id与内部token, 初始化反向查找与全局pair统计
6. BPE merge: 寻找best_pair然后更新受影响的word和pair, 达到词表上限即退出
<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| `cs336_basics/bpe.py` | `train_bpe(input_path, vocab_size, special_tokens)` | BPE 训练主函数；含 `find_chunk_boundaries` 分块辅助函数 |
| `tests/adapters.py` | `run_train_bpe` | 测试适配器，调用上述实现 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| `uv run pytest tests/test_train_bpe.py` | 3 passed in 0.99s（2026-11-14） | `test_train_bpe_speed` 在 1.5s 门限内通过；`test_train_bpe` 的 merges/vocab 与参考实现一致；`test_train_bpe_special_tokens` 通过 |
| `uv run python -m cProfile -s cumtime`（corpus.en, vocab 500） | `train_bpe` 累计耗时 0.63s | profile 定位到预分词阶段重复编译正则的瓶颈，已改为预编译 pattern 复用（修复前 speed 测试耗时 2.31s，未过门限） |

**说明 / 作答：**
后续将实现下述功能
1. 代码重构, 封装一些部分
2. 添加更多边界处理与异常机制
3. 实现多进程
4. 优化代码运行速度与内存存储, 如寻找best_pair可以使用优先队列
5. 部分代码使用c++编写来提高效率

<!-- 待填写 -->

- [x] 已按本题原文要求完成全部交付物。

#### 评审意见（2026-11-14，已根据增量更新版代码复评）
**评分：代码部分约 14 / 15（书面部分待补）**

已达标：
- 三个测试全过（1.09s），merges 与参考实现逐条一致——tie-breaking、special token 处理、pre-token 边界均验证正确
- 通过 profiling 定位并修复正则重编译瓶颈（2.31s → 0.99s），过程已记录在测试结果表
- 合并循环已实现增量更新：`pair2word` 反向索引 + `pair_freq` 只在初始化建一次，每轮仅遍历含 best_pair 的受影响 word，先删旧 pair 贡献再合并加回新贡献，计数归零的 pair 及时清理——与实现思路第 5–6 条一致（复评确认，此前误判为全量重算，已更正）

遗留事项：
1. **多进程未启用**：`num_processes = 1`，预分词仍为串行；TinyStories 规模下讲义 hint 建议 multiprocessing 预分词以进入 2 分钟门限。
2. **优先队列的坑**（后续计划第 4 条）：pair 计数动态变化，朴素堆不支持更新堆内元素，需 lazy deletion 或其他失效条目处理方案。
3. corpus.en 规模下增量更新优势未显现（0.64s，与全量版持平属正常），真正的收益要在 TinyStories 规模验证。

---

<a id="problem-train_bpe_tinystories"></a>
### 04. 在 TinyStories 上训练 BPE（2 分）
**原题：Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)**  
**来源：§2.5，PDF 第 9–10 页。**
#### 英文题目要求与交付物
**(a)** Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
**Resource requirements:** ≤ 30 minutes (no GPUs), ≤ 30 GB RAM
Hint You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:
**Hint fact (a):** The `<|endoftext|>` token delimits documents in the data files.
**Hint fact (b):** The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.
**Deliverable:** A one-to-two sentence response.
**(b)** Profile your code. What part of the tokenizer training process takes the most time?
**Deliverable:** A one-to-two sentence response.

#### 中文翻译
**(a)** 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词表大小为 10,000。确保将 TinyStories ` ` 特殊 token 加入词表。将得到的词表和合并规则序列化到磁盘以便后续检查。训练花了多少时间和内存？词表中最长的 token 是什么？它合理吗？
**资源要求：** ≤ 30 分钟（无 GPU），≤ 30 GB 内存
提示 使用多进程做预分词，再加上下面两个事实，你应该能让 BPE 训练在 2 分钟内完成：
**提示事实 (a)：** ` ` token 在数据文件中作为文档分隔符。
**提示事实 (b)：** ` ` token 在应用 BPE 合并之前作为特殊情况处理。
**交付物：** 一至两句话回答。
**(b)** 对你的代码做性能分析。分词器训练过程中哪部分最耗时？
**交付物：** 一至两句话回答。

#### 答题区域（待填写）
**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**
- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->
##### (a)
训练时间、内存、最长 token 及解释（1–2 句话）：

<!-- 待填写 -->

##### (b)
性能分析：最耗时的环节（1–2 句话）：

<!-- 待填写 -->

词表文件：<!-- 待填写 -->
Merges 文件：<!-- 待填写 -->
Profile 文件：<!-- 待填写 -->
- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-train_bpe_expts_owt"></a>
### 05. 在 OpenWebText 上训练 BPE（2 分）
**原题：Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)**  
**来源：§2.5，PDF 第 10 页。**
#### 英文题目要求与交付物
**(a)** Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
**Resource requirements:** ≤ 12 hours (no GPUs), ≤ 100 GB RAM
**Deliverable:** A one-to-two sentence response.
**(b)** Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
**Deliverable:** A one-to-two sentence response.

#### 中文翻译
**(a)** 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词表大小为 32,000。将得到的词表和合并规则序列化到磁盘以便后续检查。词表中最长的 token 是什么？它合理吗？
**资源要求：** ≤ 12 小时（无 GPU），≤ 100 GB 内存
**交付物：** 一至两句话回答。
**(b)** 比较和对比在 TinyStories 与 OpenWebText 上训练得到的分词器。
**交付物：** 一至两句话回答。

#### 答题区域（待填写）
**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**
- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->
##### (a)
最长 token 及解释（1–2 句话）：

<!-- 待填写 -->

##### (b)
TinyStories 与 OpenWebText 分词器对比（1–2 句话）：

<!-- 待填写 -->

词表文件：<!-- 待填写 -->
Merges 文件：<!-- 待填写 -->
- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-tokenizer"></a>
### 06. 实现分词器（15 分）
**原题：Problem (tokenizer): Implementing the tokenizer (15 points)**  
**来源：§2.6，PDF 第 11–12 页。**
#### 英文题目要求与交付物
**Deliverable:** Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface:
`def __init__(self, vocab, merges, special_tokens=None)` Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
special_tokens: list[str] | None = None
`def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:
vocab_filepath: str
merges_filepath: str
special_tokens: list[str] | None = None
`def encode(self, text: str) -> list[int]` Encode an input text into a sequence of token IDs.
`def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
`def decode(self, ids: list[int]) -> str` Decode a sequence of token IDs into text.
To test your Tokenizer against our provided tests, you will first need to implement the test adapter at `adapters.get_tokenizer`. Then, run `uv run pytest tests/test_tokenizer.py`. Your implementation should be able to pass all tests.

#### 中文翻译
**交付物：** 实现一个 Tokenizer 类，给定词表和合并规则列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果尚不在词表中则追加）。我们推荐以下接口：
`def __init__(self, vocab, merges, special_tokens=None)` 从给定词表、合并规则列表以及（可选的）特殊 token 列表构造分词器。该函数应接受以下参数：
`vocab: dict[int, bytes]`
`merges: list[tuple[bytes, bytes]]`
`special_tokens: list[str] | None = None`
`def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` 类方法，从序列化的词表和合并规则列表（与你 BPE 训练代码输出的格式相同）以及（可选的）特殊 token 列表构造并返回一个 Tokenizer。该方法还应接受以下参数：
`vocab_filepath: str`
`merges_filepath: str`
`special_tokens: list[str] | None = None`
`def encode(self, text: str) -> list[int]` 将输入文本编码为 token ID 序列。
`def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` 给定一个字符串的可迭代对象（如 Python 文件句柄），返回一个惰性产出 token ID 的生成器。这是对无法直接载入内存的大文件进行内存高效分词所必需的。
`def decode(self, ids: list[int]) -> str` 将 token ID 序列解码为文本。
要用我们提供的测试来测试你的 Tokenizer，首先需要实现 `adapters.get_tokenizer` 测试适配器。然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过全部测试。
#### 原文给定的公式、代码或关联条件
**原文相关约束（§2.6，PDF pp. 10–12）：** 编码时按训练时的先后顺序应用 merges，不跨 pre-token 边界合并；特殊 token 保持为单个 token。流式编码需要惰性生成 token，避免文件大小线性增长的内存占用，并保持与整体编码一致的分词结果。解码时先拼接 token 对应的 bytes，再执行 UTF-8 解码；无效字节用 U+FFFD 替换，原文明确给出 `errors='replace'`。
#### 答题区域（待填写）
**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-tokenizer_experiments"></a>
### 07. 分词器实验（4 分）
**原题：Problem (tokenizer_experiments): Experiments with tokenizers (4 points)**  
**来源：§2.7，PDF 第 12–13 页。**
#### 英文题目要求与交付物
**(a)** Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
**Deliverable:** A one-to-two sentence response.
**(b)** What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
**Deliverable:** A one-to-two sentence response.
**(c)** Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
**Deliverable:** A one-to-two sentence response.
**(d)** Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
**Deliverable:** A one-to-two sentence response.

#### 中文翻译
**(a)** 从 TinyStories 和 OpenWebText 各采样 10 篇文档。使用你之前训练的 TinyStories 与 OpenWebText 分词器（词表大小分别为 10K 和 32K），将这些采样文档编码为整数 ID。每个分词器的压缩率（bytes/token）是多少？
**交付物：** 一至两句话回答。
**(b)** 如果用你的 TinyStories 分词器对 OpenWebText 样本分词会发生什么？比较压缩率和/或定性地描述发生了什么。
**交付物：** 一至两句话回答。
**(c)** 估计你的分词器的吞吐量（如 bytes/second）。对 Pile 数据集（825GB 文本）分词需要多长时间？
**交付物：** 一至两句话回答。
**(d)** 用你的 TinyStories 和 OpenWebText 分词器，将各自的训练集与开发集编码为整数 token ID 序列。我们稍后用它训练语言模型。我们建议将 token ID 序列化为 dtype 为 uint16 的 NumPy 数组。为什么 uint16 是合适的选择？
**交付物：** 一至两句话回答。

#### 答题区域（待填写）
**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**
- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->
##### (a)
两个分词器各自的 bytes/token（1–2 句话）：

<!-- 待填写 -->

##### (b)
TinyStories 分词器编码 OpenWebText 的比较（1–2 句话）：

<!-- 待填写 -->

##### (c)
吞吐量、825 GB Pile 所需时间估计（1–2 句话）：

<!-- 待填写 -->

##### (d)
训练/开发集序列化位置，以及选择 uint16 的解释（1–2 句话）：

<!-- 待填写 -->

| 数据集 | 划分 | token 数 | dtype | 输出文件 |
| --- | --- | --- | --- | --- |
| TinyStories | train | 待填写 | 待填写 | 待填写 |
| TinyStories | dev | 待填写 | 待填写 | 待填写 |
| OpenWebText | train | 待填写 | 待填写 | 待填写 |
| OpenWebText | dev | 待填写 | 待填写 | 待填写 |

- [ ] 已按本题原文要求完成全部交付物。

---

## 3. Transformer 语言模型架构
<a id="problem-linear"></a>
### 08. 实现线性层（1 分）
**原题：Problem (linear): Implementing the linear module (1 point)**  
**来源：§3.3.2，PDF 第 18 页。**
#### 英文题目要求与交付物
**Deliverable:** Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear module, except for not having a bias argument or parameter. We recommend the following interface:
`def __init__(self, in_features, out_features, device=None, dtype=None)` Construct a linear transformation module. This function should accept the following parameters:
in_features: int    final dimension of the input
out_features: int    final dimension of the output
device: torch.device | None = None     Device to store the parameters on
dtype: torch.dtype | None = None     Data type of the parameters
`def forward(self, x: torch.Tensor) -> torch.Tensor` Apply the linear transformation to the input.
Make sure to:
- subclass nn.Module
- call the superclass constructor
- construct and store your parameter as $W$ (not $W^\top$ ), putting it in an nn.Parameter
- of course, don’t use nn.Linear or nn.functional.linear
For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to initialize the weights.
To test your Linear module, implement the test adapter at `adapters.run_linear`. The adapter should load the given weights into your Linear module. You can use Module.load_state_dict for this purpose. Then, run `uv run pytest -k test_linear`.

#### 中文翻译
**交付物：** 实现一个继承自 torch.nn.Module 的 Linear 类，执行线性变换。你的实现应遵循 PyTorch 内置 nn.Linear 模块的接口，但没有 bias 参数。我们推荐以下接口：
`def __init__(self, in_features, out_features, device=None, dtype=None)` 构造一个线性变换模块。该函数应接受以下参数：
`in_features: int`    输入的最后一维大小
`out_features: int`    输出的最后一维大小
`device: torch.device | None = None`     参数存储的设备
`dtype: torch.dtype | None = None`     参数的数据类型
`def forward(self, x: torch.Tensor) -> torch.Tensor` 对输入应用线性变换。
务必：
- 继承 nn.Module
- 调用父类构造函数
- 将参数构造并存储为 $W$（而不是 $W^\top$），放入 nn.Parameter
- 当然，不要使用 nn.Linear 或 nn.functional.linear
初始化时，使用上文给出的设置并用 `torch.nn.init.trunc_normal_` 初始化权重。
要测试你的 Linear 模块，实现测试适配器 `adapters.run_linear`。该适配器应将给定权重加载到你的 Linear 模块中，可以为此使用 `Module.load_state_dict`。然后运行 `uv run pytest -k test_linear`。
#### 原文给定的公式、代码或关联条件
**原文给定（§3.2.1、§3.3.1–3.3.2，PDF pp. 17–18）：**
$$y=Wx \tag{3}$$
数学记号使用列向量，$W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$；按 PyTorch 行向量排列时对应 $y=xW^\top$（式 1）。没有 bias。原文初始化：线性权重服从 $\mathcal N(0,\sigma^2)$，$\sigma^2=2/(d_{\mathrm{in}}+d_{\mathrm{out}})$，截断至 $[-3\sigma,3\sigma]$，使用 `torch.nn.init.trunc_normal_`。
#### 答题区域（待填写）
**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-embedding"></a>
### 09. 实现嵌入层（1 分）
**原题：Problem (embedding): Implement the embedding module (1 point)**  
**来源：§3.3.3，PDF 第 18–19 页。**
#### 英文题目要求与交付物
**Deliverable:** Implement the Embedding class that inherits from torch.nn.Module and performs an embedding lookup. Your implementation should follow the interface of PyTorch’s built-in nn.Embedding module. We recommend the following interface:
`def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)` Construct an embedding module. This function should accept the following parameters:
num_embeddings: int    Size of the vocabulary
embedding_dim: int    Dimension of the embedding vectors, i.e., $d_{\mathrm{model}}$
device: torch.device | None = None     Device to store the parameters on
dtype: torch.dtype | None = None     Data type of the parameters
`def forward(self, token_ids: torch.Tensor) -> torch.Tensor` Lookup the embedding vectors for the given token IDs.
Make sure to:
- subclass nn.Module
- call the superclass constructor
- initialize your embedding matrix as an nn.Parameter
- store the embedding matrix with the d_model being the final dimension
- of course, don’t use nn.Embedding or nn.functional.embedding
Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to initialize the weights.
To test your implementation, implement the test adapter at `adapters.run_embedding`. Then, run `uv run pytest -k test_embedding`.

#### 中文翻译
**交付物：** 实现一个继承自 torch.nn.Module 的 Embedding 类，执行嵌入查表。你的实现应遵循 PyTorch 内置 nn.Embedding 模块的接口。我们推荐以下接口：
`def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)` 构造一个嵌入模块。该函数应接受以下参数：
`num_embeddings: int` 词表大小
`embedding_dim: int` 嵌入向量的维度，即 $d_{\mathrm{model}}$
`device: torch.device | None = None`     参数存储的设备
`dtype: torch.dtype | None = None`     参数的数据类型
`def forward(self, token_ids: torch.Tensor) -> torch.Tensor` 查找给定 token ID 的嵌入向量。
务必：
- 继承 nn.Module
- 调用父类构造函数
- 将嵌入矩阵初始化为 nn.Parameter
- 存储的嵌入矩阵以 d_model 作为最后一维
- 当然，不要使用 nn.Embedding 或 nn.functional.embedding
同样，使用上文给出的设置做初始化，并用 `torch.nn.init.trunc_normal_` 初始化权重。
要测试你的实现，实现测试适配器 `adapters.run_embedding`。然后运行 `uv run pytest -k test_embedding`。
#### 原文给定的公式、代码或关联条件
**原文给定（§3.3.1、§3.3.3，PDF pp. 17–19）：** 嵌入矩阵形状为 `(vocab_size, d_model)`，输入 token IDs 的形状为 `(batch_size, sequence_length)`，输出为 `(batch_size, sequence_length, d_model)`。嵌入权重按 $\mathcal N(0,1)$ 初始化并截断至 $[-3,3]$，使用 `torch.nn.init.trunc_normal_`。
#### 答题区域（待填写）
**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-rmsnorm"></a>
### 10. 实现 RMSNorm（1 分）
**原题：Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)**  
**来源：§3.4.1，PDF 第 20 页。**
#### 英文题目要求与交付物
**Deliverable:** Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
`def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)` Construct the RMSNorm module. This function should accept the following parameters:
d_model: int   Hidden dimension of the model
eps: float = 1e-5    Epsilon value for numerical stability
device: torch.device | None = None      Device to store the parameters on
dtype: torch.dtype | None = None      Data type of the parameters
`def forward(self, x: torch.Tensor) -> torch.Tensor` Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
Note: Remember to upcast your input to torch.float32 before performing the normalization (and later downcast to the original dtype), as described above.
To test your implementation, implement the test adapter at `adapters.run_rmsnorm`. Then, run `uv run pytest -k test_rmsnorm`.

#### 中文翻译
**交付物：** 将 RMSNorm 实现为 torch.nn.Module。我们推荐以下接口：
`def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)` 构造 RMSNorm 模块。该函数应接受以下参数：
`d_model: int`   模型的隐藏维度
`eps: float = 1e-5`    用于数值稳定性的 epsilon 值
`device: torch.device | None = None`      参数存储的设备
`dtype: torch.dtype | None = None`     参数的数据类型
`def forward(self, x: torch.Tensor) -> torch.Tensor` 处理形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回相同形状的张量。
注意：如上文所述，记得在执行归一化前将输入上转为 torch.float32（之后下转回原始 dtype）。
要测试你的实现，实现测试适配器 `adapters.run_rmsnorm`。然后运行 `uv run pytest -k test_rmsnorm`。
#### 原文给定的公式、代码或关联条件
**原文给定公式与框架（PDF pp. 19–20）：
$$\operatorname{RMSNorm}(a_i)=\frac{a_i}{\operatorname{RMS}(a)}g_i,\qquad
\operatorname{RMS}(a)=\sqrt{\frac{1}{d_{\mathrm{model}}}\sum_{i=1}^{d_{\mathrm{model}}}a_i^2+\varepsilon}. \tag{4}$$

$g_i$ 是可学习 gain；原文要求初始化为全 1。以下是原文已有的框架，未填入实现：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# Your code here performing RMSNorm
...
result = ...

# Return the result in the original dtype
return result.to(in_dtype)
```

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-positionwise_feedforward"></a>

### 11. 实现 SwiGLU 前馈网络（2 分）

**原题：Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)**  
**来源：§3.4.2，PDF 第 22 页。**

#### 英文题目要求与交付物

**Deliverable:** Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.

Note: in this particular case, you should feel free to use torch.sigmoid in your implementation for numerical stability. You should set $d_{\mathrm{ff}}$ to approximately $\frac{8}{3}\times$ $d_{\mathrm{model}}$ in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. To test your implementation against our provided tests, you will need to implement the test adapter at `adapters.run_swiglu`. Then, run `uv run pytest -k test_swiglu` to test your implementation.

#### 中文翻译
**交付物：** 实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。
注意：在这个特定情况下，你可以放心地在实现中使用 torch.sigmoid 以获得数值稳定性。你应在实现中将 $d_{\mathrm{ff}}$ 设为约 $\frac{8}{3}\times d_{\mathrm{model}}$，同时确保内层前馈的维度是 64 的倍数以充分利用硬件。要用我们提供的测试测试你的实现，需要实现测试适配器 `adapters.run_swiglu`。然后运行 `uv run pytest -k test_swiglu` 测试你的实现。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF pp. 21–22）：**

$$\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}. \tag{5}$$

$$\operatorname{GLU}(x,W_1,W_2)=\sigma(W_1x)\odot W_2x. \tag{6}$$

$$\operatorname{FFN}(x)=\operatorname{SwiGLU}(x,W_1,W_2,W_3)=W_2\bigl(\operatorname{SiLU}(W_1x)\odot W_3x\bigr). \tag{7}$$

$x\in\mathbb R^{d_{\mathrm{model}}}$，$W_1,W_3\in\mathbb R^{d_{\mathrm{ff}}\times d_{\mathrm{model}}}$，$W_2\in\mathbb R^{d_{\mathrm{model}}\times d_{\mathrm{ff}}}$；不使用 bias。$d_{\mathrm{ff}}\approx\frac83d_{\mathrm{model}}$，取附近的 64 倍数。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-rope"></a>

### 12. 实现旋转位置编码 RoPE（2 分）

**原题：Problem (rope): Implement RoPE (2 points)**  
**来源：§3.4.3，PDF 第 23 页。**

#### 英文题目要求与交付物

**Deliverable:** Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor. The following interface is recommended:

`def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)` Construct the RoPE module and create buffers if needed.

theta: float   Θ value for the RoPE

d_k: int   dimension of query and key vectors

max_seq_len: int    Maximum sequence length that will be input

device: torch.device | None = None      Device to store the buffer on

`def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor` Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note that you should tolerate $x$ with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of $x$ along the sequence dimension. You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.

To test your implementation, complete `adapters.run_rope` and make sure it passes `uv run pytest -k test_rope`.

#### 中文翻译
**交付物：** 实现一个对输入张量应用 RoPE 的类 RotaryPositionalEmbedding。推荐以下接口：
`def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)` 构造 RoPE 模块，并在需要时创建缓冲区。
`theta: float`   RoPE 的 $\Theta$ 值
`d_k: int`   query 和 key 向量的维度
`max_seq_len: int`    将输入的最大序列长度
`device: torch.device | None = None`      缓冲区存储的设备
`def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor` 处理形状为 (..., seq_len, d_k) 的输入张量并返回相同形状的张量。注意，你应容忍具有任意数量批次维度的 $x$。你应假设 token 位置是形状为 (..., seq_len) 的张量，指定 $x$ 沿序列维度的 token 位置。你应使用 token 位置沿序列维度切片你（可能预计算的）cos 和 sin 张量。
要测试你的实现，完成 `adapters.run_rope`，并确保通过 `uv run pytest -k test_rope`。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF pp. 22–23）：**

$$q^{\prime(i)}=R^iq^{(i)}=R^iW_qx^{(i)},\qquad
\theta_{i,k}=\frac{i}{\Theta^{(2k-2)/d}},\quad k\in\{1,\ldots,d/2\}.$$

$$R_k^i=\begin{pmatrix}\cos\theta_{i,k}&-\sin\theta_{i,k}\\\sin\theta_{i,k}&\cos\theta_{i,k}\end{pmatrix}. \tag{8}$$

$$R^i=\operatorname{diag}(R_1^i,R_2^i,\ldots,R_{d/2}^i). \tag{9}$$

原文要求利用矩阵结构高效变换，不必构造完整 $d\times d$ 矩阵。余弦和正弦可在层与批次间复用，可用 `self.register_buffer(..., persistent=False)` 缓存；这些量不作为 `nn.Parameter`，本层没有可学习参数。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-softmax"></a>

### 13. 实现 Softmax（1 分）

**原题：Problem (softmax): Implement softmax (1 point)**  
**来源：§3.4.4，PDF 第 24 页。**

#### 英文题目要求与交付物

**Deliverable:** Write a function to apply the softmax operation on a tensor. Your function should take two parameters: a tensor and a dimension $i$, and apply softmax to the $i$-th dimension of the input tensor. The output tensor should have the same shape as the input tensor, but its $i$-th dimension will now have a normalized probability distribution. Use the trick of subtracting the maximum value in the $i$-th dimension from all elements of the $i$-th dimension to avoid numerical stability issues.

To test your implementation, complete `adapters.run_softmax` and make sure it passes `uv run pytest -k test_softmax_matches_pytorch`.

#### 中文翻译
**交付物：** 编写一个对张量应用 softmax 操作的函数。你的函数应接受两个参数：一个张量和维度 $i$，并对输入张量的第 $i$ 维应用 softmax。输出张量形状与输入相同，但其第 $i$ 维将成为归一化的概率分布。使用从第 $i$ 维所有元素中减去该维最大值的技巧来避免数值稳定性问题。
要测试你的实现，完成 `adapters.run_softmax`，并确保通过 `uv run pytest -k test_softmax_matches_pytorch`。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF pp. 23–24）：**

$$\operatorname{softmax}(v)_i=\frac{\exp(v_i)}{\sum_{j=1}^{n}\exp(v_j)}. \tag{10}$$

原文指出 softmax 对所有输入加同一常数不变，并要求先减去指定维度的最大值以保证数值稳定。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-scaled_dot_product_attention"></a>

### 14. 实现缩放点积注意力（5 分）

**原题：Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)**  
**来源：§3.4.4，PDF 第 24–25 页。**

#### 英文题目要求与交付物

**Deliverable:** Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like dimensions (if provided). The implementation should return an output with the shape (batch_size, ..., seq_len, d_v). See Section 3.2 for a discussion on batch-like dimensions.

Your implementation should also support an optional user-provided boolean mask of shape (seq_len, seq_len). The attention probabilities of positions with a mask value of True should collectively sum to 1, and the attention probabilities of positions with a mask value of False should be zero.

To test your implementation against our provided tests, you will need to implement the test adapter at `adapters.run_scaled_dot_product_attention`. `uv run pytest -k test_scaled_dot_product_attention` tests your implementation on third-order input tensors, while `uv run pytest -k test_4d_scaled_dot_product_attention` tests your implementation on fourth-order input tensors.

#### 中文翻译
**交付物：** 实现缩放点积注意力函数。你的实现应处理形状为 (batch_size, ..., seq_len, d_k) 的 key 和 query，以及形状为 (batch_size, ..., seq_len, d_v) 的 value，其中 ... 表示任意数量的其他批次类维度（如果提供的话）。实现应返回形状为 (batch_size, ..., seq_len, d_v) 的输出。批次类维度的讨论见第 3.2 节。
你的实现还应支持可选的用户提供的形状为 (seq_len, seq_len) 的布尔掩码。掩码值为 True 的位置，其注意力概率之和应为 1；掩码值为 False 的位置，其注意力概率应为零。
要用我们提供的测试测试你的实现，需要实现测试适配器 `adapters.run_scaled_dot_product_attention`。`uv run pytest -k test_scaled_dot_product_attention` 用三阶输入张量测试你的实现，`uv run pytest -k test_4d_scaled_dot_product_attention` 用四阶输入张量测试你的实现。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF p. 24）：**

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V. \tag{11}$$

$Q\in\mathbb R^{n\times d_k}$，$K\in\mathbb R^{m\times d_k}$，$V\in\mathbb R^{m\times d_v}$。布尔 mask $M\in\{\mathrm{True},\mathrm{False}\}^{n\times m}$ 中，`True` 表示可关注，`False` 表示不可关注；原文说明可在 softmax 前向不可关注位置加 $-\infty$。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-multihead_self_attention"></a>

### 15. 实现因果多头自注意力（5 分）

**原题：Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)**  
**来源：§3.4.5，PDF 第 25–26 页。**

#### 英文题目要求与交付物

**Deliverable:** Implement causal multi-head self-attention as a torch.nn.Module. Your implementation should accept (at least) the following parameters:

d_model: int Dimensionality of the Transformer block inputs.

num_heads: int Number of heads to use in multi-head self-attention.

Following A. Vaswani et al. [8], set $d_k=d_v=\frac{d_{\mathrm{model}}}{h}$. To test your implementation against our provided tests, implement the test adapter at `adapters.run_multihead_self_attention`. Then, run `uv run pytest -k test_multihead_self_attention` to test your implementation.

#### 中文翻译
**交付物：** 将因果多头自注意力实现为 torch.nn.Module。你的实现应接受（至少）以下参数：
`d_model: int` Transformer 块输入的维度。
`num_heads: int` 多头自注意力中使用的头数。
遵循 A. Vaswani et al. [8]，设 $d_k=d_v=\frac{d_{\mathrm{model}}}{h}$。要用我们提供的测试测试你的实现，实现测试适配器 `adapters.run_multihead_self_attention`。然后运行 `uv run pytest -k test_multihead_self_attention` 测试你的实现。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF p. 25）：**

$$\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h). \tag{12}$$

$$\mathrm{head}_i=\operatorname{Attention}(Q_i,K_i,V_i). \tag{13}$$

$$\operatorname{MultiHeadSelfAttention}(x)=W_O\operatorname{MultiHead}(W_Qx,W_Kx,W_Vx). \tag{14}$$

$W_Q,W_K\in\mathbb R^{hd_k\times d_{\mathrm{model}}}$，$W_V\in\mathbb R^{hd_v\times d_{\mathrm{model}}}$，$W_O\in\mathbb R^{d_{\mathrm{model}}\times hd_v}$。原文要求 Q、K、V 投影共使用三次矩阵乘法；因果 mask 只允许位置 $i$ 关注 $j\le i$。RoPE 仅施加到 query 和 key，不施加到 value；每个 head 使用相同的旋转，head 维作为批次维处理。脚注 5 的可选挑战：将 Q、K、V 投影合并到同一个权重矩阵，只做一次矩阵乘法。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-transformer_block"></a>

### 16. 实现 Transformer 块（3 分）

**原题：Problem (transformer_block): Implement the Transformer block (3 points)**  
**来源：§3.5，PDF 第 26 页。**

#### 英文题目要求与交付物

Implement the pre-norm Transformer block as described in Section 3.4 and illustrated in Figure 2. Your Transformer block should accept (at least) the following parameters.

d_model: int Dimensionality of the Transformer block inputs.

num_heads: int Number of heads to use in multi-head self-attention.

d_ff: int Dimensionality of the position-wise feed-forward inner layer.

To test your implementation, implement the adapter `adapters.run_transformer_block`. Then run `uv run pytest -k test_transformer_block` to test your implementation.

**Deliverable:** Transformer block code that passes the provided tests.

#### 中文翻译
实现 3.4 节所述、图 2 所示的 pre-norm Transformer 块。你的 Transformer 块应接受（至少）以下参数。
`d_model: int` Transformer 块输入的维度。
`num_heads: int` 多头自注意力中使用的头数。
`d_ff: int` 逐位置前馈内层的维度。
要测试你的实现，实现适配器 `adapters.run_transformer_block`。然后运行 `uv run pytest -k test_transformer_block` 测试你的实现。
**交付物：** 通过所提供测试的 Transformer 块代码。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF pp. 19、26，Figure 2）：** 每个子层先 RMSNorm，再执行 MHA 或 SwiGLU，最后加残差；输入输出均为 `(batch_size, sequence_length, d_model)`。原文给出的第一子层公式为：

$$y=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)). \tag{15}$$

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-transformer_lm"></a>

### 17. 实现完整 Transformer 语言模型（3 分）

**原题：Problem (transformer_lm): Implementing the Transformer LM (3 points)**  
**来源：§3.5，PDF 第 26–27 页。**

#### 英文题目要求与交付物

Time to put it all together! Implement the Transformer language model as described in Section 3.1 and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, as well as these additional parameters:

vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.

context_length: int The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.

num_layers: int The number of Transformer blocks to use.

To test your implementation against our provided tests, you will first need to implement the test adapter at `adapters.run_transformer_lm`. Then, run `uv run pytest -k test_transformer_lm` to test your implementation.

**Deliverable:** A Transformer LM module that passes the above tests.

#### 中文翻译
是时候把所有部分组合起来了！实现 3.1 节所述、图 1 所示的 Transformer 语言模型。你的实现至少应接受上述 Transformer 块的所有构造参数，以及这些额外参数：
`vocab_size: int` 词表大小，用于确定 token 嵌入矩阵的维度。
`context_length: int` 最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
`num_layers: int` 使用的 Transformer 块数量。
要用我们提供的测试测试你的实现，首先需要实现测试适配器 `adapters.run_transformer_lm`。然后运行 `uv run pytest -k test_transformer_lm` 测试你的实现。
**交付物：** 通过上述测试的 Transformer LM 模块。

#### 原文给定的公式、代码或关联条件

**原文给定架构（§3.1、§3.5，PDF pp. 13–14、26–27，Figure 1）：** Token Embedding → `num_layers` 个 Pre-norm Transformer Block → 最终 RMSNorm → 线性 LM head，输出未归一化的 next-token logits。输入形状为 `(batch_size, sequence_length)`，输出 logits 形状为 `(batch_size, sequence_length, vocab_size)`。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-transformer_accounting"></a>

### 18. Transformer 资源核算（5 分）

**原题：Problem (transformer_accounting): Transformer LM resource accounting (5 points)**  
**来源：§3.5，PDF 第 27–28 页。**

#### 英文题目要求与交付物

**(a)** Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:

vocab_size:   50,257

context_length:    1,024

num_layers:   48

d_model:   1,600

num_heads:    25

d_ff:   4,288 (the nearest multiple of 64 to $\frac{8}{3}\times 1{,}600$)

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

**Deliverable:** A one-to-two sentence response.

**(b)** Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

**Deliverable:** A list of matrix multiplies (with descriptions), and the total number of FLOPs required.

**(c)** Based on your analysis above, which parts of the model require the most FLOPs?

**Deliverable:** A one-to-two sentence response.

**(d)** Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

**Deliverable:** For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

**(e)** Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

**Deliverable:** A one-to-two sentence response.

#### 中文翻译
**(a)** 考虑一个使用本作业架构的 GPT-2 XL 尺寸的模型，配置如下：
`vocab_size:`   50,257
`context_length:`    1,024
`num_layers:`   48
`d_model:`   1,600
`num_heads:`    25
`d_ff:`   4,288（$\frac{8}{3}\times 1{,}600$ 最接近的 64 倍数）
假设我们用这个配置构造模型。模型有多少可训练参数？假设每个参数用单精度浮点数表示，仅加载这个模型需要多少内存？
**交付物：** 一至两句话回答。
**(b)** 找出完成该 GPT-2 XL 形状模型一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列有 context_length 个 token。
**交付物：** 矩阵乘法清单（带描述），以及所需的总 FLOPs。
**(c)** 基于你上面的分析，模型的哪些部分需要最多的 FLOPs？
**交付物：** 一至两句话回答。
**(d)** 用 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型规模增大，Transformer LM 的哪些部分占总 FLOPs 的比例变多或变少？
**交付物：** 对每个模型，给出模型组件的分解及其 FLOPs（占一次前向传播总 FLOPs 的比例）。此外，用一至两句话描述改变模型规模如何改变各组件的 FLOPs 占比。
**(e)** 取 GPT-2 XL 并把上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型各组件的 FLOPs 相对占比如何变化？
**交付物：** 一至两句话回答。

#### 原文给定的公式、代码或关联条件

#### 答题区域（待填写）

##### (a)

可训练参数数目、float32 参数内存（1–2 句话）：

<!-- 待填写 -->


##### (b)

矩阵乘法清单、各项描述与总 FLOPs：

<!-- 待填写 -->


##### (c)

主要 FLOPs 来源（1–2 句话）：

<!-- 待填写 -->


##### (d)

GPT-2 small / medium / large 的组件 FLOPs 占比，以及规模变化解释：

<!-- 待填写 -->


##### (e)

上下文长度改为 16,384 后的总 FLOPs 与组件占比变化（1–2 句话）：

<!-- 待填写 -->


- [ ] 已按本题原文要求完成全部交付物。

---

## 4. 训练组件与优化器

<a id="problem-cross_entropy"></a>

### 19. 实现交叉熵损失（1 分）

**原题：Problem (cross_entropy): Implement cross-entropy (1 point)**  
**来源：§4.1，PDF 第 29 页。**

#### 英文题目要求与交付物

**Deliverable:** Write a function to compute the cross-entropy loss, which takes in predicted logits ($o_i$ ) and targets ($x_{i+1}$ ) and computes the cross-entropy $\ell_i=-\log\operatorname{softmax}(o_i)[x_{i+1}]$. Your function should handle the following:

- Subtract the largest element for numerical stability.

- Cancel out log and exp whenever possible.

- Handle any additional batch dimensions and return the average across the batch. As with Section 3.2, we assume batch-like dimensions always come first, before the vocabulary size dimension.

Implement `adapters.run_cross_entropy`, then run `uv run pytest -k test_cross_entropy` to test your implementation.

#### 中文翻译
**交付物：** 编写一个计算交叉熵损失的函数，接收预测 logits（$o_i$）和目标（$x_{i+1}$），计算交叉熵 $\ell_i=-\log\operatorname{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下情况：
- 为数值稳定性减去最大元素。
- 尽可能约去 log 和 exp。
- 处理任何额外的批次维度，并返回批次上的平均值。与 3.2 节一样，我们假设批次类维度总是排在最前面，在词表大小维度之前。
实现 `adapters.run_cross_entropy`，然后运行 `uv run pytest -k test_cross_entropy` 测试你的实现。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF pp. 28–29）：**

$$\ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_\theta(x_{i+1}\mid x_{1:i}). \tag{16}$$

$$p(x_{i+1}\mid x_{1:i})=\operatorname{softmax}(o_i)[x_{i+1}]=\frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\mathrm{vocab\_size}}\exp(o_i[a])}. \tag{17}$$

原文另给出供评估使用的 perplexity 定义：

$$\operatorname{perplexity}=\exp\left(\frac1m\sum_{i=1}^{m}\ell_i\right). \tag{18}$$

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-learning_rate_tuning"></a>

### 20. SGD 学习率实验（1 分）

**原题：Problem (learning_rate_tuning): Tuning the learning rate (1 point)**  
**来源：§4.2.1，PDF 第 31 页。**

#### 英文题目要求与交付物

As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?

**Deliverable:** A one-to-two sentence response with the behaviors you observed.

#### 中文翻译
正如我们将看到的，对训练影响最大的超参数之一是学习率。让我们在上面的玩具示例中实际看看。用另外三个学习率值：1e1、1e2 和 1e3 运行上面的 SGD 示例，只跑 10 次训练迭代。每个学习率下损失会发生什么？它衰减得更快、更慢，还是发散（即在训练过程中上升）？
**交付物：** 一至两句话回答，说明你观察到的行为。

#### 原文给定的公式、代码或关联条件

**原文给定的 SGD 更新式（PDF pp. 29–30）：**

$$\theta_{t+1}\leftarrow\theta_t-\alpha_t\nabla L(\theta_t;B_t). \tag{19}$$

题目中的 SGD 示例使用以下随步数衰减的更新：

$$\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{t+1}}\nabla L(\theta_t;B_t). \tag{20}$$

**原文已有示例代码（供运行题目要求的实验，不是本模板新增的答案）：**

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
```

上面保留原示例的 `lr=1` 和 100 次迭代；本题要求自行改为 `1e1`、`1e2`、`1e3`，各运行 10 次迭代并记录观察。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

| 学习率 | 10 次迭代的 loss 记录/文件 | 观察 |
| --- | --- | --- |
| `1e1` | 待填写 | 待填写 |
| `1e2` | 待填写 | 待填写 |
| `1e3` | 待填写 | 待填写 |

1–2 句话总结：

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-adamw"></a>

### 21. 实现 AdamW（2 分）

**原题：Problem (adamw): Implement AdamW (2 points)**  
**来源：§4.3，PDF 第 32 页。**

#### 英文题目要求与交付物

**Deliverable:** Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Your class should take the learning rate $\alpha$ in `__init__`, as well as the $\beta$, $\varepsilon$ and $\lambda$ hyperparameters. To help you keep state, the base Optimizer class gives you a dictionary self.state, which maps nn.Parameter objects to a dictionary that stores any information you need for that parameter (for AdamW, this would be the moment estimates). Implement `adapters.get_adamw_cls` and make sure it passes `uv run pytest -k test_adamw`.

#### 中文翻译
**交付物：** 将 AdamW 优化器实现为 torch.optim.Optimizer 的子类。你的类应在 `__init__` 中接收学习率 $\alpha$，以及 $\beta$、$\varepsilon$ 和 $\lambda$ 超参数。为了帮你保存状态，Optimizer 基类提供了一个字典 self.state，它将 nn.Parameter 对象映射到一个字典，存储你需要的关于该参数的任何信息（对 AdamW 来说就是矩估计）。实现 `adapters.get_adamw_cls`，并确保通过 `uv run pytest -k test_adamw`。

#### 原文给定的公式、代码或关联条件

**原文 Algorithm 1（PDF p. 32）：** 以下是教程已给出的伪代码，保留其权重衰减顺序，未加入实现。

1. 初始化参数 $\theta$，同形状的一阶矩 $m\leftarrow0$ 和二阶矩 $v\leftarrow0$。
2. 对 $t=1,\ldots,T$：采样 batch $B_t$，计算 $g\leftarrow\nabla_\theta\ell(\theta;B_t)$。
3. $\alpha_t\leftarrow\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$。
4. $\theta\leftarrow\theta-\alpha\lambda\theta$（Apply weight decay）。
5. $m\leftarrow\beta_1m+(1-\beta_1)g$。
6. $v\leftarrow\beta_2v+(1-\beta_2)g^2$。
7. $\theta\leftarrow\theta-\alpha_t\frac{m}{\sqrt v+\varepsilon}$。

原文说明 $t$ 从 1 开始，$\varepsilon$ 例如 $10^{-8}$。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-adamw_accounting"></a>

### 22. AdamW 训练资源核算（2 分）

**原题：Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)**  
**来源：§4.3，PDF 第 32–33 页。**

#### 英文题目要求与交付物

Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

**(a)** How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume $d_{\mathrm{ff}}=\frac{8}{3}d_{\mathrm{model}}$.

For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block

  - RMSNorm(s)

  - Multi-head self-attention sublayer: $QKV$ projections, $QK^\top$ matrix multiply, softmax, weighted sum of values, output projection.

  - Position-wise feed-forward (SwiGLU): $W_1$ , $W_2$ , SiLU on the gate branch, element-wise product, $W_3$

- final RMSNorm

- output embedding

- cross-entropy on logits

**Deliverable:** An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

**(b)** Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

**Deliverable:** An expression that looks like $a\cdot\texttt{batch_size}+b$ for numerical values $a$, $b$, and a number representing the maximum batch size.

**(c)** How many FLOPs does running one step of AdamW take?

**Deliverable:** An algebraic expression, with a brief justification.

**(d)** Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput [A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for “float32” (actually TensorFloat-32, which in reality is “bfloat19”) operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following J. Kaplan et al. [25] and J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass.

**Deliverable:** The number of hours training would take, with a brief justification.

#### 中文翻译
让我们计算运行 AdamW 需要多少内存和计算量。假设所有张量都使用 float32。
**(a)** 运行 AdamW 需要多少峰值内存？请基于参数、激活、梯度和优化器状态的内存占用分解你的答案。用 batch_size 和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表示你的答案。假设 $d_{\mathrm{ff}}=\frac{8}{3}d_{\mathrm{model}}$。
为简单起见，计算激活的内存占用时只考虑以下组件：
- Transformer 块
  - RMSNorm
  - 多头自注意力子层：$QKV$ 投影、$QK^\top$ 矩阵乘法、softmax、value 的加权和、输出投影。
  - 逐位置前馈（SwiGLU）：$W_1$、$W_2$、gate 分支上的 SiLU、逐元素乘积、$W_3$
- 最终 RMSNorm
- 输出嵌入
- logits 上的交叉熵
**交付物：** 参数、激活、梯度、优化器状态各自的代数表达式，以及总量。
**(b)** 将你的答案代入 GPT-2 XL 形状的模型，得到一个只依赖 batch_size 的表达式。在 80GB 内存内你能使用的最大 batch size 是多少？
**交付物：** 形如 $a\cdot\texttt{batch_size}+b$ 的表达式（$a$、$b$ 为数值），以及一个表示最大 batch size 的数。
**(c)** 运行一步 AdamW 需要多少 FLOPs？
**交付物：** 一个代数表达式，附简要说明。
**(d)** 模型 FLOPs 利用率（MFU）定义为观测吞吐量（tokens/second）与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 对“float32”（实际上是 TensorFloat-32，现实中是“bfloat16”）操作的理论峰值为 495 teraFLOP/s。假设你能达到 50% MFU，在单张 H100 上、batch size 为 1024、训练 GPT-2 XL 400K 步需要多长时间？遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设反向传播的 FLOPs 是正向传播的两倍。
**交付物：** 训练所需的小时数，附简要说明。

#### 原文给定的公式、代码或关联条件

**关联原题条件：** GPT-2 XL 配置见 `transformer_accounting` 的 (a)；本题明确假定 $d_{\mathrm{ff}}=\frac83d_{\mathrm{model}}$，计算时应保留自己采用的口径说明。AdamW 算法见上一题给出的原文伪代码。本题未给出的内存和 FLOPs 结果均留空。

#### 答题区域（待填写）

##### (a)

参数、激活、梯度、优化器状态和总峰值内存的代数表达式：

<!-- 待填写 -->


##### (b)

GPT-2 XL 的 a × batch_size + b 表达式与 80 GB 内最大 batch size：

<!-- 待填写 -->


##### (c)

单次 AdamW step 的 FLOPs 代数式与简要说明：

<!-- 待填写 -->


##### (d)

训练小时数与简要依据：

<!-- 待填写 -->


- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-learning_rate_schedule"></a>

### 23. 实现预热与余弦学习率调度（1 分）

**原题：Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup (1 point)**  
**来源：§4.4，PDF 第 33–34 页。**

#### 英文题目要求与交付物

Write a function that takes $t$, $\alpha_{\max}$ , $\alpha_{\min}$ , $T_w$ and $T_c$ , and returns the learning rate $\alpha_t$ according to the scheduler defined above. Then implement `adapters.get_lr_cosine_schedule` and make sure it passes `uv run pytest -k test_get_lr_cosine_schedule`.

#### 中文翻译
编写一个函数，接收 $t$、$\alpha_{\max}$、$\alpha_{\min}$、$T_w$ 和 $T_c$，按上面定义的调度器返回学习率 $\alpha_t$。然后实现 `adapters.get_lr_cosine_schedule`，并确保通过 `uv run pytest -k test_get_lr_cosine_schedule`。

#### 原文给定的公式、代码或关联条件

**原文给定调度定义（PDF p. 33）：**

$$\alpha_t=\begin{cases}
\dfrac{t}{T_w}\alpha_{\max},&t<T_w\quad\text{(Warm-up)},\\
\alpha_{\min}+\dfrac12\left(1+\cos\left(\dfrac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max}-\alpha_{\min}),&T_w\le t\le T_c\quad\text{(Cosine annealing)},\\
\alpha_{\min},&t>T_c\quad\text{(Post-annealing)}.
\end{cases}$$

原文脚注 8：有时也采用学习率重新升高的 restart 调度以越过局部最小值；本题实现上面给定的调度。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-gradient_clipping"></a>

### 24. 实现梯度裁剪（1 分）

**原题：Problem (gradient_clipping): Implement gradient clipping (1 point)**  
**来源：§4.5，PDF 第 34 页。**

#### 英文题目要求与交付物

Write a function that implements gradient clipping. Your function should take a list of parameters and a maximum $\ell_2$-norm. It should modify each parameter gradient in place. Use $\varepsilon$ = $10^{-6}$ (the PyTorch default). Then, implement the adapter `adapters.run_gradient_clipping` and make sure it passes `uv run pytest -k test_gradient_clipping`.

#### 中文翻译
编写一个实现梯度裁剪的函数。你的函数应接收一个参数列表和最大 $\ell_2$ 范数。它应原地修改每个参数的梯度。使用 $\varepsilon=10^{-6}$（PyTorch 默认值）。然后实现适配器 `adapters.run_gradient_clipping`，并确保通过 `uv run pytest -k test_gradient_clipping`。

#### 原文给定的公式、代码或关联条件

**原文给定规则（PDF p. 34）：** 对所有参数的总梯度 $g$ 计算 $\lVert g\rVert_2$；若小于最大值 $M$ 则不变，否则将 $g$ 乘以 $M/(\lVert g\rVert_2+\varepsilon)$。原文指定 $\varepsilon=10^{-6}$，原地修改各参数梯度。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

## 5. 训练循环

<a id="problem-data_loading"></a>

### 25. 实现数据加载（2 分）

**原题：Problem (data_loading): Implement data loading (2 points)**  
**来源：§5.1，PDF 第 34–35 页。**

#### 英文题目要求与交付物

**Deliverable:** Write a function that takes a numpy array $x$ (integer array with token IDs), a batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both tensors should have shape (batch_size, context_length) containing token IDs, and both should be placed on the requested device. To test your implementation against our provided tests, you will first need to implement the test adapter at `adapters.run_get_batch`. Then, run `uv run pytest -k test_get_batch` to test your implementation.

#### 中文翻译
**交付物：** 编写一个函数，接收一个 numpy 数组 $x$（包含 token ID 的整数数组）、一个 batch_size、一个 context_length 和一个 PyTorch 设备字符串（如 'cpu' 或 'cuda:0'），返回一对张量：采样得到的输入序列和对应的 next-token 目标。两个张量的形状均为 (batch_size, context_length)，包含 token ID，且都应放在请求的设备上。要用我们提供的测试测试你的实现，首先需要实现测试适配器 `adapters.run_get_batch`。然后运行 `uv run pytest -k test_get_batch` 测试你的实现。

#### 原文给定的公式、代码或关联条件

**原文相关要求（PDF pp. 34–35）：** 一个 batch 包含长度相同的输入和 next-token targets。原文示例：$B=1,m=3$ 时，可以有 $([x_2,x_3,x_4],[x_3,x_4,x_5])$。训练采样时使用内存映射模式：`np.memmap`，或在数组由 `np.save` 保存时使用 `np.load(..., mmap_mode='r')`；读取 dtype 必须与保存的数据一致，检查 token ID 范围是否合理。原文低资源提示给出的设备：CPU 使用 `'cpu'`，Apple Silicon 使用 `'mps'`。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-checkpointing"></a>

### 26. 实现检查点保存与恢复（1 分）

**原题：Problem (checkpointing): Implement model checkpointing (1 point)**  
**来源：§5.2，PDF 第 36 页。**

#### 英文题目要求与交付物

Implement the following two functions to load and save checkpoints:

`def save_checkpoint(model, optimizer, iteration, out)` should dump all the state from the model, optimizer and iteration into the file-like object out. You can use the state_dict method of both the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later.

This function expects the following parameters:

model: torch.nn.Module

optimizer: torch.optim.Optimizer

iteration: int

out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

`def load_checkpoint(src, model, optimizer)` should load a checkpoint from src (path or file-like object), and then recover the model and optimizer states from that checkpoint. Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the load_state_dict method in both the model and optimizer to return them to their previous states.

This function expects the following parameters:

src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

model: torch.nn.Module

optimizer: torch.optim.Optimizer

Implement the `adapters.run_save_checkpoint` and `adapters.run_load_checkpoint` adapters, and make sure they pass `uv run pytest -k test_checkpointing`.

#### 中文翻译
实现以下两个函数来加载和保存检查点：
`def save_checkpoint(model, optimizer, iteration, out)` 应将模型、优化器和迭代次数的所有状态转储到文件类对象 out 中。你可以使用模型和优化器的 state_dict 方法获取它们各自的状态，并使用 torch.save(obj, out) 将 obj 转储到 out（PyTorch 在此支持路径或文件类对象）。典型做法是让 obj 为一个字典，但你可用任何格式，只要之后能加载你的检查点即可。
该函数接收以下参数：
`model: torch.nn.Module`
`optimizer: torch.optim.Optimizer`
`iteration: int`
`out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
`def load_checkpoint(src, model, optimizer)` 应从 src（路径或文件类对象）加载检查点，然后从该检查点恢复模型和优化器的状态。你的函数应返回保存到检查点中的迭代次数。你可以用 torch.load(src) 恢复你在 save_checkpoint 实现中保存的内容，并用模型和优化器的 load_state_dict 方法使它们回到之前的状态。
该函数接收以下参数：
`src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
`model: torch.nn.Module`
`optimizer: torch.optim.Optimizer`
实现 `adapters.run_save_checkpoint` 和 `adapters.run_load_checkpoint` 适配器，并确保通过 `uv run pytest -k test_checkpointing`。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-training_together"></a>

### 27. 整合训练循环（4 分）

**原题：Problem (training_together): Put it together (4 points)**  
**来源：§5.3，PDF 第 36–37 页。**

#### 英文题目要求与交付物

**Deliverable:** Write a script that runs a training loop to train your model on user-provided input. In particular, we recommend that your training script allow for (at least) the following:

- Ability to configure and control the various model and optimizer hyperparameters.

- Memory-efficient loading of large training and validation datasets with np.memmap.

- Serializing checkpoints to a user-provided path.

- Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases). [Footnote 9: wandb.ai]

#### 中文翻译
**交付物：** 编写一个脚本，运行训练循环，在用户提供的输入上训练你的模型。特别地，我们建议你的训练脚本（至少）支持以下功能：
- 能够配置和控制各种模型和优化器超参数。
- 用 np.memmap 内存高效地加载大型训练集和验证集。
- 将检查点序列化到用户提供的路径。
- 定期记录训练和验证性能（如输出到控制台和/或 Weights & Biases 之类的外部服务）。[脚注 9: wandb.ai]

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

## 6. 文本生成

<a id="problem-decoding"></a>

### 28. 实现文本生成与采样（3 分）

**原题：Problem (decoding): Decoding (3 points)**  
**来源：§6，PDF 第 38 页。**

#### 英文题目要求与交付物

**Deliverable:** Implement a function to decode from your language model. We recommend that you support the following features:

- Generate completions for a user-provided prompt (i.e., take in some $x_{1\ldots t}$ and sample a completion until you hit an `<|endoftext|>` token).

- Allow the user to control the maximum number of generated tokens.

- Given a desired temperature value, apply softmax temperature scaling to the predicted next-token distributions before sampling.

- Top-$p$ sampling ([A. Holtzman et al., 2020] also referred to as nucleus sampling), given a user-specified threshold value.

#### 中文翻译
**交付物：** 实现一个从你的语言模型解码（生成）的函数。我们建议你支持以下功能：
- 为用户提供的 prompt 生成补全（即接收某个 $x_{1\ldots t}$ 并采样补全，直到遇到 ` ` token）。
- 允许用户控制生成的最大 token 数。
- 给定期望的温度值，在采样前对预测的 next-token 分布应用 softmax 温度缩放。
- Top-$p$ 采样（[A. Holtzman et al., 2020]，也称核采样），使用用户指定的阈值。

#### 原文给定的公式、代码或关联条件

**原文给定（PDF pp. 37–38）：**

$$P(x_{t+1}=i\mid x_{1\ldots t})=\frac{\exp(v_i)}{\sum_j\exp(v_j)}. \tag{21}$$

$$v=\operatorname{TransformerLM}(x_{1\ldots t})_t\in\mathbb R^{\mathrm{vocab\_size}}. \tag{22}$$

$$\operatorname{softmax}(v,\tau)_i=\frac{\exp(v_i/\tau)}{\sum_{j=1}^{\mathrm{vocab\_size}}\exp(v_j/\tau)}. \tag{23}$$

$$P(x_{t+1}=i\mid q)=\begin{cases}\dfrac{q_i}{\sum_{j\in V(p)}q_j},&i\in V(p),\\0,&\text{otherwise}.\end{cases} \tag{24}$$

$V(p)$ 为累计概率至少达到 $p$ 的最小索引集合；原文说明可按概率从大到小排序后选择。生成在遇到 `<|endoftext|>` 或用户指定的最大生成 token 数时停止。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**说明 / 作答：**

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

## 7. 实验与消融

<a id="problem-experiment_log"></a>

### 29. 建立实验日志（3 分）

**原题：Problem (experiment_log): Experiment logging (3 points)**  
**来源：§7.1，PDF 第 38–39 页。**

#### 英文题目要求与交付物

For your training and evaluation code, create experiment tracking infrastructure that allows you to track your experiments and loss curves with respect to gradient steps and wall-clock time.

**Deliverable:** Logging infrastructure code for your experiments and an experiment log (a document of all the things you tried) for the assignment problems below in this section.

#### 中文翻译
为你的训练和评估代码创建实验跟踪基础设施，使你能够跟踪实验以及损失曲线随梯度步数和实际时间的变化。
**交付物：** 用于实验的日志基础设施代码，以及本节下面各作业题的实验日志（一份记录你尝试过的所有事情的文档）。

#### 原文给定的公式、代码或关联条件

**原文相关要求（PDF p. 38）：** 定期评估 validation loss，并同时记录梯度步数和实际经过时间；后续本节所有实验都要保留日志与学习曲线。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

| 实验编号 | 尝试内容 | 配置/随机种子 | 梯度步数 | 实际耗时 | 最终/最佳验证 loss | 曲线/日志 |
| --- | --- | --- | --- | --- | --- | --- |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-learning_rate"></a>

### 30. 调整模型训练学习率（3 分）

**原题：Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)**  
**来源：§7.2.3，PDF 第 40–41 页。**

#### 英文题目要求与交付物

The learning rate is one of the most important hyperparameters to tune. Taking the base model you’ve trained, answer the following questions:

**(a)** Perform a hyperparameter sweep over the learning rates and report the final losses (or note divergence if the optimizer diverges).

**Deliverable:** Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.

**Deliverable:** A model with validation loss (per-token) on TinyStories of at most 1.45

**(b)** Folk wisdom is that the best learning rate is “at the edge of stability.” Investigate how the point at which learning rates diverge is related to your best learning rate.

**Deliverable:** Learning curves of increasing learning rate which include at least one divergent run and an analysis of how this relates to convergence rates.

#### 中文翻译
学习率是最重要的需要调整的超参数之一。以你训练的基础模型为准，回答以下问题：
**(a)** 对学习率做超参数扫描，报告最终损失（如果优化器发散则注明发散）。
**交付物：** 多个学习率对应的学习曲线。解释你的超参数搜索策略。
**交付物：** 一个在 TinyStories 上验证损失（per-token）不超过 1.45 的模型
**(b)** 民间经验认为最佳学习率“在稳定边缘”。研究学习率发散的位置与你最佳学习率之间的关系。
**交付物：** 递增学习率的学习曲线，其中至少包含一次发散运行，并分析这与收敛速率的关系。

#### 原文给定的公式、代码或关联条件

**原文给定的 TinyStories 基础配置（§7.2.1，PDF p. 39）：**

| 参数 | 原文给定值 |
| --- | --- |
| `vocab_size` | 10,000 |
| `context_length` | 256 |
| `d_model` | 512 |
| `d_ff` | 1,344，约为 $\frac83d_{\mathrm{model}}$ 且为 64 的倍数 |
| RoPE $\Theta$ | 10,000 |
| `num_layers` | 4 |
| `num_heads` | 16 |
| 总处理 token 数 | 327,680,000，约等于 `batch_size × total_step_count × context_length` |

学习率、warmup、AdamW 的 $(\beta_1,\beta_2,\varepsilon)$ 和 weight decay 由学生实验调整。原文建议自行考察词表大小和上下文长度对行为、每步耗时和 perplexity 的影响。以上是题目已有配置，未预填实验结论。

**原文低资源提示（PDF pp. 40–41）：** 以下为本版教程给定的替代设置与提示，按原文保留，不作为本模板新增建议。

- CPU/MPS 可将总处理 token 数降低到约 40,000,000，将 validation loss 目标由 1.45 放宽到 2.00。
- 原文 staff 示例：`32 × 5000 × 256 = 40,960,000` tokens；M4 Max、36 GB RAM 上 CPU 约 1 小时 22 分，MPS 约 36 分；第 5000 步 validation loss 为 1.80。这些是原文参考结果，不是你的实验结果。
- 使用 $N$ 个训练步时，让余弦衰减在第 $N$ 步恰好到达最小学习率。
- 原文对 PyTorch 2.9.0 的 MPS 提示：不要启用 `torch.set_float32_matmul_precision('high')`，其后端可能使用有问题的内核造成训练不稳定。
- 原文给出的编译示例：CPU 为 `model = torch.compile(model)`；MPS 为 `model = torch.compile(model, backend="aot_eager")`。原文注明 PyTorch 2.9.0 的 MPS 尚不支持 Inductor 编译。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

##### (a)

搜索策略、各学习率最终 loss / 是否发散、学习曲线、达标模型路径：

<!-- 待填写 -->


##### (b)

稳定边界实验、至少一次发散运行及收敛速率分析：

<!-- 待填写 -->


采用的资源配置（标准 / 原文低资源替代配置）：<!-- 待填写 -->

达标验证 loss：<!-- 待填写 -->

模型 checkpoint：<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-batch_size_experiment"></a>

### 31. Batch size 对比实验（1 分）

**原题：Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)**  
**来源：§7.2.3，PDF 第 41 页。**

#### 英文题目要求与交付物

Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.

**Deliverable:** Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.

**Deliverable:** A few sentences discussing your findings on batch sizes and their impacts on training.

#### 中文翻译
将你的 batch size 从 1 一直变化到 GPU 显存上限。中间至少再试几个 batch size，包括 64 和 128 这样的典型值。
**交付物：** 不同 batch size 运行的学习曲线。如有必要，应重新优化学习率。
**交付物：** 用几句话讨论你对 batch size 及其对训练影响的发现。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

| Batch size | 调整后的学习率 | 最终 loss | 耗时 | 曲线 |
| --- | --- | --- | --- | --- |
| 1 | 待填写 | 待填写 | 待填写 | 待填写 |
| 64 | 待填写 | 待填写 | 待填写 | 待填写 |
| 128 | 待填写 | 待填写 | 待填写 | 待填写 |
| 其他 / 显存上限 | 待填写 | 待填写 | 待填写 | 待填写 |

对 batch size 影响的简评：

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-generate"></a>

### 32. 展示模型生成文本（1 分）

**原题：Problem (generate): Generate text (1 point)**  
**来源：§7.2.3，PDF 第 42 页。**

#### 英文题目要求与交付物

Using your decoder and your trained checkpoint, report the text generated by your model. You may need to manipulate decoder parameters (temperature, top-p, etc.) to get fluent outputs.

**Deliverable:** Text dump of at least 256 tokens of text (or until the first `<|endoftext|>` token), and a brief comment on the fluency of this output and at least two factors which affect how good or bad this output is.

#### 中文翻译
使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要调整解码器参数（temperature、top-p 等）才能得到流畅的输出。
**交付物：** 至少 256 个 token 的文本转储（或直到第一个 ` ` token），并简要评论这段输出的流畅度，以及至少两个影响输出好坏的因素。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

Checkpoint、prompt、temperature、top-p：<!-- 待填写 -->

生成 token 数 / 是否遇到首个 `<|endoftext|>`：<!-- 待填写 -->

生成文本（至少 256 tokens，或至首个 `<|endoftext|>`）：

```text
（待粘贴自己的生成文本）
```

流畅度简评：<!-- 待填写 -->

影响质量的因素 1：<!-- 待填写 -->

影响质量的因素 2：<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-layer_norm_ablation"></a>

### 33. 去除 RMSNorm（1 分）

**原题：Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)**  
**来源：§7.3，PDF 第 42 页。**

#### 英文题目要求与交付物

Remove all of the RMSNorms from your Transformer and train. What happens at the previous optimal learning rate? Can you get stability by using a lower learning rate?

**Deliverable:** A learning curve for when you remove RMSNorms and train, as well as a learning curve for the best learning rate.

**Deliverable:** A few sentences of commentary on the impact of RMSNorm.

#### 中文翻译
从你的 Transformer 中移除所有 RMSNorm 并训练。在之前的最佳学习率下会发生什么？使用更低的学习率能否获得稳定性？
**交付物：** 一条移除 RMSNorm 训练的学习曲线，以及一条最佳学习率下的学习曲线。
**交付物：** 用几句话评论 RMSNorm 的影响。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

原最佳学习率下的曲线：<!-- 待填写 -->

降低学习率后最佳曲线：<!-- 待填写 -->

RMSNorm 影响简评：

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-pre_norm_ablation"></a>

### 34. 将 Pre-norm 改为 Post-norm（1 分）

**原题：Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)**  
**来源：§7.3，PDF 第 43 页。**

#### 英文题目要求与交付物

Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.

**Deliverable:** A learning curve for a post-norm Transformer, compared to the pre-norm one.

#### 中文翻译
把你的 pre-norm Transformer 实现改成 post-norm。用 post-norm 模型训练，看看会发生什么。
**交付物：** 一条 post-norm Transformer 的学习曲线，与 pre-norm 的对比。

#### 原文给定的公式、代码或关联条件

**原文给定比较架构（PDF pp. 42–43）：**

Pre-norm：

$$z=x+\operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x)). \tag{25}$$
$$y=z+\operatorname{FFN}(\operatorname{RMSNorm}(z)). \tag{26}$$

Post-norm：

$$z=\operatorname{RMSNorm}(x+\operatorname{MultiHeadSelfAttention}(x)). \tag{27}$$
$$y=\operatorname{RMSNorm}(z+\operatorname{FFN}(z)). \tag{28}$$

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

Pre-norm / Post-norm 比较曲线：<!-- 待填写 -->

观察记录：<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-no_pos_emb"></a>

### 35. 移除位置编码 NoPE（1 分）

**原题：Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)**  
**来源：§7.3，PDF 第 43 页。**

#### 英文题目要求与交付物

Modify your Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.

**Deliverable:** A learning curve comparing the performance of RoPE and NoPE.

#### 中文翻译
修改你的带 RoPE 的 Transformer 实现，完全移除位置编码信息，看看会发生什么。
**交付物：** 一条比较 RoPE 与 NoPE 性能的学习曲线。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

RoPE / NoPE 比较曲线：<!-- 待填写 -->

观察记录：<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-swiglu_ablation"></a>

### 36. 比较 SwiGLU 与 SiLU（1 分）

**原题：Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)**  
**来源：§7.3，PDF 第 43–44 页。**

#### 英文题目要求与交付物

**Deliverable:** A learning curve comparing the performance of SwiGLU and SiLU feed-forward networks, with approximately matched parameter counts.

**Deliverable:** A few sentences discussing your findings.

#### 中文翻译
**交付物：** 一条比较 SwiGLU 与 SiLU 前馈网络性能的学习曲线，参数量大致匹配。
**交付物：** 用几句话讨论你的发现。

#### 原文给定的公式、代码或关联条件

**原文给定的实验定义（PDF p. 43）：**

$$\operatorname{FFN}_{\mathrm{SiLU}}(x)=W_2\operatorname{SiLU}(W_1x). \tag{29}$$

默认 SwiGLU 使用三组权重矩阵，$d_{\mathrm{ff}}\approx\frac83d_{\mathrm{model}}$ 且为 64 的倍数；本题无 GLU 的 SiLU 对照使用两组权重矩阵，并设 $d_{\mathrm{ff}}=4d_{\mathrm{model}}$，使参数量近似匹配。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

两种模型参数量与 d_ff：<!-- 待填写 -->

SwiGLU / SiLU 比较曲线：<!-- 待填写 -->

发现简评：

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-main_experiment"></a>

### 37. 在 OpenWebText 上训练模型（2 分）

**原题：Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)**  
**来源：§7.4，PDF 第 45 页。**

#### 英文题目要求与交付物

Train your language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does this model do?

**Deliverable:** A learning curve of your language model on OpenWebText. Describe the difference in losses from TinyStories – how should we interpret these losses?

**Deliverable:** Generated text from OpenWebText LM, in the same format as the TinyStories outputs. How is the fluency of this text? Why is the output quality worse even though we have the same model and compute budget as TinyStories?

#### 中文翻译
用与 TinyStories 相同的模型架构和总训练迭代次数，在 OpenWebText 上训练你的语言模型。这个模型表现如何？
**交付物：** 你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 的损失差异——我们应如何解读这些损失？
**交付物：** OpenWebText 语言模型生成的文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？为什么即使我们拥有与 TinyStories 相同的模型和计算预算，输出质量却更差？

#### 原文给定的公式、代码或关联条件

**原文补充（PDF pp. 44–45）：** 本实验可能需要重新调整学习率或 batch size。低资源提示：GPU 资源有限的线上学生可继续在 TinyStories 上测试改动，以 validation loss 衡量效果；原文说明 OpenWebText 训练至流畅需要较长时间。生成输出沿用 `generate` 的格式。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

OpenWebText 学习曲线：<!-- 待填写 -->

与 TinyStories 的 loss 差异及解释：

<!-- 待填写 -->

Checkpoint、prompt、temperature、top-p：<!-- 待填写 -->

生成 token 数 / 是否遇到首个 `<|endoftext|>`：<!-- 待填写 -->

生成文本（沿用 TinyStories 输出格式）：

```text
（待粘贴自己的生成文本）
```

流畅度、同模型与计算预算下输出质量差异的解释：

<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

<a id="problem-leaderboard"></a>

### 38. 自选改进与排行榜（6 分）

**原题：Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)**  
**来源：§7.5，PDF 第 45–46 页。**

#### 英文题目要求与交付物

You will train a model under the leaderboard rules above with the goal of minimizing the validation loss of your language model within 0.75 B200-hours.

**Deliverable:** The final validation loss that was recorded, an associated learning curve that clearly shows a wall-clock-time x-axis that is less than 45 minutes, and a description of what you did. We expect a leaderboard submission to beat at least the naive baseline of a 5.0 loss. Submit to the leaderboard here: github.com/stanford-cs336/assignment1-basics-leaderboard.

#### 中文翻译
你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200-hours 内最小化你的语言模型的验证损失。
**交付物：** 记录的最终验证损失、一条明确显示横轴为实际时间且少于 45 分钟的学习曲线，以及对你所做工作的描述。我们期望排行榜提交至少优于 5.0 损失的朴素基线。在此提交排行榜：github.com/stanford-cs336/assignment1-basics-leaderboard。

#### 原文给定的公式、代码或关联条件

**原文排行榜规则（PDF p. 45）：**

- **Runtime:** Your submission can run for at most 45 minutes on a B200. You might want to enforce this in your submission script if you use either SLURM or Modal.
- **Data:** You may only use the OpenWebText training dataset that we provide.
- Otherwise, you are free to do whatever your heart desires.

原文建议先在 OpenWebText 小子集或 TinyStories 上测试，再进行完整的 45 分钟运行。题名的 **10 B200 hrs** 是本题实验预算标注；单次排行榜训练限制仍为 **0.75 B200-hours / 45 minutes**。提交地址：[assignment1-basics-leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard)。

#### 答题区域（待填写）

**实现思路：**

<!-- 待填写：由自己说明设计、数据结构、边界处理或实验实现 -->

**代码文件与入口：**

| 文件路径 | 类 / 函数 / 脚本入口 | 说明 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

**测试结果：**

| 测试命令 / 检查方法 | 结果 | 日志 / 备注 |
| --- | --- | --- |
| 待填写 | 待运行 | 待填写 |

**实验配置与产物：**

- 数据集 / 划分 / 样本范围：<!-- 待填写 -->
- 硬件、运行环境、随机种子：<!-- 待填写 -->
- 关键超参数与运行命令：<!-- 待填写 -->
- 运行时间、内存 / 显存：<!-- 待填写 -->
- 曲线、日志或输出文件：<!-- 待填写 -->

采用的改动和训练设置：

<!-- 待填写 -->

最终验证 loss：<!-- 待填写 -->

B200 实际运行时间：<!-- 待填写 -->

横轴为实际时间且少于 45 分钟的学习曲线：<!-- 待填写 -->

训练代码 / 配置 / checkpoint：<!-- 待填写 -->

排行榜提交链接：<!-- 待填写 -->

- [ ] 已按本题原文要求完成全部交付物。

---

## 最终提交核对

- [ ] 38 道 Problem 均已完成；所有字母子问逐项作答。
- [ ] 书面回答满足原文句数、图表、公式、生成文本长度等要求。
- [ ] 编程题代码可通过各题列出的测试，adapters 已接入。
- [ ] 实验日志包含所有尝试，曲线能对应配置、步数与实际时间。
- [ ] `writeup.pdf` 已排版，`code.zip` 由原文提交脚本生成。
- [ ] 如提交排行榜，附最终验证 loss、少于 45 分钟的实际时间曲线和提交链接。
