"""分块并行编码文本，输出无文件头的 uint16 token 数据。

在 assignment1-basics 目录运行：
uv run python -m cs336_basics.preprocess \
    --input_path data/TinyStoriesV2-GPT4-train.txt \
    --output_path data/tinystories_train.bin \
    --tokenizer_path data/tinystories_bpe.pkl
"""

from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


_tokenizer: Tokenizer


#初始化tokenizer，获取词表与合并规则，我们的词表与合并规则在同一个文件里，所以这里只需要传递一个路径，如果分开则此接口需要修改
def init_worker(tokenizer_path: str):
    global _tokenizer
    # 每个进程只加载一次词表和合并规则，后续分块复用同一个分词器。
    _tokenizer = Tokenizer.from_files(
        vocab_filepath=tokenizer_path,
        merges_filepath=tokenizer_path,
        special_tokens=["<|endoftext|>"],
    )
    if max(_tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("token ID 超出 uint16 范围，不能保存为当前格式")

#真正对数据进行处理的函数，防止文件过大，只读取start到end处的数据，由process控制流程
def encode_chunk(input_path: str, start: int, end: int) -> np.ndarray:
    # 只向进程传递文件位置，避免主进程读取、复制和传输大段原始文本。
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return np.asarray(_tokenizer.encode(text), dtype=np.uint16)

#workers最多同时使用的线程数，每个线程处理的数据大小，并不精确会根据<|endoftext|>调整
def preprocess(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    workers: int = 4,
    chunk_mb: int = 4,
) -> int:
    #记录当前时间，用于计算耗时
    started = perf_counter()
    #获取输入文件的大小，单位是字节
    file_size = Path(input_path).stat().st_size
    chunk_bytes = chunk_mb * 1024 * 1024
    #向上取整计算需要多少块
    num_chunks = max(1, (file_size + chunk_bytes - 1) // chunk_bytes)
    with open(input_path, "rb") as f:
        # 边界位于特殊标记之前，不拆开 UTF-8 字符或普通预分词片段。
        # 特殊标记仍保留在下一个块中，拼接编码结果与整篇编码保持一致。
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    #准备输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    #把相邻边界配成区间
    spans = iter(zip(boundaries[:-1], boundaries[1:]))
    token_count = 0
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),#指定通过 spawn 方式创建进程：启动新的 Python 解释器运行子进程。每个子进程拥有自己独立的内存和全局变量。
            initializer=init_worker, #指定每个工作进程启动时，先调用一次 init_worker()
            initargs=(tokenizer_path,),#指定初始化函数的参数，也就是把 tokenizer_path 传给 init_worker()
        ) as pool,
        open(output_path, "wb") as output,
        #创建进度条
        tqdm(total=file_size, unit="B", unit_scale=True, desc=Path(input_path).name) as progress,
    ):
        # 双端队列，保存已提交但尚未fiery结果的任务，最多保留两倍进程数的任务，使待写结果的内存占用不会随全文长度增长。
        #submit 会安排子进程执行 encode_chunk(input_path, start, end)，并立即返回 Future
        #Future 是一个代表“任务执行状态和将来结果”的对象。具体见note.md
        pending = deque(
            (pool.submit(encode_chunk, input_path, start, end), end - start)
            #从 spans 中最多取出 2 * workers 个区间，为每个区间提交一个任务
            for start, end in islice(spans, 2 * workers)
        )
        while pending:
            # 按提交顺序写入，即使后面的块先完成，也不会打乱原文顺序。
            future, source_bytes = pending.popleft()
            #取出处理后的数据
            token_ids = future.result()
            #写入输出文件
            token_ids.tofile(output)
            token_count += token_ids.size
            #推进进度条
            progress.update(source_bytes)

            #每写完一块，在从span中取出一块进行补交，如果span中没有内容就为None
            span = next(spans, None)
            if span is not None:
                start, end = span
                pending.append((pool.submit(encode_chunk, input_path, start, end), end - start))

    #计算总耗时
    elapsed = perf_counter() - started
    print(f"完成：{token_count:,} tokens，uint16，耗时 {elapsed:.1f}s，输出 {output_path}")
    return token_count


def main():
    parser = ArgumentParser(description="将文本分块编码为 uint16 token 二进制文件")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk_mb", type=int, default=4)
    args = parser.parse_args()
    #vars把对象属性转换成字典，**展开字典，将key作为参数名value作为参数值
    preprocess(**vars(args))


if __name__ == "__main__":
    main()
