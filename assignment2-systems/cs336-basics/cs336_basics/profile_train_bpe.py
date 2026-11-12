"""带阶段耗时、进度和内存统计的 BPE 训练。

运行：uv run -m cs336_basics.profile_train_bpe
"""

import re
import regex
from collections import Counter,defaultdict
from pathlib import Path
from time import perf_counter, process_time

import psutil

from cs336_basics.pretokenization_example import find_chunk_boundaries


def _record_memory(stats: dict, process: psutil.Process) -> int:
    """记录进程驻留内存；Windows 的峰值工作集覆盖本进程整个生命周期。"""
    memory = process.memory_info()
    stats["rss_bytes"] = memory.rss
    stats["peak_sampled_rss_bytes"] = max(
        stats.get("peak_sampled_rss_bytes", 0), memory.rss
    )
    if hasattr(memory, "peak_wset"):
        stats["process_peak_rss_bytes"] = memory.peak_wset
    return memory.rss


def preokenization(
    filepath:str = "tests/fixtures/tinystories_sample.txt",
    num_processes:int = 4,
    split_special_token:str = "<|endoftext|>",
    stats: dict | None = None,
):
    split_bytes = split_special_token.encode("utf-8")
    boundary_started = perf_counter() if stats is not None else 0.0
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_bytes)
        if stats is not None:
            stats["timings_seconds"]["chunk_boundaries"] += perf_counter() - boundary_started
            stats["chunks_total"] = len(boundaries) - 1
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            read_started = perf_counter() if stats is not None else 0.0
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            if stats is not None:
                stats["timings_seconds"]["read_decode"] += perf_counter() - read_started
                stats["bytes_read"] += f.tell() - start
            yield chunk


def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str],
    *,
    stats: dict | None = None,
    verbose: bool = False,
):
    """训练 BPE；传入 stats 可收集性能统计，verbose=True 可显示进度。"""
    if verbose and stats is None:
        stats = {}
    profiling = stats is not None
    started = perf_counter() if profiling else 0.0
    cpu_started = process_time() if profiling else 0.0
    process = psutil.Process() if profiling else None
    if profiling:
        stats.update({
            "input_path": str(Path(input_path).resolve()),
            "input_bytes": Path(input_path).stat().st_size,
            "target_vocab_size": vocab_size,
            "special_tokens": list(special_tokens),
            "requested_chunks": 128,
            "chunks_processed": 0,
            "bytes_read": 0,
            "timings_seconds": {
                "setup": 0.0,
                "pretokenization_total": 0.0,
                "chunk_boundaries": 0.0,
                "read_decode": 0.0,
                "pretokenize_count": 0.0,
                "word_sequences": 0.0,
                "index_build": 0.0,
                "merging_total": 0.0,
                "pair_selection": 0.0,
                "merge_update": 0.0,
            },
        })

    def progress(message: str):
        if not profiling:
            return
        rss = _record_memory(stats, process)
        if verbose:
            print(
                f"[{perf_counter() - started:.1f}s] {message} | "
                f"RSS={rss / 1024 ** 3:.2f} GiB",
                flush=True,
            )

    progress("初始化词表和正则")
    vocab: dict[int,bytes] = {
        i:bytes([i]) for i in range(256)
    }
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    #GPT-2预分词规则
    pretoken_pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pattern = "|".join(
        re.escape(token)
        for token in sorted(special_tokens,key=len,reverse=True)
    )

    #创建计数器
    pretoken_counts:Counter[bytes] = Counter()

    if profiling:
        stats["timings_seconds"]["setup"] = perf_counter() - started
    pretoken_started = perf_counter() if profiling else 0.0
    last_progress = pretoken_started
    progress("开始读取语料并预分词")
    for chunk_number, chunk in enumerate(
        preokenization(filepath=input_path, num_processes=128, stats=stats), 1
    ):
        chunk_started = perf_counter() if profiling else 0.0
        if special_tokens:
            parts = re.split(pattern,chunk)
        else:
            parts = [chunk]

        for part in parts:
            for match in pretoken_pattern.finditer(part):
                token_bytes = match.group().encode("utf-8")
                pretoken_counts[token_bytes] +=1

        if profiling:
            now = perf_counter()
            stats["timings_seconds"]["pretokenize_count"] += now - chunk_started
            stats["chunks_processed"] = chunk_number
            if now - last_progress >= 10 or chunk_number == stats["chunks_total"]:
                percent = 100 * stats["bytes_read"] / max(stats["input_bytes"], 1)
                speed = stats["bytes_read"] / 1024 ** 2 / max(now - pretoken_started, 1e-9)
                progress(
                    f"预分词 {chunk_number}/{stats['chunks_total']} 块，"
                    f"读取 {percent:.1f}%，{speed:.2f} MiB/s，"
                    f"不同单元 {len(pretoken_counts):,} 个"
                )
                last_progress = now

    if profiling:
        stats["timings_seconds"]["pretokenization_total"] = perf_counter() - pretoken_started
        stats["distinct_pretokens"] = len(pretoken_counts)
        stats["pretoken_occurrences"] = sum(pretoken_counts.values())
    progress("预分词完成，开始转换字节序列")
    sequences_started = perf_counter() if profiling else 0.0
    #把pretoken_counts = {b"hello": 3,b"ab": 2,}变成word_counts = {(b"h", b"e", b"l", b"l", b"o"): 3,(b"a", b"b"): 2,}
    word_conuts = {
        tuple(bytes([byte]) for byte in token_bytes):count
        for token_bytes,count in pretoken_counts.items()
    }

    if profiling:
        stats["timings_seconds"]["word_sequences"] = perf_counter() - sequences_started
    progress("字节序列转换完成，开始建立相邻对索引")
    index_started = perf_counter() if profiling else 0.0
    # 单元编号 → 当前 token 序列
    word_tokens: dict[int, tuple[bytes, ...]] = {}
    # 单元编号 → 该单元在语料中的出现次数
    word_freqs: dict[int, int] = {}
    # 相邻对 → 包含这个对的单元编号集合
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    #全局计数器
    pair_counts:Counter[tuple[bytes,bytes]] = Counter()

    for word_id, (tokens, count) in enumerate(word_conuts.items()):
        word_tokens[word_id] = tokens
        word_freqs[word_id] = count

        for pair in zip(tokens[:-1],tokens[1:]):
            pair_to_words[pair].add(word_id)
            pair_counts[pair] += count

    if profiling:
        stats["timings_seconds"]["index_build"] = perf_counter() - index_started
        stats["initial_distinct_pairs"] = len(pair_counts)
    progress(f"索引建立完成，相邻对 {len(pair_counts):,} 种，开始合并")
    merges:list[tuple[bytes,bytes]] = []

    merging_started = perf_counter() if profiling else 0.0
    last_progress = merging_started
    target_merges = max(vocab_size - len(vocab), 0)
    while len(vocab) < vocab_size:

        if not pair_counts:
            break

        selection_started = perf_counter() if profiling else 0.0
        #优先选择频次最高的对，同频时选择字典序最大的
        best_pair = max(
            pair_counts,
            key = lambda pair:(pair_counts[pair],pair),
        )

        if profiling:
            update_started = perf_counter()
            stats["timings_seconds"]["pair_selection"] += update_started - selection_started
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        affected_word_ids = list(pair_to_words[best_pair])

        for word_id in affected_word_ids:
            tokens = word_tokens[word_id]
            count = word_freqs[word_id]

            #合并前，各相邻对在当前单元中出现的次数
            old_counts = Counter(zip(tokens[:-1],tokens[1:]))
            old_pairs = set(old_counts)

            #从左到右执行不重叠合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_tokens = tuple(new_tokens)

            #合并后，各相邻对在当前单元中出现的次数
            new_counts = Counter(zip(new_tokens[:-1],new_tokens[1:]))
            new_pairs = set(new_counts)

            #根据当前单元变化，修正全局频次。（|表并集）
            for pair in old_pairs|new_pairs:
                delta = new_counts[pair] - old_counts[pair]

                if delta:
                    pair_counts[pair] += delta*count

                    if pair_counts[pair] == 0:
                        del pair_counts[pair]

            #从已消失的相邻对中移除该单元编号，old_pair - new_pairs为合并后消失的相邻对
            for pair in old_pairs - new_pairs:
                pair_to_words[pair].discard(word_id)

                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            #给新增的相邻对登记该单元编号
            for pair in new_pairs - old_pairs:
                pair_to_words[pair].add(word_id)

            word_tokens[word_id] = new_tokens

        if profiling:
            now = perf_counter()
            stats["timings_seconds"]["merge_update"] += now - update_started
            stats["merges_completed"] = len(merges)
            if now - last_progress >= 10 or len(merges) % 100 == 0:
                progress(
                    f"合并 {len(merges):,}/{target_merges:,}，词表 {len(vocab):,}，"
                    f"本轮影响 {len(affected_word_ids):,} 个单元，"
                    f"活跃相邻对 {len(pair_counts):,} 种"
                )
                last_progress = now

    if profiling:
        timings = stats["timings_seconds"]
        timings["merging_total"] = perf_counter() - merging_started
        stats["training_wall_seconds"] = perf_counter() - started
        stats["training_cpu_seconds"] = process_time() - cpu_started
        stats["cpu_percent_one_core"] = (
            100 * stats["training_cpu_seconds"] / max(stats["training_wall_seconds"], 1e-9)
        )
        stats["vocab_size"] = len(vocab)
        stats["merges_completed"] = len(merges)
        stats["final_distinct_pairs"] = len(pair_counts)
        stats["pretokenization_mib_per_second"] = (
            stats["bytes_read"] / 1024 ** 2 / max(timings["pretokenization_total"], 1e-9)
        )
        longest_token = max(vocab.values(), key=len)
        stats["longest_token_repr"] = repr(longest_token)
        stats["longest_token_bytes"] = len(longest_token)
        progress("BPE 训练完成")
    return vocab,merges


# if __name__ == "__main__":
#     import pickle

#     print("开始训练 BPE……", flush=True)

#     vocab, merges = train_bpe(
#         input_path="data/TinyStoriesV2-GPT4-train.txt",
#         vocab_size=10000,
#         special_tokens=["<|endoftext|>"],
#     )

#     # 保存结果，bytes 类型可以直接保存
#     with open("data/tinystories_bpe.pkl", "wb") as f:
#         pickle.dump({"vocab": vocab, "merges": merges}, f)

#     print(f"训练完成：词表 {len(vocab)} 项，合并 {len(merges)} 次")


if __name__ == "__main__":
    import json
    import pickle

    job_started = perf_counter()
    job_cpu_started = process_time()
    stats = {"status": "running"}
    output_path = Path("data/owt_bpe_profiled.pkl")
    report_path = output_path.with_suffix(".stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("开始训练 OpenWebText BPE……", flush=True)

    try:
        vocab, merges = train_bpe(
            input_path="data/owt_train.txt",
            vocab_size=32000,
            special_tokens=["<|endoftext|>"],
            stats=stats,
            verbose=True,
        )

        print("开始保存词表和合并规则……", flush=True)
        save_started = perf_counter()
        with output_path.open("wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges}, f)
        stats["timings_seconds"]["serialization"] = perf_counter() - save_started
        stats["output_path"] = str(output_path.resolve())
        stats["output_bytes"] = output_path.stat().st_size
        stats["status"] = "completed"
    except BaseException as error:
        stats["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        stats["error_type"] = type(error).__name__
        stats["error_message"] = str(error)
        raise
    finally:
        stats["job_wall_seconds_excluding_report"] = perf_counter() - job_started
        stats["job_cpu_seconds"] = process_time() - job_cpu_started
        _record_memory(stats, psutil.Process())
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计摘要已保存：{report_path}", flush=True)

    print(f"训练完成：词表 {len(vocab)} 项，合并 {len(merges)} 次")
    for name, seconds in stats["timings_seconds"].items():
        print(f"  {name}: {seconds:.3f} 秒")
    peak = stats.get("process_peak_rss_bytes", stats["peak_sampled_rss_bytes"])
    peak_label = "进程生命周期峰值工作集" if "process_peak_rss_bytes" in stats else "检查点采样的最大 RSS（下限）"
    print(f"训练墙钟时间：{stats['training_wall_seconds']:.3f} 秒")
    print(f"训练 CPU 时间：{stats['training_cpu_seconds']:.3f} 秒")
    print(f"{peak_label}：{peak / 1024 ** 3:.2f} GiB")
    print("说明：预分词总时间包含边界查找、读取解码和预分词计数；合并总时间包含选对和更新。")
