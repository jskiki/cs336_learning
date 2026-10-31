"""uv run pytest tests/test_train_bpe.py -v"""

import re
import regex
from collections import Counter,defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


def preokenization(
    filepath:str = "tests/fixtures/tinystories_sample.txt",
    num_processes:int = 4,
    split_special_token:str = "<|endoftext|>",
):
    split_bytes = split_special_token.encode("utf-8")
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_bytes)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk


def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str],
):
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

    for chunk in preokenization(filepath=input_path, num_processes=128):
        if special_tokens:
            parts = re.split(pattern,chunk)
        else:
            parts = [chunk]

        for part in parts:
            for match in pretoken_pattern.finditer(part):
                token_bytes = match.group().encode("utf-8")
                pretoken_counts[token_bytes] +=1

    #把pretoken_counts = {b"hello": 3,b"ab": 2,}变成word_counts = {(b"h", b"e", b"l", b"l", b"o"): 3,(b"a", b"b"): 2,}
    word_conuts = {
        tuple(bytes([byte]) for byte in token_bytes):count
        for token_bytes,count in pretoken_counts.items()
    }

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

    
    merges:list[tuple[bytes,bytes]] = []

    while len(vocab) < vocab_size:

        if not pair_counts:
            break

        #优先选择频次最高的对，同频时选择字典序最大的
        best_pair = max(
            pair_counts,
            key = lambda pair:(pair_counts[pair],pair),
        )

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
    import pickle

    print("开始训练 OpenWebText BPE……", flush=True)

    vocab, merges = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    with open("data/owt_bpe.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    print(f"训练完成：词表 {len(vocab)} 项，合并 {len(merges)} 次")
