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

    for chunk in preokenization(filepath=input_path):
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

    merges:list[tuple[bytes,bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts:Counter[tuple[bytes,bytes]] = Counter()
        for tokens,count in word_conuts.items():
            for pair in zip(tokens[:-1],tokens[1:]):
                pair_counts[pair] +=count

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

        #保存本轮合并后的token序列及其出现次数
        updated_counts:Counter[tuple[bytes,...]] = Counter()

        for tokens,count in word_conuts.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if(
                    i+1 < len(tokens)
                    and (tokens[i],tokens[i+1]) == best_pair
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            updated_counts[tuple(new_tokens)] += count

        word_conuts = updated_counts
    return vocab,merges


