"""字节级 BPE 分词器的接口框架。

完成 TODO 后，在 tests/adapters.py 的 get_tokenizer 中构造并返回此类的实例，
再运行：uv run pytest tests/test_tokenizer.py
"""
from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator

import regex


class Tokenizer:
    """使用已经训练好的词表和合并规则，完成文本编码与解码。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """构造分词器，并准备编码和解码所需的数据。

        参数：
            vocab: token ID 到字节串的映射。
            merges: 按训练时产生的先后顺序排列的合并规则。
            special_tokens: 必须作为完整 token 处理的特殊标记字符串。
                如果尚未存在于词表中，需要追加到词表。
        """
        # 复制传入的数据，避免追加特殊标记时修改调用方的词表。
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = (
            list(special_tokens) if special_tokens is not None else []
        )

        # 编码时使用的反向词表：token 字节串 → token ID。
        self.token_to_id: dict[bytes, int] = {}

        # 合并优先级：相邻 token 对 → 在 merges 中的位置。
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}

        # TODO：处理特殊标记，保留已有 ID，只为缺失的标记分配新 ID。
        if self.special_tokens:
            existing_tokens = set(self.vocab.values())
            next_id = max(self.vocab,default=-1)+1 #self.vocab遍历字典，遍历的是key

            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing_tokens:
                    self.vocab[next_id] = token_bytes
                    existing_tokens.add(token_bytes)
                    next_id +=1
        # TODO：补充反向词表、合并优先级及预分词所需的规则。
        self.token_to_id = {
            token_bytes:token_id
            for token_id,token_bytes in self.vocab.items()
        }
        self.merge_ranks = {
            pair:rank for rank,pair in enumerate(self.merges)
        }

        self.pretoken_pattern = regex.compile(
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )
        self.special_pattern = (
            regex.compile(
                "("
                + "|".join(
                    regex.escape(token)
                    for token in sorted(
                        set(self.special_tokens), key=len, reverse=True
                    )
                )
                + ")"
            ) #加（）是捕获组
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """从序列化文件加载词表、合并规则，并返回分词器实例。

        参数：
            vocab_filepath: 保存词表的文件路径。
            merges_filepath: 保存合并规则的文件路径。
            special_tokens: 可选的特殊标记列表。

        当前训练脚本使用 pickle，将 vocab 和 merges 保存在同一个字典中。
        实现时需与该保存格式对应；可以约定两个参数传入同一个文件路径，
        也可以先将词表和合并规则分别导出为文件。
        """
        # TODO：按训练输出的格式读取数据。
        #现在的写法是vocab和merges保存在同一个文件
        with open(vocab_filepath,"rb") as f:
            data = pickle.load(f)
        # TODO：使用 cls 构造实例，并将 special_tokens 传入。
        return cls(
            vocab = data["vocab"],
            merges = data["merges"],
            special_tokens=special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        """将输入文本编码为 token ID 列表。

        特殊标记需要保留为完整 token，普通文本采用与训练时一致的预分词规则。
        """
        # TODO：区分特殊标记与普通文本，处理特殊标记有共同前缀的情况。
        # TODO：对普通文本预分词，转换为 UTF-8 字节并应用已有的合并规则。
        # TODO：将得到的 token 字节串映射为 ID，按输入顺序返回。
        from heapq import heapify, heappop, heappush
        
        if not text:
            return []
        token_to_id = self.token_to_id
        rank_get = self.merge_ranks.get
        finditer = self.pretoken_pattern.finditer #迭代器
        #提前准备256种单字节
        byte_tokens = tuple(bytes((i,)) for i in range(256))
        #缓存“预分词片段->编码结果”
        cache: dict[str,tuple[int,...]] = {}
        result: list[int] = []

        parts = (
            self.special_pattern.split(text)
            if self.special_pattern is not None
            else [text]
        )

        for part_index,part in enumerate(parts):
            #处理特殊标记，特殊标记只会出现在基数位置，因split还会保留空字符串，正常文本段只会出现子在偶数字段
            if part_index & 1:
                result.append(token_to_id[part.encode("utf-8")])
                continue
            for match in finditer(part):
                #取整个预分词片段，如果片段里还有捕获组用group(1)、group(2)...取出
                word = match.group()
                #去缓存看一下
                cached = cache.get(word)
                if cached is not None:
                    result.extend(cached)
                    continue
                tokens = [byte_tokens[b] for b in word.encode("utf-8")]
                size = len(tokens)
                #previous和following这两个数组用来模拟双向链表
                #每个节点前一个有效索引节点
                previous = list(range(-1,size - 1))
                #每个节点后一个有效索引节点
                following = list(range(1,size)) + [-1]
                #判断候选是否因节点变化而失效
                versions = [0]*size
                #候选
                heap = []

                for left in range(size-1):
                    right = left + 1
                    rank = rank_get((tokens[left],tokens[right]))

                    if rank is not None:
                        #按照rank建堆
                        heap.append((rank,left,right,0,0))

                heapify(heap)

                while heap:
                    _,left,right,left_version,right_version = heappop(heap)
                    #成立则说明是在旧状态下生成的，已经失效
                    if(
                        versions[left] != left_version
                        or versions[right] != right_version
                    ):
                        continue
                    #合并两个结果并把结果放到左节点
                    tokens[left] += tokens[right]
                    #左节点内容改变，旧候选失效
                    versions[left] += 1
                    #右节点被移除，设计它的候选也失效
                    versions[right] += 1
                    #找到右节点后面的节点
                    successor = following[right]
                    #跳过右节点
                    following[left] = successor
                    #如果successor存在，把它的前驱改成左节点
                    if successor != -1:
                        previous[successor] = left
                    #分别检查：1.左节点的前驱->左节点，2.左节点->它的新后继
                    for start in (previous[left],left):
                        #没有前驱时跳过第一个候选
                        if start == -1:
                            continue
                        end = following[start]
                        #没有后继也跳过
                        if end == -1:
                            continue
                        rank = rank_get((tokens[start],tokens[end]))
                        if rank is not None:
                            heappush(heap,(rank,start,end,versions[start],versions[end]),)
                            #这并不会主动删除堆中过期的节点，只有在弹出时才会删除，这叫惰性删除
                word_ids = []
                index = 0
                while index != -1:
                    word_ids.append(token_to_id[tokens[index]])
                    index = following[index]

                encoded = tuple(word_ids)
                cache[word] = encoded

                result.extend(encoded)
        return result


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """按需读取字符串可迭代对象，惰性产出 token ID。

        例如可以接收以文本模式打开的文件句柄。
        应避免一次性读取或拼接整个输入，以便处理大型文件。
        """
        # TODO：逐步读取输入，按需产出 ID，避免累计所有片段或全部编码结果。
        encode = self.encode
        for text in iterable:
            yield from encode(text)

    def decode(self, ids: list[int]) -> str:
        """将 token ID 列表解码为文本。

        先拼接所有 token 的字节内容，再整体进行 UTF-8 解码。
        对畸形 UTF-8 字节序列使用 U+FFFD（替换字符）处理。
        """
        # TODO：按顺序查词表并拼接字节，不要逐个 token 单独解码。
        # TODO：整体解码时使用 errors="replace"。
        vocab = self.vocab
        return b"".join(vocab[token_id] for token_id in ids).decode("utf-8",errors="replace")

