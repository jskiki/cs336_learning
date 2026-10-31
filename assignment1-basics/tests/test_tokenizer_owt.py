"""用已训练的 OWT 分词器检查文本往返，并测量编码速度和 bytes/token。

默认读取验证集前 100 万个字符；设置 OWT_MAX_CHARS=0 可测试全量。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import pytest

from cs336_basics.tokenizer import Tokenizer


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = DATA_DIR / "owt_bpe_profiled.pkl"
VALID_PATH = DATA_DIR / "owt_valid.txt"
SPECIAL_TOKEN = "<|endoftext|>"


@pytest.fixture(scope="module")
def owt_tokenizer() -> Tokenizer:
    assert MODEL_PATH.is_file(), f"找不到训练好的分词器：{MODEL_PATH}"
    return Tokenizer.from_files(
        vocab_filepath=str(MODEL_PATH),
        merges_filepath=str(MODEL_PATH),
        special_tokens=[SPECIAL_TOKEN],
    )


def _documents(path: Path, max_chars: int) -> Iterator[str]:
    """分块读取，保留文档末尾的特殊标记；未读完的文档留到下一块。

    max_chars=0 表示读完整个文件。newline="" 保留文件原始换行符。
    """
    pending = ""
    chars_read = 0
    with path.open("r", encoding="utf-8", newline="") as source:
        while max_chars == 0 or chars_read < max_chars:
            read_size = 65536 if max_chars == 0 else min(65536, max_chars - chars_read)
            chunk = source.read(read_size)
            if not chunk:
                break
            chars_read += len(chunk)
            parts = (pending + chunk).split(SPECIAL_TOKEN)
            pending = parts.pop()
            for part in parts:
                yield part + SPECIAL_TOKEN

    if pending:
        yield pending


def test_owt_model_roundtrip(owt_tokenizer: Tokenizer) -> None:
    """检查自己的模型文件能加载，且能处理 Unicode 和特殊标记。"""
    text = f"Hello, OWT! 中文与 emoji 🙃\n{SPECIAL_TOKEN}{SPECIAL_TOKEN}下一篇文章。"
    ids = owt_tokenizer.encode(text)
    assert owt_tokenizer.decode(ids) == text
    special_id = owt_tokenizer.token_to_id[SPECIAL_TOKEN.encode("utf-8")]
    assert ids.count(special_id) == 2


def test_owt_validation_roundtrip_and_metrics(owt_tokenizer: Tokenizer) -> None:
    """逐文档验证；耗时只统计 encode/decode，不包含文件读取和模型加载。"""
    assert VALID_PATH.is_file(), f"找不到 OWT 验证集：{VALID_PATH}"
    max_chars = int(os.environ.get("OWT_MAX_CHARS", "1000000"))
    assert max_chars >= 0, "OWT_MAX_CHARS 必须是非负整数，0 表示全量"

    total_chars = total_bytes = total_tokens = segments = 0
    encode_seconds = decode_seconds = 0.0

    for text in _documents(VALID_PATH, max_chars):
        segments += 1
        start = perf_counter()
        ids = owt_tokenizer.encode(text)
        encode_seconds += perf_counter() - start

        start = perf_counter()
        decoded = owt_tokenizer.decode(ids)
        decode_seconds += perf_counter() - start
        assert decoded == text, f"第 {segments} 个文档片段编码解码后与原文不一致"

        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)

    assert total_tokens > 0, "未产生任何 token，请检查验证集是否为空"
    scope = "完整验证集" if max_chars == 0 else f"验证集前最多 {max_chars:,} 个字符"
    print(f"\n测试范围：{scope}")
    print(f"文档片段数：{segments:,}（采样结尾可能是不完整文档）")
    print(f"字符数：{total_chars:,}；UTF-8 字节数：{total_bytes:,}")
    print(f"Token 数：{total_tokens:,}")
    print(f"编码耗时：{encode_seconds:.3f} 秒；解码耗时：{decode_seconds:.3f} 秒")
    print(f"编码速度：{total_bytes / encode_seconds / 1_000_000:.3f} MB/s")
    print(f"Token 速度：{total_tokens / encode_seconds:,.0f} tokens/s")
    print(f"平均字节数：{total_bytes / total_tokens:.3f} bytes/token")
