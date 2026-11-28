"""Prepare and verify a small DCLM token dataset in a Modal CPU container."""

import json
from pathlib import Path

import modal

DATASET = "mlfoundations/dclm-baseline-1.0-parquet"
REVISION = "817d6752765f6a41261085171dd546b104f60626"
SHARD = (
    "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/"
    "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/"
    "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.parquet"
)
SPLIT_SEED = 67
DOCUMENT_LIMITS = {"train": 1000, "val": 100}
END_OF_TEXT = "<|endoftext|>"
VOLUME_ROOT = Path("/data")
OUTPUT_DIR = VOLUME_ROOT / "dclm-pilot"
TOKENIZER_PATH = Path("/root/owt_bpe_profiled.pkl")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub", "numpy", "pyarrow", "regex"
)
if modal.is_local():
    assignment1 = Path(__file__).resolve().parents[2] / "assignment1-basics"
    image = (
        image.add_local_file(assignment1 / "cs336_basics/tokenizer.py", "/root/tokenizer.py")
        .add_local_file(assignment1 / "data/owt_bpe_profiled.pkl", str(TOKENIZER_PATH))
    )
app = modal.App("cs336-a3-dclm-pilot")
volume = modal.Volume.from_name("cs336-a3-dclm")


def verify_dataset(path: Path, source: dict) -> dict:
    import numpy as np

    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["source"] != source:
        raise ValueError(f"Existing dataset at {path} was prepared with a different configuration")

    for split, expected_documents in DOCUMENT_LIMITS.items():
        split_path = path / f"{split}.bin"
        count = manifest["tokens"][split]
        if manifest["documents"][split] != expected_documents or count <= 0 or split_path.stat().st_size != count * 2:
            raise ValueError(f"Unexpected document or token count in {split_path}")
        token_ids = np.memmap(split_path, dtype="<u2", mode="r")
        if int(token_ids.max()) >= manifest["vocab_size"] or int(token_ids[-1]) != manifest["eos_token_id"]:
            raise ValueError(f"Invalid token IDs in {split_path}")
    return manifest


@app.function(image=image, volumes={str(VOLUME_ROOT): volume}, cpu=2, memory=2048, timeout=900)
def prepare_sample() -> dict:
    import hashlib
    import random
    from tempfile import TemporaryDirectory

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizer import Tokenizer

    source = {
        "dataset": DATASET,
        "revision": REVISION,
        "shard": SHARD,
        "split_seed": SPLIT_SEED,
        "tokenizer_sha256": hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
    }
    if OUTPUT_DIR.exists():
        return verify_dataset(OUTPUT_DIR, source)

    tokenizer = Tokenizer.from_files(TOKENIZER_PATH, TOKENIZER_PATH, special_tokens=[END_OF_TEXT])
    vocab_size = max(tokenizer.vocab) + 1
    if vocab_size > 2**16:
        raise ValueError("Tokenizer vocabulary exceeds the uint16 token format")
    (eos_token_id,) = tokenizer.encode(END_OF_TEXT)
    documents = dict.fromkeys(DOCUMENT_LIMITS, 0)
    token_counts = dict.fromkeys(DOCUMENT_LIMITS, 0)
    split_rng = random.Random(SPLIT_SEED)

    shard_path = hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=REVISION, token=False)
    parquet = pq.ParquetFile(shard_path)

    with TemporaryDirectory(prefix=".dclm-pilot-", dir=VOLUME_ROOT) as tmp:
        staging = Path(tmp) / "sample"
        staging.mkdir()
        with (staging / "train.bin").open("wb") as train, (staging / "val.bin").open("wb") as val:
            outputs = {"train": train, "val": val}
            for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    if not text or not text.strip():
                        continue
                    split = "val" if split_rng.random() < 0.1 else "train"
                    if documents[split] >= DOCUMENT_LIMITS[split]:
                        continue
                    token_ids = tokenizer.encode(text)
                    token_ids.append(eos_token_id)
                    np.asarray(token_ids, dtype="<u2").tofile(outputs[split])
                    documents[split] += 1
                    token_counts[split] += len(token_ids)
                    if documents == DOCUMENT_LIMITS:
                        break
                if documents == DOCUMENT_LIMITS:
                    break

        manifest = {
            "source": source,
            "dtype": "uint16-le",
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "documents": documents,
            "tokens": token_counts,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_dataset(staging, source)
        staging.rename(OUTPUT_DIR)

    volume.commit()
    return manifest


@app.local_entrypoint()
def main():
    print(json.dumps(prepare_sample.remote(), indent=2))
