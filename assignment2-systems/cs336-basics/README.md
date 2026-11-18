# CS336 Assignment 1: Local Copy for Assignment 2

`cs336_basics/` and `pyproject.toml` were copied from
`../../assignment1-basics/`. This copy is independent of the original assignment
1 directory; changes there are not automatically synchronized here.

Use `cs336_basics.transformer_lm.TransformerLM` for our language model and
`cs336_basics.optim.AdamW` for our optimizer. Assignment 2 installs this directory
as an editable dependency, so edits to this copy take effect locally.

`cs336_basics.layers` defines `Embedding`, `Linear`, and `RMSNorm`.
`cs336_basics.model` re-exports these layers and `TransformerLM` for the
provided FSDP tests.

`cs336_basics.optim` also contains gradient clipping and learning-rate scheduling.
`cs336_basics.generate` contains decoding and the `python -m cs336_basics.generate`
entry point for running a saved model.

`cs336_basics.train_bpe` is the entry point for the incremental BPE trainer.
From the directory containing `data/TinyStoriesV2-GPT4-train.txt`, run
`uv run python -m cs336_basics.train_bpe` to write `data/tinystories_bpe.pkl`.

The original staff package is stored separately in
`../reference/cs336-basics/` for reading and comparison. It is not installed or
used by the tests. The staff `optimizer.py` and `nn_utils.py` are only in the
reference directory.

Virtual environments, caches, datasets, and training outputs were not copied.
