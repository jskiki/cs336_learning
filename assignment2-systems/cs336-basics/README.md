# CS336 Assignment 1: Local Copy for Assignment 2

`cs336_basics/` and `pyproject.toml` were copied from
`../../assignment1-basics/`. This copy is independent of the original assignment
1 directory; changes there are not automatically synchronized here.

Use `cs336_basics.transformer_lm.TransformerLM` for our language model and
`cs336_basics.adamw.AdamW` for our optimizer. Assignment 2 installs this directory
as an editable dependency, so edits to this copy take effect locally.

`cs336_basics.model` is a compatibility import module that re-exports our
`Embedding`, `Linear`, `RMSNorm`, and `TransformerLM`. The provided FSDP tests
therefore use our layers without changes to the original test files.

The original staff package is stored separately in
`../reference/cs336-basics/` for reading and comparison. It is not installed or
used by the tests. Our optimizer is available in `cs336_basics.adamw`;
the staff `optimizer.py` and `nn_utils.py` are only in the reference directory.

Virtual environments, caches, datasets, and training outputs were not copied.
