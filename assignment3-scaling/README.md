# CS336 Spring 2026 Assignment 3: Scaling

For a full description of the assignment, see the assignment handout at
[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Local experiments

Section 2.1 uses `data/isoflops_curves.json`. For local PyTorch experiments,
the model package is `cs336_scaling.torch_model`. Run it in an environment with
PyTorch installed. The upstream JAX code under `cs336_scaling/training/` is
kept as a reference and its worker relies on the original course service.

## DCLM preprocessing

Install dependencies:

```sh
uv sync --extra server
```

To download tokenized data:

```sh
uv run modal run scripts/1_download_tokenized_data.py
```
