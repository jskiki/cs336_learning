from io import BytesIO

import pytest
import torch

from cs336_basics.checkpoint import save_checkpoint


@pytest.mark.parametrize("use_buffer", [False, True])
def test_save_checkpoint(tmp_path, use_buffer):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    iteration = 42

    out = BytesIO() if use_buffer else tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, iteration, out)

    if use_buffer:
        out.seek(0)

    saved = torch.load(out, weights_only=True)

    assert set(saved) == {"model_state", "opt_state", "iteration"}
    assert saved["iteration"] == iteration

    torch.testing.assert_close(
        saved["model_state"],
        model.state_dict(),
    )
    assert saved["opt_state"]["param_groups"] == optimizer.state_dict()["param_groups"]