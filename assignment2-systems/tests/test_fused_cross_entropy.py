import pytest
import torch
from torch.nn import functional as F

from cs336_systems.fused_cross_entropy import fused_linear_cross_entropy


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for mixed-dtype addmm"),
        ),
    ],
)
@pytest.mark.parametrize("chunk_size", [16, 64])
def test_weight_gradient_accumulation_matches_dense(device, dtype, chunk_size):
    if device == "cuda":
        pytest.importorskip("triton")
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 is required")

    torch.manual_seed(0)
    hidden = torch.randn(37, 64, device=device, dtype=dtype, requires_grad=True)
    weight = (torch.randn(257, 64, device=device, dtype=dtype) * 0.1).requires_grad_()
    targets = torch.randint(257, (37,), device=device)
    actual = fused_linear_cross_entropy(hidden, weight, targets, chunk_size)
    expected = F.cross_entropy((hidden @ weight.T).float(), targets)
    loss_rtol, loss_atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-4, 5e-4)
    torch.testing.assert_close(actual, expected, rtol=loss_rtol, atol=loss_atol)

    actual_gradients = torch.autograd.grad(actual, (hidden, weight))
    expected_gradients = torch.autograd.grad(expected, (hidden, weight))
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (3e-2, 1e-3)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)
