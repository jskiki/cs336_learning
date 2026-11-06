"""CPU 单批次过拟合：uv run python -m cs336_basics.overfit_batch。

使用 TinyStories 样本文本和 data/tinystories_bpe.pkl 验证训练链路。
"""

from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import train_step
from cs336_basics.transformer_lm import TransformerLM


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    # 小矩阵计算使用单线程，避免线程调度开销盖过实际计算。
    torch.set_num_threads(1)

    batch_size = 2
    context_length = 16
    steps = 500
    log_interval = 25

    project_dir = Path(__file__).resolve().parents[1]
    tokenizer_path = project_dir / "data/tinystories_bpe.pkl"
    text_path = project_dir / "tests/fixtures/tinystories_sample.txt"
    # 训练结果将 vocab 和 merges 存在同一个 pickle 中，因此两个路径相同。
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(tokenizer_path),
        merges_filepath=str(tokenizer_path),
        special_tokens=["<|endoftext|>"],
    )
    vocab_size = len(tokenizer.vocab)
    # 只需一小段真实文本，不必为这个实验编码整个数据集。
    with text_path.open(encoding="utf-8") as f:
        text = f.read(8192)
    dataset = np.asarray(tokenizer.encode(text), dtype=np.int64)
    # 只采样一次。get_batch 将连续片段错开一位，构造下一 token 目标。
    inputs, targets = get_batch(dataset, batch_size, context_length, device="cpu")
    print(f"数据来源: {text_path.name}，词表大小={vocab_size}，样本 token 数={len(dataset)}")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=32,
        num_layers=1,
        num_heads=4,
        d_ff=96,
        rope_theta=10000,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # 固定学习率、关闭权重衰减，便于观察模型是否能记住这个 batch。
    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    initial_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.train()

    with torch.no_grad():
        initial_loss = cross_entropy(model(inputs), targets).item()
    print(f"初始 loss={initial_loss:.6f}，固定 batch 形状={tuple(inputs.shape)}", flush=True)

    for step in range(1, steps + 1):
        # 与正式训练共用更新逻辑，这里始终传入同一批 inputs、targets。
        loss, logits = train_step(model, optimizer, inputs, targets, max_grad_norm=1.0)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"第 {step} 步 loss 出现 NaN/Inf")

        # 这是本实验需要验证的梯度连通性和数值状态，不要求梯度每个元素都非零。
        for name, p in model.named_parameters():
            if p.grad is None:
                raise RuntimeError(f"参数 {name} 没有梯度")
            if not torch.isfinite(p.grad).all().item():
                raise RuntimeError(f"参数 {name} 的梯度出现 NaN/Inf")

        should_log = step == 1 or step % log_interval == 0 or step == steps
        if should_log:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters()])
            ).item()
            accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
            # loss、准确率来自更新前的前向计算；此时保留的梯度已经裁剪。
            print(
                f"step={step:3d} loss={loss.item():.6f} "
                f"accuracy={accuracy:.2%} grad_norm_after_clip={grad_norm:.6f}",
                flush=True,
            )

    # 最后重新前向计算，评估完成全部更新后的模型。
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        final_loss = cross_entropy(predictions, targets).item()
        predicted_ids = predictions.argmax(dim=-1)
        final_accuracy = (predicted_ids == targets).float().mean().item()
        update_norm = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(p - initial_weights[name])
                for name, p in model.named_parameters()
            ])
        ).item()

    print(f"\nloss: {initial_loss:.6f} -> {final_loss:.6f}")
    print(f"最终准确率: {final_accuracy:.2%}，参数更新总范数: {update_norm:.6f}")
    print(f"第一条序列输入: {tokenizer.decode(inputs[0].tolist())!r}")
    print(f"第一条序列目标: {tokenizer.decode(targets[0].tolist())!r}")
    print(f"第一条序列逐位置预测: {tokenizer.decode(predicted_ids[0].tolist())!r}")
    passed = final_loss < 0.1 and final_accuracy == 1.0 and update_norm > 0.0
    print("单批次过拟合通过" if passed else "尚未达到目标，请检查上面的训练日志")


if __name__ == "__main__":
    main()
