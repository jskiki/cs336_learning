from argparse import ArgumentParser
from pathlib import Path
import json
import time

import torch
import numpy as np

from cs336_basics.data import get_batch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint

def get_hparam():
    parser = ArgumentParser(description="训练 Transformer 语言模型")

    # 数据：data_dtype 表示原始二进制 token 文件的存储类型。
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    #磁盘上的token数据用那种整数类型存储
    parser.add_argument(
        "--data_dtype",
        choices=["uint16", "uint32", "int32", "int64"],
        default="uint16",
    )

    # 模型与批次。
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=344)
    parser.add_argument("--rope_theta", type=float, default=1e4)
    parser.add_argument("--eps", type=float, default=1e-5)

    # 优化器。
    parser.add_argument("--lr", type=float, default=1e-3)
    #metavar 就是给帮助信息中的参数值起个展示名称
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.999),
        metavar=("BETA1", "BETA2"),
    )
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    #梯度裁剪的最大范数
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 训练时长与学习率调度。
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)

    # 日志、验证和检查点。
    #每训练多少步记录一次日志如loss,学习率，耗时等
    parser.add_argument("--log_interval", type=int, default=10)
    #每训练多少步在验证集上观察模型表现
    parser.add_argument("--eval_interval", type=int, default=100)
    #每次验证使用多少个batch，通常将这些batch的loss取平均
    parser.add_argument("--eval_batches", type=int, default=10)
    #每训练多少步保存一次检查点
    parser.add_argument("--save_interval", type=int, default=500)
    #保存检查点的目标路径
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest.pt",
    )
    #启动时要加载的检查点路径；None 表示不加载已有检查点
    parser.add_argument("--resume_path", type=str, default=None)

    # 运行配置。
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=42)

    #把参数保存到args，通过args.batch_size就能访问
    args = parser.parse_args()

    # 统一默认值与命令行输入产生的数据类型。
    args.betas = tuple(args.betas)
    args.dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # 未单独指定衰减终点时，使用总训练步数。
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters

    for name in (
        "vocab_size",
        "batch_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "max_iters",
        "log_interval",
        "eval_interval",
        "eval_batches",
        "save_interval",
    ):
        #getattr()是python的内置函数，根据字符串形式的属性名获取对象的属性值
        if getattr(args, name) <= 0:
            parser.error(f"--{name} 必须大于 0")

    if args.d_model % args.num_heads != 0:
        parser.error("d_model 必须能被 num_heads 整除")

    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("使用 RoPE 时，每个注意力头的维度必须为偶数")

    if not all(0 <= beta < 1 for beta in args.betas):
        parser.error("betas 中的两个值必须在 [0, 1) 范围内")

    if not 0 <= args.min_lr <= args.lr:
        parser.error("需要满足 0 <= min_lr <= lr")

    if not 0 <= args.warmup_iters < args.cosine_cycle_iters:
        parser.error("需要满足 0 <= warmup_iters < cosine_cycle_iters")

    return args


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行一次参数更新，返回本次更新前的 loss 和 logits。"""
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(inputs=logits, targets=targets)
    loss.backward()
    gradient_clipping(parameters=model.parameters(), max_l2_norm=max_grad_norm)
    optimizer.step()

    # 调用方只用结果记录指标，detach 避免日志持有计算图；梯度保留供调试检查。
    return loss.detach(), logits.detach()


def train():
    args = get_hparam()

    # 在模型初始化和数据采样之前设置随机种子。
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = np.memmap(
        args.train_path,
        dtype=args.data_dtype,
        mode="r",
    )

    val_dataset = np.memmap(
        args.val_path,
        dtype=args.data_dtype,
        mode="r",
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.eps,
        device=args.device,
        dtype=args.dtype,
    )
    # 在云端 CUDA 训练时启用编译。
    if torch.device(args.device).type == "cuda":
        model.compile(backend="inductor")

    # 整个训练过程复用同一个优化器，保留矩估计等历史状态。
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )

    # iteration 表示已经完成的训练步数。如果指定了恢复的路径，就恢复，并把已经完成的步数赋值给start_iteration,否则就赋值为0从头开始计算
    start_iteration = (
        load_checkpoint(args.resume_path, model, optimizer)
        if args.resume_path is not None
        else 0
    )
    #确保保存检查点的目录存在没有就创建
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    #把参数转换成字典并复制一份
    config = vars(args).copy()
    # torch.dtype 不能直接写入 JSON，转换为与命令行参数一致的字符串。torch.float32->"torch.float32"->"float32"
    config["dtype"] = str(args.dtype).removeprefix("torch.")
    with (Path(args.checkpoint_path).parent / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    #切换到训练模式
    model.train()
    start_time = time.perf_counter()
    # 每次启动重新记录；不同实验使用不同的检查点目录。
    with (Path(args.checkpoint_path).parent / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for iteration in range(start_iteration, args.max_iters):
            inputs, targets = get_batch(
                dataset=train_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )

            lr = get_lr_cosine_schedule(
                it=iteration,
                max_learning_rate=args.lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.cosine_cycle_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss, _ = train_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                max_grad_norm=args.max_grad_norm,
            )

            completed_steps = iteration + 1
            is_last_step = completed_steps == args.max_iters

            #定期打印并保存训练进度
            if completed_steps % args.log_interval == 0 or is_last_step:
                train_loss = loss.item()
                record = {
                    "step": completed_steps,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "train_loss": train_loss,
                    "lr": lr,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(
                    f"step={completed_steps}/{args.max_iters} "
                    f"train_loss={train_loss:.4f} "
                    f"lr={lr:.6g}",
                    flush=True,
                )

            #定期进行评估
            if completed_steps % args.eval_interval == 0 or is_last_step:
                model.eval()
                val_losses = []

                #关闭梯度记录，不构建反向传播计算图，可以减少内存和计算开销
                with torch.no_grad():
                    for _ in range(args.eval_batches):
                        val_inputs, val_targets = get_batch(
                            dataset=val_dataset,
                            batch_size=args.batch_size,
                            context_length=args.context_length,
                            device=args.device,
                        )
                        val_logits = model(val_inputs)
                        val_losses.append(
                            cross_entropy(
                                inputs=val_logits,
                                targets=val_targets,
                            )
                        )

                val_loss = torch.stack(val_losses).mean().item()
                model.train()

                record = {
                    "step": completed_steps,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "val_loss": val_loss,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(
                    f"step={completed_steps} val_loss={val_loss:.4f}",
                    flush=True,
                )

            #定期保存检查点
            if completed_steps % args.save_interval == 0 or is_last_step:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_steps,
                    out=args.checkpoint_path,
                )
                print(
                    f"checkpoint saved: {args.checkpoint_path} "
                    f"(step={completed_steps})",
                    flush=True,
                )


if __name__ == "__main__":
    train()
