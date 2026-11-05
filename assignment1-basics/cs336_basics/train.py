from argparse import ArgumentParser

import torch


def get_hparam():
    parser = ArgumentParser(description="训练 Transformer 语言模型")

    # 数据：data_dtype 表示原始二进制 token 文件的存储类型。
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
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
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.9, 0.999),
        metavar=("BETA1", "BETA2"),
    )
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 训练时长与学习率调度。
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)

    # 日志、验证和检查点。
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/latest.pt",
    )
    parser.add_argument("--resume_path", type=str, default=None)

    # 运行配置。
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=42)

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

