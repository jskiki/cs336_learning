"""加载训练目录中的配置和检查点，复用 decoding 生成文本。"""

from argparse import ArgumentParser
import json
from pathlib import Path

import torch

from cs336_basics.decoding import generate_text
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


def main():
    parser = ArgumentParser(description="使用训练好的 TinyStories 模型续写文本")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--tokenizer_path", default="data/tinystories_bpe.pkl")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(4)

    # 必须用训练时的结构重建模型，再加载学到的参数。
    config = json.loads((args.run_dir / "config.json").read_text())
    model = TransformerLM(
        **{key: config[key] for key in (
            "vocab_size", "context_length", "d_model", "num_layers",
            "num_heads", "d_ff", "rope_theta", "eps",
        )},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    checkpoint = torch.load(
        args.run_dir / "latest.pt", map_location="cpu", weights_only=True,
    )
    # 推理只需要模型权重，不需要恢复 AdamW 状态。
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    model.to(args.device).eval()

    tokenizer = Tokenizer.from_files(
        args.tokenizer_path, args.tokenizer_path,
        special_tokens=["<|endoftext|>"],
    )
    print(generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ))


if __name__ == "__main__":
    main()
