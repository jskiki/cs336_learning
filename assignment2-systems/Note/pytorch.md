import torch.distributed as dist

import torch.multiprocessing as mp

### 1. `mp.spawn` 启动工作进程

```
mp.spawn(
    fn=distributed_demo,
    args=(world_size,),
    nprocs=world_size,
    join=True,
)
```

| 参数                  | 含义                                                         |
| --------------------- | ------------------------------------------------------------ |
| `fn=distributed_demo` | 每个子进程要执行的函数。传进去的是函数本身，所以这里没有加括号调用它。 |
| `args=(world_size,)`  | 传给 `fn` 的额外参数。`spawn` 会把进程编号放在最前面，实际调用 `fn(rank, *args)`。逗号让 `(world_size,)` 成为单元素元组。 |
| `nprocs=world_size`   | 要启动的子进程数量。这里是 4 个，编号为 `0～3`。             |
| `join=True`           | 主进程等待所有子进程结束后，才从 `spawn` 返回。              |

所以这里 rank 2 的子进程实际调用的是 `distributed_demo(2, 4)`。其中 `2` 由 `spawn` 传入，`4` 来自你提供的 `args`。[PyTorch 多进程文档](https://docs.pytorch.org/docs/stable/multiprocessing.html)

### 2. `dist.init_process_group` 建立进程组

```
dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

| 参数                    | 含义                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `"gloo"`                | 通信后端，决定用哪套库实现进程间通信。这里用 Gloo 做 CPU 通信，多 GPU 训练通常用 `"nccl"`。 |
| `rank=rank`             | 当前进程在组内的编号。每个进程不同，例如 rank 2 传入 `2`。   |
| `world_size=world_size` | 组内总共有多少个进程。这里所有进程都传入 `4`。               |

前面设置的 `MASTER_ADDR="localhost"` 和 `MASTER_PORT="29500"` 告诉进程到哪里建立初始连接。可以记成 **`rank` 是“我是几号”，`world_size` 是“一共几个人”，而地址和端口是“去哪里会合”**。[PyTorch 进程组初始化](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)

### 3. `dist.all_reduce` 对张量做归约

```
dist.all_reduce(data, async_op=False)
```

- **`data`** 是当前进程参与归约的张量。操作结束后，结果直接写回这个张量。
- **`op`** 在代码里省略了，默认是 `dist.ReduceOp.SUM`，所以各进程的张量会逐元素相加。
- **`group`** 也省略了，所以使用刚才初始化的默认进程组，4 个进程都参与。
- **`async_op=False`** 表示同步调用。在这个 CPU/Gloo 示例中，调用返回时归约已经完成，下一行就可以打印结果。设成 `True` 时，调用会返回一个 `Work` 对象，可以稍后用 `work.wait()` 等待。[PyTorch `all_reduce` 文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)