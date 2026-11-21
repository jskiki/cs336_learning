import os

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    #Gloo 是 PyTorch 用来实现多进程通信的底层库
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

#注意观察输出顺序，理解init_process_group()，all_reduce(data, async_op=False)，join=True这三种等待
if __name__ == "__main__":
    world_size = 4
    #distributed_demo第一个参数必须是整数rank
    spawn(
        fn=distributed_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )