import os
from typing import BinaryIO,IO
import torch
import torch.nn as nn
from torch.optim import Optimizer

#保存模型和优化器的状态（会影响后续计算的信息）以及迭代次数到out里
def save_checkpoint(
        model:nn.Module,
        optimizer:Optimizer,
        iteration:int,
        out:str|os.PathLike|BinaryIO|IO[bytes],
) ->None:
    model_state = model.state_dict()
    opt_state = optimizer.state_dict()
    saved = {
        "model_state" : model_state,
        "opt_state" : opt_state,
        "iteration" : iteration,
    }
    torch.save(saved,out)

#从src里取出模型和优化器的状态并返回当前的迭代次数
def load_checkpoint(
        src:str | os.PathLike | BinaryIO | IO[bytes],
        model:nn.Module,
        optimizer:Optimizer,
) ->int:
    load_state = torch.load(src)
    model.load_state_dict(load_state["model_state"])
    optimizer.load_state_dict(load_state["opt_state"])
    return load_state["iteration"]
