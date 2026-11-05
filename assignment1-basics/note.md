```bash
# 查看改动
git status
git diff

# 暂存当前目录下的全部改动
git add .

# 提交到本地仓库
git commit -m "Complete Embedding implementation and test adapter"

# 推送到远程仓库
git push

#如果本地和仓库都发生了修改，应执行以下操作
1.如果本地的修改已经准备好提交
git status

# 提交本地修改，文件名按实际情况填写
git add note.md
git commit -m "更新学习笔记"

# 获取远程更新，将本地提交接到远程最新提交之后
git pull --rebase

# 推送
git push


2.如果本地的修改还没完成不想提交
git stash push -m "暂存未完成修改"

git pull --ff-only

git stash pop
```



# 新增内容

1.train_bpe.py

2.fast_train_bpe.py

3.show.py

4.tokenizer.py

5.liner.py

6.linear.py

7.swiglu.py







在数值表示上，一维的长度为d的张量与1*d的二维张量表示完全相同，但是二维的在计算时会进行广播所以注意二者的区别

nn.ModuleList 会将其中的块注册为子模块，这样 PyTorch 才能：
- 通过 model.parameters() 找到它们的参数。
- 在 state_dict() 中生成 layers.0.*、layers.1.* 等权重名称。
- 随整个模型统一迁移设备和数据类型。



### 交叉熵损失函数的详细解释

```py
def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """根据预测 logits 和目标 token ID，返回平均交叉熵损失。

    参数：
        inputs: 未经 softmax 的浮点 logits，形状为 (..., vocab_size)。
        targets: 整数类别索引，形状为 (...)，与 inputs 的前导维度一致。
            每个索引在 [0, vocab_size) 内，与 inputs 位于同一设备。

    返回值：
        对所有前导维度取平均后的标量 Tensor，形状为 ()。
        例如 inputs 为 (batch_size, seq_len, vocab_size) 时，
        targets 为 (batch_size, seq_len)，每个 token 位置参与平均。

    完成标准：
        按题目要求减去最大元素，保证数值稳定性。
        尽可能消去相互抵消的 log 和 exp 运算。
        支持任意数量的前导批次维度，最后一维始终是词表维度。
        不原地修改输入，保留从损失到 inputs 的 autograd 计算图。
    """
    # 平移量只用于数值稳定；视为常量不会改变损失对 logits 的梯度。
    #detach():得到一个与inputs共享数据。但脱离自动求导计算图的张量。它不复制数据，也不会让原来的 inputs 停止计算梯度。keepdim保留维度进行广播
    shifted = inputs - inputs.detach().amax(dim=-1, keepdim=True)
    #targets.unsqueeze(-1):在最后一维添加一个维度。shifted.gather(dim=-1, index=...)：沿最后一维按索引取值。
    #
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    return (log_normalizer - target_logits).mean()

```

假设 **batch 中有 2 条序列，每条有 3 个预测位置，词表里有 4 个 token**：

```
# inputs.shape = (2, 3, 4)
# 分别对应：(batch_size, seq_len, vocab_size)
inputs = torch.tensor([
    [
        [1.0, 2.0, 3.0, 4.0],  # 第 1 条序列，第 1 个位置
        [4.0, 3.0, 2.0, 1.0],  # 第 1 条序列，第 2 个位置
        [1.0, 3.0, 2.0, 0.0],  # 第 1 条序列，第 3 个位置
    ],
    [
        [2.0, 1.0, 4.0, 3.0],  # 第 2 条序列，第 1 个位置
        [0.0, 2.0, 1.0, 3.0],  # 第 2 条序列，第 2 个位置
        [3.0, 1.0, 0.0, 2.0],  # 第 2 条序列，第 3 个位置
    ],
])

# targets.shape = (2, 3)
# 每个位置的正确 token ID，范围是 0～3
targets = torch.tensor([
    [3, 0, 1],
    [2, 1, 0],
])
```

例如，`targets[1, 1] = 1` 表示：**第 2 条序列的第 2 个位置，正确答案是词表中 ID 为 1 的 token**，对应的原始分数为 `inputs[1, 1, 1] = 2.0`。

减去每个位置的最大值后：

```
# shifted.shape = (2, 3, 4)
shifted = tensor([
    [
        [-3., -2., -1.,  0.],
        [ 0., -1., -2., -3.],
        [-2.,  0., -1., -3.],
    ],
    [
        [-2., -3.,  0., -1.],
        [-3., -1., -2.,  0.],
        [ 0., -2., -3., -1.],
    ],
])
```

`targets.unsqueeze(-1)` 把每个目标索引放进一个长度为 1 的维度：

```
# 形状：(2, 3, 1)
tensor([
    [[3], [0], [1]],
    [[2], [1], [0]],
])
```

接着执行：

```
shifted.gather(dim=-1, index=targets.unsqueeze(-1))
```

得到：

```
# 形状：(2, 3, 1)
tensor([
    [[ 0.], [ 0.], [ 0.]],
    [[ 0.], [-1.], [ 0.]],
])
```

也就是**每个位置从 4 个词表分数中，按照 targets 选出 1 个分数**。再 `.squeeze(-1)`，形状就变成 `(2, 3)`。

这句代码计算的是：**每个位置的所有词表分数先取指数，再求和，最后取自然对数**。

\[ \text{log\_normalizer} = \log\left(\sum_{a=1}^{\text{vocab\_size}}e^{\text{shifted}[a]}\right) \]

分三步看。假设某个位置有 3 个词表分数：

```
shifted = tensor([-2.0, -1.0, 0.0])
```

1. **`.exp()`**：逐元素计算 \(e^x\)。

   ```
   tensor([0.1353, 0.3679, 1.0000])
   ```

2. **`.sum(dim=-1)`**：沿最后的词表维度求和。

   ```
   tensor(1.5032)
   ```

3. **`.log()`**：对结果取自然对数。

   ```
   tensor(0.4076)
   ```

如果 `shifted` 的形状为 `(batch_size, seq_len, vocab_size)`，形状变化为：

```
shifted                  (batch_size, seq_len, vocab_size)
.exp()                   (batch_size, seq_len, vocab_size)
.sum(dim=-1)             (batch_size, seq_len)
.log()                   (batch_size, seq_len)
```

它为什么出现在交叉熵中？因为：

\[ -\log\frac{e^{z_y}}{\sum_a e^{z_a}} = \underbrace{\log\sum_a e^{z_a}}_{\text{log\_normalizer}} - \underbrace{z_y}_{\text{target\_logits}}. \]

所以，下一句 `log_normalizer - target_logits` 就得到**每个位置的交叉熵损失**。





## np.memmap

：数据保存在磁盘文件中，但你可以像操作普通数组一样，通过索引和切片访问它，无须预先把整个文件读入内存。[NumPy 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

例如，你有一个 20 GB 的 token 文件，而机器只有 8 GB 内存。通过内存映射，可以先建立文件与虚拟地址空间的映射；访问某段数据时，操作系统再按需将相关的内存页载入物理内存。因此，**建立映射不等于把整个文件加载到内存中**。

理解它时，有几个关键点：

- **使用方式接近普通数组。** 可以查看形状、读取元素、获取切片；`np.memmap` 本身是 `np.ndarray` 的子类。
- **`dtype` 决定如何解释文件里的字节。** 如果原始文件保存的是 `uint16`，读取时却指定 `int32`，就会按错误方式解释数据，而不是自动完成类型转换。
- **训练数据通常使用只读模式 `mode='r'`。** 另外，`'r+'` 允许修改原文件，`'w+'` 会创建或覆盖文件。
- **它依然会占用内存。** 访问的数据需要进入内存；将数据复制成新数组，也会产生额外内存开销。[参数与模式说明](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

加载方式取决于文件格式：

| 文件格式                                  | 对应方式                                                     |
| ----------------------------------------- | ------------------------------------------------------------ |
| 原始二进制文件，例如直接保存的 token 字节 | 使用 `np.memmap`，自行指定正确的 `dtype`，必要时指定形状和偏移量 |
| 由 `np.save` 保存的 `.npy` 文件           | 使用 `np.load`，设置 `mmap_mode='r'`，由文件头提供形状和类型信息 |

`.npy` 文件包含文件头，不能直接把整个文件当作纯 token 字节来解释。[`np.load` 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.load.html)

结合你的作业，**内存映射应在调用 `get_batch` 之前完成**。`get_batch` 接收到的 `dataset` 可以是普通数组，也可以是映射数组；它只需访问本批次所需的数据，再生成指定设备上的张量。这样，完整数据集留在磁盘上，训练时按批次读取。



## 关于data.py里部分代码的解释

```py
    start_indices = randint(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size,
    )
        # 每段多取一个 token，用于构造下一 token 目标。
    indices = start_indices[:, None] + arange(context_length + 1)
    batch = as_tensor(dataset[indices], dtype=long, device=device)

    return batch[:, :-1], batch[:, 1:]
```

数据是一个特别长的一维np数组，然后我们随机采样batch_size的合法起始位置，比如采取到`[2, 10, 5]`  ，context_length为4那这个batch里的三条数据就是data里的2-6、10-14、5-9


### 为什么要这样做g

这样做，是为了**从长序列中随机取出固定长度的训练片段，让模型学习预测下一个 token**。

假设数据是：

> 我 今天 去 公园 看见 一只 小猫 ……

取出一段后，输入与目标错开一个位置：

- 输入：`我 今天 去 公园`
- 目标：`今天 去 公园 看见`

模型在每个位置都学习“根据前面的内容，预测下一个 token”。因此，一条长度为 `context_length` 的输入，还需要后面额外一个 token 才能构造完整目标。

你选中的代码就是在决定**每条训练片段从哪里开始**：

- `size=batch_size`：选出一个批次需要的起点数量。
- `low=0`：允许从数组开头开始。
- `high=len(dataset) - context_length`：限制起点，确保输入及目标都不会越界。
- 随机选择：让训练不断接触数据集中的不同片段，同时保持片段内部的 token 顺序。

**随机的是片段的位置，片段内部的顺序保持不变**，模型才能学习语言中的前后关系。



这样一个片段可能被多次选中，在有限次的采样中有的片段可能从来没被选中过



```py
indices = start_indices[:, None] + arange(context_length + 1)
```

`arange（）`

numpy库里的一个方法
arange(x)生成从0到x不包含x的np数组

`start_indices[:, None]` 添加一个维度，把形状从 `(3,)` 变成 `(3, 1)`：

```
[[ 2],
 [10],
 [ 5]]
```

`arange(context_length + 1)` 生成偏移量：

```
[0, 1, 2, 3, 4]
```

相加时，NumPy 通过**广播**，让每个起点分别加上所有偏移量：

```python
indices =
[[ 2,  3,  4,  5,  6],
 [10, 11, 12, 13, 14],
 [ 5,  6,  7,  8,  9]]
```