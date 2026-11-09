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



### torch.compile

默认情况下，PyTorch 通常按照程序的执行顺序，逐个调度张量运算。这叫 **eager execution，即时执行**。

例如：

```
y = x * 2
z = y + 1
out = torch.relu(z)
```

`torch.compile` 会尝试捕获这些运算及其依赖关系，形成**计算图**，再交给编译后端处理。后端可以分析整段计算，寻找优化机会。

比如，某些后端可以把上面的多个操作合并成一个计算内核，减少调度和中间数据读写的开销。

这里的 **JIT（Just-In-Time，即时编译）**，意味着编译通常发生在程序运行过程中：第一次实际调用模型时进行追踪和编译，后续复用结果。所以前几次运行可能更慢，之后才可能获得收益。

**`backend` 参数有什么作用？**

`backend` 的意思是“后端”，在这里指定：**捕获计算图后，由哪套机制负责处理和执行它。**

常见选项可以这样理解：

| `backend`     | 主要行为                                              |
| ------------- | ----------------------------------------------------- |
| `"eager"`     | 捕获计算图，然后用现有 PyTorch 算子执行，主要用于调试 |
| `"aot_eager"` | 进一步生成前向和反向计算图，仍用现有 PyTorch 算子执行 |
| `"inductor"`  | 在计算图基础上进一步优化，并生成高效的计算代码        |

其中，**Inductor（完整名称 TorchInductor）是 `torch.compile` 的默认编译后端**。因此：

```
model = torch.compile(model)
```

默认相当于：

```
model = torch.compile(model, backend="inductor")
```

用法

选项什么时候选择

`"inductor"`模型已能正确运行，希望加速训练或推理

`"eager"`检查“捕获计算图”这一步是否正常

`"aot_eager"`检查前向、反向图生成是否正常；或按特定环境的兼容性说明使用

### 计算图

它能进行算子融合等优化，并生成适合目标设备执行的代码。

计算图就是把一次计算拆成若干步骤，用节点和箭头表示各步骤之间的数据依赖。

区分两种图：

| 图          | 主要用途             | 主要存储形式                |
| ----------- | -------------------- | --------------------------- |
| Autograd 图 | 反向传播，计算梯度   | 互相连接的反向计算节点      |
| FX 图       | 分析、变换和编译程序 | `Graph` 对象中的一组 `Node` |



### 关于线程池操作的一些解释

调用pool.submit后会返回一个Future对象

**`Future` 是一个代表“任务执行状态和将来结果”的对象。** 你可以把它理解成提交任务后拿到的“取号凭证”。

在你的代码中：

```
future = pool.submit(encode_chunk, input_path, start, end)
```

`submit()` 将编码任务交给进程池，然后返回 `Future`，主进程可以继续提交其他任务。

此时，任务可能处于：

```
等待执行 → 正在执行 → 已完成
```

通过这个凭证，可以做这些事：

| 操作              | 作用                                 |
| ----------------- | ------------------------------------ |
| `future.done()`   | 查询任务是否已经结束，立即返回布尔值 |
| `future.result()` | 获取结果；尚未结束就等待             |
| `future.cancel()` | 尝试取消尚未开始的任务               |
## Json操作

| 函数           | 作用                             | 输入 → 输出                 |
| -------------- | -------------------------------- | --------------------------- |
| `json.dumps()` | 把 Python 对象转换成 JSON 字符串 | Python 对象 → `str`         |
| `json.loads()` | 把 JSON 内容解析成 Python 对象   | JSON 字符串等 → Python 对象 |
| `json.dump()`  | 把 Python 对象写入 JSON 文件     | Python 对象 → 文件          |
| `json.load()`  | 从 JSON 文件读取 Python 对象     | 文件 → Python 对象          |

### example1

**使用 `json.dumps()`，把 Python 对象转成 JSON 字符串。**

```python
import json

user = {
    "name": "小明",
    "age": 20,
    "is_student": True,
    "address": None,
}

text = json.dumps(user, ensure_ascii=False)

print(text)
print(type(text))
```

输出：

```python
{"name": "小明", "age": 20, "is_student": true, "address": null}
<class 'str'>
```

转换之后，`text` 是字符串，已经不是字典了

`dumps()` 常用参数如下：

| 参数                    | 作用                                              |
| ----------------------- | ------------------------------------------------- |
| `ensure_ascii=False`    | 让中文等非 ASCII 字符直接显示                     |
| `indent=2`              | 使用 2 个空格缩进，方便阅读                       |
| `sort_keys=True`        | 按键排序                                          |
| `separators=(",", ":")` | 去掉分隔符旁多余的空格                            |
| `allow_nan=False`       | 遇到 `NaN`、正负无穷大时报错，避免输出非标准 JSON |
| `default=函数`          | 指定不支持的对象如何转换                          |

### example2

**使用 `json.loads()`，把 JSON 字符串转成 Python 对象。**

```python
import json

text = """
{
  "name": "小明",
  "age": 20,
  "skills": ["Python", "SQL"],
  "is_student": true
}
"""

user = json.loads(text)

print(type(user))          # <class 'dict'>
print(user["name"])        # 小明
print(user["skills"][0])   # Python
print(user["is_student"])  # True
```

解析完成后，就可以正常使用字典、列表等 Python 操作：

```python
user["age"] += 1
user["skills"].append("Git")

print(user["age"])     # 21
print(user["skills"])  # ['Python', 'SQL', 'Git']
```

读取可能不存在的键，可以使用 `get()`：

```python
city = user.get("city", "未知")
print(city)  # 未知
```

`loads()` 接收 `str`，也支持包含 JSON 数据的 `bytes` 和 `bytearray`。它不接收一个已经解析好的字典：

```python
data = {"name": "小明"}

# 错误：data 已经是字典，无需再次解析
# json.loads(data)
```

### example3

 **使用 `json.dump()`，将数据写入 JSON 文件。**

```python
import json

users = [
    {"name": "小明", "age": 20},
    {"name": "小红", "age": 22},
]

with open("users.json", "w", encoding="utf-8") as file:
    json.dump(
        users,
        file,
        ensure_ascii=False,
        indent=2,
    )
```

生成的 `users.json` 内容是：

```python
[
  {
    "name": "小明",
    "age": 20
  },
  {
    "name": "小红",
    "age": 22
  }
]
```

这里有几个关键点：

- `"w"` 表示写入模式；文件不存在时创建，存在时会清空原内容。
- `encoding="utf-8"` 指定文件编码。
- `with` 会在代码块结束后关闭文件。
- `ensure_ascii=False` 控制 JSON 文本的字符表示方式，**并不负责设置文件编码**。

`dump()` 负责写入文件，返回值是 `None`。需要获得字符串时，应使用 `dumps()`。

### example4

**使用 `json.load()`，读取 JSON 文件。**

```py
import json

with open("users.json", "r", encoding="utf-8") as file:
    users = json.load(file)

print(type(users))       # <class 'list'>
print(users[0]["name"])  # 小明
```

返回什么 Python 类型，取决于文件里的 JSON 顶层是什么：

```python
{"name": "小明"}
```

读取后是 `dict`。

```python
[{"name": "小明"}, {"name": "小红"}]
```

读取后是 `list`。

可以把这两组操作放在一起理解：

```python
# 字符串
text = json.dumps(data)
data = json.loads(text)

# 文件
json.dump(data, file)
data = json.load(file)
```





### with open() as

```
with open("data.json", "r", encoding="utf-8") as input_file:
```

| 模式  | 含义             | 文件不存在时 | 文件已存在时           |
| ----- | ---------------- | ------------ | ---------------------- |
| `"r"` | 只读，默认模式   | 报错         | 读取内容，不修改       |
| `"w"` | 只写             | 创建文件     | **先清空原内容**       |
| `"a"` | 追加写入         | 创建文件     | 在末尾追加，保留原内容 |
| `"x"` | 创建新文件并写入 | 创建文件     | 报错，避免覆盖         |


main() 解析参数
  ├─ --gpu B200：用于选择云端 GPU
  └─ --batch_size 32：放入 trainer_arguments
          ↓
run_training(trainer_arguments, run_name)
          ↓
training_command(arguments, run_dir)
          ↓
subprocess.run() 启动 train.py
          ↓
train.py 解析最终参数

check=False 决定的是：训练程序失败退出后，subprocess.run() 是否立刻抛出异常。

程序退出时会返回一个数字，叫“退出码”：
- 0：通常表示成功。
- 非 0：通常表示失败，例如训练因显存不足而退出。
代码是：
result = subprocess.run(..., check=False)
volume.commit()
result.check_returncode()
执行顺序是：
1. 等待训练程序结束；即使训练失败，也先返回 result，继续往下执行。
2. volume.commit() 保存已经生成的日志和检查点。
3. result.check_returncode() 检查退出码；非 0 时再抛出异常，报告失败。
如果改成 check=True，训练失败时会在第一行直接抛出异常，后面的 volume.commit() 就执行不到了。

check=False 不会让训练忽略错误继续训练，它只是推迟在父进程中报告失败的时机。

# Moadl使用方法

### uv给项目添加Python包

uv add --dev modal

### 登陆

uv run modal setup
需要代理，直接问AI，让他查代理端口并给出设置的命令

### 创建持久化存储Volume

uv run modal volume create cs336-data --version=2

部分含义`uv run`在当前项目的 Python 环境中运行后面的命令`modal`调用 Modal 命令行工具`volume`操作 Modal 的持久化存储`create`创建一个新的 Volume`cs336-data`你给这个 Volume 起的名字，可以自行命名`--version=2`使用 Volume 文件系统的 **v2 版本**，不是创建两个 Volume

### 上传数据

uv run modal volume put cs336-data \
  data/tinystories_train.bin /tinystories_train.bin

uv run modal volume put cs336-data \
  data/tinystories_valid.bin /tinystories_valid.bin

cs336-data：刚创建的 Volume。
data/...：本地文件路径。
/...：文件在 Volume 内部的保存路径。

### 查看

uv run modal volume ls cs336-data