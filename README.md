# FUEL-JAX

面向 JAX / PyTorch 的算子差分测试小框架。当前主要覆盖 `dataset/jax_rules.yaml` 里的一元 / 二元算子输入生成与执行对比。

## 快速开始

### 1) 安装依赖（推荐使用 uv）

```bash
uv sync
```

### 2) 生成输入

> `gen` 子命令需要 **完整 JAX 名称**（如 `jax.lax.pow`）。

```bash
# 单个算子
.venv/bin/python -m fuel_jax.main gen --op-name jax.lax.pow --shape 2,3 --seed 0

# 生成全部规则中的算子
.venv/bin/python -m fuel_jax.main gen --op-name all --seed 0
```

默认输出到：
- 单个算子：`input/<op_name>/00.npz`
- 全部算子：`input/<op_name>/00.npz`

### 3) 执行对比

> `exec` 使用 **完整 JAX 名称**（如 `jax.lax.abs`）。

```bash
# 运行单个算子
.venv/bin/python -m fuel_jax.main exec --op-name jax.lax.abs --device gpu --mode compiler --test-id 0

# 执行全部
.venv/bin/python -m fuel_jax.main exec --op-name all --device gpu --mode compiler --test-id 0
```

输出路径默认：
```
output/<op_name>/<precision>/jax_<device>.npz
output/<op_name>/<precision>/torch_<device>.npz
```

### 4) 验证结果

```bash
.venv/bin/python -m fuel_jax.main validate --op-name jax.lax.abs
.venv/bin/python -m fuel_jax.main validate --op-name all
```

## 规则文件（输入生成）

`dataset/jax_rules.yaml` 定义输入生成策略。主要结构：

- `definitions.strategies`：策略模板（uniform / normal / int）
- `unary_operators` / `binary_operators`：每个算子的输入与生成策略

示例（标准正态 + 限制域）：

```yaml
- op_name: jax.lax.log
  input: [x]
  generation: {x: *strat_pos}   # 正数

- op_name: jax.lax.integer_pow
  input: [x, y]
  generation:
    x: *strat_normal
    y: *strat_int_scalar         # 标量整数
```

支持的 strategy：
- `uniform`：`range: [low, high]`
- `normal`：`mean`, `std`
- `int`：`range: [low, high]`（标量整数）

## 目录结构

```
Fuel-JAX/
  dataset/
    jax_rules.yaml
    jax2torch_map.csv
  fuel_jax/
    main.py
    generator/generate.py
    difftesting/exec.py
    difftesting/validate.py
    script/jax_script.py
    script/torch_script.py
    utils/utils.py
```

## 设计说明与当前限制

- **算子命名与目录一致**：
  - 全流程使用 `jax.lax.xxx`，目录也使用完整名称（如 `input/jax.lax.abs/`）。
- **PyTorch 映射依赖 `jax2torch_map.csv`**：
  `torch_script.py` 根据映射解析 `torch.xxx` / `torch.special.xxx` / `torch.Tensor.xxx`。如需表达式映射，需要后续补充自定义处理。
- **JAX/Torch 设备检测逻辑互换**：
  `get_jax_device` 依赖 torch，`get_torch_device` 依赖 jax，可能导致错误设备选择。
- **`integer_pow` 需要 Python int**：
  JAX 要求幂次是静态 int，但当前执行路径会把标量保存为 ndarray 后再转为 JAX array，可能报错。需要在执行时特判。
- **`convert_element_type` / `bitcast_convert_type` 未被生成器正确处理**：
  规则中用 `signature` 描述，但生成器只读取 `input`。
- **torch 输入参数顺序依赖 dict 迭代顺序**：
  `torch_script.py` 用 `inp.values()` 组装参数，若输入顺序错乱会导致调用错误（建议按规则中的 `input` 顺序传参）。

## 下一步建议

1) 在执行层为 `integer_pow` 做特判（把 `y` 转成 Python int）
2) 修正设备检测函数（JAX 检测 JAX，Torch 检测 Torch）
3) 根据规则中的 `input` 顺序组装 torch 参数，避免 dict 顺序偏差
4) 扩展规则覆盖到更多非 elementwise 算子

---

如果你希望我把上述限制逐条修复，我可以按优先级直接改代码。接下来先修哪一条？
