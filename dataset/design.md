JAX

```yaml
# jax_lax_ops_full.yaml
schema_version: "3.0"
description: "Comprehensive operator definition for JAX LAX module with fuzzing strategies."

# ==============================================================================
# 0. 全局定义 (Global Definitions)
# 使用 YAML 锚点 (&) 避免重复定义，确保类型集合的一致性
# ==============================================================================
definitions:
  # 常用生成策略模板
  strategies:
    # 0. 标准生成 (无限制)
    std_uniform: &strat_uniform
      strategy: uniform
      range: [-10.0, 10.0]

    # 1. 正数域 (用于 log, sqrt)
    positive_only: &strat_pos
      strategy: uniform
      range: [0.0001, 100.0]

    # 2. [0, 1) 范围 (用于概率/反三角)
    unit_interval: &strat_unit
      strategy: uniform
      range: [-0.999, 0.999]

    # 3. >= 1 范围 (用于 acosh)
    large_pos: &strat_large_pos
      strategy: uniform
      range: [1.0001, 100.0]
    
    # 4. 非零 (用于除法分母)
    non_zero: &strat_nonzero
      strategy: uniform
      range: [-10.0, 10.0]
      exclude_values: [0.0, -0.0]

# ==============================================================================
# 1. 一元算子 (Unary Operators)
# [cite_start]文档来源: [cite: 24, 30, 34, 40] 等
# ==============================================================================
operators:

  # --- 1.1 纯数学函数 (无特殊定义域) ---
  - op_name: jax.lax.abs
    category: unary_math
    [cite_start]doc_ref: "Elementwise absolute value: |x| [cite: 24]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *num_all
    generation:
      inputs:
        x: *strat_uniform

  - op_name: jax.lax.neg
    category: unary_math
    [cite_start]doc_ref: "Elementwise negation: -x [cite: 95]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *num_all
    generation:
      inputs:
        x: *strat_uniform

  # --- 1.2 有定义域限制的数学函数 ---
  - op_name: jax.lax.log
    category: unary_math
    [cite_start]doc_ref: "Elementwise natural logarithm [cite: 95]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *fp_all
    generation:
      inputs:
        x: *strat_pos  # 必须 > 0

  - op_name: jax.lax.acosh
    category: unary_math
    [cite_start]doc_ref: "Elementwise inverse hyperbolic cosine [cite: 24]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *fp_all
    generation:
      inputs:
        x: *strat_large_pos # 必须 >= 1

  # --- 1.3 整数位运算 ---
  - op_name: jax.lax.clz
    category: unary_bitwise
    [cite_start]doc_ref: "Elementwise count-leading-zeros [cite: 34]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *int_all  # 严禁浮点
    generation:
      inputs:
        x: *strat_uniform

  - op_name: jax.lax.population_count
    category: unary_bitwise
    [cite_start]doc_ref: "Elementwise popcount [cite: 30]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *int_all
    generation:
      inputs:
        x: *strat_uniform

  # --- 1.4 类型转换/属性判断 ---
  - op_name: jax.lax.is_finite
    category: unary_logic
    [cite_start]doc_ref: "Elementwise isfinite [cite: 89]"
    signature:
      inputs: [{name: x, type: tensor}]
    constraints:
      supported_dtypes: *fp_all
      output_dtype_rule: ALWAYS_BOOL # 输出强制为 Bool
    generation:
      inputs:
        x: 
          strategy: special_values 
          options: [INF, NAN, 0.0, 1.0] # 专门测试特殊值

# ==============================================================================
# 2. 二元算子 (Binary Operators)
# [cite_start]文档来源: [cite: 24, 30, 40, 58]
# ==============================================================================

  # --- 2.1 基础算术 ---
  - op_name: jax.lax.add
    category: binary_math
    [cite_start]doc_ref: "Elementwise addition: x + y [cite: 24]"
    signature:
      inputs: [{name: x, type: tensor}, {name: y, type: tensor}]
    constraints:
      supported_dtypes: *num_all
      shape_rule: BROADCASTABLE # 允许广播
    generation:
      inputs:
        x: *strat_uniform
        y: *strat_uniform

  - op_name: jax.lax.div
    category: binary_math
    [cite_start]doc_ref: "Elementwise division: x / y [cite: 40]"
    signature:
      inputs: [{name: x, type: tensor}, {name: y, type: tensor}]
    constraints:
      supported_dtypes: *fp_all
    generation:
      inputs:
        x: *strat_uniform
        y: *strat_nonzero # 分母防 0

  # --- 2.2 比较运算 (输出 Bool) ---
  - op_name: jax.lax.eq
    category: binary_comparison
    [cite_start]doc_ref: "Elementwise equals: x = y [cite: 58]"
    signature:
      inputs: [{name: x, type: tensor}, {name: y, type: tensor}]
    constraints:
      supported_dtypes: *any_type
      output_dtype_rule: ALWAYS_BOOL
    generation:
      inputs:
        x: *strat_uniform
        y: *strat_uniform

  # --- 2.3 逻辑位运算 ---
  - op_name: jax.lax.bitwise_and
    category: binary_bitwise
    [cite_start]doc_ref: "Elementwise AND: x & y [cite: 30]"
    signature:
      inputs: [{name: x, type: tensor}, {name: y, type: tensor}]
    constraints:
      supported_dtypes: *int_all # 仅限整数
    generation:
      inputs:
        x: *strat_uniform
        y: *strat_uniform

# ==============================================================================
# 3. 复杂/线性代数算子 (Complex/Linalg)
# 需要处理形状依赖和静态参数
# ==============================================================================

  - op_name: jax.lax.dot
    category: linalg
    [cite_start]doc_ref: "General dot product [cite: 40]"
    signature:
      inputs:
        - {name: lhs, type: tensor}
        - {name: rhs, type: tensor}
      attributes:
        - name: precision
          type: enum
          [cite_start]options: [DEFAULT, HIGH, HIGHEST] # [cite: 407-416]
          default: DEFAULT
    constraints:
      supported_dtypes: *fp_all
    
    # 复杂的形状生成协议
    generation:
      shape_constraints:
        # 定义变量
        variables: 
          M: {min: 1, max: 64}
          K: {min: 1, max: 64}
          N: {min: 1, max: 64}
        
        # 定义场景
        scenarios:
          - id: matrix_mul
            shapes: {lhs: [M, K], rhs: [K, N]} # (M,K) * (K,N) -> (M,N)
          - id: vector_dot
            shapes: {lhs: [K], rhs: [K]}       # (K) * (K) -> Scalar
      
      inputs:
        lhs: *strat_uniform
        rhs: *strat_uniform

  - op_name: jax.lax.clamp
    category: ternary_math
    [cite_start]doc_ref: "Elementwise clamp [cite: 34]"
    signature:
      inputs:
        - {name: min, type: scalar_or_tensor}
        - {name: x,   type: tensor}
        - {name: max, type: scalar_or_tensor}
    constraints:
      supported_dtypes: *num_all
    generation:
      # 策略：保证 min <= max，否则逻辑无意义
      inputs:
        min: {strategy: constant, value: -5.0}
        max: {strategy: constant, value: 5.0}
        x:   {strategy: uniform, range: [-10.0, 10.0]}
```







