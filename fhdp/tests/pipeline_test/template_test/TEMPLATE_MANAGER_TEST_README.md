# Template Manager Test Suite

## 概述

本测试套件用于验证FHDP系统中Template Manager的功能，特别是基于template的模型划分功能。

## 文件说明

### 1. `test_template_manager.py`
完整的单元测试套件，验证Template Manager的所有核心功能。

**测试类别：**
- **TestTemplateGeneration**: 测试从成功Pipeline生成Template
- **TestTemplateMatching**: 测试Template匹配性能和准确性
- **TestModelPartitioning**: 测试基于Template的模型划分
- **TestTemplateManagerIntegration**: 测试Template Manager与FHDP系统的集成
- **TestPerformanceMetrics**: 测试性能指标合规性

**运行方式：**
```bash
cd /path/to/fhdp/tests
python test_template_manager.py
```

### 2. `demo_template_partitioning.py`
交互式演示脚本，展示如何使用Template进行模型划分。

**演示内容：**
- Demo 1: 2-Stage Pipeline划分
- Demo 2: 3-Stage Pipeline划分
- Demo 3: Template匹配性能
- Demo 4: 性能基准测试

**运行方式：**
```bash
cd /path/to/fhdp/tests
python demo_template_partitioning.py
```

## 核心功能测试

### 1. Template生成
从成功的Pipeline执行中学习并生成Template：

```python
# Pipeline执行成功后
pipeline = Pipeline(
    pipeline_id="pipeline_001",
    template_id="temp_001",
    vehicles=["v1", "v2", "v3"],
    stages=["s0", "s1", "s2"],
    start_time=time.time() - 30.0,
    expected_completion=time.time()
)

# 生成Template
template = template_generator.generate_template_from_pipeline(
    pipeline, 
    success_rate=0.95
)
```

**生成的Template包含：**
- `template_id`: 唯一标识符
- `resource_requirements`: 各阶段的资源需求列表
- `expected_duration`: 预期执行时间
- `communication_pattern`: 阶段间通信模式
- `training_config`: 训练配置
- `model_fragment_size`: 每个模型片段的大小

### 2. Template匹配
快速查找最适合车辆组合的Template（<5ms）：

```python
# 给定一组车辆
vehicles = [vehicle1, vehicle2, vehicle3]

# 查找最佳Template
template = template_manager.find_template_for_vehicles(vehicles)
```

**匹配算法：**
- 基于资源要求的哈希索引
- 篮子（Basket）组织
- LRU缓存优化
- 灵活长度匹配

### 3. 模型划分
根据Template将模型划分为多个片段：

```python
# 创建Partitioner
partitioner = TemplateBasedPartitioner(template)

# 划分模型
fragments = partitioner.partition_model(model)

# fragments是参数字典的列表
# fragments[0]: Stage 0的参数
# fragments[1]: Stage 1的参数
# fragments[2]: Stage 2的参数
```

**划分策略：**
- **策略1**: 基于模型定义的阶段划分（如果模型提供了`get_stage_params`方法）
- **策略2**: 基于参数数量的通用划分

**示例模型结构（3-Stage Pipeline）：**

```
Stage 0 (AGX Orin - High):
  - conv1.weight, conv1.bias
  - bn1.weight, bn1.bias
  - conv2.weight, conv2.bias
  - bn2.weight, bn2.bias

Stage 1 (Orin Nano - Medium):
  - conv3.weight, conv3.bias
  - bn3.weight, bn3.bias
  - conv4.weight, conv4.bias
  - bn4.weight, bn4.bias

Stage 2 (AGX Orin - High):
  - fc1.weight, fc1.bias
  - bn5.weight, bn5.bias
  - fc2.weight, fc2.bias
  - bn6.weight, bn6.bias
  - fc3.weight, fc3.bias
```

### 4. 通信模式
Template定义了Pipeline中各阶段的通信模式：

```python
# Template中的通信模式
communication_pattern = [(0, 1), (1, 2), (0, 2)]

# 可视化为通信图
partitioner = TemplateBasedPartitioner(template)
print(partitioner.visualize_communication_pattern())
```

**输出示例：**
```
Communication Pattern (from Template)
============================================================

Adjacency Matrix:
   S0  S1  S2
S0 0  1  1 
S1 1  0  1 
S2 1  1  0 

Direct Connections:
  Stage 0 <---> Stage 1  # 相邻阶段通信
  Stage 1 <---> Stage 2  # 相邻阶段通信
  Stage 0 <---> Stage 2  # 跳跃连接（直接通信）

Connectivity per Stage:
  Stage 0: 2 connections
  Stage 1: 2 connections
  Stage 2: 2 connections
============================================================
```

### 5. 性能验证
验证FHDP性能指标的合规性：

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| Template查找 | < 5ms | `test_template_lookup_latency()` |
| Pipeline组成 | < 1.5s | `test_pipeline_composition_time()` |
| 内存使用 | < 50MB | `test_memory_efficiency()` |

## 测试覆盖范围

### TemplateGenerator测试
- ✅ 从Pipeline生成Template
- ✅ 资源模式提取
- ✅ 通信模式推断
- ✅ 训练配置创建
- ✅ 模型片段大小估计
- ✅ 合成Template生成

### TemplateMatcher测试
- ✅ Template添加到Matcher
- ✅ Template查找性能（<5ms）
- ✅ Template缓存优化
- ✅ 篮子组织验证
- ✅ 资源匹配评分

### ModelPartitioning测试
- ✅ 2-Stage模型划分
- ✅ 3-Stage模型划分
- ✅ 通信图提取
- ✅ 片段大小一致性
- ✅ 划分验证

### TemplateManager集成测试
- ✅ Template发现
- ✅ Pipeline注册
- ✅ Template学习
- ✅ 统计跟踪

### 性能指标测试
- ✅ Template查找延迟
- ✅ Pipeline组成时间
- ✅ 内存效率

## 运行测试

### 运行完整测试套件
```bash
cd /path/to/fhdp/tests
python test_template_manager.py
```

**预期输出：**
```
test_template_from_pipeline (test_template_manager.TestTemplateGeneration) ... ok
test_resource_pattern_extraction (test_template_manager.TestTemplateGeneration) ... ok
test_communication_pattern_inference (test_template_manager.TestTemplateGeneration) ... ok
test_synthetic_template_generation (test_template_manager.TestTemplateGeneration) ... ok
test_template_addition (test_template_manager.TestTemplateMatching) ... ok
test_template_lookup_performance (test_template_manager.TestTemplateMatching) ... ok
test_template_caching (test_template_manager.TestTemplateMatching) ... ok
test_basket_organization (test_template_manager.TestTemplateMatching) ... ok
test_model_partition_2_stage (test_template_manager.TestModelPartitioning) ... ok
test_model_partition_3_stage (test_template_manager.TestModelPartitioning) ... ok
test_communication_graph_extraction (test_template_manager.TestModelPartitioning) ... ok
test_fragment_size_adherence (test_template_manager.TestModelPartitioning) ... ok
test_template_discovery (test_template_manager.TestTemplateManagerIntegration) ... ok
test_pipeline_registration (test_template_manager.TestTemplateManagerIntegration) ... ok
test_template_learning (test_template_manager.TestTemplateManagerIntegration) ... ok
test_statistics_tracking (test_template_manager.TestTemplateManagerIntegration) ... ok
test_template_lookup_latency (test_template_manager.TestPerformanceMetrics) ... ok
test_pipeline_composition_time (test_template_manager.TestPerformanceMetrics) ... ok
test_memory_efficiency (test_template_manager.TestPerformanceMetrics) ... ok

======================================================================
Test Summary
======================================================================
Tests run: 22
Successes: 22
Failures: 0
Errors: 0
======================================================================
```

### 运行演示脚本
```bash
cd /path/to/fhdp/tests
python demo_template_partitioning.py
```

**预期输出：**
```
======================================================================
FHDP Template-Based Model Partitioning Demo
======================================================================

======================================================================
DEMO 1: 2-Stage Pipeline Partitioning
======================================================================

Fragment Information:
----------------------------------------------------------------------

Stage 0:
  Layers: 8
  Parameters: 12,352
  Size: 0.05 MB
  Layer names: conv1.weight, conv1.bias, bn1.weight, bn1.bias

Stage 1:
  Layers: 12
  Parameters: 2,961,792
  Size: 11.31 MB
  Layer names: conv3.weight, conv3.bias, bn3.weight, bn3.bias

Validation: ✓ PASS

Communication Pattern (from Template)
======================================================================
...
```

## 测试场景

### 场景1: 2-Stage Pipeline（AGX Orin + Orin Nano）
```
Stage 0: AGX Orin (High Resource)
  - 早期卷积层
  - 参数: ~12K
  - 内存: ~0.05 MB

Stage 1: Orin Nano (Medium Resource)
  - 深度卷积层
  - 参数: ~3M
  - 内存: ~11 MB

通信: Stage 0 <-> Stage 1
```

### 场景2: 3-Stage Pipeline（AGX Orin + Orin Nano + AGX Orin）
```
Stage 0: AGX Orin (High Resource)
  - 早期卷积层
  - 参数: ~12K

Stage 1: Orin Nano (Medium Resource)
  - 深度卷积层
  - 参数: ~3M

Stage 2: AGX Orin (High Resource)
  - 全连接层
  - 参数: ~1.5M

通信: 
  - Stage 0 <-> Stage 1
  - Stage 1 <-> Stage 2
  - Stage 0 <-> Stage 2 (跳跃连接)
```

## 扩展和自定义

### 自定义模型划分

如果您有自己的模型，可以添加阶段划分方法：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 定义层 ...
    
    def forward(self, x):
        # ... 前向传播 ...
        return x
    
    def get_stage_parameters(self, stage: int) -> Dict[str, nn.Parameter]:
        """返回指定阶段的参数"""
        if stage == 0:
            return {
                'layer1.weight': self.layer1.weight,
                'layer1.bias': self.layer1.bias,
                # ... 更多参数 ...
            }
        # ... 更多阶段 ...
```

### 自定义通信模式

创建自定义Template：

```python
template = PipelineTemplate(
    template_id="my_custom_template",
    resource_requirements=[
        ResourceClass.HIGH,    # Stage 0
        ResourceClass.MEDIUM,   # Stage 1
        ResourceClass.LOW,      # Stage 2
        ResourceClass.HIGH     # Stage 3
    ],
    expected_duration=20.0,
    communication_pattern=[
        (0, 1),  # 0 -> 1
        (1, 2),  # 1 -> 2
        (2, 3),  # 2 -> 3
        (0, 2),  # 跳跃连接: 0 -> 2
        (1, 3),  # 跳跃连接: 1 -> 3
    ],
    training_config=TrainingConfig(
        epochs=2,
        batch_size=32,
        learning_rate=0.001
    ),
    model_fragment_size=25 * 1024 * 1024  # 25MB per fragment
)
```

## 性能基准

在测试环境中运行基准测试：

```bash
python demo_template_partitioning.py
```

**查看Demo 4的性能输出：**
```
======================================================================
DEMO 4: Performance Benchmarking
======================================================================

Benchmarking model partitioning:
----------------------------------------------------------------------
  Partition 1: 2.45 ms
  Partition 100: 2.38 ms

Statistics:
  Average: 2.41 ms
  Min: 2.30 ms
  Max: 2.55 ms

Benchmarking template lookup:
----------------------------------------------------------------------
  Average: 1.23 ms
  Max: 1.45 ms
  Target: <5.0 ms
  Compliant: ✓ YES
```

## 故障排除

### 问题1: 导入错误
```
ModuleNotFoundError: No module named 'edge_server'
```
**解决方案：**
```bash
cd /path/to/fhdp
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tests/test_template_manager.py
```

### 问题2: PyTorch版本不兼容
```
AttributeError: module 'torch' has no attribute '...'
```
**解决方案：** 确保使用PyTorch 1.10+：
```bash
pip install --upgrade torch
```

### 问题3: 性能测试失败
```
AssertionError: 7.5 not less than 5.0
```
**解决方案：** 这是性能测试失败，可能由于系统负载高。可以：
1. 关闭其他应用程序
2. 增加性能阈值
3. 跳过性能测试：`python -m unittest test_template_manager -k "not performance"`

## 与EVO1集成

要将Template Manager集成到EVO1项目中：

1. **使用EVO1模型**：
```python
from EVO1.model import EVO1Model

model = EVO1Model()
partitioner = TemplateBasedPartitioner(template)
fragments = partitioner.partition_model(model)
```

2. **自定义模型阶段**：
EVO1模型需要实现`get_stage_parameters`方法以支持精确划分。

3. **与Pipeline训练协调器集成**：
```python
from EVO1.pipeline_parallel_federated.coordinator import FHDPipelineCoordinator

coordinator = FHDPipelineCoordinator()
coordinator.use_template_manager(template_manager)
```

## 相关文档

- **FHDP架构**: `/fhdp/docs/ARCHITECTURE.md`
- **Template Manager实现**: `/fhdp/edge_server/template_manager.py`
- **核心类型定义**: `/fhdp/core/types.py`
- **Pipeline训练测试**: `/fhdp/tests/pipeline_test/`

## 贡献

如果您发现测试中的问题或有改进建议，请：

1. 记录失败案例的详细信息
2. 提供系统环境信息
3. 提交issue或PR

## 许可证

本测试套件遵循与FHDP项目相同的许可证。
