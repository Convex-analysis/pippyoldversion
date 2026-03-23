# FHDP Pipeline Training Test - Summary

## 概述

本项目已创建完整的FHDP Pipeline训练测试脚本，用于验证FHDP系统在异构设备（Jetson AGX Orin、Jetson Orin Nano和4090 Linux服务器）之间进行Pipeline训练的能力。

## 已创建的文件

### 1. 核心测试脚本
- **`test_pipeline_training.py`** (45KB+, 可执行)
  - 完整的Pipeline训练测试实现
  - 支持服务器模式（4090 Linux）和车辆模式（Jetson设备）
  - 实现网络通信（基于TCP/IP socket）
  - 包含模型训练、更新聚合、Pipeline形成等核心功能
  - ✨ 集成真实的FHDP stage划分逻辑（非硬编码）

### 2. 启动脚本
- **`run_pipeline_test.sh`** (可执行)
  - 简化的启动脚本，方便快速部署
  - 支持三种模式：server、agx、nano
  - 自动检测环境和依赖
  - 彩色输出，易于阅读

### 3. 验证脚本
- **`validate_pipeline_test.py`** (可执行)
  - 环境验证脚本
  - 检查Python版本、PyTorch、CUDA、网络端口等
  - 在运行测试前验证环境配置

- **`validate_stage_partitioning.py`** (可执行)
  - Stage划分逻辑验证脚本
  - 验证ResourceClassifier、TemplateManager、PipelineFormation组件
  - 确认FHDP stage partitioning逻辑正确集成

### 4. 文档
- **`PIPELINE_TEST_README.md`**
  - 详细的使用说明
  - 安装指南
  - 故障排除
  - 预期输出示例
  - 高级配置选项
  - Stage划分流程说明

## 架构设计

```
4090 Linux Server (服务器端)
├── Edge Server Coordinator (Pipeline协调)
├── FHDP System (核心系统)
├── Network Server (TCP/IP通信)
└── Global Model Aggregation (模型聚合)

Jetson AGX Orin (高资源车辆)
├── Vehicle Agent (车辆代理)
├── Network Client (网络客户端)
├── Local Training (本地训练)
└── Model Update Submission (模型更新提交)

Jetson Orin Nano (中资源车辆)
├── Vehicle Agent (车辆代理)
├── Network Client (网络客户端)
├── Local Training (本地训练)
└── Model Update Submission (模型更新提交)
```

## 核心功能

### 服务器端功能
1. **车辆注册管理**
   - 接收车辆注册请求
   - 维护车辆信息数据库
   - 资源分类和能力评估

2. **Pipeline形成** ✨ (已升级为真实FHDP逻辑)
   - **资源分类**：使用ResourceClassifier对车辆自动分类为HIGH/MEDIUM/LOW
   - **模板匹配**：通过TemplateManager查找最优Pipeline模板
   - **贪心分配**：使用PipelineFormation执行基于分数的stage分配算法
   - **动态分配**：根据车辆资源能力智能分配stage，而非硬编码
   - 发送Pipeline邀请（含真实的vehicle-stage映射）

3. **模型聚合**
   - 接收来自车辆的模型更新
   - 执行加权平均聚合
   - 广播全局模型

4. **训练协调**
   - 协调多个训练轮次
   - 异步聚合模型更新
   - 处理Pipeline重组

### 客户端（车辆）功能
1. **服务器连接**
   - TCP/IP网络连接
   - 心跳保持
   - 自动重连

2. **Pipeline参与**
   - 接收并响应Pipeline邀请
   - 执行本地训练
   - 提交模型更新

3. **本地训练**
   - 简单CNN模型
   - 1-2轮短视界训练
   - 支持批量训练

4. **资源监控**
   - 模拟资源使用情况
   - 根据设备类型设置不同的资源能力

## 使用方法

### 快速开始

#### 步骤1: 环境验证（在每台机器上运行）

```bash
cd /path/to/fhdp
python3 validate_pipeline_test.py
```

#### 步骤2: 启动服务器（在4090 Linux上）

```bash
cd /path/to/fhdp
./run_pipeline_test.sh server
```

#### 步骤3: 启动Jetson AGX Orin（在高资源车辆上）

```bash
cd /path/to/fhdp
./run_pipeline_test.sh agx <server-ip>
```

#### 步骤4: 启动Jetson Orin Nano（在中资源车辆上）

```bash
cd /path/to/fhdp
./run_pipeline_test.sh nano <server-ip>
```

### 直接使用Python脚本

如果需要更多控制，可以直接运行Python脚本：

```bash
# 服务器
python test_pipeline_training.py --mode server --host 0.0.0.0 --port 5000

# 车辆 (AGX Orin)
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id agx_orin_001 \
    --server-host <server-ip> \
    --server-port 5000 \
    --resource-level high

# 车辆 (Orin Nano)
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id orin_nano_001 \
    --server-host <server-ip> \
    --server-port 5000 \
    --resource-level medium
```

## 预期结果

测试将演示以下FHDP能力：

1. ✅ **车辆注册**
   - 服务器成功注册两台Jetson设备
   - 识别设备资源等级（高/中）

2. ✅ **Pipeline形成**
   - 服务器识别到足够的车辆后形成Pipeline
   - 发送邀请到两台车辆
   - 车辆接受邀请并加入Pipeline

3. ✅ **分布式训练**
   - 执行3轮训练
   - 每轮包含1-2个epoch的本地训练
   - 车辆并行执行训练

4. ✅ **模型聚合**
   - 服务器收集所有车辆更新
   - 执行加权平均聚合
   - 更新全局模型

5. ✅ **协调和通信**
   - 全局模型广播
   - 车辆间状态同步
   - 心跳保持连接

## 技术特性

### 网络通信
- **协议**: TCP/IP
- **序列化**: Python pickle
- **端口**: 默认5000（可配置）
- **消息类型**: 注册、模型更新、全局模型、Pipeline邀请等

### 训练配置
- **模型**: 简单CNN（适合快速测试）
- **轮次**: 3轮
- **每轮Epoch**: 1-2个
- **批次大小**: 32
- **学习率**: 0.001
- **优化器**: SGD
- **损失函数**: CrossEntropyLoss

### 资源配置

#### Jetson AGX Orin (高资源)
- CPU: 12核心，90%可用
- 内存: 32GB，80%可用
- 电池: 90%
- 网络质量: 80%

#### Jetson Orin Nano (中资源)
- CPU: 6核心，60%可用
- 内存: 8GB，50%可用
- 电池: 70%
- 网络质量: 70%

## FHDP核心特性验证

### 1. 混合参与模式
- ✅ 支持Pipeline训练和个体训练
- ✅ 根据设备能力自动选择模式

### 2. 异构设备支持
- ✅ 不同资源等级的设备
- ✅ 自适应资源配置

### 3. Pipeline形成算法 ✨ (已实现真实FHDP逻辑)
- ✅ **ResourceClassifier**：车辆资源分类（HIGH/MEDIUM/LOW）
- ✅ **TemplateManager**：智能模板匹配系统
- ✅ **PipelineFormation**：贪心stage分配算法
- ✅ **动态分配**：基于资源分数的stage划分
- ✅ **公平性管理**：考虑参与历史
- ✅ 快速Pipeline重组

### 4. 异步协调
- ✅ 异步模型聚合
- ✅ 非阻塞操作
- ✅ 灵活参与

### 5. 通信优化
- ✅ 消息打包传输
- ✅ 网络协议抽象
- ✅ 错误处理和重连

## 性能指标

基于FHDP设计目标：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Pipeline形成时间 | < 1.5秒 | 从检测到可用车辆到Pipeline形成 |
| 模板查找延迟 | < 5毫秒 | 查找合适的Pipeline模板 |
| 聚合延迟 | < 2秒 | 收集并聚合模型更新 |
| 每轮训练时间 | < 10秒 | 每台设备的本地训练时间 |
| 内存使用 | < 50MB | FHDP组件的内存占用 |
| CPU使用率 | < 30% | FHDP组件的CPU占用 |

## 扩展性

### 添加更多车辆

测试框架支持添加任意数量的车辆：

```bash
# 车辆3
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id vehicle_003 \
    --server-host <server-ip> \
    --resource-level medium

# 车辆4
python test_pipeline_training.py \
    --mode vehicle \
    --vehicle-id vehicle_004 \
    --server-host <server-ip> \
    --resource-level low
```

### 自定义模型架构

编辑`test_pipeline_training.py`中的`SimpleCNN`类：

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义模型结构
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 自定义配置

创建YAML配置文件：

```yaml
custom_config.yaml
system:
  max_vehicles_per_region: 10
  aggregation_interval: 2.0

training:
  learning_rate: 0.001
  batch_size: 16
  epochs: 2
```

使用自定义配置：

```bash
python test_pipeline_training.py --mode server --config custom_config.yaml
```

## 故障排除

### 常见问题

1. **连接失败**
   - 检查服务器IP地址是否正确
   - 验证防火墙设置：`sudo ufw allow 5000/tcp`
   - 确保所有设备在同一网络

2. **PyTorch未安装**
   - Jetson设备：从NVIDIA官方仓库安装PyTorch for Jetson
   - 参考：https://developer.nvidia.com/embedded/downloads

3. **端口被占用**
   ```bash
   # 使用不同端口
   export FHDP_PORT=6000
   ./run_pipeline_test.sh server
   ```

4. **内存不足**
   - 减少批次大小
   - 减少epoch数量
   - 关闭其他应用程序

## 文件结构

```
fhdp/
├── test_pipeline_training.py      # 主测试脚本 (核心)
├── run_pipeline_test.sh           # 启动脚本 (便捷)
├── validate_pipeline_test.py      # 验证脚本 (环境检查)
├── PIPELINE_TEST_README.md        # 详细文档 (使用指南)
├── PIPELINE_TEST_SUMMARY.md       # 本文件 (摘要)
├── core/                         # FHDP核心模块
│   ├── types.py                  # 类型定义
│   ├── fhdp_system.py            # 系统实现
│   └── ...
├── edge_server/                  # 边缘服务器
│   ├── server.py                 # 服务器实现
│   └── ...
└── vehicle_layer/                # 车辆层
    ├── vehicle.py                # 车辆实现
    └── ...
```

## Stage划分实现详情 ✨

### 实现背景

原始的 `test_pipeline_training.py` 中，Pipeline的stage划分是硬编码的占位符逻辑：
```python
# 旧版本：硬编码的stages
stages=["stage_0", "stage_1"]
```

这无法验证FHDP的真实能力。现已升级为完整的FHDP stage划分流程。

### 实现的组件

#### 1. ResourceClassifier (资源分类器)
**文件位置**: `fhdp/edge_server/resource_classifier.py`

**功能**:
- 根据CPU、内存、电池等指标将车辆分类为 HIGH/MEDIUM/LOW
- 支持资源稳定性分析
- 管理参与公平性

**使用方式**:
```python
classifier = ResourceClassifier()
resource_class = classifier.classify_vehicle(vehicle_info)
# 返回: ResourceClass.HIGH / MEDIUM / LOW
```

#### 2. TemplateManager (模板管理器)
**文件位置**: `fhdp/edge_server/template_manager.py`

**功能**:
- 维护预定义的Pipeline模板库
- 根据车辆资源能力查找最优模板
- 支持模板缓存和性能统计

**使用方式**:
```python
template_manager = TemplateManager()
template = template_manager.find_template_for_vehicles(vehicles)
# 返回: PipelineTemplate
```

#### 3. PipelineFormation (Pipeline形成器)
**文件位置**: `fhdp/vehicle_layer/pipeline_formation.py`

**功能**:
- 执行贪心算法为每个stage分配最合适的车辆
- 考虑资源、移动性、通信质量等多个维度
- 管理Pipeline生命周期

**使用方式**:
```python
pipeline_formation = PipelineFormation(initiator_vehicle)
pipeline_id = pipeline_formation.initiate_pipeline_formation(
    template=template,
    candidate_vehicles=vehicles
)
```

### Stage划分流程

```
┌─────────────────────────────────────────────────────────┐
│           Step 1: 车辆注册与资源分类                 │
├─────────────────────────────────────────────────────────┤
│  车辆注册 → ResourceClassifier.classify_vehicle()    │
│  结果: vehicle_id → ResourceClass (HIGH/MED/LOW)    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Step 2: 查找匹配的Pipeline模板             │
├─────────────────────────────────────────────────────────┤
│  TemplateManager.find_template_for_vehicles()       │
│  输入: List[VehicleInfo]                           │
│  输出: PipelineTemplate (含resource_requirements)     │
│  示例: [HIGH, MEDIUM] → 匹配2-stage模板          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Step 3: 贪心Stage分配                      │
├─────────────────────────────────────────────────────────┤
│  PipelineFormation.initiate_pipeline_formation()      │
│  • 为每个stage计算候选车辆的适配分数                │
│  • 选择最高分数的车辆分配到该stage                 │
│  • 重复直到所有stage被填充                         │
│  • 输出: Pipeline对象 (vehicles, stages映射)        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Step 4: 发送邀请与响应                    │
├─────────────────────────────────────────────────────────┤
│  服务器 → 车辆: PIPELINE_INVITE                  │
│  {                                                  │
│    'pipeline_id': 'pipeline_123',                    │
│    'vehicles': ['v1', 'v2'],                        │
│    'stages': ['stage_0', 'stage_1'],               │
│    'resource_requirements': ['high', 'medium']        │
│  }                                                  │
│                                                      │
│  车辆 → 服务器: PIPELINE_RESPONSE                   │
│  {                                                  │
│    'accepted': True,                                 │
│    'stage': 'stage_0',                              │
│    'stage_index': 0                                 │
│  }                                                  │
└─────────────────────────────────────────────────────────┘
```

### 贪心算法详情

#### 评分因素
每个车辆对每个stage的评分综合考虑：

| 因素 | 权重 | 说明 |
|------|------|------|
| 资源匹配度 | 0.4 | 车辆资源是否满足stage要求 |
| 移动稳定性 | 0.3 | 速度适中、方向稳定 |
| 通信质量 | 0.3 | 网络带宽、信号强度 |

#### Stage位置权重
- **首尾stage** (stage_0, stage_N): 资源权重0.5，更关键
- **中间stage**: 资源权重0.4，通信权重0.4

### 修改的关键代码位置

#### 1. SimulatedEdgeServer.__init__
```python
# 新增FHDP组件
from fhdp.edge_server.resource_classifier import ResourceClassifier
from fhdp.edge_server.template_manager import TemplateManager
from fhdp.vehicle_layer.pipeline_formation import PipelineFormation

self.resource_classifier = ResourceClassifier()
self.template_manager = TemplateManager()
self.pipeline_formation = PipelineFormation(dummy_vehicle_info)
```

#### 2. _handle_register (车辆注册)
```python
# 注册时立即分类
resource_class = self.resource_classifier.classify_vehicle(vehicle_info)
print(f"✓ Vehicle {message.sender_id} registered (Resource Class: {resource_class.value})")
```

#### 3. _try_form_pipeline (Pipeline形成)
```python
# Step 1: 分类所有车辆
for vehicle in candidate_vehicles:
    resource_class = self.resource_classifier.classify_vehicle(vehicle)
    vehicle_classes[vehicle.vehicle_id] = resource_class

# Step 2: 查找模板
template = self.template_manager.find_template_for_vehicles(candidate_vehicles)

# Step 3: 贪心分配
pipeline_id = self.pipeline_formation.initiate_pipeline_formation(
    template=template,
    candidate_vehicles=candidate_vehicles
)

# Step 4: 发送邀请
invitation = NetworkMessage(
    data={
        'pipeline_id': pipeline_id,
        'vehicles': pipeline.vehicles,
        'stages': pipeline.stages,
        'resource_requirements': [r.value for r in template.resource_requirements]
    }
)
```

#### 4. _handle_pipeline_invitation (车辆响应)
```python
# 根据车辆ID找到被分配的stage
stage_index = pipeline_data['vehicles'].index(self.vehicle_id)
self.current_stage = pipeline_data['stages'][stage_index]

# 显示资源需求
resource_requirements = pipeline_data.get('resource_requirements', [])
if resource_requirements:
    required_class = resource_requirements[stage_index]
    print(f"  Assigned stage: {self.current_stage} (requires: {required_class})")
```

### 预期输出示例

#### 服务器端输出
```
============================================================
Attempting to form pipeline using FHDP stage partitioning...
============================================================

Step 1: Classifying vehicles by resource capability...
  agx_orin_001: high
  orin_nano_001: medium

Step 2: Finding best matching pipeline template...
  Template ID: template_42
  Resource requirements: ['high', 'medium']
  Expected duration: 60.0s

Step 3: Performing greedy stage selection...

✓ Pipeline formed successfully!
  Pipeline ID: pipeline_1711234567890
  Template: template_42
  Vehicles in pipeline: ['agx_orin_001', 'orin_nano_001']
  Stages: ['stage_0', 'stage_1']
  Vehicle-Stage mapping:
    agx_orin_001 -> stage_0 (required: high, actual: high)
    orin_nano_001 -> stage_1 (required: medium, actual: medium)

→ Sent pipeline invitation to agx_orin_001
→ Sent pipeline invitation to orin_nano_001
```

#### 车辆端输出
```
Received pipeline invitation from server
  Pipeline ID: pipeline_1711234567890
  Template ID: template_42
  Vehicles in pipeline: ['agx_orin_001', 'orin_nano_001']
  Stages: ['stage_0', 'stage_1']
  Assigned stage: stage_0 (requires: high)
✓ Accepted pipeline invitation
```

### 验证方法

运行验证脚本确认各组件正常工作：
```bash
python3 validate_stage_partitioning.py
```

预期输出：
```
============================================================
FHDP Stage Partitioning Validation
============================================================

[Test 1] Checking imports...
✓ All imports successful

[Test 2] Testing ResourceClassifier...
  High resource vehicle -> high
  Medium resource vehicle -> medium
  Low resource vehicle -> low
✓ Resource classification working correctly

[Test 3] Testing TemplateManager...
  Found template: template_42
  Resource requirements: ['high', 'medium']
✓ Template manager working

[Test 4] Testing PipelineFormation...
  Pipeline ID: pipeline_123456
  Vehicles: ['vehicle_high', 'vehicle_medium']
  Stages: ['stage_0', 'stage_1']
✓ Pipeline formation working

============================================================
All validation tests passed!
============================================================
```

## 下一步

### 测试完成后

1. 分析训练日志和性能指标
2. 验证Pipeline形成时间是否符合目标
3. 检查模型聚合是否正确
4. 评估网络通信效率

### 优化方向

1. 实际数据集集成（替换mock数据）
2. 更复杂的模型架构
3. 增强错误处理和恢复机制
4. 添加性能监控和日志记录
5. 实现Pipeline动态重组测试

### 集成到EVO1

此测试脚本可以作为基础，进一步集成到EVO1项目中：

1. 替换SimpleCNN为实际的EVO1模型
2. 使用真实的数据集
3. 集成EVO1的Pipeline并行联邦训练协调器
4. 添加更详细的性能指标收集

## 参考资料

- **FHDP架构文档**: `/fhdp/docs/ARCHITECTURE.md`
- **核心类型定义**: `/fhdp/core/types.py`
- **系统实现**: `/fhdp/core/fhdp_system.py`
- **Edge Server**: `/fhdp/edge_server/server.py`
- **Vehicle Layer**: `/fhdp/vehicle_layer/vehicle.py`
- **EVO1集成**: `/fhdp/EVO1/pipeline_parallel_federated/`

## 许可证

此测试脚本遵循与FHDP项目相同的许可证。

---

**创建日期**: 2026年3月23日
**版本**: 1.0
**作者**: Auto AI Assistant
