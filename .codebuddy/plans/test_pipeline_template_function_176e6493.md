---
name: test_pipeline_template_function
overview: 将 test_pipeline_training_refactored.py 中的 SimpleCNN 替换为 EVO1Driving 模型，并为其适配 Jetson 设备的配置（Orin Nano/AGX Orin 差异化参数），同时增加完整的 Template 功能测试：模板查找计时与 Basket 统计、训练后反馈回写、以及 --mode test-template 独立本地测试模式。
todos:
  - id: replace-model-and-dataloader
    content: 将 SimpleCNN 替换为 EVO1Driving：新增 TestModelConfig 和 _build_model_config() 工厂函数，替换 create_mock_driving_data_loader（返回 images/mask/state/controls），更新 server/vehicle __init__ 的模型初始化（AGX=batch4/views3/wp20，Nano=batch2/views2/wp10+set_stage1_mode，Server=batch8），更新广播 payload 加入 num_views/image_size/max_waypoints
    status: completed
  - id: rewrite-training-loop
    content: 重写 _train_locally() 解包新格式并调用 model.forward/compute_loss，加梯度裁剪，optimizer 改为 AdamW，移除 criterion，更新 _handle_global_model 从 payload 读取 num_views/image_size/max_waypoints 创建匹配数据加载器
    status: completed
    dependencies:
      - replace-model-and-dataloader
  - id: enhance-template-observability
    content: 在 _try_form_pipeline() 注入 perf_counter 延迟计时、Top-3 候选评分和 Basket 统计打印；在 _aggregate_model_updates() 第3轮后调用 register_successful_pipeline() 并打印前后 Basket 统计对比
    status: completed
  - id: standalone-template-test-and-argparse
    content: 新增 run_template_standalone_test()：4 组 Mock 车辆 100 次延迟基准测试（avg/max/p99）、Top-3 评分验证、在线学习统计、PASS/FAIL 汇总表格；扩展 argparse choices 加入 'test-template' 并在 main() 添加对应分支
    status: completed
    dependencies:
      - enhance-template-observability
---

## 用户需求

在 `test_pipeline_training_refactored.py` 中，将测试模型从 SimpleCNN 升级为 EVO1Driving，充分利用 Jetson AGX Orin 与 Orin Nano 的硬件差异进行自适应配置，同时对 FHDP TemplateManager 的模板功能进行完整测试和集成。

## 产品概述

修改 `test_pipeline_training_refactored.py`，将 SimpleCNN（~90K 参数）替换为 EVO1Driving（fallback 实现，约 25-30M 参数，无需 HuggingFace 权重下载），实现基于真实自动驾驶模型架构的联邦流水线训练测试，并完整集成 TemplateManager 的可观测性与反馈闭环。

## 核心功能

1. **EVO1Driving 模型集成**

- 使用 fallback 实现：InternVL3Embedder（Conv2d 骨干）+ FlowmatchingActionHead（6 层 Transformer, d_model=512）+ StateEncoder + ControlHead（LSTM）+ ConfidenceEstimator
- 自动驾驶风格的 Mock 数据：多视角图像 `[B,N,3,H,W]` + 车辆状态 `[B,12]` + 控制序列 `[B,T,3]`
- 损失函数：`control_loss + 0.5*waypoint_loss + confidence_loss`（来自 `compute_loss()`）

2. **Jetson 硬件自适应配置**

- AGX Orin（resource_level=high）：batch_size=4，num_views=3，max_waypoints=20，全量参数可训练
- Orin Nano（resource_level=medium）：batch_size=2，num_views=2，max_waypoints=10，调用 `set_stage1_mode()` 冻结 VL 骨干仅训练动作头
- 4090 服务器（server）：batch_size=8，num_views=3，max_waypoints=20，full config

3. **模板查找可观测性（_try_form_pipeline）**

- 对 `find_template_for_vehicles()` 用 `time.perf_counter()` 计时，输出延迟（ms）
- 调用 `matcher.find_best_template()` 获取 Top-3 候选及评分
- 打印 Basket 统计（total_baskets、total_templates、avg_success_rate、cache_hit_rate）

4. **训练完成后的反馈回写（_aggregate_model_updates）**

- 第 3 轮聚合后调用 `register_successful_pipeline()` 反馈结果
- 打印反馈前后 Basket 统计对比，验证模板系统的在线学习效果

5. **独立本地模板测试模式（--mode test-template）**

- 不依赖网络，适合在任一设备上单独运行
- 4 组 Mock 车辆（HIGH+HIGH、HIGH+MEDIUM、MEDIUM+MEDIUM、HIGH+MEDIUM+HIGH）各 100 次查找，统计 avg/max/p99 延迟，与 <5ms 阈值对比
- 模拟 Pipeline 执行后调用 `register_successful_pipeline()` 验证在线学习
- 输出 PASS/FAIL 总结报告

## 技术栈

- Python 3.8+、PyTorch（CPU/CUDA 自适应）、NumPy，与项目完全一致
- `fhdp.EVO1.model.evo1_driving`：EVO1Driving、EVO1DrivingOutput、ModelConfig（inline，需补充 max_steering/max_speed）
- `fhdp.edge_server.template_manager`：TemplateManager、TemplateMatcher（find_best_template、get_template_statistics）
- `fhdp.core.types`：VehicleInfo、Pipeline、ResourceClass、TrainingConfig
- `fhdp.core.constants`：TEMPLATE_LOOKUP_LATENCY_THRESHOLD = 0.005s

---

## 实现方案

### 整体策略

对 `test_pipeline_training_refactored.py` 进行**精确定点修改**，全部变更集中在单一文件内，不引入新文件。分为两类改动：

1. **模型层替换**：SimpleCNN → EVO1Driving，数据格式从 MNIST 风格改为多视角驾驶风格，训练循环适配新接口
2. **模板层注入**：在已有的三个挂载点注入 Template 可观测性和反馈逻辑，并新增 `--mode test-template` 独立测试分支

### 关键设计决策

**1. TestModelConfig 合并设计**

inline `ModelConfig`（evo1_driving.py）缺少 `max_steering`、`max_speed` 字段，而 `_controls_to_waypoints()` 会访问这两个字段，直接使用会崩溃。解决方案：在测试脚本中定义 `TestModelConfig` dataclass，继承/合并 inline ModelConfig 的所有字段，并补充 `max_steering=0.6`、`max_speed=30.0`、`num_views`、`horizon` 等 config.py 中的字段。此方法不修改任何 FHDP 库文件，完全向后兼容。

**2. 硬件自适应模型配置**

Orin Nano 内存（8GB）有限，通过调用 `set_stage1_mode()` 冻结 VL 骨干（InternVL3Embedder 参数），只训练动作头（~6M 参数），显著降低显存占用和梯度计算量。AGX Orin 及服务器使用全量参数。num_views 在 Nano 上从 3 降为 2，进一步减少前向计算量。

**3. 数据序列化兼容**

EVO1Driving 的 state_dict 键名较长（如 `vl_embedder.vision_encoder.0.weight`），但 `_serialize_state_dict`/`_deserialize_state_dict` 基于 dict 遍历，完全兼容，无需修改。

**4. 聚合逻辑兼容**

`_aggregate_model_updates()` 中的 FedAvg 平均逻辑遍历 `updates[0]['model_state'].keys()` 并逐 key 平均，与具体模型结构无关，对 EVO1Driving 的大 state_dict 同样适用，无需修改聚合核心逻辑。

**5. 模板测试独立性**

`run_template_standalone_test()` 在函数内部直接 import 并实例化 TemplateManager，与 server/vehicle 类完全解耦，可在任一机器上单独执行。

---

## 实现细节

### 改动点 1：模型与数据层（最大改动）

**新增 `TestModelConfig` dataclass**（在 EVO1Driving import 区段之后）：

```python
@dataclass
class TestModelConfig:
    # 继承 inline ModelConfig 字段
    vision_encoder: str = "OpenGVLab/InternVL3-1B"
    image_size: int = 224
    max_waypoints: int = 20
    action_dim: int = 8
    per_action_dim: int = 7
    sequence_length: int = 32
    hidden_dim: int = 4096
    # 补充 _controls_to_waypoints 需要的字段
    max_steering: float = 0.6
    max_speed: float = 30.0
    # 驾驶扩展字段
    num_views: int = 3
    horizon: int = 20
    action_hidden_dim: int = 512
```

**`create_mock_driving_data_loader()`** 替换 `create_mock_data_loader()`：

- 参数：`num_samples, batch_size, num_views=3, image_size=224, max_waypoints=20`
- 每次 yield：`(images[B,N,3,H,W], image_mask[B,N], state[B,12], controls[B,T,3])`，值为标准正态分布随机数，controls 用 tanh 限幅到 [-1, 1]

**`_build_model_config(resource_level, device)`** 工厂函数：

```
high   → TestModelConfig(max_waypoints=20, num_views=3) + device='cuda'
medium → TestModelConfig(max_waypoints=10, num_views=2) + device='cuda'  
server → TestModelConfig(max_waypoints=20, num_views=3) + device='cpu'（仅用于聚合）
```

**Server `__init__`**：

- `self.global_model = EVO1Driving(_build_model_config('server'), device='cpu')`

**Vehicle `__init__`**：

- `self.model_config = _build_model_config(resource_level, 'cuda')`
- `self.model = EVO1Driving(self.model_config, device='cuda')`
- Orin Nano（medium）额外调用 `self.model.set_stage1_mode()`
- `self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)`
- 移除 `self.criterion`

**Training config 广播 payload** 中增加：

```python
'num_views': 3,       # server side
'image_size': 224,
'max_waypoints': 20
```

### 改动点 2：训练循环（_train_locally）

```python
# 解包新格式
for batch_idx, (images, image_mask, state, target_controls) in enumerate(train_loader):
    images = images.to(device); image_mask = image_mask.to(device)
    state = state.to(device); target_controls = target_controls.to(device)
    
    self.optimizer.zero_grad()
    output = self.model(images, image_mask, state,
                        mode="training", future_controls=target_controls)
    losses = self.model.compute_loss(output, target_controls)
    loss = losses['total_loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
    
    # 打印分损失
    print(f"  ctrl={losses['control_loss']:.4f} wp={losses['waypoint_loss']:.4f} "
          f"conf={losses['confidence_loss']:.4f}")
```

`_handle_global_model` 中的 `_deserialize_state_dict` 调用保持不变，仅解包 payload 中的 `num_views`/`image_size`/`max_waypoints` 字段创建匹配的数据加载器。

### 改动点 3：Template 可观测性注入

**`_try_form_pipeline()` 模板查找段（原 ~480 行）**：

```python
# Timed lookup
t0 = time.perf_counter()
template = self.template_manager.find_template_for_vehicles(candidate_vehicles)
lookup_ms = (time.perf_counter() - t0) * 1000
threshold_ms = TEMPLATE_LOOKUP_LATENCY_THRESHOLD * 1000
status = "PASS" if lookup_ms < threshold_ms else "FAIL"
print(f"  Template lookup latency: {lookup_ms:.2f}ms (threshold {threshold_ms:.0f}ms) [{status}]")

# Top-3 candidates with scores
top_candidates = self.template_manager.matcher.find_best_template(candidate_vehicles, max_candidates=3)
for rank, (t, score) in enumerate(top_candidates, 1):
    print(f"  Candidate #{rank}: {t.template_id} | score={score:.3f} | "
          f"req={[r.value for r in t.resource_requirements]}")

# Basket statistics
stats = self.template_manager.get_template_statistics()
print(f"  [Basket Stats] baskets={stats['total_baskets']} templates={stats['total_templates']} "
      f"avg_success={stats['avg_success_rate']:.3f} cache_hit={stats['cache_hit_rate']:.3f}")
```

**`_aggregate_model_updates()` 第3轮分支（原 ~640 行）**：

```python
if round_num >= 3:
    # Template feedback
    stats_before = self.template_manager.get_template_statistics()
    elapsed = time.time() - self.active_pipeline.start_time
    self.template_manager.register_successful_pipeline(
        self.active_pipeline, success=True, duration=elapsed)
    stats_after = self.template_manager.get_template_statistics()
    print(f"[Template Feedback] duration={elapsed:.1f}s")
    print(f"  Before: templates={stats_before['total_templates']} "
          f"avg_success={stats_before['avg_success_rate']:.3f}")
    print(f"  After:  templates={stats_after['total_templates']} "
          f"avg_success={stats_after['avg_success_rate']:.3f}")
    ...
```

### 改动点 4：独立测试 + argparse

`run_template_standalone_test(resource_level='medium')` 函数：

- 4 组 mock vehicle groups，每组 100 次 `find_best_template()` 调用
- `time.perf_counter()` 精确计时，numpy 计算 avg/max/p99
- 与 `TEMPLATE_LOOKUP_LATENCY_THRESHOLD` 对比输出 PASS/FAIL
- 调用 `register_successful_pipeline()` 统计模板数量变化
- 汇总报告表格：Group | Avg(ms) | Max(ms) | P99(ms) | BestScore | Status

argparse 扩展：

```python
choices=['server', 'vehicle', 'test-template']
```

main() 新增分支：

```python
elif args.mode == 'test-template':
    run_template_standalone_test(resource_level=args.resource_level)
    sys.exit(0)
```

---

## 目录结构

```
fhdp/tests/pipeline_test/
└── test_pipeline_training_refactored.py   # [MODIFY] 唯一修改文件
    # 修改点 1 (行 ~1-170): 
    #   - 新增 EVO1Driving import（fhdp.EVO1.model.evo1_driving）
    #   - 删除 SimpleCNN class，新增 TestModelConfig dataclass
    #   - 新增 _build_model_config() 工厂函数
    #   - 替换 create_mock_data_loader → create_mock_driving_data_loader
    # 修改点 2 (行 ~175-260, server __init__):
    #   - global_model = EVO1Driving(server_config)
    # 修改点 3 (行 ~480-550, _try_form_pipeline):
    #   - 注入 Template 延迟计时/Top-3 评分/Basket 统计
    # 修改点 4 (行 ~596-650, _aggregate_model_updates):
    #   - 第3轮后注入 register_successful_pipeline() 及前后统计对比
    # 修改点 5 (行 ~715-810, vehicle __init__):
    #   - model = EVO1Driving(device_config), set_stage1_mode for Nano
    #   - optimizer = AdamW, 移除 criterion
    # 修改点 6 (行 ~1042-1104, _train_locally):
    #   - 完整重写适配 EVO1Driving forward/compute_loss
    # 修改点 7 (行 ~1160-1282, run_template_standalone_test + main):
    #   - 新增 run_template_standalone_test() 顶层函数
    #   - argparse choices 加入 'test-template'
    #   - main() 加入对应分支
```