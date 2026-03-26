---
name: test_pipeline_template_function
overview: 修改 test_pipeline_training_refactored.py，增加 Template 功能专项测试支持：在 server 的 pipeline 形成流程中添加详细的 template 查找/匹配日志和性能计时，以及在训练结束后打印 template 统计信息；在 vehicle 端训练完成后调用 register_successful_pipeline 反馈结果；增加 --test-template 模式，用于在本机单独验证 TemplateManager 的查找速度、Basket 组织、匹配分数等功能（不需要网络连接）。
todos:
  - id: enhance-try-form-pipeline
    content: 在 _try_form_pipeline() 中增加模板查找计时、Top-3 候选评分打印及 Basket 统计输出
    status: pending
  - id: feedback-after-training
    content: 在 _aggregate_model_updates() 第3轮聚合后调用 register_successful_pipeline() 并打印反馈前后 Basket 统计对比
    status: pending
  - id: standalone-template-test
    content: 新增 run_template_standalone_test() 函数，实现 4 组 Mock 车辆的延迟基准测试、匹配评分验证、在线学习验证及汇总报告
    status: pending
  - id: wire-argparse-and-main
    content: 扩展 argparse choices 加入 test-template，在 main() 中添加对应分支调用 run_template_standalone_test()
    status: pending
    dependencies:
      - standalone-template-test
---

## 用户需求

在 `test_pipeline_training_refactored.py` 中，利用 FHDP 现有的 TemplateManager 对 Pipeline 模板功能进行完整测试和集成，包括三个方向：

## 产品概述

增强 `test_pipeline_training_refactored.py` 脚本，使其具备模板系统的完整可观测性与反馈闭环，并新增一个无需网络的本地模板测试模式。

## 核心功能

1. **模板查找可观测性（_try_form_pipeline）**

- 对 `find_template_for_vehicles()` 计时，输出查找延迟（毫秒）
- 调用底层 `matcher.find_best_template()` 获取匹配分数，输出 Top-N 候选及评分
- 打印 TemplateManager 的 Basket 统计（篮子数、模板总数、平均成功率、缓存命中率）

2. **训练完成后的反馈回写（_aggregate_model_updates）**

- 第 3 轮聚合完成后，调用 `register_successful_pipeline()` 将本次 Pipeline 执行结果反馈给模板系统
- 打印反馈前后的 Basket 统计，直观对比模板系统的学习效果

3. **独立本地模板测试模式（--mode test-template）**

- 不依赖任何网络，仅在本地运行，适合在 Jetson 或 4090 任一机器上单独执行
- 创建 4 组 Mock 车辆（HIGH+HIGH、HIGH+MEDIUM、MEDIUM+MEDIUM、HIGH+MEDIUM+HIGH）
- 对每组车辆进行 100 次模板查找，统计平均/最大延迟，验证是否满足 <5ms 要求
- 打印各组的最佳模板 ID、匹配分数、Basket 来源
- 模拟完整的 Pipeline 执行后调用 `register_successful_pipeline()`，验证模板系统的在线学习能力
- 最终输出总结报告，标注各项指标是否达标（PASS/FAIL）

## 技术栈

- 与项目完全一致：Python 3.8+、PyTorch、NumPy
- 直接复用 `fhdp.edge_server.template_manager`（TemplateManager / TemplateMatcher / TemplateGenerator）
- 直接复用 `fhdp.core.types`（VehicleInfo、Pipeline、PipelineTemplate、ResourceClass、TrainingConfig）
- 直接复用 `fhdp.core.constants`（TEMPLATE_LOOKUP_LATENCY_THRESHOLD = 0.005）

---

## 实现方案

### 整体策略

对 `test_pipeline_training_refactored.py` 进行**最小侵入式改动**，仅在三个精确位置注入逻辑，并在 `main()` 末尾添加一个新的 `--mode` 分支，所有新增逻辑封装为私有辅助方法，不改变原有服务器/车辆逻辑。

### 三处改动定位

| 改动点 | 原有行号 | 内容 |
| --- | --- | --- |
| `_try_form_pipeline()` 中模板查找段 | 480-496 | 注入延迟计时 + 调用 `find_best_template()` 获取 Top-3 候选评分 + 打印 Basket 统计 |
| `_aggregate_model_updates()` 最终轮 | 640-646 | 第 3 轮聚合结束后调用 `register_successful_pipeline()`，打印前后统计对比 |
| `main()` 的 argparse + if/else | 1208-1277 | 新增 `--mode test-template`，调用独立函数 `run_template_standalone_test()` |


### 关键设计决策

1. **延迟测量**：直接调用 `self.template_manager.matcher.find_best_template(candidate_vehicles)` 并用 `time.perf_counter()` 包裹（比 `time.time()` 精度更高），与 `TEMPLATE_LOOKUP_LATENCY_THRESHOLD` 常量对比后输出 PASS/FAIL。

2. **Basket 统计打印**：复用 `template_manager.get_template_statistics()` 方法，该方法已返回 `total_baskets`、`total_templates`、`avg_success_rate`、`cache_hit_rate`、`memory_usage`，直接格式化打印。

3. **`register_successful_pipeline()` 调用时机**：必须在 `_aggregate_model_updates()` 的第 3 轮分支中调用，此时 `self.active_pipeline` 已确保存在，训练耗时 = `time.time() - self.active_pipeline.start_time`。

4. **独立测试模式**：新增顶层函数 `run_template_standalone_test()`，不依赖网络，仅使用 `TemplateManager`、`TemplateGenerator`、`VehicleInfo`、`Pipeline`，通过函数内部构造 Mock 数据，主循环直接调用该函数后退出，避免干扰原有 server/vehicle 流程。

5. **Mock Pipeline 构造**：测试模式中需构造 `Pipeline` 对象调用 `register_successful_pipeline()`，使用 `start_time = time.time() - elapsed`、`expected_completion = time.time()` 来模拟真实执行时长。

---

## 实现细节

- **`_try_form_pipeline()` 修改**：
- 在调用 `find_template_for_vehicles()` 前后用 `time.perf_counter()` 计时
- 额外调用 `self.template_manager.matcher.find_best_template(candidate_vehicles, max_candidates=3)` 获取候选列表（find_template_for_vehicles 内部已调用，但未暴露分数）
- 打印 Top-3 候选的 template_id 与 score
- 打印 Basket 统计（调用 `get_template_statistics()`）

- **`_aggregate_model_updates()` 修改**：
- 在 `round_num >= 3` 分支的 `_print_summary()` 之前，调用 `register_successful_pipeline()`
- 打印 "Before feedback" 和 "After feedback" 两次 Basket 统计对比

- **`run_template_standalone_test()` 函数**：
- 4 组 Mock 车辆：2 × (HIGH+HIGH)、(HIGH+MEDIUM)、(MEDIUM+MEDIUM)、(HIGH+MEDIUM+HIGH)
- 每组执行 100 次查找，用 `time.perf_counter()` 记录每次耗时，计算 avg/max/p99
- 调用 `matcher.find_best_template()` 获取分数，打印首选模板信息
- 模拟 Pipeline 完成，调用 `register_successful_pipeline()`，统计模板数变化
- 最终打印汇总报告

- **argparse 扩展**：`choices=['server', 'vehicle', 'test-template']`，在 `main()` 中添加对应 elif 分支，调用 `run_template_standalone_test()` 后 `sys.exit(0)`

- **不引入任何新文件**：所有修改集中在 `test_pipeline_training_refactored.py` 一个文件中

---

## 目录结构

```
fhdp/tests/pipeline_test/
└── test_pipeline_training_refactored.py   # [MODIFY] 唯一修改文件
    # 修改点 1: _try_form_pipeline() 中增加延迟计时和 Basket 统计打印
    # 修改点 2: _aggregate_model_updates() 第3轮后增加 register_successful_pipeline() 调用
    # 修改点 3: 新增顶层函数 run_template_standalone_test()
    # 修改点 4: main() 中 argparse choices 添加 'test-template'，新增 elif 分支
```