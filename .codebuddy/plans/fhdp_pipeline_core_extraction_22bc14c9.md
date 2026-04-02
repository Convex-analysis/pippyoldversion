---
name: fhdp_pipeline_core_extraction
overview: 基于 pipeline_proto.py 抽取可复用的 1F1B 调度与流水线通信/执行逻辑，形成可扩展的核心 API/类。
todos:
  - id: scope-deps-audit
    content: 使用 [subagent:code-explorer] 梳理抽取逻辑的调用链与依赖边界
    status: completed
  - id: core-pipeline-module
    content: 新增核心模块并实现 sequence_id、1F1B 调度与序列管理封装
    status: completed
    dependencies:
      - scope-deps-audit
  - id: refactor-proto
    content: 改造 pipeline_proto.py 使用核心 API，保留现有训练与通信行为
    status: completed
    dependencies:
      - core-pipeline-module
  - id: exports-docs
    content: 更新 core/__init__.py 导出新 API 并补充必要注释
    status: completed
    dependencies:
      - core-pipeline-module
---

## User Requirements

- 分析并抽取 `fhdp/tests/pipeline_test/pipeline_proto.py` 中可复用能力，增强核心流水线并行与 1F1B 调度支持
- 抽取内容包含：micro-batch 划分与 warmup/steady/cooldown 逻辑、序列消息管理（sequence_id 与 handler 注册/注销）、通用阶段执行逻辑
- 抽取后的逻辑不固定为 stage0/stage1 两阶段，需更具可扩展性与可复用性
- 以可复用 API/类的形式提供给其他脚本调用
- 无新增或改动界面展示

## Product Overview

- 将原型脚本中的流水线执行与调度逻辑沉淀到核心模块，形成通用的流水线运行与调度能力

## Core Features

- 通用 1F1B 调度与 micro-batch 阶段划分
- 可复用的序列消息管理与 sequence_id 生成
- 可配置的通用阶段执行骨架（支持多阶段扩展）

## Tech Stack Selection

- 语言：Python（基于现有 FHDP 代码结构）
- 训练/张量：PyTorch（脚本中已使用）
- 通信：现有 `PipelineMessage` 与 `PipelineCommunicationManager`（`fhdp/core/cross_platform_comm.py`）

## Implementation Approach

- 将原型脚本中的调度与序列管理逻辑抽取为核心模块，形成可复用 API。
- 提供可扩展的 1F1B 调度策略与 micro-batch 阶段判定，按 `num_stages` 与 `stage_index` 生成 warmup/steady/cooldown 阶段信息。
- 封装序列消息注册/注销与 sequence_id 生成，避免业务代码重复拼接与清理逻辑。
- `pipeline_proto.py` 仅保留具体训练/数据加载与最小业务组装，改用核心 API 驱动。
- 性能：调度计算为 O(micro_batches)，避免重复构造/销毁序列 handler；保留现有消息通道与序列缓冲机制，降低改动风险。

## Implementation Notes (Execution Details)

- 复用 `PipelineCommunicationManager` 的 sequence handler 机制，抽取为轻量封装，不修改通信层协议。
- 1F1B 阶段判定需处理 `micro_batches < num_stages` 的边界，保证阶段标签稳定。
- 保持 `pipeline_proto.py` 的功能行为一致，避免改变消息格式与训练张量流转。
- 保留现有日志输出结构，新增日志不包含张量数据。

## Architecture Design

- 新增核心模块负责调度与序列管理：`pipeline_parallel.py`
- 原型脚本调用核心模块 API，作为具体示例实现
- 通信仍通过 `cross_platform_comm.py` 的 `PipelineCommunicationManager` 完成

## Directory Structure Summary

本次实现新增核心流水线调度与序列管理模块，并改造原型脚本使用新 API。

/Volumes/HardDriveMac/EXP/pippyoldversion/
├── fhdp/
│   ├── core/
│   │   ├── pipeline_parallel.py  # [NEW] 核心流水线并行支持模块。包含 sequence_id 生成、1F1B 调度策略、micro-batch 阶段判定、序列 handler 管理封装，以及通用阶段运行骨架。
│   │   └── **init**.py           # [MODIFY] 导出新增核心 API，便于外部调用。
│   └── tests/
│       └── pipeline_test/
│           └── pipeline_proto.py # [MODIFY] 改用核心 API 进行 1F1B 调度与序列管理，保留数据加载与具体训练逻辑。

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 搜索并核实跨文件依赖与既有通信/调度模式，确保抽取接口与现有架构一致
- Expected outcome: 明确可复用逻辑边界与调用链，避免引入不一致的核心 API