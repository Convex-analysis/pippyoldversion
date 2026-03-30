---
name: pipeline_proto_resnet18
overview: 为 FHDP 新增独立 `pipeline_proto.py` 原型脚本：固定 AGX=Stage0、Orin=Stage1、服务器仅协调，使用 `timm` 的 ResNet‑18 与 `torchvision.datasets.FakeData` 做单 micro‑batch 的流水线前向/反向通信，并预留后续 template 管理接口。
todos:
  - id: scan-proto-patterns
    content: 确认 cross_platform_comm 与测试脚本的消息类型与序列化模式
    status: completed
  - id: implement-pipeline-proto
    content: 实现 pipeline_proto.py：固定拓扑、ResNet-18 拆分、激活/梯度传输闭环
    status: completed
    dependencies:
      - scan-proto-patterns
  - id: wire-control-flow
    content: 补齐服务器协调流程、训练轮次控制、心跳与状态日志
    status: completed
    dependencies:
      - implement-pipeline-proto
  - id: document-run-steps
    content: 补充脚本内运行说明与参数示例（server/AGX/Orin）
    status: completed
    dependencies:
      - wire-control-flow
---

## 用户需求

- 固定 pipeline 拓扑：AGX=Stage0、Orin=Stage1，服务器仅协调不计算。
- 原型使用 ResNet-18 进行测试；不依赖 pippy，仅使用 FHDP 通信与消息路由。
- 采用可扩展设计，后续可引入 template 管理 pipeline 划分，并扩展到多 micro-batch/1F1B。

## 产品概述

- 新增一个独立的 pipeline 原型脚本，演示单 micro-batch 的前向激活传输与反向梯度回传的最小闭环。

## 核心特性

- 固定两阶段 pipeline 的端到端通信流程（激活/梯度）。
- 服务器协调与状态广播，不参与计算。
- 使用 ResNet-18 与 FakeData 生成可重复的训练数据流。

## 技术栈选择

- 语言：Python（沿用现有项目）
- 深度学习：PyTorch
- 模型：timm.models.resnet 中的 ResNet-18
- 通信：fhdp/core/cross_platform_comm.py（现有 FHDP 通信栈）

## 实现方案

- 新增独立脚本 `pipeline_proto.py`，复用 `cross_platform_comm` 的消息路由与连接管理。
- 固定拓扑映射：AGX -> Stage0，Orin -> Stage1；服务器仅负责连接、注册、路由与训练轮次协调。
- 使用单 micro-batch 进行前向激活传输（Stage0 -> Stage1）与反向梯度回传（Stage1 -> Stage0），以最小闭环验证协议。
- 为后续模板接入预留结构（例如 pipeline_id、stage_id、template_id 字段）但不改变现有 PipelineFormation 逻辑。

## 实现细节（执行要点）

- 复用 `CrossPlatformMessage` 的消息类型进行“activation/gradient/control”载荷传输，避免新增协议。
- 数据序列化采用张量转 list 的 JSON 方式，保持与现有测试脚本一致，避免兼容性问题。
- 服务器不参与模型实例化与计算，只广播控制消息与转发路由。
- 确保训练轮次和 sequence_id 的一致性，避免重复处理与竞争。

## 架构设计

- Server：注册/协调/路由/心跳，生成 pipeline 指令与训练轮次。
- Stage0(AGX)：运行 ResNet-18 前半段，输出激活并接收梯度进行反向与更新。
- Stage1(Orin)：运行 ResNet-18 后半段，计算 loss，反向得到输入梯度并回传给 Stage0。

## 目录结构

project-root/
├── fhdp/tests/pipeline_test/
│   └── pipeline_proto.py  # [NEW] 独立 pipeline 原型脚本。实现固定拓扑通信、ResNet-18 拆分、单 micro-batch 前向/反向传输与训练轮次协调。需复用 cross_platform_comm 的消息发送与路由模式，并预留 template 扩展字段。