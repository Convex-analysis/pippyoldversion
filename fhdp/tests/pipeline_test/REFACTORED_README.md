# FHDP Pipeline Training Test - Refactored Version

## 概述

重构后的测试脚本使用 FHDP 内置的 `cross_platform_comm.py` 进行网络通信，解决了原版本中 `pickle data was truncated` 的问题。

## 主要改进

### 1. 使用 FHDP 内置通信层
- **原版问题**：自己实现的 `NetworkServer`/`NetworkClient`，使用固定 40KB 缓冲区接收数据
- **新版解决**：使用 `cross_platform_comm.py` 的长度前缀协议（4字节长度头 + 循环接收）
- **优势**：
  - 不会因大模型消息被截断
  - 支持 zlib 压缩，减少网络传输量
  - 连接池管理，提高性能
  - 平台自适应优化（Jetson vs x86）

### 2. 修复的 Bug
- `pipeline_formation.py` 第 230 行的类型比较错误：
  ```python
  # 错误: 列表和整数永远不相等
  if len(selected_candidates) == template.resource_requirements:

  # 正确: 比较长度
  if len(selected_candidates) == len(template.resource_requirements):
  ```

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_pipeline_training_refactored.py` | 重构后的测试脚本，使用 FHDP 内置通信层 |
| `run_pipeline_test_refactored.sh` | 重构后的启动脚本 |

## 启动流程

### 前提条件

1. **Mac 建立 SSH 端口转发**（每次重启 Mac 或断开后执行）：
   ```bash
   ssh -L 0.0.0.0:5001:localhost:5001 xta@219.216.64.173 -N -f
   ```

2. **验证转发已建立**：
   ```bash
   ps aux | grep "ssh -L"
   ```

### 第一步：Linux 服务器启动 Server
```bash
cd ~/fhdp/tests/pipeline_test
export FHDP_PORT=5001
./run_pipeline_test_refactored.sh server
```

### 第二步：Jetson 启动 Vehicle

**Jetson AGX Orin：**
```bash
cd ~/fhdp/tests/pipeline_test
export FHDP_PORT=5001
./run_pipeline_test_refactored.sh agx 219.216.65.34
```

**Jetson Orin Nano：**
```bash
cd ~/fhdp/tests/pipeline_test
export FHDP_PORT=5001
./run_pipeline_test_refactored.sh nano 219.216.65.34
```

> **注意**：连接目标是 Mac 的 IP `219.216.65.34`，而不是 Linux 服务器的 IP `219.216.64.173`。

## 网络架构

```
Jetson AGX Orin  ──┐
                   ├──→ Mac (219.216.65.34:5001) ──SSH隧道──→ Linux服务器 (219.216.64.173:5001)
Jetson Orin Nano ──┘
```

## FHDP 内置通信层特性

### 1. 长度前缀协议
```python
# 发送时：先发送4字节长度头，再发送数据
message_length = len(data).to_bytes(4, byteorder='big')
conn.send(message_length)
conn.send(data)

# 接收时：先读4字节长度，再循环读取完整数据
length_data = client_socket.recv(4)
message_length = int.from_bytes(length_data, byteorder='big')
message_data = b''
while len(message_data) < message_length:
    chunk = client_socket.recv(min(message_length - len(message_data), 4096))
    message_data += chunk
```

### 2. 压缩支持
- 使用 `zlib` 压缩消息数据
- 对于模型参数等大消息，压缩率可达 50-70%

### 3. 连接池
- 复用 TCP 连接，减少握手开销
- 自动清理过期连接

### 4. 平台自适应
- **Jetson Orin**：高压缩，批量 50 条消息，5 秒超时
- **Jetson Nano**：高压缩，批量 20 条消息，10 秒超时
- **x86 Linux**：高压缩，批量 100 条消息，3 秒超时

## 故障排查

| 问题 | 检查命令 | 解决方法 |
|------|----------|----------|
| Server 未运行 | `ps aux \| grep test_pipeline` | 重新执行第一步 |
| Mac 转发未建立 | `ps aux \| grep "ssh -L"` | 重新建立 SSH 端口转发 |
| Jetson 连不上 Mac | `nc -zv 219.216.65.34 5001` | 检查 Mac 防火墙是否屏蔽了 5001 端口 |
| 端口 5001 被占用 | `lsof -i :5001` | `kill -9 <PID>` 释放端口 |
| pickle 截断错误 | 无 | 新版使用长度前缀协议，已修复 |

## 关键参数速查

| 参数 | 值 |
|------|----|
| Linux 服务器 IP | `219.216.64.173` |
| Linux 服务器用户名 | `xta` |
| Mac IP | `219.216.65.34` |
| 通信端口 | `5001` |
| Jetson 连接目标 | `219.216.65.34`（Mac，非服务器） |

## 与原版的区别

| 特性 | 原版 | 重构版 |
|------|------|--------|
| 网络通信 | 自定义 `NetworkServer`/`NetworkClient` | FHDP `cross_platform_comm.py` |
| 接收方式 | `recv(40KB)` 固定缓冲 | 4字节长度头 + 循环读取 |
| 序列化 | pickle | JSON + zlib压缩 |
| 连接管理 | 无连接池 | 有连接池 |
| 平台优化 | 无 | Jetson/x86 自适应 |
| 大消息支持 | ❌ 会被截断 | ✅ 完整传输 |
