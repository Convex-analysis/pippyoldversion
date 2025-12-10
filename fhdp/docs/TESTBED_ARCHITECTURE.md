# FHDP异构测试平台完整架构设计

## 🏗️ **测试平台架构概览**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FHDP 测试平台集群架构                          │
├─────────────────────────────────────────────────────────────────┤
│  管理节点 (Control Node)                                        │
│  ├── Kubernetes Master (k3s)                                   │
│  ├── Ansible Control Engine                                     │
│  ├── Prometheus + Grafana监控栈                                 │
│  ├── ELK日志聚合系统                                            │
│  └── NFS/分布式存储                                             │
├─────────────────────────────────────────────────────────────────┤
│  计算节点层 (Compute Nodes)                                     │
│  ├── Jetson Orin Nano集群 (边缘计算层)                          │
│  │   ├── Jetson-01 (Stage-1 Pipeline)                          │
│  │   ├── Jetson-02 (Stage-2 Pipeline)                          │
│  │   ├── Jetson-03 (Stage-3 Pipeline)                          │
│  │   └── Jetson-04 (备用/热备份)                               │
│  └── x86 PC集群 (云端训练层)                                    │
│      ├── PC-01 (Edge Server + Aggregator)                      │
│      ├── PC-02 (训练节点 + Model Store)                         │
│      └── PC-03 (数据预处理 + Pipeline Manager)                   │
├─────────────────────────────────────────────────────────────────┤
│  网络基础设施                                                   │
│  ├── 10GbE高速以太网 (数据平面)                                 │
│  ├── 1GbE管理网络 (控制平面)                                    │
│  ├── Wi-Fi 6 (V2X仿真)                                         │
│  └── 专用V2X测试设备                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 **核心工具栈选择**

### **1. 集群管理软件**

#### **主要选择：Kubernetes (k3s) + Docker**
```yaml
# 推荐理由：
优势：
  - 轻量级Kubernetes发行版，适合资源受限的Jetson设备
  - 原生支持ARM64和x86_64混合架构
  - 内置网络存储和负载均衡
  - 丰富的生态系统和Helm charts

版本要求：
  - k3s: v1.28+ (支持ARM64和x86_64)
  - Docker: 24.0+ (容器运行时)
  - Helm: 3.12+ (包管理)
```

#### **备选方案：Docker Swarm**
```yaml
适用场景：
  - 学习成本较低的小型集群
  - 不需要复杂调度的简单工作负载
  - 快速原型验证

限制：
  - 调度能力有限
  - 不适合复杂的FHDP流水线编排
```

### **2. 设备配置管理**

#### **主要选择：Ansible + AWX**
```yaml
Ansible Playbooks结构：
├── inventory/
│   ├── jetson_nodes.yml      # Jetson设备清单
│   ├── pc_nodes.yml          # PC节点清单
│   └── cluster_vars.yml      # 集群变量
├── playbooks/
│   ├── setup_base.yml        # 基础环境配置
│   ├── install_docker.yml    # Docker安装
│   ├── install_k3s.yml       # k3s集群部署
│   ├── configure_fhdp.yml    # FHDP应用部署
│   └── monitoring_setup.yml  # 监控系统配置
├── roles/
│   ├── jetson_optimization/  # Jetson特定优化
│   ├── network_config/       # 网络配置
│   └── security_hardening/   # 安全加固
└── group_vars/
    ├── jetson.yml           # Jetson通用变量
    └── pc_nodes.yml         # PC节点变量

AWX (Ansible Tower开源版)：
  - Web界面管理Playbooks
  - RBAC权限控制
  - 调度执行和日志记录
  - API集成能力
```

### **3. 远程访问解决方案**

#### **推荐方案：WireGuard VPN + ZeroTier**
```yaml
WireGuard (主VPN方案)：
优势：
  - 高性能，低CPU开销
  - 现代加密协议
  - 简单配置
  - 适合Jetson设备

配置示例：
[Interface]
PrivateKey = <jetson-private-key>
Address = 10.0.0.101/24
ListenPort = 51820

[Peer]
PublicKey = <control-node-public-key>
Endpoint = control.example.com:51820
AllowedIPs = 10.0.0.0/24, 192.168.1.0/24

ZeroTier (备选方案)：
优势：
  - 零配置NAT穿透
  - SD-WAN特性
  - 管理控制台
```

#### **SSH管理：ClusterSSH + tmux**
```bash
# 批量SSH管理工具
apt install clusterssh tmux

# 配置文件 ~/.clusterssh/config
tag_jetson=jetson-01 jetson-02 jetson-03 jetson-04
tag_pcs=pc-01 pc-02 pc-03
tag_all=jetson-01 jetson-02 jetson-03 jetson-04 pc-01 pc-02 pc-03

# 使用示例
cssh jetson-01 jetson-02 jetson-03    # 同时连接多个Jetson
cssh tag_jetson                       # 连接所有Jetson设备
cssh tag_all                          # 连接所有节点
```

### **4. 监控和日志系统**

#### **监控栈：Prometheus + Grafana + Node Exporter**
```yaml
# docker-compose.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:v2.40.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    
  grafana:
    image: grafana/grafana:9.3.0
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
    
  node-exporter:
    image: prom/node-exporter:v1.5.0
    ports:
      - "9100:9100"
    volumes:
      - /:/host:ro,rslave
      
  jetson-stats:
    image: dustinr/jetson-stats:latest
    ports:
      - "9101:9100"
    privileged: true
    volumes:
      - /:/host:ro,rslave

volumes:
  prometheus_data:
  grafana_data:
```

#### **日志聚合：ELK Stack (Elasticsearch + Logstash + Kibana)**
```yaml
# 轻量级替代方案：Grafana Loki + Promtail
loki:
  image: grafana/loki:2.7.0
  ports:
    - "3100:3100"
  volumes:
    - loki_data:/loki

promtail:
  image: grafana/promtail:2.7.0
  volumes:
    - /var/log:/var/log:ro
    - ./promtail-config.yml:/etc/promtail/config.yml
```

---

## 🚀 **设备供应和初始化流程**

### **1. Jetson Orin Nano初始化**

#### **基础镜像准备**
```bash
# 创建自定义Jetson镜像
#!/bin/bash

# 1. 下载JetPack SDK
wget https://developer.nvidia.com/embedded/jetpack-sdk-55

# 2. 创建基础镜像
sudo ./flash.sh jetson-orin-nano-devkit internal

# 3. 优化系统配置
cat << EOF | sudo tee /etc/systemd/system/jetson-optimization.service
[Unit]
Description=Jetson Performance Optimization
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/optimize-jetson.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable jetson-optimization
```

#### **Docker和k3s安装脚本**
```bash
#!/bin/bash
# install_k3s_jetson.sh

# 禁用swap
sudo dphys-swapfile swapoff
sudo dphys-swapfile uninstall
sudo update-rc.d dphys-swapfile remove

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安装k3s (ARM64版本)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker" sh -

# 配置GPU支持
echo 'nvidia' | sudo tee -a /etc/modules
sudo modprobe nvidia
```

### **2. PC节点配置**

#### **自动化部署脚本**
```bash
#!/bin/bash
# setup_pc_node.sh

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装k3s worker
sudo k3s agent --server https://control-node:6443 --token ${K3S_TOKEN}

# 安装监控代理
docker run -d --name node-exporter \
  -p 9100:9100 \
  --restart unless-stopped \
  prom/node-exporter
```

---

## 📊 **性能监控和调优**

### **1. FHDP特定监控指标**

#### **Prometheus配置**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'jetson-nodes'
    static_configs:
      - targets: 
        - 'jetson-01:9100'
        - 'jetson-02:9100'
        - 'jetson-03:9100'
    metrics_path: /metrics
    params:
      format: ['prometheus']
      
  - job_name: 'fhdp-system'
    static_configs:
      - targets: ['pc-01:8080']
    metrics_path: /metrics
    scrape_interval: 5s
    
  - job_name: 'jetson-gpu'
    static_configs:
      - targets: ['jetson-01:9101', 'jetson-02:9101']
    scrape_interval: 2s
```

#### **Grafana仪表板**
```json
{
  "dashboard": {
    "title": "FHDP集群性能监控",
    "panels": [
      {
        "title": "流水线吞吐量",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fhdp_pipeline_completed_total[5m])",
            "legendFormat": "{{vehicle_id}}"
          }
        ]
      },
      {
        "title": "GPU利用率 (Jetson)",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "网络带宽使用",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(network_transmit_bytes_total[5m])",
            "legendFormat": "{{interface}} - TX"
          }
        ]
      }
    ]
  }
}
```

### **2. 自动化测试和基准**

#### **持续集成测试**
```yaml
# .github/workflows/test.yml
name: FHDP集成测试

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: 运行单元测试
        run: python -m pytest tests/unit/
        
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: 部署测试集群
        run: |
          docker-compose -f test-cluster.yml up -d
          sleep 30
          
      - name: 运行集成测试
        run: |
          python -m pytest tests/integration/
          
      - name: 清理测试环境
        run: docker-compose -f test-cluster.yml down
        
  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - name: 性能基准测试
        run: python scripts/benchmark.py
```

---

## 🔐 **安全和访问控制**

### **1. 网络安全配置**

#### **防火墙规则 (UFW)**
```bash
#!/bin/bash
# configure_firewall.sh

# 基础规则
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允许SSH管理
sudo ufw allow 22/tcp

# 允许k3s集群通信
sudo ufw allow 6443/tcp  # Kubernetes API
sudo ufw allow 8472/udp  # Flannel VXLAN
sudo ufw allow 10250/tcp # Kubelet

# 允许监控端口
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3000/tcp  # Grafana

# 允许WireGuard
sudo ufw allow 51820/udp

# 启用防火墙
sudo ufw --force enable
```

#### **证书管理 (certbot)**
```bash
#!/bin/bash
# setup_ssl.sh

# 安装certbot
sudo apt install certbot python3-certbot-nginx

# 为Grafana生成SSL证书
sudo certbot --nginx -d grafana.testbed.local

# 为API服务器生成证书
sudo certbot certonly --standalone -d api.testbed.local

# 自动续期
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

---

## 📚 **学习路径和资源**

### **1. 核心技能学习路径**

#### **阶段1：基础容器化 (2-3周)**
```yaml
Week 1: Docker基础
  - [ ] Docker基础概念和命令
  - [ ] Dockerfile编写最佳实践
  - [ ] Docker Compose多容器应用
  - [ ] 镜像构建和优化

推荐资源：
  - Docker官方教程
  - "Docker Deep Dive" book
  - 实践：构建FHDP应用镜像

Week 2: Kubernetes基础
  - [ ] Kubernetes架构和核心概念
  - [ ] Pod, Service, Deployment
  - [ ] ConfigMap和Secret管理
  - [ ] k3s安装和配置

推荐资源：
  - "Kubernetes in Action"
  - k3s官方文档
  - 实践：部署FHDP到k3s集群

Week 3: 高级Kubernetes
  - [ ] Ingress和网络策略
  - [ ] 存储卷和持久化
  - [ ] Helm包管理
  - [ ] 监控和日志收集

推荐资源：
  - Helm官方文档
  - "Kubernetes Monitoring"指南
```

#### **阶段2：集群管理 (2-3周)**
```yaml
Week 4-5: Ansible自动化
  - [ ] Ansible基础语法和模块
  - [ ] Playbook编写和角色组织
  - [ ] AWX/Tower使用
  - [ ] 实践：自动化部署FHDP集群

推荐资源：
  - "Ansible Up & Running"
  - Ansible官方最佳实践
  - AWX文档和教程

Week 6: 网络和VPN
  - [ ] WireGuard配置和管理
  - [ ] ZeroTier网络管理
  - [ ] 网络故障排查
  - [ ] 实践：构建VPN测试网络

推荐资源：
  - WireGuard官方文档
  - 网络基础教程
```

#### **阶段3：监控和运维 (2周)**
```yaml
Week 7: 监控系统
  - [ ] Prometheus指标收集
  - [ ] Grafana仪表板设计
  - [ ] 告警规则配置
  - [ ] 实践：FHDP性能监控

Week 8: 日志和故障排查
  - [ ] ELK/Loki日志聚合
  - [ ] 日志分析和查询
  - [ ] 系统故障排查
  - [ ] 实践：FHDP日志分析

推荐资源：
  - "Prometheus: Up & Running"
  - Grafana官方教程
  - Linux性能调优指南
```

### **2. Jetson特定技能**

#### **Jetson优化专题 (1-2周)**
```yaml
Week 9: Jetson平台特性
  - [ ] NVIDIA Jetson SDK和JetPack
  - [ ] GPU加速和CUDA编程
  - [ ] 电源管理和性能模式
  - [ ] 实践：优化FHDP在Jetson上性能

推荐资源：
  - NVIDIA Jetson官方文档
  - "Jetson Programming Guide"
  - CUDA编程教程
```

### **3. 实践项目建议**

#### **项目1：最小可用集群 (1周)**
- 搭建2节点k3s集群 (1个PC + 1个Jetson)
- 部署基础的FHDP应用
- 配置基础监控
- 验证基本功能

#### **项目2：完整测试环境 (2周)**
- 4节点Jetson集群 + 3节点PC集群
- 完整的CI/CD流水线
- 全面的监控和日志系统
- 性能基准测试

#### **项目3：生产就绪环境 (3-4周)**
- 高可用配置
- 安全加固和访问控制
- 自动化运维脚本
- 文档和操作手册

---

## 🛠️ **部署时间线**

### **Phase 1: 基础设施 (1-2周)**
```
Week 1: 硬件采购和网络配置
Week 2: 基础软件安装和VPN配置
```

### **Phase 2: 集群搭建 (1-2周)**
```
Week 3: k3s集群部署和验证
Week 4: Ansible自动化配置
```

### **Phase 3: 监控和测试 (1周)**
```
Week 5: 监控系统部署和FHDP集成测试
```

### **Phase 4: 优化和文档 (1周)**
```
Week 6: 性能调优和文档编写
```

---

## 📋 **检查清单**

### **硬件检查**
- [ ] Jetson Orin Nano ×4 (含电源、存储)
- [ ] x86 PC ×3 (满足最低配置要求)
- [ ] 10GbE网络交换机
- [ ] 备用电源和UPS
- [ ] 散热和机架设备

### **软件检查**
- [ ] Docker版本兼容性
- [ ] k3s ARM64支持
- [ ] 操作系统镜像和备份
- [ ] 许可证和合规检查

### **网络检查**
- [ ] IP地址规划和VLAN配置
- [ ] 防火墙规则和访问控制
- [ ] VPN证书和密钥管理
- [ ] 带宽和延迟测试

### **安全检查**
- [ ] SSH密钥配置
- [ ] 用户权限管理
- [ ] 数据加密和备份
- [ ] 安全审计配置

---

## 🎯 **预期成果**

完成此测试平台后，将具备：

1. **完整的FHDP异构集群环境**：支持Jetson和PC混合部署
2. **自动化运维能力**：一键部署、配置管理、监控告警
3. **性能基准测试能力**：系统化性能评估和优化
4. **可扩展架构**：支持从开发到生产的完整生命周期
5. **文档和培训材料**：便于团队协作和知识传承

这个测试平台将成为FHDP系统的强大支撑环境，为算法优化、性能调优和系统验证提供坚实的基础。