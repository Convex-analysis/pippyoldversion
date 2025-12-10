#!/bin/bash
# FHDP监控和日志系统自动配置脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker和Docker Compose
check_prerequisites() {
    log_info "检查系统依赖..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    log_success "系统依赖检查完成"
}

# 创建监控目录结构
create_directories() {
    log_info "创建监控目录结构..."
    
    MONITORING_DIR="$HOME/fhdp-monitoring"
    mkdir -p "$MONITORING_DIR"/{prometheus,grafana/provisioning/{dashboards,datasources},loki,alertmanager,promtail}
    
    # 设置权限
    chown -R $USER:$USER "$MONITORING_DIR"
    
    log_success "监控目录创建完成: $MONITORING_DIR"
}

# 生成Docker Compose配置
generate_docker_compose() {
    local monitoring_dir=$1
    log_info "生成Docker Compose配置..."
    
    cat > "$monitoring_dir/docker-compose.yml" <<'EOF'
version: '3.8'

services:
  # Prometheus - 指标收集
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: fhdp-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/rules:/etc/prometheus/rules
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    restart: unless-stopped
    networks:
      - monitoring

  # Grafana - 可视化
  grafana:
    image: grafana/grafana:9.3.0
    container_name: fhdp-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=fhdp123!
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    restart: unless-stopped
    networks:
      - monitoring
    depends_on:
      - prometheus

  # Node Exporter - 系统指标
  node-exporter:
    image: prom/node-exporter:v1.5.0
    container_name: fhdp-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /:/host:ro,rslave
    command:
      - '--path.rootfs=/host'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
      - '--collector.processes'
    restart: unless-stopped
    networks:
      - monitoring

  # cAdvisor - 容器指标
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.46.0
    container_name: fhdp-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    restart: unless-stopped
    networks:
      - monitoring

  # Alertmanager - 告警管理
  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: fhdp-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    restart: unless-stopped
    networks:
      - monitoring

  # Loki - 日志聚合
  loki:
    image: grafana/loki:2.7.0
    container_name: fhdp-loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki/loki.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped
    networks:
      - monitoring

  # Promtail - 日志收集
  promtail:
    image: grafana/promtail:2.7.0
    container_name: fhdp-promtail
    ports:
      - "9080:9080"
    volumes:
      - ./promtail/promtail.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    restart: unless-stopped
    networks:
      - monitoring

  # Pushgateway - 批量任务指标
  pushgateway:
    image: prom/pushgateway:v1.5.1
    container_name: fhdp-pushgateway
    ports:
      - "9091:9091"
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
  loki_data:

networks:
  monitoring:
    driver: bridge
EOF

    log_success "Docker Compose配置生成完成"
}

# 生成Prometheus配置
generate_prometheus_config() {
    local monitoring_dir=$1
    log_info "生成Prometheus配置..."
    
    # 主配置文件
    cat > "$monitoring_dir/prometheus/prometheus.yml" <<'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'fhdp-testbed'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus自监控
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # 系统监控
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']
    scrape_interval: 10s

  # 容器监控
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['host.docker.internal:8080']
    scrape_interval: 10s

  # FHDP集群监控 (需要根据实际IP调整)
  - job_name: 'fhdp-jetson-cluster'
    static_configs:
      - targets: 
        - '192.168.100.11:9100'  # jetson-01
        - '192.168.100.12:9100'  # jetson-02
        - '192.168.100.13:9100'  # jetson-03
        - '192.168.100.14:9100'  # jetson-04
    metrics_path: /metrics
    scrape_interval: 5s
    scrape_timeout: 5s

  - job_name: 'fhdp-pc-cluster'
    static_configs:
      - targets:
        - '192.168.100.21:9100'  # pc-01
        - '192.168.100.22:9100'  # pc-02
        - '192.168.100.23:9100'  # pc-03
    metrics_path: /metrics
    scrape_interval: 5s

  # FHDP应用监控
  - job_name: 'fhdp-edge-servers'
    static_configs:
      - targets:
        - '192.168.100.21:8080'  # pc-01: edge server
        - '192.168.100.22:8080'  # pc-02: edge server
    metrics_path: /metrics
    scrape_interval: 3s
    scrape_timeout: 3s

  # Kubernetes API Server (如果在k3s节点上)
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
    - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
    - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
      action: keep
      regex: default;kubernetes;https

  # 推送网关
  - job_name: 'pushgateway'
    honor_labels: true
    static_configs:
      - targets: ['pushgateway:9091']
EOF

    # 告警规则
    mkdir -p "$monitoring_dir/prometheus/rules"
    cat > "$monitoring_dir/prometheus/rules/fhdp.yml" <<'EOF'
groups:
  - name: fhdp_cluster_health
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space on {{ $labels.instance }}"
          description: "Disk space is below 10%"

      - alert: NodeDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Node {{ $labels.instance }} is down"
          description: "Node has been down for more than 2 minutes"

      - alert: FHDPServiceDown
        expr: up{job=~"fhdp-.*"} == 0
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "FHDP service {{ $labels.job }} on {{ $labels.instance }} is down"
          description: "FHDP service has been down for more than 3 minutes"

      - alert: HighNetworkLatency
        expr: probe_duration_seconds > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High network latency to {{ $labels.instance }}"
          description: "Network latency is above 100ms for more than 5 minutes"

  - name: fhdp_pipeline_metrics
    rules:
      - alert: LowPipelineThroughput
        expr: rate(fhdp_pipeline_completed_total[5m]) < 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low pipeline throughput"
          description: "Pipeline completion rate is below 0.1 per second for 10 minutes"

      - alert: HighPipelineFailureRate
        expr: rate(fhdp_pipeline_failed_total[5m]) / rate(fhdp_pipeline_completed_total[5m]) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High pipeline failure rate"
          description: "Pipeline failure rate is above 20% for 5 minutes"

      - alert: GPUUtilizationHigh
        expr: nvidia_gpu_utilization_gpu > 90
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High GPU utilization on {{ $labels.instance }}"
          description: "GPU utilization is above 90% for more than 10 minutes"
EOF

    log_success "Prometheus配置生成完成"
}

# 生成Alertmanager配置
generate_alertmanager_config() {
    local monitoring_dir=$1
    log_info "生成Alertmanager配置..."
    
    cat > "$monitoring_dir/alertmanager/alertmanager.yml" <<'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@fhdp.local'
  smtp_auth_username: 'alerts@fhdp.local'
  smtp_auth_password: 'your_password'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
    - url: 'http://127.0.0.1:5001/'

  - name: 'critical-alerts'
    email_configs:
    - to: 'admin@fhdp.local'
      subject: '[CRITICAL] FHDP Alert: {{ .GroupLabels.alertname }}'
      body: |
        {{ range .Alerts }}
        Alert: {{ .Annotations.summary }}
        Description: {{ .Annotations.description }}
        {{ end }}
    webhook_configs:
    - url: 'http://127.0.0.1:5001/critical'

  - name: 'warning-alerts'
    email_configs:
    - to: 'ops@fhdp.local'
      subject: '[WARNING] FHDP Alert: {{ .GroupLabels.alertname }}'
      body: |
        {{ range .Alerts }}
        Alert: {{ .Annotations.summary }}
        Description: {{ .Annotations.description }}
        {{ end }}
    webhook_configs:
    - url: 'http://127.0.0.1:5001/warning'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
EOF

    log_success "Alertmanager配置生成完成"
}

# 生成Grafana配置
generate_grafana_config() {
    local monitoring_dir=$1
    log_info "生成Grafana配置..."
    
    # 数据源配置
    cat > "$monitoring_dir/grafana/provisioning/datasources/prometheus.yml" <<'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: 5s
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      maxLines: 1000
    editable: true

  - name: Alertmanager
    type: alertmanager
    access: proxy
    url: http://alertmanager:9093
    jsonData:
      implementation: prometheus
    editable: true
EOF

    # 仪表板配置
    cat > "$monitoring_dir/grafana/provisioning/dashboards/dashboards.yml" <<'EOF'
apiVersion: 1

providers:
  - name: 'FHDP Dashboards'
    orgId: 1
    folder: 'FHDP'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    log_success "Grafana配置生成完成"
}

# 生成Loki配置
generate_loki_config() {
    local monitoring_dir=$1
    log_info "生成Loki配置..."
    
    cat > "$monitoring_dir/loki/loki.yml" <<'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF

    # Promtail配置
    cat > "$monitoring_dir/promtail/promtail.yml" <<'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log

    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          expressions:
            tag:
          source: attrs
      - regex:
          expression: (?P<container_name>(?:[^|]*))\|
          source: tag
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
      - output:
          source: output

  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*log

  - job_name: fhdp
    static_configs:
      - targets:
          - localhost
        labels:
          job: fhdp
          __path__: /var/log/fhdp/*.log
EOF

    log_success "Loki配置生成完成"
}

# 生成FHDP仪表板
generate_fhdp_dashboard() {
    local monitoring_dir=$1
    log_info "生成FHDP Grafana仪表板..."
    
    mkdir -p "$monitoring_dir/grafana/dashboards"
    
    cat > "$monitoring_dir/grafana/dashboards/fhdp-overview.json" <<'EOF'
{
  "dashboard": {
    "id": null,
    "title": "FHDP集群概览",
    "tags": ["fhdp", "cluster"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "活跃节点数量",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"fhdp-.*\"}",
            "legendFormat": "{{ instance }}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {
                "options": {
                  "0": {
                    "text": "DOWN",
                    "color": "red"
                  },
                  "1": {
                    "text": "UP",
                    "color": "green"
                  }
                },
                "type": "value"
              }
            ],
            "thresholds": {
              "steps": [
                {
                  "color": "green",
                  "value": null
                },
                {
                  "color": "red",
                  "value": 80
                }
              ]
            }
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "集群CPU使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{ instance }}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "max": 100,
            "min": 0,
            "unit": "percent"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "内存使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "{{ instance }}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "max": 100,
            "min": 0,
            "unit": "percent"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "网络带宽",
        "type": "graph",
        "targets": [
          {
            "expr": "irate(node_network_receive_bytes_total[5m]) * 8",
            "legendFormat": "{{ instance }} - RX",
            "refId": "A"
          },
          {
            "expr": "irate(node_network_transmit_bytes_total[5m]) * 8",
            "legendFormat": "{{ instance }} - TX",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "unit": "bps"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 16
        }
      },
      {
        "id": 5,
        "title": "磁盘使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_filesystem_size_bytes{mountpoint=\"/\"} - node_filesystem_avail_bytes{mountpoint=\"/:\"}) / node_filesystem_size_bytes{mountpoint=\"/\"} * 100",
            "legendFormat": "{{ instance }}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "max": 100,
            "min": 0,
            "unit": "percent"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 16
        }
      },
      {
        "id": 6,
        "title": "FHDP流水线吞吐量",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fhdp_pipeline_completed_total[5m])",
            "legendFormat": "{{ vehicle_id }}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "unit": "reqps"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 24
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    log_success "FHDP仪表板生成完成"
}

# 生成管理脚本
generate_management_scripts() {
    local monitoring_dir=$1
    log_info "生成管理脚本..."
    
    # 启动脚本
    cat > "$monitoring_dir/start.sh" <<'EOF'
#!/bin/bash
echo "启动FHDP监控系统..."
docker-compose up -d

echo "等待服务启动..."
sleep 30

echo "检查服务状态..."
docker-compose ps

echo
echo "访问地址:"
echo "Grafana: http://localhost:3000 (admin/fhdp123!)"
echo "Prometheus: http://localhost:9090"
echo "Alertmanager: http://localhost:9093"
echo "Loki: http://localhost:3100"
EOF

    # 停止脚本
    cat > "$monitoring_dir/stop.sh" <<'EOF'
#!/bin/bash
echo "停止FHDP监控系统..."
docker-compose down

echo "清理数据卷 (可选)?"
read -p "删除所有数据? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose down -v
    echo "所有数据已清理"
fi
EOF

    # 备份脚本
    cat > "$monitoring_dir/backup.sh" <<'EOF'
#!/bin/bash
BACKUP_DIR="$HOME/fhdp-monitoring-backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

echo "备份监控数据..."

# 备份Prometheus数据
docker exec fhdp-prometheus tar -czf - /prometheus | gzip > "$BACKUP_DIR/prometheus_$DATE.tar.gz"

# 备份Grafana配置
docker exec fhdp-grafana tar -czf - /var/lib/grafana | gzip > "$BACKUP_DIR/grafana_$DATE.tar.gz"

# 备份配置文件
tar -czf "$BACKUP_DIR/configs_$DATE.tar.gz" prometheus grafana alertmanager

echo "备份完成: $BACKUP_DIR"
EOF

    # 设置执行权限
    chmod +x "$monitoring_dir"/*.sh
    
    log_success "管理脚本生成完成"
}

# 验证配置
verify_config() {
    local monitoring_dir=$1
    log_info "验证监控配置..."
    
    # 检查配置文件语法
    log_info "检查Prometheus配置..."
    docker run --rm -v "$monitoring_dir/prometheus:/etc/prometheus" prom/prometheus:v2.40.0 --config.file=/etc/prometheus/prometheus.yml --dry-run
    
    log_info "检查Alertmanager配置..."
    docker run --rm -v "$monitoring_dir/alertmanager:/etc/alertmanager" prom/alertmanager:v0.25.0 --config.file=/etc/alertmanager/alertmanager.yml --dry-run
    
    log_info "检查Loki配置..."
    docker run --rm -v "$monitoring_dir/loki:/etc/loki" grafana/loki:2.7.0 --config.file=/etc/loki/local-config.yaml --dry-run
    
    log_success "配置验证完成"
}

# 主函数
main() {
    echo "FHDP监控系统自动配置脚本"
    echo "=========================="
    echo
    
    MONITORING_DIR="$HOME/fhdp-monitoring"
    
    check_prerequisites
    create_directories
    generate_docker_compose "$MONITORING_DIR"
    generate_prometheus_config "$MONITORING_DIR"
    generate_alertmanager_config "$MONITORING_DIR"
    generate_grafana_config "$MONITORING_DIR"
    generate_loki_config "$MONITORING_DIR"
    generate_fhdp_dashboard "$MONITORING_DIR"
    generate_management_scripts "$MONITORING_DIR"
    verify_config "$MONITORING_DIR"
    
    echo
    log_success "FHDP监控系统配置完成！"
    echo
    log_info "下一步操作:"
    echo "1. 进入监控目录: cd $MONITORING_DIR"
    echo "2. 启动监控系统: ./start.sh"
    echo "3. 访问Grafana: http://localhost:3000 (admin/fhdp123!)"
    echo "4. 查看状态: docker-compose ps"
    echo "5. 查看日志: docker-compose logs -f [service_name]"
    echo
    log_info "注意: 请根据实际网络环境调整prometheus.yml中的节点IP地址"
}

# 运行主函数
main "$@"