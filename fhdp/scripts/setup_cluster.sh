#!/bin/bash
# FHDP集群自动化部署脚本
# 支持：Jetson Orin Nano + x86 PC集群
# 依赖：Ansible, Docker, k3s

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [[ $(uname -m) == "aarch64" ]]; then
            ARCH="arm64"
        else
            ARCH="amd64"
        fi
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker未安装，将自动安装..."
        install_docker
    else
        log_success "Docker已安装: $(docker --version)"
    fi
    
    # 检查Ansible
    if ! command -v ansible &> /dev/null; then
        log_warning "Ansible未安装，将自动安装..."
        install_ansible
    else
        log_success "Ansible已安装: $(ansible --version | head -n1)"
    fi
    
    # 检查系统资源
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    TOTAL_CPU=$(nproc)
    
    log_info "系统资源:"
    log_info "  - CPU: $TOTAL_CPU cores"
    log_info "  - Memory: ${TOTAL_MEM}MB"
    log_info "  - Architecture: $ARCH"
    
    if [[ $TOTAL_MEM -lt 4096 ]]; then
        log_warning "内存少于4GB，可能影响性能"
    fi
}

# 安装Docker
install_docker() {
    log_info "安装Docker..."
    
    if [[ "$ARCH" == "arm64" ]]; then
        # Jetson特定安装
        sudo apt update
        sudo apt install -y docker.io docker-compose
    else
        # 标准Linux安装
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
    fi
    
    # 添加用户到docker组
    sudo usermod -aG docker $USER
    sudo systemctl enable docker
    sudo systemctl start docker
    
    log_success "Docker安装完成"
}

# 安装Ansible
install_ansible() {
    log_info "安装Ansible..."
    
    sudo apt update
    sudo apt install -y python3-pip python3-dev
    pip3 install ansible ansible-core
    
    log_success "Ansible安装完成"
}

# 安装k3s
install_k3s() {
    local role=$1  # "server" or "agent"
    local server_url=$2
    local token=$3
    
    log_info "安装k3s ($role)..."
    
    if [[ "$role" == "server" ]]; then
        # 安装k3s server
        curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker --tls-san $(hostname)" sh -
        
        # 获取token
        K3S_TOKEN=$(sudo cat /var/lib/rancher/k3s/server/node-token)
        echo "K3S_TOKEN=$K3S_TOKEN" > ~/.k3s_token
        log_info "K3s Token已保存到 ~/.k3s_token"
        
        # 安装kubectl
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/$ARCH/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        
    else
        # 安装k3s agent
        if [[ -z "$server_url" || -z "$token" ]]; then
            log_error "Agent模式需要server_url和token参数"
            exit 1
        fi
        
        curl -sfL https://get.k3s.io | K3S_URL="https://$server_url:6443" K3S_TOKEN="$token" INSTALL_K3S_EXEC="--docker" sh -
    fi
    
    log_success "k3s安装完成"
}

# 配置网络
configure_networking() {
    log_info "配置网络..."
    
    # 创建专用网络配置
    sudo tee /etc/netplan/99-fhdp-network.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: true
      optional: true
    eth1:  # 如果有第二张网卡用于集群通信
      dhcp4: false
      addresses: [192.168.100.10/24]
      optional: true
EOF
    
    # 应用网络配置
    if command -v netplan &> /dev/null; then
        sudo netplan apply
    fi
    
    # 配置防火墙
    sudo ufw --force reset
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # 允许必要端口
    sudo ufw allow 22/tcp          # SSH
    sudo ufw allow 6443/tcp        # Kubernetes API
    sudo ufw allow 8472/udp        # Flannel VXLAN
    sudo ufw allow 10250/tcp       # Kubelet
    sudo ufw allow 30000-32767/tcp # NodePort services
    
    sudo ufw --force enable
    log_success "网络配置完成"
}

# 配置SSH密钥
setup_ssh_keys() {
    log_info "配置SSH密钥..."
    
    if [[ ! -f ~/.ssh/id_rsa ]]; then
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    fi
    
    log_info "SSH公钥内容 (复制到其他节点):"
    cat ~/.ssh/id_rsa.pub
    echo
}

# 安装监控工具
install_monitoring() {
    log_info "安装监控工具..."
    
    # 创建监控目录
    mkdir -p ~/fhdp-monitoring/{prometheus,grafana,loki}
    
    # 下载docker-compose配置
    cat > ~/fhdp-monitoring/docker-compose.yml <<'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: fhdp-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:9.3.0
    container_name: fhdp-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped

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
    restart: unless-stopped

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

volumes:
  prometheus_data:
  grafana_data:
EOF
    
    # 创建Prometheus配置
    mkdir -p ~/fhdp-monitoring/prometheus
    cat > ~/fhdp-monitoring/prometheus/prometheus.yml <<'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['host.docker.internal:8080']

  - job_name: 'fhdp-cluster'
    static_configs:
      - targets: 
        - '192.168.100.11:9100'  # jetson-01
        - '192.168.100.12:9100'  # jetson-02
        - '192.168.100.13:9100'  # jetson-03
        - '192.168.100.14:9100'  # jetson-04
        - '192.168.100.21:9100'  # pc-01
        - '192.168.100.22:9100'  # pc-02
        - '192.168.100.23:9100'  # pc-03
    metrics_path: /metrics
    scrape_interval: 5s
EOF
    
    log_success "监控工具配置完成"
    log_info "启动监控: cd ~/fhdp-monitoring && docker-compose up -d"
}

# 配置Jetson特定优化
optimize_jetson() {
    if [[ ! -f /etc/nv_tegra_release ]]; then
        log_info "不是Jetson设备，跳过Jetson优化"
        return
    fi
    
    log_info "配置Jetson特定优化..."
    
    # 创建性能优化脚本
    sudo tee /usr/local/bin/optimize-jetson.sh > /dev/null <<'EOF'
#!/bin/bash
# Jetson性能优化脚本

# 设置最大性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 配置风扇控制
echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm

# 设置CPU调度器
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 优化内存管理
echo 100 | sudo tee /proc/sys/vm/swappiness
echo 1 | sudo tee /proc/sys/vm/drop_caches

# 禁用不必要的服务
sudo systemctl disable nvphs
sudo systemctl disable nvfan-control
EOF
    
    sudo chmod +x /usr/local/bin/optimize-jetson.sh
    
    # 创建systemd服务
    sudo tee /etc/systemd/system/jetson-optimization.service > /dev/null <<EOF
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
    
    sudo systemctl enable jetson-optimization
    
    # 立即应用优化
    sudo /usr/local/bin/optimize-jetson.sh
    
    log_success "Jetson优化配置完成"
}

# 部署FHDP应用
deploy_fhdp() {
    log_info "部署FHDP应用到Kubernetes..."
    
    # 创建namespace
    kubectl create namespace fhdp --dry-run=client -o yaml | kubectl apply -f -
    
    # 创建ConfigMap
    kubectl create configmap fhdp-config \
        --from-file=config/default_config.yaml \
        --namespace=fhdp \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # 部署FHDP核心服务
    cat > ~/fhdp-deployment.yml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fhdp-edge-server
  namespace: fhdp
  labels:
    app: fhdp-edge-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fhdp-edge-server
  template:
    metadata:
      labels:
        app: fhdp-edge-server
    spec:
      containers:
      - name: fhdp-server
        image: fhdp/edge-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: MAX_VEHICLES
          value: "100"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: fhdp-edge-server-service
  namespace: fhdp
spec:
  selector:
    app: fhdp-edge-server
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: LoadBalancer
EOF
    
    kubectl apply -f ~/fhdp-deployment.yml
    
    log_success "FHDP应用部署完成"
    log_info "查看状态: kubectl get pods -n fhdp"
}

# 验证集群状态
verify_cluster() {
    log_info "验证集群状态..."
    
    # 检查k3s状态
    if systemctl is-active --quiet k3s; then
        log_success "k3s服务运行正常"
    else
        log_error "k3s服务未运行"
        return 1
    fi
    
    # 检查节点状态
    kubectl get nodes
    kubectl get pods --all-namespaces
    
    # 检查Docker状态
    if systemctl is-active --quiet docker; then
        log_success "Docker服务运行正常"
    else
        log_error "Docker服务未运行"
        return 1
    fi
    
    log_success "集群验证完成"
}

# 生成集群配置报告
generate_report() {
    log_info "生成集群配置报告..."
    
    cat > ~/fhdp-cluster-report.txt <<EOF
FHDP集群配置报告
生成时间: $(date)

=== 系统信息 ===
操作系统: $(uname -a)
架构: $(uname -m)
内存: $(free -h | grep Mem)
存储: $(df -h /)

=== Docker信息 ===
Docker版本: $(docker --version)
容器数量: $(docker ps -q | wc -l)
镜像数量: $(docker images -q | wc -l)

=== Kubernetes信息 ===
k3s状态: $(systemctl is-active k3s)
Kubernetes版本: $(kubectl version --client --short)
节点数量: $(kubectl get nodes --no-headers | wc -l)

=== 网络信息 ===
IP地址: $(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}')
防火墙状态: $(sudo ufw status | head -n1)

=== FHDP服务状态 ===
命名空间: $(kubectl get namespace fhdp --ignore-not-found)
Pod状态: $(kubectl get pods -n fhdp --ignore-not-found --no-headers | wc -l)

=== 监控服务 ===
Prometheus: $(docker ps --filter "name=prometheus" --format "table {{.Names}}\t{{.Status}}" | grep -v NAMES)
Grafana: $(docker ps --filter "name=grafana" --format "table {{.Names}}\t{{.Status}}" | grep -v NAMES)

=== 访问地址 ===
Grafana: http://$(hostname -I | awk '{print $1}'):3000 (admin/admin123)
Prometheus: http://$(hostname -I | awk '{print $1}'):9090
Kubernetes Dashboard: kubectl proxy --address=0.0.0.0 --accept-hosts='^*$'

EOF
    
    log_success "报告已生成: ~/fhdp-cluster-report.txt"
    cat ~/fhdp-cluster-report.txt
}

# 主函数
main() {
    echo "FHDP集群自动化部署脚本"
    echo "========================"
    echo
    
    # 解析命令行参数
    ROLE=${1:-"server"}
    SERVER_URL=${2:-""}
    TOKEN=${3:-""}
    
    if [[ "$ROLE" != "server" && "$ROLE" != "agent" ]]; then
        echo "用法: $0 [server|agent] [server_url] [token]"
        echo "示例:"
        echo "  $0 server                    # 安装k3s master节点"
        echo "  $0 agent 192.168.100.10 token  # 安装k3s worker节点"
        exit 1
    fi
    
    # 执行部署步骤
    check_system_requirements
    configure_networking
    setup_ssh_keys
    
    if [[ "$ROLE" == "server" ]]; then
        install_k3s server
        install_monitoring
        sleep 10
        deploy_fhdp
    else
        install_k3s agent "$SERVER_URL" "$TOKEN"
    fi
    
    optimize_jetson
    verify_cluster
    generate_report
    
    log_success "FHDP集群部署完成！"
    
    if [[ "$ROLE" == "server" ]]; then
        echo
        log_info "下一步操作:"
        echo "1. 复制K3s Token到其他节点: cat ~/.k3s_token"
        echo "2. 在其他节点运行: sudo bash setup_cluster.sh agent <server_ip> <token>"
        echo "3. 启动监控: cd ~/fhdp-monitoring && docker-compose up -d"
        echo "4. 访问Grafana: http://$(hostname -I | awk '{print $1}'):3000"
    fi
}

# 运行主函数
main "$@"